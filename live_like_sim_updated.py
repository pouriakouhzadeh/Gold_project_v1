#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Live-like simulator for the user's gold project.
#
# What it does
# ------------
# • Replays the last N (e.g. 4000) closed 30‑minute candles as if MetaTrader were
#   feeding the model in real‑time.
# • For each step `t`, it writes four *temporary* CSVs (M5/M15/M30/H1) containing
#   data up to and including candle `t` (as MT4 would).
# • It then loads & cleans them via PREPARE_DATA_FOR_TRAIN, **drops the last row**
#   (to eliminate the unstable row at `t`), and calls `ready_incremental(...)`
#   to produce a single, batch‑identical feature row for `t−1`.
# • Uses the saved model (best_model.pkl) to predict with calibrated probabilities,
#   applies saved thresholds (pos_thr / neg_thr), and compares with the *true*
#   label built from the main timeframe close: y(t−1) = 1{close_t > close_{t−1}}.
# • Prints running stats (wins/losses/unpred), Balanced Accuracy, and coverage.
#
# Design notes
# ------------
# • This script **reuses** one PREPARE_DATA_FOR_TRAIN instance across the loop so
#   `ready_incremental` can maintain its internal two‑row buffer.
# • It assumes your raw CSV filenames match the project's defaults:
#   XAUUSD_M5.csv, XAUUSD_M15.csv, XAUUSD_M30.csv, XAUUSD_H1.csv
#   (case‑sensitive, with a 'time' column in each). If your symbol differs,
#   pass --symbol accordingly (it only affects log tags).
# • If your ModelSaver stored different keys, the loader tries several fallbacks.
#
# Usage
# -----
# python3 live_like_sim_updated.py \
#   --base-data-dir /home/pouria/gold_project9 \
#   --model /home/pouria/gold_project9/best_model.pkl \
#   --n-test 4000 \
#   --fast-mode 1 \
#   --audit 200 \
#   --log-file live_like_real.log

import os
import sys
import gc
import json
import time
import shutil
import pickle
import argparse
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd

# Project imports
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

# ---------------------------- Logging ----------------------------

def setup_logger(log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("live_like_sim")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    # Clear existing handlers (if re-run)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

# ----------------------- Model loader ----------------------------

def load_model_bundle(path: str):
    """
    Loads best_model.pkl saved by ModelSaver().save_full(...).

    Returns:
        pipeline  : the sklearn/imb pipeline (fitted)
        window    : int
        neg_thr   : float
        pos_thr   : float
        train_cols: list[str] or None
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    bundle = {}
    if isinstance(obj, dict):
        bundle = obj
        pipeline = (
            bundle.get("pipeline")
            or bundle.get("model")
            or bundle.get("clf")
            or bundle.get("estimator")
        )
    else:
        # Plain pipeline pickled
        pipeline = obj

    if pipeline is None:
        raise ValueError("Could not find a fitted pipeline inside best_model.pkl")

    window = int(bundle.get("window_size") or bundle.get("window") or 1)
    neg_thr = float(bundle.get("neg_thr", 0.5))
    pos_thr = float(bundle.get("pos_thr", 0.5))

    # Columns as used at TRAIN TIME (post-window, with _tminus* suffixes)
    train_cols = (
        bundle.get("train_window_cols")
        or bundle.get("final_cols")
        or bundle.get("cols")
        or bundle.get("feats")
        or None
    )
    if train_cols is not None and not isinstance(train_cols, list):
        train_cols = list(train_cols)

    return pipeline, window, neg_thr, pos_thr, train_cols

# ------------------- Data helpers (CSV IO) -----------------------

DEFAULT_FILES = {
    "5T":  "XAUUSD_M5.csv",
    "15T": "XAUUSD_M15.csv",
    "30T": "XAUUSD_M30.csv",  # main TF
    "1H":  "XAUUSD_H1.csv",
}

def _read_raw_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"'time' column missing in {csv_path}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df

def write_truncated_csvs(base_dir: str, out_dir: str, t_end: pd.Timestamp) -> dict[str, str]:
    """
    Reads the 4 raw timeframe CSVs from base_dir, filters rows where time<=t_end,
    writes them into out_dir using the project’s default filenames.

    Returns a filepaths dict usable by PREPARE_DATA_FOR_TRAIN(filepaths=...).
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for tf, fname in DEFAULT_FILES.items():
        src = os.path.join(base_dir, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing required CSV: {src}")
        df = _read_raw_csv(src)
        df = df[df["time"] <= t_end]
        dst = os.path.join(out_dir, fname)
        # Overwrite each step (atomic write via temp then replace to avoid partials)
        tmp = dst + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, dst)
        paths[tf] = dst
    return paths

# -------------------------- Metrics ------------------------------

class RollingStats:
    def __init__(self) -> None:
        self.total = 0
        self.unpred = 0
        self.pred = 0
        self.correct = 0
        self.incorrect = 0
        # Confusion on predicted subset only
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, y_true: int | None, y_pred: int) -> None:
        self.total += 1
        if y_pred == -1 or (y_true is None):
            # treat unknown y as unpred (skip contributing to confusion)
            self.unpred += 1
            return
        self.pred += 1
        if y_pred == y_true:
            self.correct += 1
        else:
            self.incorrect += 1
        # For balanced accuracy, y in {0,1}
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 1 and y_pred == 0:
            self.fn += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1

    def coverage(self) -> float:
        return 0.0 if self.total == 0 else self.pred / self.total

    def bal_acc(self) -> float:
        # Balanced accuracy on the predicted subset
        pos_den = self.tp + self.fn
        neg_den = self.tn + self.fp
        if pos_den == 0 or neg_den == 0:
            return 0.0
        tpr = self.tp / max(1, pos_den)
        tnr = self.tn / max(1, neg_den)
        return 0.5 * (tpr + tnr)

    def summary(self) -> str:
        return (f"size={self.total} · conf={self.coverage():.2f} · "
                f"BalAcc={self.bal_acc():.4f} · "
                f"Correct={self.correct} Incorrect={self.incorrect} Unpredict={self.unpred}")

# --------------------------- Main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Live-like simulator for gold project")
    ap.add_argument("--base-data-dir", required=True, help="Folder containing raw CSVs (XAUUSD_M*.csv)")
    ap.add_argument("--model", required=True, help="best_model.pkl")
    ap.add_argument("--n-test", type=int, default=600, help="How many last 30T bars to replay")
    ap.add_argument("--symbol", default="XAUUSD", help="Symbol tag for logs")
    ap.add_argument("--fast-mode", type=int, default=1, help="Set 1 to skip drift scan & bad-col detection (faster)")
    ap.add_argument("--audit", type=int, default=100, help="Print every k steps")
    ap.add_argument("--log-file", default="", help="Optional log file")
    ap.add_argument("--tmp-dir", default="./_live_tmp", help="Temp directory for per-step CSVs")
    args = ap.parse_args()

    logger = setup_logger(args.log_file if args.log_file else None)

    # ---- Load model & metadata ----
    pipeline, window, neg_thr, pos_thr, train_cols = load_model_bundle(args.model)
    if train_cols is None:
        logger.warning("Train column list not found in model bundle; "
                       "the script will fall back to 'as is' columns from PREP.")
    logger.info("==> Loaded model: window=%d · neg_thr=%.3f · pos_thr=%.3f", window, neg_thr, pos_thr)

    # ---- Build a reference FULL merged dataset once, to compute y_true by timestamp ----
    # Use the original base CSVs directly (no truncation), in fast mode for speed.
    ref_prep = PREPARE_DATA_FOR_TRAIN(filepaths=None, main_timeframe="30T",
                                      verbose=False, fast_mode=bool(args.fast_mode))
    full_df = ref_prep.load_data()
    tcol = "30T_time"
    close_col = "30T_close"
    if tcol not in full_df.columns or close_col not in full_df.columns:
        raise RuntimeError(f"Missing required columns in merged data: {tcol}, {close_col}")
    full_df[tcol] = pd.to_datetime(full_df[tcol], errors="coerce")
    y_full = ((full_df[close_col].shift(-1) - full_df[close_col]) > 0).astype(int)
    y_by_time = pd.Series(y_full.values, index=full_df[tcol].values)

    # We'll replay the last N indices of the FULL merged dataset.
    n_total = len(full_df)
    if args.n_test > n_total - 10:
        args.n_test = max(1, n_total - 10)
        logger.warning("n-test too large for dataset; clipped to %d", args.n_test)

    start_idx = n_total - args.n_test
    logger.info("==> Starting live-like replay: last %d bars (idx from %d to %d)",
                args.n_test, start_idx, n_total - 1)

    # Prepare one PREP instance that points to *temporary* per-step CSVs.
    # We overwrite those CSVs at each step, so the same PREP object can be reused.
    os.makedirs(args.tmp_dir, exist_ok=True)
    tmp_paths = {tf: os.path.join(args.tmp_dir, fn) for tf, fn in DEFAULT_FILES.items()}
    live_prep = PREPARE_DATA_FOR_TRAIN(filepaths=tmp_paths, main_timeframe="30T",
                                       verbose=False, fast_mode=bool(args.fast_mode))

    stats = RollingStats()
    step = 0

    try:
        for i in range(start_idx, n_total):
            step += 1
            t_end = pd.to_datetime(full_df.iloc[i][tcol])
            # 1) Write truncated timeframe CSVs up to and including t_end
            write_truncated_csvs(args.base_data_dir, args.tmp_dir, t_end)

            # 2) Load & process (merge) them → then drop last row (unstable `t`), keep up to `t-1`
            merged = live_prep.load_data()
            if merged.empty or len(merged) < 3:
                # need at least a few rows to form diffs/windows
                stats.update(None, -1)
                continue

            merged = merged.iloc[:-1].copy()  # drop `t` (unstable); now last row is `t-1`
            if merged.empty:
                stats.update(None, -1)
                continue

            # 3) Ready incremental → returns last 1xN features (aligned to `t-1`)
            X_step, feats = live_prep.ready_incremental(
                data_window=merged,
                window=window,
                selected_features=train_cols if train_cols else []
            )

            if X_step is None or X_step.empty:
                # likely the first call; buffer warm-up
                stats.update(None, -1)
                continue

            # Ensure we only pass columns used at train time (order preserved)
            if train_cols:
                cols_used = [c for c in train_cols if c in X_step.columns]
                if not cols_used:
                    # fallback to all available
                    cols_used = list(X_step.columns)
            else:
                cols_used = list(X_step.columns)

            X_used = X_step[cols_used]

            # 4) Predict with thresholds
            prob = float(pipeline.predict_proba(X_used)[:, 1][0])
            if prob <= neg_thr:
                pred = 0
            elif prob >= pos_thr:
                pred = 1
            else:
                pred = -1

            # 5) True label for `t-1` (the time of the last row in `merged`)
            t_pred_time = pd.to_datetime(merged.iloc[-1][tcol])
            y_true = int(y_by_time.get(t_pred_time)) if t_pred_time in y_by_time.index else None

            stats.update(y_true, pred)

            if args.audit and (step % args.audit == 0):
                logging.getLogger("live_like_sim").info(
                    "[%s] step=%d time=%s prob=%.3f pred=%s y=%s · %s",
                    args.symbol, step, t_pred_time, prob,
                    {-1:"∅",0:"0",1:"1"}[pred],
                    ("_" if y_true is None else str(y_true)),
                    stats.summary()
                )

            # 6) Periodic GC & cleanup (temp files are constantly overwritten; no need to delete here)
            if (step % 500) == 0:
                gc.collect()

        # Final summary
        logging.getLogger("live_like_sim").info("==> Finished.\n%s", stats.summary())

    finally:
        # Clean temp directory to leave no trace
        try:
            shutil.rmtree(args.tmp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
