#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live-like simulator with per-step reporting.
- Mirrors real-time behavior (no look-ahead)
- Prints a progress line at EVERY step with running metrics
- Saves a detailed CSV log: live_like_log.csv

Run as:
  python3 live_like_simulator_verbose.py
"""

from __future__ import annotations
import os
import sys
import csv
import math
import json
import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# ---- project imports ----
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN   # type: ignore
from ModelSaver import ModelSaver                           # type: ignore

# ===================== CONFIG =====================
DATA_DIR     = "/home/pouria/gold_project9"   # ← مسیر همان دیتای Train
SYMBOL       = "XAUUSD"
MAIN_TF      = "30T"
MODEL_PATH   = "best_model.pkl"               # ← خروجی ModelSaver
N_STEPS      = 2000                           # ← تعداد گام‌های انتهایی
PRINT_EVERY  = 1                              # ← هر چند گام یکبار گزارش کند (1 = هر گام)
CSV_LOG_FILE = "live_like_log.csv"            # ← فایل خروجی لاگ
VERBOSE      = True
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger("live_like_sim_verbose")

def _build_filepaths(data_dir: str, symbol: str) -> dict:
    return {
        "30T": f"{data_dir}/{symbol}_M30.csv",
        "15T": f"{data_dir}/{symbol}_M15.csv",
        "5T":  f"{data_dir}/{symbol}_M5.csv",
        "1H":  f"{data_dir}/{symbol}_H1.csv",
    }

@dataclass
class LoadedModel:
    pipeline: any
    window_size: int
    neg_thr: float
    pos_thr: float
    train_window_cols: List[str]
    train_distribution: Optional[str]
    model_dir: str

def load_model_payload(path: str) -> LoadedModel:
    base_dir = os.path.abspath(os.path.dirname(path))
    payload = ModelSaver(filename=os.path.basename(path), model_dir=base_dir).load_full()

    pipeline   = payload["pipeline"]
    window_sz  = int(payload["window_size"])
    neg_thr    = float(payload["neg_thr"])
    pos_thr    = float(payload["pos_thr"])
    cols_order = list(payload.get("train_window_cols") or payload.get("feats") or [])
    dist_name  = payload.get("train_distribution", None)

    LOGGER.info("[model] window_size=%s  neg_thr=%.4f  pos_thr=%.4f  cols=%d",
                window_sz, neg_thr, pos_thr, len(cols_order))

    return LoadedModel(
        pipeline=pipeline,
        window_size=window_sz,
        neg_thr=neg_thr,
        pos_thr=pos_thr,
        train_window_cols=cols_order,
        train_distribution=dist_name,
        model_dir=base_dir
    )

def compute_true_targets(raw: pd.DataFrame, main_tf: str) -> pd.Series:
    close_col = f"{main_tf}_close"
    if close_col not in raw.columns:
        raise KeyError(f"Missing column: {close_col}")
    return (raw[close_col].shift(-1) - raw[close_col] > 0).astype("int64")

def find_true_y_at_time(y_full: pd.Series, t_index: pd.Timestamp, raw: pd.DataFrame, main_tf: str) -> Optional[int]:
    tcol = f"{main_tf}_time" if f"{main_tf}_time" in raw.columns else "time"
    loc = raw.index[raw[tcol] == t_index].tolist()
    if not loc:
        return None
    idx = loc[0]
    if idx >= len(y_full):
        return None
    val = y_full.iloc[idx]
    if pd.isna(val):
        return None
    return int(val)

def classify_with_band(prob1: float, neg_thr: float, pos_thr: float) -> int:
    if prob1 <= neg_thr:
        return 0
    if prob1 >= pos_thr:
        return 1
    return -1  # No-Trade

def main():
    # 1) Load model
    if not os.path.isfile(MODEL_PATH):
        LOGGER.error("Model file not found: %s", os.path.abspath(MODEL_PATH))
        sys.exit(1)
    mdl = load_model_payload(MODEL_PATH)

    # 2) Prepare PREP with same main timeframe & logic
    filepaths = _build_filepaths(DATA_DIR, SYMBOL)
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=MAIN_TF, verbose=False)

    # 3) Load merged data with your stable pipeline (resample-safe etc.)
    raw = prep.load_data()
    tcol = f"{MAIN_TF}_time" if f"{MAIN_TF}_time" in raw.columns else "time"
    raw[tcol] = pd.to_datetime(raw[tcol])
    raw.sort_values(tcol, inplace=True)
    raw.reset_index(drop=True, inplace=True)

    total = len(raw)
    start_t = max(0, total - N_STEPS)
    LOGGER.info("[sim] total=%d, start_index=%d, steps=%d", total, start_t, total - start_t)

    # 4) Full true y over raw
    y_full = compute_true_targets(raw, MAIN_TF)

    # 5) Prepare CSV log
    with open(CSV_LOG_FILE, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "step_idx", "timestamp", "prob1", "pred", "true",
            "is_correct", "cum_trades", "cum_correct", "cum_incorrect",
            "cum_abstain", "running_accuracy_on_trades", "running_balanced_accuracy",
            "coverage"
        ])

        # Accumulators
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        correct_n = incorrect_n = abstain_n = 0

        for t in range(start_t, total):
            window = mdl.window_size
            known = raw.iloc[:t].copy()

            # Build LIVE-like features up to 'known'
            X_full, _, feats, price_raw, t_idx = prep.ready(
                known,
                window=window,
                selected_features=mdl.train_window_cols,  # exact column order
                mode="predict",
                with_times=True,
                predict_drop_last=True
            )

            if X_full.empty or t_idx is None or len(t_idx) == 0:
                continue

            # Feed last available row (what you'd have in real-time)
            X_last = X_full.tail(1).copy()
            t_feat = pd.to_datetime(t_idx.iloc[-1])

            # Predict probability and map to {-1,0,1}
            prob1 = float(mdl.pipeline.predict_proba(X_last)[:, 1][0])
            y_hat = classify_with_band(prob1, mdl.neg_thr, mdl.pos_thr)

            # True label at the same timestamp
            y_true = find_true_y_at_time(y_full, t_feat, raw, MAIN_TF)
            if y_true is None:
                continue

            # Accumulate
            y_true_all.append(y_true)
            y_pred_all.append(y_hat)

            # Update counters
            if y_hat == -1:
                abstain_n += 1
                is_correct = None  # N/A
            else:
                if y_hat == y_true:
                    correct_n += 1
                    is_correct = 1
                else:
                    incorrect_n += 1
                    is_correct = 0

            # Running metrics (on trades only)
            y_true_arr = np.array(y_true_all, dtype=int)
            y_pred_arr = np.array(y_pred_all, dtype=int)
            trade_mask = (y_pred_arr != -1)

            if trade_mask.any():
                run_acc = accuracy_score(y_true_arr[trade_mask], y_pred_arr[trade_mask])
                run_bacc = balanced_accuracy_score(y_true_arr[trade_mask], y_pred_arr[trade_mask])
                coverage = float(trade_mask.mean())
            else:
                run_acc = 0.0
                run_bacc = 0.0
                coverage = 0.0

            cum_trades = int(trade_mask.sum())

            # Console line (every step by default)
            if PRINT_EVERY and ((len(y_true_all) % PRINT_EVERY) == 0):
                LOGGER.info(
                    "step=%d/%d  ts=%s  prob=%.4f  pred=%2d  true=%d  "
                    "trades=%d  correct=%d  incorrect=%d  abstain=%d  "
                    "acc=%.4f  bacc=%.4f  cov=%.2f%%",
                    t, total-1, t_feat, prob1, y_hat, y_true,
                    cum_trades, correct_n, incorrect_n, abstain_n,
                    run_acc, run_bacc, 100.0 * coverage
                )

            # CSV row
            writer.writerow([
                t, t_feat.isoformat(), f"{prob1:.6f}", y_hat, y_true,
                ("" if is_correct is None else is_correct),
                cum_trades, correct_n, incorrect_n, abstain_n,
                f"{run_acc:.6f}", f"{run_bacc:.6f}", f"{coverage:.6f}"
            ])

    # Final summary (read last line from counters)
    LOGGER.info("=== DONE === CSV saved: %s", os.path.abspath(CSV_LOG_FILE))
    LOGGER.info("Note: Final running metrics are printed in the last console line above.")
    LOGGER.info("Tip: open CSV in a spreadsheet to inspect step-by-step behavior.")

if __name__ == "__main__":
    main()
