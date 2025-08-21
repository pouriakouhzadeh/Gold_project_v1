#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live-like rolling simulation (windowed) with 4 timeframes.
- At each iteration:
  1) Take a rolling window of the last N rows (default 4000) up to a cutoff time (main TF = 30T)
  2) Write 4 temp CSVs (30T, 15T, 5T, 1H) filtered to [start_time .. cutoff_time]
  3) Run PREPARE_DATA_FOR_TRAIN in predict mode with predict_drop_last=True
  4) Take the LAST row prob, apply thresholds -> decision {0, 1, ∅}
  5) Compare to true label of the NEXT interval [cutoff -> cutoff+1] from main 30T data
  6) Print per-iteration metrics and running summary

Requirements:
- Model artifacts saved by ModelSaver().save_full(...)
- Your prepare_data_for_train.py uses the "next-interval" label logic you just implemented.
"""

from __future__ import annotations
import os, sys, shutil, json, math, tempfile, warnings, argparse, logging
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Project imports (must be in PYTHONPATH) ---
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from ModelSaver import ModelSaver     # relies on your existing implementation

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live-like rolling backtest")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9",
                   help="Base directory where raw CSVs exist (XAUUSD_M30.csv, ...)")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix of CSV files")
    p.add_argument("--model-dir", default=".", help="Directory containing saved model artifacts")
    p.add_argument("--window-rows", type=int, default=4000, help="Rows per rolling window (main TF)")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Limit number of iterations (for quick runs). Default: all possible.")
    p.add_argument("--keep-tmp", action="store_true", help="Keep temp CSVs for debugging")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

# ------------------------- Helpers -------------------------
TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}

def expect_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'time' column is datetime and expected raw columns exist."""
    if "time" not in df.columns:
        # Try common alternatives if needed (add more if your files differ)
        for cand in ("Time", "timestamp", "datetime", "Date"):
            if cand in df.columns:
                df = df.rename(columns={cand: "time"})
                break
    if "time" not in df.columns:
        raise KeyError("CSV must contain a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def cut_by_time(df: pd.DataFrame, start, end) -> pd.DataFrame:
    m = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[m].copy()

def load_artifacts(model_dir: str):
    """
    Load everything saved by your ModelSaver.save_full(...)
    Expected keys (by your GA code): pipeline, hyperparams, window_size, neg_thr, pos_thr,
    feats, feat_mask, train_window_cols
    """
    ms = ModelSaver()
    arte = ms.load_full(model_dir)   # Assumes your ModelSaver has a symmetric load_full
    # Normalise keys
    pipeline = arte.get("pipeline") or arte.get("model") or arte.get("pipeline_obj")
    window   = int(arte.get("window_size") or arte.get("window") or 1)
    neg_thr  = float(arte.get("neg_thr", 0.5))
    pos_thr  = float(arte.get("pos_thr", 0.5))
    final_cols = arte.get("train_window_cols") or arte.get("final_cols") or arte.get("feats") or []
    if not isinstance(final_cols, list):
        final_cols = list(final_cols)
    return pipeline, window, neg_thr, pos_thr, final_cols

def decide(prob: float, neg_thr: float, pos_thr: float):
    """Return -1 (∅), 0 (short/cash), or 1 (long)."""
    if prob <= neg_thr:
        return 0
    if prob >= pos_thr:
        return 1
    return -1

# ------------------------- Main -------------------------
def main():
    args = parse_args()
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    log = logging.getLogger("live-like")

    # --- Resolve base CSV paths (your file names: XAUUSD_M30.csv, ... in data-dir) ---
    base_csvs = {
        "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
        "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
        "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
        "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
    }

    # --- Read full raw CSVs once ---
    raw_df: Dict[str, pd.DataFrame] = {}
    for tf, path in base_csvs.items():
        if not os.path.isfile(path):
            log.warning("Missing CSV for %s: %s (this TF will be skipped)", tf, path)
            continue
        df = pd.read_csv(path)
        df = expect_cols(df)
        raw_df[tf] = df

    # Require main timeframe 30T
    if "30T" not in raw_df:
        log.error("Main timeframe 30T CSV is required but missing.")
        sys.exit(1)

    main_df = raw_df["30T"]

    # --- Load model artefacts ---
    try:
        pipeline, window, neg_thr, pos_thr, final_cols = load_artifacts(args.model_dir)
    except Exception as e:
        log.exception("Failed to load model artifacts from %s", args.model_dir)
        sys.exit(1)

    log.info("Model loaded: window=%d, thr=(neg=%.3f,pos=%.3f), final_cols=%d",
             window, neg_thr, pos_thr, len(final_cols))

    # --- Iteration bounds on main TF (need next bar for true label) ---
    N = len(main_df)
    if N < args.window_rows + 2:
        log.error("Not enough rows in main TF: have %d, need >= %d", N, args.window_rows + 2)
        sys.exit(1)

    start_idx = args.window_rows - 1                   # inclusive
    end_idx   = N - 2                                  # we need i+1 for true label
    total_iters = (end_idx - start_idx + 1)
    if args.max_iters is not None:
        total_iters = min(total_iters, args.max_iters)

    # --- Accumulators ---
    wins = losses = unpred = preds = 0

    # --- Temp directory for CSVs per iteration ---
    tmp_root = tempfile.mkdtemp(prefix="live_like_")
    log.info("Temp root: %s", tmp_root)

    try:
        for k in range(total_iters):
            i = start_idx + k          # cutoff index in 30T
            cutoff_time = pd.to_datetime(main_df.loc[i, "time"])
            start_time  = pd.to_datetime(main_df.loc[i - (args.window_rows - 1), "time"])

            # 1) Write 4 timeframe CSVs limited to [start_time .. cutoff_time]
            # Filenames must match prepare_data_for_train expectations: *_30T.csv etc.
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            os.makedirs(iter_dir, exist_ok=True)
            tmp_paths = {}
            for tf, df in raw_df.items():
                sub = cut_by_time(df, start_time, cutoff_time)
                if sub.empty:
                    continue
                out_name = f"{args.symbol}_{tf}.csv"   # e.g., XAUUSD_30T.csv
                out_path = os.path.join(iter_dir, out_name)
                # Ensure 'time' column is kept and first
                cols = ["time"] + [c for c in sub.columns if c != "time"]
                sub[cols].to_csv(out_path, index=False)
                tmp_paths[tf] = out_path

            if "30T" not in tmp_paths:
                # No main tf rows → skip
                continue

            # 2) Prepare with PREPARE_DATA_FOR_TRAIN in predict mode
            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths=tmp_paths, main_timeframe="30T",
                verbose=False, fast_mode=True, strict_disk_feed=True
            )
            merged = prep.load_data()
            if merged.empty:
                continue

            # ensure there is at least window rows to form features:
            # We'll let ready() handle windowing; we want full X and then take last row's prob.
            X_live, _, _, _ = prep.ready(
                merged,
                window=window,
                selected_features=final_cols,
                mode="predict",
                predict_drop_last=True  # ← remove the unstable last (cutoff) row
            )
            if X_live.empty:
                # Not enough rows to construct stable features
                continue

            # 3) Predict last stable row prob (corresponds to previous bar → predicts [cutoff → cutoff+1])
            try:
                probs = pipeline.predict_proba(X_live)[:, 1]
            except Exception:
                # Some pipelines need reshape/columns; ensure column order matches final_cols
                X_live = X_live[final_cols]
                probs = pipeline.predict_proba(X_live)[:, 1]
            p_last = float(probs[-1])

            # Decision
            pred = decide(p_last, neg_thr, pos_thr)

            # 4) True label for [cutoff → cutoff+1] from main_df
            # y_true = 1 if close[i+1] > close[i] else 0
            c0 = float(main_df.loc[i,   "close"])
            c1 = float(main_df.loc[i+1, "close"])
            y_true = 1 if (c1 - c0) > 0 else 0

            # 5) Update metrics and print
            if pred == -1:
                unpred += 1
                verdict = "UNPRED"
            else:
                preds += 1
                if pred == y_true:
                    wins += 1
                    verdict = "WIN"
                else:
                    losses += 1
                    verdict = "LOSS"

            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            print(
                f"[{k+1:>5}/{total_iters}] @ {cutoff_time}  "
                f"p={p_last:.3f} → pred={pred if pred!=-1 else '∅'}  "
                f"true={y_true}  → {verdict}   "
                f"| cum P={preds} W={wins} L={losses} U={unpred} Acc={acc:.2f}%"
            )

            # 6) Cleanup iteration folder unless debugging requested
            if not args.keep_tmp:
                shutil.rmtree(iter_dir, ignore_errors=True)

        # Final summary
        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        print("\n========== SUMMARY ==========")
        print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
        print(f"Accuracy (predicted only): {acc:.2f}%")
        print("================================")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()



# python3 live_like_sim.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9   \
#   --window-rows 4000 \
#   --max-iters 500    \
#   --verbose
