#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# live_like_sim.py
"""
Live-like rolling simulation (windowed) with 4 timeframes.

Loop (for each cutoff):
  1) Take a rolling window of the last N rows (main TF = 30T) up to cutoff_time (inclusive)
  2) Write 4 temp CSVs (30T, 15T, 5T, 1H) filtered to [start_time .. cutoff_time]
  3) PREPARE_DATA_FOR_TRAIN(..., mode="predict", predict_drop_last=True)  ← drop unstable last row
  4) Take the LAST row prob, apply thresholds -> decision {BUY=1, SELL=0, NONE=-1}
  5) Compare to true label of the NEXT interval [cutoff -> cutoff+1] from 30T data
  6) Log per-iteration metrics + running summary

Notes:
- Uses the exact feature columns selected at train-time (train_window_cols / feats).
- Uses strict_disk_feed=True to avoid reading beyond cutoff_time.
"""

from __future__ import annotations
import os, sys, shutil, tempfile, warnings, argparse, logging
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# --- Project imports ---
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live-like rolling backtest")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9",
                   help="Base directory where raw CSVs exist (XAUUSD_M30.csv, ...)")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix of CSV files")
    p.add_argument("--model-dir", default=".", help="Directory containing saved model artifacts (best_model.pkl)")
    p.add_argument("--window-rows", type=int, default=4000,
                   help="Rows per rolling window on main TF (30T).")
    p.add_argument("--tail-iters", type=int, default=4000,
                   help="Run ONLY the last K iterations (e.g., last 4000 predictions).")
    p.add_argument("--keep-tmp", action="store_true", help="Keep per-iteration temp CSVs for debugging")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

# ------------------------- Helpers -------------------------
TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}

def expect_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'time' column exists and is datetime; sort by time."""
    if "time" not in df.columns:
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

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    """Return -1 (NONE), 0 (SELL/CASH), or 1 (BUY)."""
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

    # --- Resolve base CSV paths (XAUUSD_M30.csv, ...) ---
    base_csvs = {
        "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
        "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
        "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
        "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
    }

    # --- Read all raw CSVs once ---
    raw_df: Dict[str, pd.DataFrame] = {}
    for tf, path in base_csvs.items():
        if not os.path.isfile(path):
            log.warning("Missing CSV for %s: %s (skipping this TF)", tf, path)
            continue
        df = pd.read_csv(path)
        df = expect_cols(df)
        raw_df[tf] = df

    if "30T" not in raw_df:
        log.error("Main timeframe (30T) CSV is required but missing.")
        sys.exit(1)

    main_df = raw_df["30T"]

    # --- Load model artefacts (best_model.pkl) ---
    try:
        payload = joblib.load(os.path.join(args.model_dir, "best_model.pkl"))
        pipeline   = payload["pipeline"]
        window     = int(payload.get("window_size", 1))
        neg_thr    = float(payload.get("neg_thr", 0.5))
        pos_thr    = float(payload.get("pos_thr", 0.5))
        final_cols = payload.get("train_window_cols") or payload.get("feats") or []
        if not isinstance(final_cols, list):
            final_cols = list(final_cols)
    except Exception:
        log.exception("Failed to load model artifacts from %s", args.model_dir)
        sys.exit(1)

    log.info("Model loaded: window=%d, thr=(neg=%.3f,pos=%.3f), final_cols=%d",
             window, neg_thr, pos_thr, len(final_cols))

    # --- Bounds on main TF (need next bar for true label) ---
    N = len(main_df)
    need_min = args.window_rows + 2
    if N < need_min:
        log.error("Not enough rows in main TF: have %d, need >= %d", N, need_min)
        sys.exit(1)

    # Base range (earliest possible start after warmup) to last usable index (N-2)
    base_start = args.window_rows - 1
    end_idx    = N - 2

    # If tail-iters is set, shift the start so we simulate the *last* K iterations
    if args.tail_iters is not None and args.tail_iters > 0:
        start_idx = max(base_start, end_idx - args.tail_iters + 1)
    else:
        start_idx = base_start

    total_iters = end_idx - start_idx + 1
    if total_iters <= 0:
        log.error("Computed total_iters <= 0 (start_idx=%d, end_idx=%d)", start_idx, end_idx)
        sys.exit(1)

    # --- Accumulators & running confusion matrix (predicted-only) ---
    wins = losses = unpred = preds = 0
    tp = tn = fp = fn = 0   # predicted-only confusion
    buy_n = sell_n = none_n = 0

    # --- Temp directory per-iteration ---
    tmp_root = tempfile.mkdtemp(prefix="live_like_")
    log.info("Temp root: %s", tmp_root)

    try:
        for k in range(total_iters):
            i = start_idx + k                   # cutoff index in 30T
            cutoff_time = pd.to_datetime(main_df.loc[i, "time"])
            start_time  = pd.to_datetime(main_df.loc[i - (args.window_rows - 1), "time"])

            # 1) Write per-iteration timeframe CSVs limited to [start_time .. cutoff_time]
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            os.makedirs(iter_dir, exist_ok=True)
            tmp_paths = {}
            for tf, df in raw_df.items():
                sub = cut_by_time(df, start_time, cutoff_time)
                if sub.empty:
                    continue
                out_name = f"{args.symbol}_{tf}.csv"   # e.g., XAUUSD_30T.csv
                out_path = os.path.join(iter_dir, out_name)
                cols = ["time"] + [c for c in sub.columns if c != "time"]
                sub[cols].to_csv(out_path, index=False)
                tmp_paths[tf] = out_path

            if "30T" not in tmp_paths:
                # No main-tf rows → skip
                continue

            # 2) Prepare features in predict mode (STRICT, FAST) and DROP the unstable last row
            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths=tmp_paths, main_timeframe="30T",
                verbose=False, fast_mode=True, strict_disk_feed=True
            )
            merged = prep.load_data()
            if merged.empty:
                continue

            X_live, _, _, _ = prep.ready(
                merged,
                window=window,
                selected_features=final_cols,   # must match train columns
                mode="predict",
                predict_drop_last=True          # ← CRUCIAL: drop the unstable cutoff row
            )
            if X_live.empty:
                continue

            # Align columns & predict last stable row
            X_live = X_live.reindex(columns=final_cols, fill_value=0.0)
            try:
                probs = pipeline.predict_proba(X_live)[:, 1]
            except Exception:
                # Some wrappers require DataFrame with same dtypes/index; already ensured columns.
                probs = pipeline.predict_proba(X_live.values)[:, 1]

            p_last = float(probs[-1])
            pred = decide(p_last, neg_thr, pos_thr)

            # 3) True label for [cutoff → cutoff+1] from main_df
            c0 = float(main_df.loc[i,   "close"])
            c1 = float(main_df.loc[i+1, "close"])
            y_true = 1 if (c1 - c0) > 0 else 0

            # 4) Update metrics
            if pred == -1:
                unpred += 1
                none_n += 1
                verdict = "UNPRED"
            else:
                preds += 1
                if pred == 1:
                    buy_n += 1
                else:
                    sell_n += 1

                if pred == y_true:
                    wins += 1
                    verdict = "WIN"
                    if y_true == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    losses += 1
                    verdict = "LOSS"
                    if y_true == 1:
                        fn += 1
                    else:
                        fp += 1

            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            coverage = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0

            # Fancy decision label for console
            dec_label = {1: "BUY ", 0: "SELL", -1: "NONE"}[pred]

            # 5) Print per-iteration line
            print(
                f"[{k+1:>5}/{total_iters}] @ {cutoff_time}  "
                f"p={p_last:.3f} → pred={dec_label}  true={y_true} → {verdict}   "
                f"| cum P={preds} W={wins} L={losses} U={unpred} "
                f"Acc={acc:.2f}% Cover={coverage:.2f}%  "
                f"| buys={buy_n} sells={sell_n} none={none_n}"
            )

            # 6) Cleanup
            if not args.keep_tmp:
                shutil.rmtree(iter_dir, ignore_errors=True)

        # Final summary
        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        coverage = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
        print("\n========== SUMMARY ==========")
        print(f"Cutoffs tested (iterations): {total_iters}")
        print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
        print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {coverage:.2f}%")
        print(f"Decisions  → BUY: {buy_n}  SELL: {sell_n}  NONE: {none_n}")
        print(f"Confusion  → TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
        print("================================\n")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
