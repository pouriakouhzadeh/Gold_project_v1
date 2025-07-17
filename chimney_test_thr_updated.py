#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_like_backtest_updated1.py – Simulated real-time prediction with ground truth and evaluation
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

# Argument parser
cli = argparse.ArgumentParser("Simulated LIVE backtest with label generation")
cli.add_argument("--model", default="best_model.pkl")
cli.add_argument("--data-dir", default=".")
cli.add_argument("--rows", type=int, default=2000)
cli.add_argument("--output", default="live_snapshot.csv")
cli.add_argument("--log", default="live_test.log")
cli.add_argument("--verbose", action="store_true")
args = cli.parse_args()

# Logging
logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(args.log, "w", "utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("live")

# Load model
model_path = Path(args.model).resolve()
if not model_path.exists():
    log.error("Model file not found: %s", model_path)
    sys.exit(1)
mdl = joblib.load(model_path)
pipe = mdl["pipeline"]
window = mdl["window_size"]
feats = mdl["feats"]
all_cols = mdl["train_window_cols"]
neg_thr, pos_thr = mdl["neg_thr"], mdl["pos_thr"]
log.info("Model loaded | window=%d | thr=(%.3f, %.3f)", window, neg_thr, pos_thr)

# Load raw data
RAW = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T": "XAUUSD_M5.csv",
    "1H":  "XAUUSD_H1.csv"
}
filepaths = {tf: Path(args.data_dir) / fname for tf, fname in RAW.items()}
missing = [str(p) for p in filepaths.values() if not p.exists()]
if missing:
    log.error("Missing CSVs: %s", ", ".join(missing))
    sys.exit(1)

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
prep = PREPARE_DATA_FOR_TRAIN(filepaths={k: str(v) for k, v in filepaths.items()}, main_timeframe="30T", verbose=False)

merged = prep.load_data()
main_time_col = f"{prep.main_timeframe}_time"
merged[main_time_col] = pd.to_datetime(merged[main_time_col])
merged.sort_values(main_time_col, inplace=True)
merged.reset_index(drop=True, inplace=True)

# Apply tail limit
if args.rows > 0:
    merged = merged.tail(args.rows + window + 1).reset_index(drop=True)

# Real-time simulation
y_true_all, y_pred_all, y_proba_all = [], [], []
rows_out = []

for i in range(window, len(merged) - 1):  # exclude last row (t+1 not available)
    window_data = merged.iloc[i - window:i + 1].copy()
    X_live, _ = prep.ready_incremental(window_data, window=window, selected_features=feats)

    if X_live.empty:
        continue

    for col in all_cols:
        if col not in X_live.columns:
            X_live[col] = np.nan
    X_live = X_live.reindex(columns=all_cols).astype("float32")

    # Model prediction
    proba = pipe.predict_proba(X_live)[0, 1]
    label = -1
    if proba <= neg_thr:
        label = 0
    elif proba >= pos_thr:
        label = 1

    # Ground truth
    cur_close = merged.iloc[i]["30T_close"]
    next_close = merged.iloc[i + 1]["30T_close"]
    y_true = int(next_close > cur_close)

    # Logs per iteration
    print(f"[{i}] y_true={y_true} | y_proba={proba:.5f} | thr_neg={neg_thr:.3f} | thr_pos={pos_thr:.3f} → pred={label}")

    if label != -1:
        y_true_all.append(y_true)
        y_pred_all.append(label)
        y_proba_all.append(proba)

    row_dict = X_live.iloc[0].to_dict()
    row_dict[main_time_col] = merged.iloc[i][main_time_col].strftime("%Y-%m-%d %H:%M:%S")
    row_dict["proba"] = proba
    row_dict["label"] = label
    row_dict["y_true"] = y_true
    rows_out.append(row_dict)

# Final evaluation
if y_pred_all:
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all)
    print("\n✅ Final Evaluation:")
    print(f"Total predictions: {len(y_pred_all)} / {len(merged) - window - 1}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"F1-score         : {f1:.4f}")
else:
    print("\n⚠️ No confident predictions were made.")

# Save snapshot
df_out = pd.DataFrame(rows_out)
df_out.to_csv(args.output, index=False)
log.info("Saved live snapshot with %d rows to %s", len(df_out), args.output)
