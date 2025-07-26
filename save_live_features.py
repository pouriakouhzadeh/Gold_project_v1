#!/usr/bin/env python3
"""
save_live_features.py
=====================

Creates a *one‑row* snapshot (`X_live_snapshot.csv`) of the latest feature
matrix **X_live** built from the live CSV candles (M5 / M15 / M30 / H1).

Key points
----------
*   Verifies that the model (`best_model.pkl` by default) and all four live
    CSV files exist before continuing.
*   Re‑builds the feature matrix with PREPARE_DATA_FOR_TRAIN so that the
    snapshot is **exactly** what the online predictor would receive.
*   Fills NaNs with zero and re‑orders columns to match the original
    training layout (`train_window_cols`).
*   Logs every step to the console (DEBUG level – drop to INFO if too chatty)
    and exits with code 1 on any fatal error.
"""

import argparse
import sys
import os
import logging
import joblib
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ──────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,                            # change to INFO if needed
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate a one‑row CSV snapshot of X_live for debugging."
)
parser.add_argument("-m", "--model", default="best_model.pkl",
                    help="Path to the trained model .pkl file")
parser.add_argument("-o", "--out", default="X_live_snapshot.csv",
                    help="Output CSV filename")
parser.add_argument("-d", "--live-dir", default=".",
                    help="Directory containing live CSV files (default: cwd)")
args = parser.parse_args()

MODEL_PATH = args.model
OUT_CSV    = args.out
LIVE_DIR   = args.live_dir.rstrip("/")

logging.info("=== save_live_features.py started ===")

# ──────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)

live_files = {
    "30T": os.path.join(LIVE_DIR, "XAUUSD.F_M30_live.csv"),
    "1H" : os.path.join(LIVE_DIR, "XAUUSD.F_H1_live.csv"),
    "15T": os.path.join(LIVE_DIR, "XAUUSD.F_M15_live.csv"),
    "5T" : os.path.join(LIVE_DIR, "XAUUSD.F_M5_live.csv"),
}
missing = [fp for fp in live_files.values() if not os.path.exists(fp)]
if missing:
    logging.error("Missing live CSV file(s):\n  " + "\n  ".join(missing))
    sys.exit(1)
logging.info("All live CSV files found.")

# ──────────────────────────────────────────
# Load model metadata
# ──────────────────────────────────────────
try:
    saved = joblib.load(MODEL_PATH)
except Exception as e:
    logging.error(f"Could not load model: {e}")
    sys.exit(1)

window_size       = saved["window_size"]
feat_list         = saved["feats"]
train_window_cols = saved["train_window_cols"]
train_raw_window  = saved.get("train_raw_window")

logging.debug(
    f"window_size={window_size}, "
    f"#selected_features={len(feat_list)}, "
    f"#train_window_cols={len(train_window_cols)}"
)

# ──────────────────────────────────────────
# Build PREPARE_DATA_FOR_TRAIN and merge
# ──────────────────────────────────────────
prep = PREPARE_DATA_FOR_TRAIN(filepaths=live_files, main_timeframe="30T")

try:
    merged_df = prep.load_data()
except Exception as e:
    logging.error(f"load_data() failed: {e}")
    sys.exit(1)

# concatenate raw window (if any) so that rolling windows line up
merged_df = merged_df.tail(window_size + 1)

# ──────────────────────────────────────────
# Generate X_live
# ──────────────────────────────────────────
try:
    if window_size == 1:
        X_live, _, _ = prep.ready(
            merged_df, window=1, selected_features=feat_list, mode="predict")
    else:
        X_live, _ = prep.ready_incremental(
            merged_df, window=window_size, selected_features=feat_list)
except Exception as e:
    logging.error(f"Feature preparation failed: {e}")
    sys.exit(1)

if X_live.empty:
    logging.error("X_live is empty – aborting.")
    sys.exit(1)

# Replace NaNs (common after re‑index) and align column order
if X_live.isna().any().any():
    logging.warning("Found NaN values – filling with 0.")
    X_live = X_live.fillna(0)

X_live.columns = [str(c) for c in X_live.columns]          # ensure str type
X_live = X_live.reindex(columns=train_window_cols, fill_value=0).astype(float)

# ──────────────────────────────────────────
# Save snapshot
# ──────────────────────────────────────────
try:
    X_live.tail(1).to_csv(OUT_CSV, index=False)
except Exception as e:
    logging.error(f"Failed to save CSV: {e}")
    sys.exit(1)

logging.info(f"Snapshot successfully written ➜ {OUT_CSV}")
