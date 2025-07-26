#!/usr/bin/env python3
"""
test_live_input.py
==================

Loads a CSV snapshot created with *save_live_features.py*, feeds it through
the stored model, and prints probability + class prediction.

Exits with non‑zero code if anything goes wrong (file missing, transform
error, etc.) so that it can be used in automated tests.
"""

import argparse
import sys
import os
import logging
import joblib
import pandas as pd

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run the stored model on a saved X_live snapshot."
)
parser.add_argument("-m", "--model", default="best_model.pkl",
                    help="Path to the model .pkl file")
parser.add_argument("-c", "--csv", default="X_live_snapshot.csv",
                    help="CSV snapshot produced by save_live_features.py")
args = parser.parse_args()

MODEL_PATH = args.model
CSV_PATH   = args.csv

logging.info("=== test_live_input.py started ===")

# ──────────────────────────────────────────
# File checks
# ──────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model not found: {MODEL_PATH}")
    sys.exit(1)
if not os.path.exists(CSV_PATH):
    logging.error(f"Snapshot CSV not found: {CSV_PATH}")
    sys.exit(1)

# ──────────────────────────────────────────
# Load model
# ──────────────────────────────────────────
try:
    saved = joblib.load(MODEL_PATH)
except Exception as e:
    logging.error(f"Could not load model: {e}")
    sys.exit(1)

pipeline = saved["pipeline"]
scaler   = saved["scaler"]

# ──────────────────────────────────────────
# Load snapshot
# ──────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
if df.empty:
    logging.error("Snapshot CSV is empty.")
    sys.exit(1)

# Ensure numeric dtype
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

X = df.values
if scaler is not None:
    try:
        X = scaler.transform(X)
    except Exception as e:
        logging.error(f"Scaler transform failed: {e}")
        sys.exit(1)

# ──────────────────────────────────────────
# Predict
# ──────────────────────────────────────────
try:
    prob = pipeline.predict_proba(X)[0, 1]
    pred = pipeline.predict(X)[0]
except Exception as e:
    logging.error(f"Model prediction failed: {e}")
    sys.exit(1)

label = "BUY" if pred == 1 else "SEL"
logging.info(f"Probability = {prob:.4f}  |  Prediction = {label} ({pred})")
