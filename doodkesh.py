#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chimney_test_thr.py – smoke-test *with* threshold filtering
────────────────────────────────────────────────────────────
Evaluates the saved model on the **whole** dataset using the current
PREPARE_DATA_FOR_TRAIN implementation but honours neg_thr / pos_thr.
"""

from __future__ import annotations
import argparse, hashlib, logging, os, inspect
from pathlib import Path
from typing import Dict

import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ───────────────────── helpers ─────────────────────
def md5_path(p: str) -> str:
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()[:8]

# ───────────────────── CLI ─────────────────────
cli = argparse.ArgumentParser("Chimney-test with threshold")
cli.add_argument("--model", default="best_model.pkl")
cli.add_argument("--data-dir", default=".")
cli.add_argument("--log",   default="chimney_test_thr.log")
cli.add_argument("--verbose", action="store_true")
args = cli.parse_args()

# ───────────────────── logging ─────────────────────
logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(args.log, "w", "utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("chimney-thr")

# ───────────────────── load model ─────────────────────
mdl  = joblib.load(args.model)
pipe = mdl["pipeline"]
win, feats, cols = mdl["window_size"], mdl["feats"], mdl["train_window_cols"]
neg_thr, pos_thr = mdl["neg_thr"], mdl["pos_thr"]
log.info("Model loaded | window=%d | feats=%d | cols=%d | thr=(%.3f, %.3f)",
         win, len(feats), len(cols), neg_thr, pos_thr)

# ───────────────────── CSV paths ─────────────────────
RAW_FILES: Dict[str, str] = {
    "30T": "XAUUSD_M30.csv", "15T": "XAUUSD_M15.csv",
    "5T" : "XAUUSD_M5.csv",  "1H" : "XAUUSD_H1.csv",
}
filepaths = {tf: os.path.join(args.data_dir, fn) for tf, fn in RAW_FILES.items()}
missing = [fp for fp in filepaths.values() if not os.path.isfile(fp)]
if missing:
    raise FileNotFoundError(f"Missing CSV(s): {missing}")

# ───────────────────── prep instance ─────────────────────
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
prep_src = inspect.getsourcefile(PREPARE_DATA_FOR_TRAIN)
prep_hash = md5_path(prep_src) if prep_src else "unknown"
log.info("PREP src=%s | hash=%s", prep_src, prep_hash)

prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T", verbose=False)

# ───────────────────── build features ─────────────────────
log.info("Building merged dataframe…")
merged = prep.load_data()
X_all, y_all, _ = prep.ready(merged, window=win, selected_features=feats, mode="train")

missing_cols = [c for c in cols if c not in X_all.columns]
extra_cols   = [c for c in X_all.columns if c not in cols]
if missing_cols:
    log.warning("Missing %d columns from training… sample=%s", len(missing_cols), missing_cols[:8])
X_eval = X_all.reindex(columns=cols).astype(float)

# ───────────────────── predict with threshold ─────────────────────
proba = pipe.predict_proba(X_eval)[:, 1]
y_pred = np.full_like(y_all.values, fill_value=-1, dtype=int)
y_pred[proba <= neg_thr] = 0
y_pred[proba >= pos_thr] = 1

mask = y_pred != -1
conf_ratio = mask.mean()
acc = accuracy_score(y_all[mask], y_pred[mask]) if mask.any() else 0.0
f1  = f1_score(y_all[mask], y_pred[mask]) if mask.any() else 0.0

log.info("Eval done | rows=%d | decided=%d (%.2f%%) | Acc=%.4f | F1=%.4f",
         len(y_all), mask.sum(), conf_ratio*100, acc, f1)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rows (total)        : {len(y_all):,}
Rows predicted      : {mask.sum():,}  ({conf_ratio*100:.2f}%)
Accuracy (thr)      : {acc:.4f}
F1-score (thr)      : {f1:.4f}
Thresholds          : neg={neg_thr:.3f}  pos={pos_thr:.3f}
Model file          : {Path(args.model).resolve()}
PREP hash           : {prep_hash}
Log                 : {Path(args.log).resolve()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
