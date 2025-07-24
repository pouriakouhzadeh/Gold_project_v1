#!/usr/bin/env python3
"""
realtime_live_backtest_fixed.py  ─────────────────────────────────────────────
Fixed version of consistency test (batch‑vs‑live) for GA‑trained model.

Key Fixes:
~~~~~~~~~~
* Uses batch processing for LIVE simulation instead of incremental
* Ensures identical feature calculation for both batch and live
* Unified data cleaning and imputation
* Improved timestamp alignment
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("best_model.pkl")
RAW_FILES = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T": "XAUUSD_M5.csv",
    "1H": "XAUUSD_H1.csv",
}
MAIN_TIMEFRAME = "30T"
TIME_COL = f"{MAIN_TIMEFRAME}_time"
SNAPSHOT_CSV = "chimney_snapshot.csv"
LIVE_CSV = "live_snapshot.csv"
COMPARE_CSV = "comparison_differences.csv"

STRICT = False   # if True ⇒ abort on any suspect row/column

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("tester")
logging.getLogger("prepare_data_for_train").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def align_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    """Add missing columns (filled with NaN) & reorder."""
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[ordered_cols]

def ensure_time_col(df: pd.DataFrame, tag: str) -> None:
    """Guarantee TIME_COL exists or use first *_time column."""
    global TIME_COL
    if TIME_COL in df.columns:
        return
    alts = [c for c in df.columns if c.endswith("_time")]
    if not alts:
        raise KeyError(f"[{tag}] No *_time column found!")
    LOG.warning("⚠️  %s: Using '%s' instead of '%s'", tag, alts[0], TIME_COL)
    df.rename(columns={alts[0]: TIME_COL}, inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# Main routine - FIXED VERSION
# ══════════════════════════════════════════════════════════════════════════════

def main(max_rows: int | None = None) -> None:
    # 1️⃣ Load model and configuration
    if not MODEL_PATH.exists():
        LOG.error("Model file %s not found!", MODEL_PATH)
        sys.exit(1)

    mdl = joblib.load(MODEL_PATH)
    pipe = mdl["pipeline"]
    window = int(mdl["window_size"])
    feats = list(mdl["feats"])
    final_cols = list(mdl.get("train_window_cols", feats))
    neg_thr = float(mdl["neg_thr"])
    pos_thr = float(mdl["pos_thr"])
    
    # Calculate required padding
    import re
    lookbacks = [int(m.group(1)) for c in final_cols 
                 if (m := re.search(r"_(\d{1,3})(?:$|_)", c))]
    max_lb = max(lookbacks) if lookbacks else 1
    PAD = window + max_lb + 10  # Extra buffer for safety
    LOG.info("📏 max-lookback=%d | PAD=%d", max_lb, PAD)

    # 2️⃣ Load and prepare data
    from prepare_data_for_train_deepseek import PREPARE_DATA_FOR_TRAIN
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={tf: str(Path(fp)) for tf, fp in RAW_FILES.items()},
        main_timeframe=MAIN_TIMEFRAME,
        verbose=False,
    )
    merged = prep.load_data()
    ensure_time_col(merged, "merged")
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    merged.sort_values(TIME_COL, inplace=True, ignore_index=True)

    # Determine rows to process
    if not max_rows or max_rows <= 0:
        max_rows = min(4000, len(merged) - PAD)
    required = max_rows + PAD
    if len(merged) < required:
        LOG.error("❌ Need %d rows but only %d available", required, len(merged))
        sys.exit(1)
    
    merged = merged.tail(required).reset_index(drop=True)
    LOG.info("📂 Merged data: %d rows, %d cols", *merged.shape)

    # 3️⃣ Batch (Chimney) processing
    X_batch, y_batch, _, _ = prep.ready(
        merged, 
        window=window, 
        selected_features=feats, 
        mode="train"
    )
    
    if X_batch.empty:
        LOG.error("❌ Batch processing returned empty feature matrix")
        sys.exit(1)
    
    X_batch = align_columns(X_batch, final_cols).astype("float32")
    X_batch.replace([np.inf, -np.inf], np.nan, inplace=True)
    batch_medians = X_batch.median()
    X_batch.fillna(batch_medians, inplace=True)
    
    times_batch = merged[TIME_COL].iloc[window:window+len(X_batch)].reset_index(drop=True)
    
    proba_batch = pipe.predict_proba(X_batch)[:, 1]
    labels_batch = np.select(
        [proba_batch <= neg_thr, proba_batch >= pos_thr],
        [0, 1],
        default=-1
    )
    
    # Save batch results
    batch_df = pd.concat([
        X_batch,
        pd.Series(times_batch, name=TIME_COL),
        pd.Series(proba_batch, name="proba"),
        pd.Series(labels_batch, name="label")
    ], axis=1)
    batch_df.to_csv(SNAPSHOT_CSV, index=False)
    
    valid_mask = labels_batch != -1
    if valid_mask.any():
        acc_batch = accuracy_score(y_batch[valid_mask], labels_batch[valid_mask])
        f1_batch = f1_score(y_batch[valid_mask], labels_batch[valid_mask])
    else:
        acc_batch = f1_batch = 0.0
        
    LOG.info("🏭 BATCH metrics | Acc=%.4f | F1=%.4f | decided=%d/%d",
             acc_batch, f1_batch, valid_mask.sum(), len(labels_batch))

    # 4️⃣ LIVE simulation using batch method - KEY FIX!
    live_data = []
    start_idx = PAD
    end_idx = len(merged) - 1
    
    LOG.info("Starting LIVE simulation (%d iterations)...", end_idx - start_idx)
    
    for idx in range(start_idx, end_idx):
        # Use a sliding window with identical processing to batch
        window_start = max(0, idx - PAD + 1)
        window_end = idx + 1
        window_data = merged.iloc[window_start:window_end].copy()
        
        # Process window with same method as batch
        X_window, _, _, _ = prep.ready(
            window_data, 
            window=window, 
            selected_features=feats, 
            mode="predict"
        )
        
        if X_window.empty:
            continue
            
        X_window = align_columns(X_window, final_cols).astype("float32")
        X_window.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_window.fillna(batch_medians, inplace=True)
        
        # Get only the last row (current prediction)
        X_current = X_window.iloc[[-1]].copy()
        proba = float(pipe.predict_proba(X_current)[0, 1])
        
        label = -1
        if proba <= neg_thr:
            label = 0
        elif proba >= pos_thr:
            label = 1
        
        # Get true label (price movement)
        current_close = merged.at[idx, f"{MAIN_TIMEFRAME}_close"]
        next_close = merged.at[idx + 1, f"{MAIN_TIMEFRAME}_close"]
        y_true = int(next_close > current_close)
        
        live_data.append({
            **X_current.iloc[0].to_dict(),
            TIME_COL: merged.at[idx, TIME_COL].strftime("%Y-%m-%d %H:%M:%S"),
            "proba": proba,
            "label": label,
            "y_true": y_true,
        })
        
        if (idx - start_idx) % 500 == 0:
            LOG.info("⏩ Processed %d/%d live points", idx - start_idx, end_idx - start_idx)
    
    # Save live results
    live_df = pd.DataFrame(live_data)
    if live_df.empty:
        LOG.error("❌ LIVE simulation produced no valid rows")
        sys.exit(1)
        
    live_df.to_csv(LIVE_CSV, index=False)
    
    # Calculate live metrics
    valid_live = live_df[live_df["label"] != -1]
    if not valid_live.empty:
        acc_live = accuracy_score(valid_live["y_true"], valid_live["label"])
        f1_live = f1_score(valid_live["y_true"], valid_live["label"])
    else:
        acc_live = f1_live = 0.0
        
    LOG.info("🎯 LIVE metrics | Acc=%.4f | F1=%.4f | decided=%d/%d",
             acc_live, f1_live, len(valid_live), len(live_df))

    # 5️⃣ Comparison and analysis
    batch_df = pd.read_csv(SNAPSHOT_CSV)
    live_df = pd.read_csv(LIVE_CSV)
    
    ensure_time_col(batch_df, "batch")
    ensure_time_col(live_df, "live")
    
    batch_df[TIME_COL] = pd.to_datetime(batch_df[TIME_COL])
    live_df[TIME_COL] = pd.to_datetime(live_df[TIME_COL])
    
    # Merge on timestamps
    comparison = pd.merge(
        batch_df, 
        live_df,
        on=TIME_COL,
        suffixes=('_batch', '_live'),
        how='inner'
    )
    
    if comparison.empty:
        LOG.error("❌ No common timestamps for comparison")
        sys.exit(1)
        
    # Find differences
    feature_diffs = []
    label_diffs = 0
    
    for _, row in comparison.iterrows():
        # Compare features
        for col in final_cols:
            if col not in row:
                continue
                
            batch_val = row[f"{col}_batch"]
            live_val = row[f"{col}_live"]
            
            if (
                pd.isna(batch_val) != pd.isna(live_val) or
                (not pd.isna(batch_val) and 
                 not np.isclose(batch_val, live_val, atol=1e-5, rtol=1e-3))
            ):
                feature_diffs.append(col)
                
        # Compare labels
        if row["label_batch"] != row["label_live"]:
            label_diffs += 1
            
    # Save comparison results
    diff_report = {
        "total_points": len(comparison),
        "label_mismatches": label_diffs,
        "mismatch_ratio": label_diffs / len(comparison),
        "common_features": len(set(final_cols) & set(comparison.columns)),
        "different_features": len(set(feature_diffs)),
        "top_different_features": pd.Series(feature_diffs).value_counts().head(20).to_dict()
    }
    with open(COMPARE_CSV, "w", encoding="utf-8") as f:
        json.dump(diff_report, f, indent=2, ensure_ascii=False)    
            
    LOG.info("📊 COMPARISON RESULTS:")
    LOG.info("   - Total points: %d", diff_report["total_points"])
    LOG.info("   - Label mismatches: %d (%.2f%%)", 
             diff_report["label_mismatches"],
             100 * diff_report["mismatch_ratio"])
    LOG.info("   - Features with differences: %d", 
             diff_report["different_features"])
    
    if diff_report["different_features"] > 0:
        LOG.info("   - Top different features:")
        for feat, count in diff_report["top_different_features"].items():
            LOG.info("      - %s: %d mismatches", feat, count)

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed Batch vs Live Consistency Test"
    )
    parser.add_argument("--max-rows", type=int, default=-1,
                        help="Number of rows to process (default: auto)")
    args = parser.parse_args()
    main(args.max_rows)