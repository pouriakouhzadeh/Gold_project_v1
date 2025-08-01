#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Chimney vs Live Comparison Tool with Progress Tracking
"""

from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV   

# Project modules
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive

# Configure logging
LOG = logging.getLogger("tester")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comparison_tool.log")
    ]
)

class ProgressTracker:
    """Utility class for tracking and displaying progress"""
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total = total_steps
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.pbar = tqdm(
            total=total_steps,
            desc=description,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
    def update(self, increment: int = 1, status: str = ""):
        """Update progress with optional status message"""
        self.current += increment
        self.pbar.set_postfix_str(status)
        self.pbar.update(increment)
        
    def close(self):
        """Complete the progress tracking"""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        LOG.info(f"{self.description} completed in {elapsed:.2f} seconds")

# ═══════════════════════════════════════════════════════
#          Core Functions with Progress Tracking
# ═══════════════════════════════════════════════════════

def build_live_estimator(fitted_pipe: Pipeline,
                        keep_calibrator: bool = True) -> ModelPipelineLive:
    """Construct live model from trained pipeline with proper calibration"""
    LOG.info("Building live estimator from trained pipeline...")
    scaler = fitted_pipe.named_steps["scaler"]
    cls_trained = fitted_pipe.named_steps["classifier"]

    if isinstance(cls_trained, CalibratedClassifierCV):
        final_clf = cls_trained
        hyper_for_live = cls_trained.estimator.get_params()
    else:
        final_clf = cls_trained
        hyper_for_live = final_clf.get_params()

    live = ModelPipelineLive(hyperparams=hyper_for_live, calibrate=False)
    live.base_pipe = Pipeline([("scaler", scaler), ("clf", final_clf)])

    if keep_calibrator and isinstance(cls_trained, CalibratedClassifierCV):
        live._calibrator = cls_trained

    LOG.info("Live estimator construction complete")
    return live

def generate_predictions(df: pd.DataFrame, model: Pipeline, 
                       window: int, all_cols: List[str],
                       neg_thr: float, pos_thr: float,
                       prep: PREPARE_DATA_FOR_TRAIN) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions with progress tracking"""
    y_true, y_pred = [], []
    close_col = f"{prep.main_timeframe}_close"
    total_rows = len(df) - window - 1
    
    LOG.info(f"Generating predictions for {total_rows} rows...")
    progress = ProgressTracker(total_rows, "Generating predictions")
    
    for idx in range(window, len(df) - 1):
        sub = df.iloc[idx - window : idx + 1].copy().reset_index(drop=True)
        X_inc, _ = prep.ready_incremental(sub, window=window, selected_features=all_cols)
        
        if X_inc.empty:
            progress.update(1, "Skipped (empty)")
            continue

        # Ensure all columns exist
        for c in all_cols:
            if c not in X_inc.columns:
                X_inc[c] = np.nan
        X_inc = X_inc[all_cols].astype("float32")

        # Handle NaN values
        if X_inc.isna().any().any():
            scaler_means = model.base_pipe.named_steps["scaler"].mean_
            mean_dict = {col: scaler_means[i] for i, col in enumerate(all_cols)}
            X_inc = X_inc.fillna(mean_dict)

        # Make prediction
        proba = model.predict_proba(X_inc)[:, 1]
        pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]

        # Store results
        y_true.append(int((df.iloc[idx + 1][close_col] - df.iloc[idx][close_col]) > 0))
        y_pred.append(int(pred))
        
        progress.update(1, f"Row {idx-window+1}/{total_rows}")
    
    progress.close()
    return np.array(y_true), np.array(y_pred)

def report_performance(y_true: np.ndarray, y_pred: np.ndarray, method_name: str):
    """Generate comprehensive performance report"""
    mask_decided = y_pred != -1
    total_predictions = len(y_pred)
    decided_count = int(mask_decided.sum())
    undecided_count = total_predictions - decided_count
    
    report = [
        "\n" + "="*60,
        f"📊 {method_name.upper()} PERFORMANCE REPORT",
        "="*60,
        f"─ Total predictions: {total_predictions}",
        f"─ Decided predictions: {decided_count} ({decided_count/total_predictions*100:.1f}%)",
        f"─ Undecided cases: {undecided_count} ({undecided_count/total_predictions*100:.1f}%)"
    ]
    
    if decided_count > 0:
        acc = accuracy_score(y_true[mask_decided], y_pred[mask_decided])
        f1 = f1_score(y_true[mask_decided], y_pred[mask_decided])
        tn, fp, fn, tp = confusion_matrix(y_true[mask_decided], y_pred[mask_decided]).ravel()
        
        report.extend([
            f"─ Correct predictions: {tp + tn} ({(tp + tn)/decided_count*100:.1f}%)",
            f"─ Incorrect predictions: {fp + fn} ({(fp + fn)/decided_count*100:.1f}%)",
            f"─ Accuracy: {acc:.4f}",
            f"─ F1-score: {f1:.4f}",
            "─ Confusion Matrix:",
            str(confusion_matrix(y_true[mask_decided], y_pred[mask_decided]))
        ])
    else:
        report.append("⚠ No decisions made in this method")
    
    report.append("="*60 + "\n")
    LOG.info("\n".join(report))

def save_snapshot(df_feat: pd.DataFrame, ts: pd.Series, time_col: str,
                 rows_lim: int, out_csv: Path):
    """Save snapshot with row limit and progress tracking"""
    LOG.info(f"Saving snapshot to {out_csv}...")
    if rows_lim and len(df_feat) > rows_lim:
        progress = ProgressTracker(2, "Trimming data")
        df_feat = df_feat.tail(rows_lim).reset_index(drop=True)
        progress.update(1, "Data trimmed")
        ts = ts.tail(rows_lim).reset_index(drop=True)
        progress.update(1, "Timestamps trimmed")
        progress.close()
    
    snap = df_feat.assign(**{time_col: ts.dt.strftime("%Y-%m-%d %H:%M:%S")})
    snap.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved (%d rows, %d cols)", out_csv.name, *snap.shape)

def chimney_snapshot(prep: PREPARE_DATA_FOR_TRAIN, window: int, feats: List[str],
                    all_cols: List[str], start: str | None, rows: int,
                    out_csv: Path, live_est: ModelPipelineLive,
                    neg_thr: float, pos_thr: float):
    """Generate chimney snapshot with performance metrics"""
    LOG.info("▶ Building CHIMNEY snapshot with performance metrics...")
    progress = ProgressTracker(4, "Chimney Snapshot")
    
    merged = prep.load_data()
    progress.update(1, "Data loaded")
    
    X_raw, y_true, _, _ = prep.ready(merged, window=window,
                                   selected_features=feats, mode="train")
    progress.update(1, "Data prepared")

    # Ensure all columns exist
    for c in all_cols:
        if c not in X_raw.columns:
            X_raw[c] = np.nan
    X_raw = X_raw.reindex(columns=all_cols).astype("float32")
    progress.update(1, "Columns verified")

    time_col = f"{prep.main_timeframe}_time"
    ts = merged[time_col].iloc[window:window+len(X_raw)].reset_index(drop=True)

    if start:
        keep = ts >= pd.Timestamp(start)
        X_raw, ts, y_true = X_raw[keep], ts[keep], y_true[keep]
    
    # Generate predictions
    y_pred = []
    pred_progress = ProgressTracker(len(X_raw), "Making predictions")
    for i in range(len(X_raw)):
        X_inc = X_raw.iloc[[i]]
        proba = live_est.predict_proba(X_inc)[:, 1]
        pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]
        y_pred.append(int(pred))
        pred_progress.update(1)
    pred_progress.close()
    
    y_pred = np.array(y_pred)
    report_performance(y_true, y_pred, "chimney method")

    # Save snapshot
    save_snapshot(X_raw, ts, time_col, rows, out_csv)
    progress.update(1, "Snapshot saved")
    progress.close()
    
    return y_true, y_pred

def live_snapshot(prep: PREPARE_DATA_FOR_TRAIN, merged: pd.DataFrame,
                 live_est: ModelPipelineLive, window: int,
                 neg_thr: float, pos_thr: float, all_cols: List[str],
                 start: str | None, rows: int, out_csv: Path):
    """Run live back-test with comprehensive reporting"""
    LOG.info("▶ Running LIVE back-test with full metrics...")
    progress = ProgressTracker(3, "Live Snapshot")
    
    # Generate predictions
    y_true, y_pred = generate_predictions(
        merged, live_est, window, all_cols, neg_thr, pos_thr, prep
    )
    progress.update(1, "Predictions generated")
    
    # Report performance
    report_performance(y_true, y_pred, "live method")
    progress.update(1, "Report generated")

    # Save snapshot
    time_col = f"{prep.main_timeframe}_time"
    ts = merged[time_col].iloc[window:window+len(y_true)].reset_index(drop=True)
    save_snapshot(
        pd.DataFrame({col: merged[col].iloc[window:window+len(y_true)].values 
                    for col in all_cols}),
        ts,
        time_col,
        rows,
        out_csv
    )
    progress.update(1, "Snapshot saved")
    progress.close()
    
    return y_true, y_pred

def diff_snapshots(ref_csv: Path, live_csv: Path, diff_txt: Path,
                  abs_tol: float = 1e-6, rel_tol: float = 1e-9):
    """Compare snapshots with enhanced difference reporting"""
    LOG.info("▶ Comparing snapshots with detailed analysis...")
    progress = ProgressTracker(5, "Comparing Snapshots")
    
    # Load data
    df_ref = pd.read_csv(ref_csv, low_memory=False)
    df_live = pd.read_csv(live_csv, low_memory=False)
    progress.update(1, "Data loaded")

    # Find time column
    time_col = next((c for c in df_ref.columns 
                   if c.endswith("_time") and c in df_live.columns), None)
    if not time_col:
        LOG.error("No common time column found")
        return

    # Clean data
    for df in (df_ref, df_live):
        df[time_col] = pd.to_datetime(df[time_col])
        df.dropna(subset=[time_col], inplace=True)
        df.drop_duplicates(subset=[time_col], keep="last", inplace=True)
    progress.update(1, "Data cleaned")

    merged = df_ref.merge(df_live, on=time_col, how="inner", 
                         suffixes=("_ref", "_live"))
    if merged.empty:
        LOG.error("No overlapping timestamps between snapshots!")
        return
    progress.update(1, "Data merged")

    # Calculate differences
    diff_cols, total_diff, worst_abs = [], 0, 0.0
    diff_values = []
    
    common_cols = [c for c in df_ref.columns 
                  if c != time_col and c in df_live.columns]
    
    compare_progress = ProgressTracker(len(common_cols), "Comparing columns")
    for col in common_cols:
        ref_vals = merged[f"{col}_ref"]
        live_vals = merged[f"{col}_live"]

        if pd.api.types.is_numeric_dtype(ref_vals):
            abs_diff = (ref_vals - live_vals).abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = abs_diff / np.maximum(ref_vals.abs(), live_vals.abs())
            mis = (abs_diff > abs_tol) & (rel_diff > rel_tol)
            
            if mis.any():
                diff_cols.append(col)
                total_diff += int(mis.sum())
                current_max = float(abs_diff[mis].max())
                if current_max > worst_abs:
                    worst_abs = current_max
                    diff_values.append((col, current_max))
        else:
            mis = (ref_vals.fillna("__") != live_vals.fillna("__"))
            if mis.any():
                diff_cols.append(col)
                total_diff += int(mis.sum())
        
        compare_progress.update(1)
    compare_progress.close()
    progress.update(1, "Differences calculated")

    # Save and report differences
    diff_txt.write_text("\n".join(diff_cols) + "\n", encoding="utf-8")
    progress.update(1, "Results saved")
    
    report = [
        "\n" + "="*60,
        "🔍 SNAPSHOT COMPARISON RESULTS",
        "="*60,
        f"─ Total differing cells: {total_diff}",
        f"─ Worst absolute difference: {worst_abs:.6f}"
    ]
    
    if diff_values:
        report.append("─ Top differing columns:")
        for col, val in sorted(diff_values, key=lambda x: x[1], reverse=True)[:5]:
            report.append(f"   • {col}: {val:.6f}")
    
    report.extend([
        f"─ Differing columns count: {len(diff_cols)} (listed in {diff_txt.name})",
        "="*60 + "\n"
    ])
    
    LOG.info("\n".join(report))
    progress.close()

# ═══════════════════════════════════════════════════════
#          Main Execution
# ═══════════════════════════════════════════════════════

def main():
    """Main execution function with proper error handling"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Chimney vs Live Comparison Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="best_model.pkl", help="Trained model file")
    parser.add_argument("--data-dir", default=".", help="Directory containing data files")
    parser.add_argument("--start", help="Start date (e.g., '2025-01-01 00:00')")
    parser.add_argument("--rows", type=int, default=2000, help="Maximum rows to save")
    parser.add_argument("--chimney-snap", default="chimney_snapshot.csv", 
                       help="Chimney output file")
    parser.add_argument("--live-snap", default="live_snapshot.csv", 
                       help="Live output file")
    parser.add_argument("--diff-columns", default="diff_columns.txt", 
                       help="Differing columns output")
    args = parser.parse_args()

    try:
        LOG.info("🚀 Starting comparison tool")
        main_progress = ProgressTracker(5, "Main Process")

        # Load model and configuration
        mdl_path = Path(args.model).resolve()
        if not mdl_path.is_file():
            raise FileNotFoundError(f"Model file {mdl_path} not found")

        payload = joblib.load(mdl_path)
        pipe_fit = payload["pipeline"]
        window = int(payload["window_size"])
        neg_thr = float(payload["neg_thr"])
        pos_thr = float(payload["pos_thr"])
        feats = payload["feats"]
        all_cols = payload["train_window_cols"]
        main_progress.update(1, "Configuration loaded")

        # Initialize models and data preparation
        live_est = build_live_estimator(pipe_fit)
        csv_dir = Path(args.data_dir).resolve()
        prep = PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T",
            filepaths={
                "30T": str(csv_dir / "XAUUSD_M30.csv"),
                "15T": str(csv_dir / "XAUUSD_M15.csv"),
                "5T": str(csv_dir / "XAUUSD_M5.csv"),
                "1H": str(csv_dir / "XAUUSD_H1.csv"),
            },
            verbose=False,
        )
        main_progress.update(1, "Components initialized")

        # Run chimney (batch) method with metrics
        chimney_true, chimney_pred = chimney_snapshot(
            prep, window, feats, all_cols, args.start, args.rows, 
            Path(args.chimney_snap), live_est, neg_thr, pos_thr
        )
        main_progress.update(1, "Chimney complete")

        # Run live method with metrics
        merged_all = prep.load_data()
        live_true, live_pred = live_snapshot(
            prep, merged_all, live_est, window, neg_thr, pos_thr,
            all_cols, args.start, args.rows, Path(args.live_snap)
        )
        main_progress.update(1, "Live complete")

        # Compare results
        diff_snapshots(
            Path(args.chimney_snap),
            Path(args.live_snap),
            Path(args.diff_columns)
        )
        main_progress.update(1, "Comparison complete")
        main_progress.close()

        LOG.info("✅ All operations completed successfully!")
        LOG.info("💾 Logs saved to comparison_tool.log")

    except Exception as e:
        LOG.error("❌ Fatal error during execution: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()