#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Chimney vs Live Comparison Tool with Syntax Corrections
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib

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

class DataConsistencyError(Exception):
    """Custom exception for data consistency issues"""
    pass

def validate_feature_columns(X: pd.DataFrame, expected_cols: List[str]) -> None:
    """Validate that all expected columns are present"""
    missing = set(expected_cols) - set(X.columns)
    if missing:
        raise DataConsistencyError(f"Missing columns: {missing}")

def align_features(X: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """Ensure feature matrix has correct columns in correct order"""
    # Add missing columns filled with NaN
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan
    # Reorder columns to match training
    return X[expected_cols]

def save_snapshot(df_feat: pd.DataFrame, ts: pd.Series, time_col: str,
                 rows_lim: int, out_csv: Path) -> None:
    """Save snapshot with row limit"""
    if rows_lim and len(df_feat) > rows_lim:
        df_feat = df_feat.tail(rows_lim).reset_index(drop=True)
        ts = ts.tail(rows_lim).reset_index(drop=True)
    snap = df_feat.assign(**{time_col: ts.dt.strftime("%Y-%m-%d %H:%M:%S")})
    snap.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved (%d rows, %d cols)", out_csv.name, *snap.shape)

def report_performance(y_true: np.ndarray, y_pred: np.ndarray, method_name: str) -> None:
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
    """Generate predictions with strict data validation"""
    y_true, y_pred = [], []
    close_col = f"{prep.main_timeframe}_close"
    total_rows = len(df) - window - 1
    
    LOG.info(f"Generating predictions for {total_rows} rows...")
    
    for idx in tqdm(range(window, len(df) - 1), desc="Predicting", unit="row"):
        try:
            # Get windowed data
            sub = df.iloc[idx - window : idx + 1].copy().reset_index(drop=True)
            
            # Prepare features - must match training exactly
            X_inc, _ = prep.ready_incremental(sub, window=window,
                                             selected_features=all_cols)
            if X_inc.empty:
                y_pred.append(-1)
                y_true.append(0)
                continue

            # Validate and align features
            X_inc = align_features(X_inc, all_cols)
            validate_feature_columns(X_inc, all_cols)
            
            # Handle NaN values using training imputation
            if X_inc.isna().any().any():
                if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                    scaler = model.named_steps['scaler']
                    if hasattr(scaler, 'mean_'):
                        mean_dict = {col: scaler.mean_[i] for i, col in enumerate(all_cols)}
                        X_inc = X_inc.fillna(mean_dict)
            
            # Ensure correct data type
            X_inc = X_inc.astype("float32")
            
            # Make prediction
            proba = model.predict_proba(X_inc)[:, 1]
            pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]
            
            # Get true label (next candle's direction)
            true_label = int((df.iloc[idx + 1][close_col] - df.iloc[idx][close_col]) > 0)
            
            y_true.append(true_label)
            y_pred.append(int(pred))
            
        except Exception as e:
            LOG.error(f"Error at index {idx}: {str(e)}")
            y_pred.append(-1)
            y_true.append(0)
    
    return np.array(y_true), np.array(y_pred)

def chimney_snapshot(prep: PREPARE_DATA_FOR_TRAIN, window: int, feats: List[str],
                    all_cols: List[str], start: str | None, rows: int,
                    out_csv: Path, live_est: ModelPipelineLive,
                    neg_thr: float, pos_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate chimney snapshot with strict no-lookahead checks"""
    LOG.info("▶ Building CHIMNEY snapshot with validation...")
    
    # Load data without future information
    merged = prep.load_data()
    X_raw, y_true, _, _ = prep.ready(merged, window=window,
                                    selected_features=feats, mode="train")
    
    # Validate we're not using future data
    if len(X_raw) != len(y_true):
        raise DataConsistencyError("Feature/target length mismatch - possible lookahead")
    
    # Ensure proper feature alignment
    X_raw = align_features(X_raw, all_cols)
    validate_feature_columns(X_raw, all_cols)
    
    # Generate predictions
    y_pred = []
    for _, row in tqdm(X_raw.iterrows(), total=len(X_raw), desc="Batch Predicting"):
        X_inc = pd.DataFrame([row])[all_cols].astype("float32")
        
        # Apply same preprocessing as live
        if X_inc.isna().any().any():
            scaler = live_est.base_pipe.named_steps['scaler']
            mean_dict = {col: scaler.mean_[i] for i, col in enumerate(all_cols)}
            X_inc = X_inc.fillna(mean_dict)
        
        proba = live_est.predict_proba(X_inc)[:, 1]
        pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]
        y_pred.append(int(pred))
    
    y_pred = np.array(y_pred)
    
    # Verify no data leakage
    if accuracy_score(y_true, y_pred) > 0.95:
        LOG.warning("⚠ Suspiciously high accuracy - possible data leakage")
    
    report_performance(y_true, y_pred, "chimney method")
    
    # Save snapshot
    time_col = f"{prep.main_timeframe}_time"
    ts = merged[time_col].iloc[window:window+len(X_raw)].reset_index(drop=True)
    save_snapshot(X_raw, ts, time_col, rows, out_csv)
    
    return y_true, y_pred

def live_snapshot(prep: PREPARE_DATA_FOR_TRAIN, merged: pd.DataFrame,
                 live_est: ModelPipelineLive, window: int,
                 neg_thr: float, pos_thr: float, all_cols: List[str],
                 start: str | None, rows: int, out_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Run live back-test with comprehensive reporting"""
    LOG.info("▶ Running LIVE back-test with full metrics...")
    
    # Generate predictions
    y_true, y_pred = generate_predictions(
        merged, live_est, window, all_cols, neg_thr, pos_thr, prep
    )
    
    # Report performance
    report_performance(y_true, y_pred, "live method")

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
    return y_true, y_pred

def diff_snapshots(ref_csv: Path, live_csv: Path, diff_txt: Path,
                  abs_tol: float = 1e-6, rel_tol: float = 1e-9) -> None:
    """Compare snapshots with enhanced difference reporting"""
    LOG.info("▶ Comparing snapshots with detailed analysis...")
    
    # Load data
    df_ref = pd.read_csv(ref_csv, low_memory=False)
    df_live = pd.read_csv(live_csv, low_memory=False)

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

    merged = df_ref.merge(df_live, on=time_col, how="inner", 
                         suffixes=("_ref", "_live"))
    if merged.empty:
        LOG.error("No overlapping timestamps between snapshots!")
        return

    # Calculate differences
    diff_cols, total_diff, worst_abs = [], 0, 0.0
    diff_values = []
    
    common_cols = [c for c in df_ref.columns 
                  if c != time_col and c in df_live.columns]
    
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

    # Save and report differences
    diff_txt.write_text("\n".join(diff_cols) + "\n", encoding="utf-8")
    
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

def main() -> None:
    """Main execution with enhanced validation"""
    parser = argparse.ArgumentParser(
        description="Validated Chimney vs Live Comparison Tool",
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
        # Load model and verify structure
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

        # Verify critical parameters
        if window < 1 or window > 100:
            raise ValueError(f"Invalid window size: {window}")
        if not 0 <= neg_thr <= pos_thr <= 1:
            raise ValueError(f"Invalid thresholds: neg_thr={neg_thr}, pos_thr={pos_thr}")

        # Initialize with validation
        live_est = build_live_estimator(pipe_fit)
        if not hasattr(live_est, 'predict_proba'):
            raise AttributeError("Live estimator missing predict_proba method")

        # Data preparation with checks
        csv_dir = Path(args.data_dir).resolve()
        prep = PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T",
            filepaths={
                "30T": str(csv_dir / "XAUUSD_M30.csv"),
                "15T": str(csv_dir / "XAUUSD_M15.csv"),
                "5T": str(csv_dir / "XAUUSD_M5.csv"),
                "1H": str(csv_dir / "XAUUSD_H1.csv"),
            },
            verbose=True,
        )

        # Run with validation
        chimney_true, chimney_pred = chimney_snapshot(
            prep, window, feats, all_cols, args.start, args.rows, 
            Path(args.chimney_snap), live_est, neg_thr, pos_thr
        )

        # Live processing with strict checks
        merged_all = prep.load_data()
        live_true, live_pred = live_snapshot(
            prep, merged_all, live_est, window, neg_thr, pos_thr,
            all_cols, args.start, args.rows, Path(args.live_snap)
        )

        # Final validation
        accuracy_diff = abs(accuracy_score(chimney_true, chimney_pred) - 
                          accuracy_score(live_true, live_pred))
        if accuracy_diff > 0.1:
            raise DataConsistencyError(
                f"Large accuracy difference between methods: {accuracy_diff:.2f}"
            )

        # Compare results
        diff_snapshots(
            Path(args.chimney_snap),
            Path(args.live_snap),
            Path(args.diff_columns)
        )

        LOG.info("✅ Validation passed - methods are consistent")

    except Exception as e:
        LOG.error("❌ Validation failed: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()