#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_chimney_vs_live_no_smote.py
────────────────────────────────
• Snapshot دودکش (batch) ← chimney_snapshot.csv
• Snapshot لایو (پنجره‌ای) ← live_snapshot.csv
• مقایسه و گزارش:
      - تعداد کل سلول‌های متفاوت
      - بدترین |Δ| عددی
      - ذخیرهٔ نام ستون‌هایی که اختلاف دارند → diff_columns.txt
"""

from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# ────────── ماژول‌های پروژه (در همان فولدر) ──────────
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive          # ← نسخهٔ بدون SMOTE

LOG = logging.getLogger("tester")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# ════════════════════════════════════════════════════════════════════════
#          Helpers
# ════════════════════════════════════════════════════════════════════════
def build_live_estimator(fitted_pipe: Pipeline) -> ModelPipelineLive:
    """استخراج scaler + classifier از Pipeline آموزش‌دیده و قرار دادن در
       ModelPipelineLive (بدون SMOTE)."""
    scaler     = fitted_pipe.named_steps["scaler"]
    classifier = fitted_pipe.named_steps["classifier"]      # Calibrated LR
    live = ModelPipelineLive(calibrate=False)               # dummy
    live.base_pipe.steps[0] = ("scaler", scaler)
    live.base_pipe.steps[1] = ("clf",    classifier)
    return live


def save_snapshot(df_feat: pd.DataFrame,
                  ts: pd.Series,
                  time_col: str,
                  rows_lim: int,
                  out_csv: Path):
    if rows_lim and len(df_feat) > rows_lim:
        df_feat, ts = (df.tail(rows_lim).reset_index(drop=True) for df in (df_feat, ts))
    snap = df_feat.assign(**{time_col: ts.dt.strftime("%Y-%m-%d %H:%M:%S")})
    snap.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved (%d rows, %d cols)", out_csv.name, *snap.shape)


# ════════════════════════════════════════════════════════════════════════
#          Build chimney snapshot  (batch)
# ════════════════════════════════════════════════════════════════════════
def chimney_snapshot(prep: PREPARE_DATA_FOR_TRAIN,
                     window: int,
                     feats: List[str],
                     all_cols: List[str],
                     start: str | None,
                     rows: int,
                     out_csv: Path):
    LOG.info("▶ Building CHIMNEY snapshot …")
    merged = prep.load_data()
    X_raw, _, _, _ = prep.ready(merged, window=window,
                                selected_features=feats, mode="train")

    # اطمینان از تمام ستون‌ها و ترتیب
    for c in all_cols:
        if c not in X_raw.columns:
            X_raw[c] = np.nan
    X_raw = X_raw.reindex(columns=all_cols).astype("float32")

    time_col = f"{prep.main_timeframe}_time"
    ts = merged[time_col].iloc[window:window+len(X_raw)].reset_index(drop=True)

    if start:
        keep = ts >= pd.Timestamp(start)
        X_raw, ts = X_raw[keep], ts[keep]

    save_snapshot(X_raw, ts, time_col, rows, out_csv)


# ════════════════════════════════════════════════════════════════════════
#          Live / back-test snapshot  (no-SMOTE pipeline)
# ════════════════════════════════════════════════════════════════════════
def live_snapshot(prep: PREPARE_DATA_FOR_TRAIN,
                  merged: pd.DataFrame,
                  live_est,
                  window: int,
                  neg_thr: float,
                  pos_thr: float,
                  all_cols: List[str],
                  start: str | None,
                  rows: int,
                  out_csv: Path):

    LOG.info("▶ Running LIVE back-test …")
    time_col  = f"{prep.main_timeframe}_time"
    close_col = f"{prep.main_timeframe}_close"

    merged[time_col] = pd.to_datetime(merged[time_col])
    if start:
        merged = merged[merged[time_col] >= pd.Timestamp(start)].reset_index(drop=True)

    snaps, y_true, y_pred = [], [], []

    for idx in range(window, len(merged)-1):      # -1 ⇒ future label موجود
        sub = merged.iloc[idx-window:idx+1].copy().reset_index(drop=True)
        X_inc, _ = prep.ready_incremental(sub, window=window,
                                          selected_features=all_cols)
        if X_inc.empty:
            continue

        for c in all_cols:
            if c not in X_inc.columns:
                X_inc[c] = np.nan
        X_inc = X_inc[all_cols].astype("float32")

        proba = live_est.predict_proba(X_inc)[:, 1]
        pred  = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]

        ts    = merged.loc[idx, time_col]
        snaps.append(pd.Series({**{time_col: ts.strftime("%Y-%m-%d %H:%M:%S")},
                                 **{c: X_inc.iloc[0][c] for c in all_cols}}))
        # real label
        y_true.append(int((merged.loc[idx+1, close_col] - merged.loc[idx, close_col]) > 0))
        y_pred.append(int(pred))

    snap_df = pd.DataFrame(snaps)
    save_snapshot(snap_df, pd.to_datetime(snap_df[time_col]), time_col, rows, out_csv)

    # متریک روى ردیف‌های تصمیم
    mask = np.array(y_pred) != -1
    if mask.any():
        acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
        f1  = f1_score(np.array(y_true)[mask],  np.array(y_pred)[mask])
        LOG.info("LIVE decided=%d  |  Acc=%.4f  F1=%.4f",
                 int(mask.sum()), acc, f1)
    else:
        LOG.warning("⚠ No decided rows in LIVE snapshot")


# ════════════════════════════════════════════════════════════════════════
#          Snapshot diff & report
# ════════════════════════════════════════════════════════════════════════
def diff_snapshots(ref_csv: Path,
                   live_csv: Path,
                   diff_txt: Path,
                   abs_tol: float = 1e-6,
                   rel_tol: float = 1e-9) -> None:
    LOG.info("▶ Comparing snapshots …")
    df_ref  = pd.read_csv(ref_csv,  low_memory=False)
    df_live = pd.read_csv(live_csv, low_memory=False)

    # تشخیص ستون زمان (اولین ستونی که با _time تمام مى‌شود و در هر دو وجود دارد)
    time_col = next((c for c in df_ref.columns if c.endswith("_time") and c in df_live.columns), None)
    if not time_col:
        LOG.error("Could not find common *_time column"); return

    for df in (df_ref, df_live):
        df[time_col] = pd.to_datetime(df[time_col])
        df.dropna(subset=[time_col], inplace=True)
        df.drop_duplicates(subset=[time_col], keep="last", inplace=True)

    merged = df_ref.merge(df_live, on=time_col, how="inner", suffixes=("_ref", "_live"))
    if merged.empty:
        LOG.error("No overlapping timestamps between snapshots!"); return

    diff_cols, total_diff, worst_abs = [], 0, 0.0

    common_cols = [c for c in df_ref.columns if c != time_col and c in df_live.columns]
    for col in common_cols:
        a = merged[f"{col}_ref"]
        b = merged[f"{col}_live"]

        if pd.api.types.is_numeric_dtype(a):
            abs_diff = (a - b).abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = abs_diff / np.maximum(a.abs(), b.abs())
            mis = (abs_diff > abs_tol) & (rel_diff > rel_tol)
            if mis.any():
                diff_cols.append(col)
                total_diff += int(mis.sum())
                worst_abs = max(worst_abs, float(abs_diff[mis].max()))
        else:
            mis = (a.fillna("__") != b.fillna("__"))
            if mis.any():
                diff_cols.append(col)
                total_diff += int(mis.sum())

    # نوشتن ستون‌های متفاوت
    diff_txt.write_text("\n".join(diff_cols) + "\n", encoding="utf-8")
    LOG.info("🔍 diff finished | diff_cells=%d | worst |Δ|=%.3g | columns listed → %s",
             total_diff, worst_abs, diff_txt.name)


# ════════════════════════════════════════════════════════════════════════
#          CLI & Main
# ════════════════════════════════════════════════════════════════════════
def cli():
    p = argparse.ArgumentParser("Chimney vs Live tester (no-SMOTE) + diff")
    p.add_argument("--model", default="best_model.pkl")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--start", help="e.g. '2025-01-01 00:00'")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--chimney-snap", default="chimney_snapshot.csv")
    p.add_argument("--live-snap",    default="live_snapshot.csv")
    p.add_argument("--diff-columns", default="diff_columns.txt")
    return p.parse_args()


def main():
    args = cli()
    mdl_path = Path(args.model).resolve()
    if not mdl_path.is_file():
        LOG.error("Model %s not found", mdl_path); sys.exit(1)

    payload   = joblib.load(mdl_path)
    pipe_fit  : Pipeline = payload["pipeline"]
    window    : int      = int(payload["window_size"])
    neg_thr   : float    = float(payload["neg_thr"])
    pos_thr   : float    = float(payload["pos_thr"])
    feats     : List[str]= payload["feats"]
    all_cols  : List[str]= payload["train_window_cols"]

    live_est = build_live_estimator(pipe_fit)

    csv_dir = Path(args.data_dir).resolve()
    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        filepaths={
            "30T": str(csv_dir / "XAUUSD_M30.csv"),
            "15T": str(csv_dir / "XAUUSD_M15.csv"),
            "5T" : str(csv_dir / "XAUUSD_M5.csv"),
            "1H" : str(csv_dir / "XAUUSD_H1.csv"),
        },
        verbose=False,
    )

    # ❶ دودکش
    chimney_snapshot(prep, window, feats, all_cols,
                     args.start, args.rows, Path(args.chimney_snap))

    # ❷ لایو
    merged_all = prep.load_data()
    live_snapshot(prep, merged_all, live_est, window,
                  neg_thr, pos_thr, all_cols,
                  args.start, args.rows, Path(args.live_snap))

    # ❸ مقایسه
    diff_snapshots(Path(args.chimney_snap),
                   Path(args.live_snap),
                   Path(args.diff_columns))

    LOG.info("🎉 All done.")


if __name__ == "__main__":
    main()
