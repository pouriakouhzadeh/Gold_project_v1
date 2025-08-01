#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_chimney_vs_live_no_smote.py
────────────────────────────────
• snapshot دودکش ← chimney_snapshot.csv      (batch / بدون SMOTE)
• snapshot لایو   ← live_snapshot.csv         (پنجره‌ای / بدون SMOTE)
• گزارش کامل:
      ─ دقّت و F1 برای هر دو اسنپ‌شات
      ─ تعداد کل، تصمیم‌گرفته، درست، غلط، بدون پیش‌بینی (-1)
      ─ تعداد برچسب‌های متفاوت بین دودکش و لایو
      ─ تعداد و فهرست فیچرهای متفاوت
      ─ نوارهای پیشرفت حین اجرا
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm                          # ← NEW
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive          # نسخهٔ بدون SMOTE

# ────────────── تنظیم لاگر ──────────────
LOG = logging.getLogger("tester")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# ════════════════════════════════════════════════════════════════
#           build_live_estimator  (بدون SMOTE)
# ════════════════════════════════════════════════════════════════
def build_live_estimator(fitted_pipe: Pipeline,
                         keep_calibrator: bool = True) -> ModelPipelineLive:
    scaler      = fitted_pipe.named_steps["scaler"]
    trained_clf = fitted_pipe.named_steps["classifier"]    # LR یا Calibrated LR

    if isinstance(trained_clf, CalibratedClassifierCV):
        final_clf   = trained_clf
        hp_for_live = trained_clf.estimator.get_params()
    else:
        final_clf   = trained_clf
        hp_for_live = final_clf.get_params()

    live = ModelPipelineLive(hyperparams=hp_for_live, calibrate=False)
    live.base_pipe = Pipeline([("scaler", scaler), ("clf", final_clf)])

    if keep_calibrator and isinstance(trained_clf, CalibratedClassifierCV):
        live._calibrator = trained_clf        # pylint: disable=protected-access
    return live

# ════════════════════════════════════════════════════════════════
#                    Snapshot utilities
# ════════════════════════════════════════════════════════════════
def save_snapshot(df_feat: pd.DataFrame, ts: pd.Series,
                  time_col: str, rows_lim: int, out_csv: Path):
    if rows_lim and len(df_feat) > rows_lim:
        df_feat, ts = (df.tail(rows_lim).reset_index(drop=True)
                       for df in (df_feat, ts))
    snap = df_feat.assign(**{time_col: ts.dt.strftime("%Y-%m-%d %H:%M:%S")})
    snap.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved  (%d rows · %d cols)", out_csv.name, *snap.shape)

# ----------------------------------------------------------------
def chimney_snapshot(prep: PREPARE_DATA_FOR_TRAIN,
                     merged: pd.DataFrame,
                     window: int, feats: List[str], all_cols: List[str],
                     start: str | None, rows: int, out_csv: Path):
    LOG.info("▶ Building CHIMNEY snapshot …")
    X_raw, _, _, _ = prep.ready(merged, window=window,
                                selected_features=feats, mode="train")

    # تضمین همهٔ ستون‌ها
    for c in all_cols:
        if c not in X_raw.columns:
            X_raw[c] = np.nan
    X_raw = X_raw.reindex(columns=all_cols).astype("float32")

    time_col = f"{prep.main_timeframe}_time"
    ts = merged[time_col].iloc[window : window + len(X_raw)].reset_index(drop=True)

    if start:
        keep = ts >= pd.Timestamp(start)
        X_raw, ts = X_raw[keep], ts[keep]

    # ── progress bar — فقط زیبایی؛ در عمل کپی ساده است
    for _ in tqdm(range(1), desc="Saving chimney snapshot", leave=False):
        save_snapshot(X_raw, ts, time_col, rows, out_csv)

# ----------------------------------------------------------------
def live_snapshot(prep: PREPARE_DATA_FOR_TRAIN,
                  merged: pd.DataFrame,
                  live_est: ModelPipelineLive,
                  window: int,
                  neg_thr: float, pos_thr: float,
                  all_cols: List[str],
                  start: str | None, rows: int, out_csv: Path):

    LOG.info("▶ Running LIVE back-test …")
    time_col, close_col = f"{prep.main_timeframe}_time", f"{prep.main_timeframe}_close"

    merged[time_col] = pd.to_datetime(merged[time_col])
    if start:
        merged = merged[merged[time_col] >= pd.Timestamp(start)]
    merged = merged.reset_index(drop=True)

    snaps, y_true, y_pred = [], [], []
    scaler_means = live_est.base_pipe.named_steps["scaler"].mean_

    loop = tqdm(range(window, len(merged) - 1),
                desc="LIVE back-test",
                unit="row", leave=False)

    for idx in loop:                                  # ← progress bar
        sub = merged.iloc[idx - window : idx + 1].copy().reset_index(drop=True)
        X_inc, _ = prep.ready_incremental(sub, window=window,
                                          selected_features=all_cols)
        if X_inc.empty:
            continue
        for c in all_cols:
            if c not in X_inc.columns:
                X_inc[c] = np.nan
        X_inc = X_inc[all_cols].astype("float32")
        if X_inc.isna().any().any():
            fill_means = {col: scaler_means[i] for i, col in enumerate(all_cols)}
            X_inc = X_inc.fillna(fill_means)

        proba = live_est.predict_proba(X_inc)[:, 1]
        pred  = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]
        ts    = merged.loc[idx, time_col]

        snaps.append(pd.Series({**{time_col: ts.strftime("%Y-%m-%d %H:%M:%S")},
                                 **{c: X_inc.iloc[0][c] for c in all_cols}}))
        y_true.append(int((merged.iloc[idx + 1][close_col] -
                           merged.iloc[idx][close_col]) > 0))
        y_pred.append(int(pred))

    snap_df = pd.DataFrame(snaps)
    save_snapshot(snap_df, pd.to_datetime(snap_df[time_col]),
                  time_col, rows, out_csv)

    return pd.Series(y_true), pd.Series(y_pred)

# ════════════════════════════════════════════════════════════════
#            Metrics for each snapshot (with progress bar)
# ════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────
def compute_metrics(df_snap: pd.DataFrame, merged: pd.DataFrame,
                    est, neg_thr: float, pos_thr: float,
                    all_cols: List[str]) -> Dict[str, int | float]:
    time_col  = f"{prep.main_timeframe}_time"
    close_col = f"{prep.main_timeframe}_close"

    # نگاشت زمان → ایندکس مرج‌شده (برای ساخت لیبل آینده)
    time_map = dict(zip(merged[time_col], merged.index))

    y_true, y_pred = [], []

    # اسکیلر (برای میانگین ستون‌ها)
    scaler = (est.named_steps["scaler"]
              if isinstance(est, Pipeline)
              else est.base_pipe.named_steps["scaler"])
    scaler_means = scaler.mean_

    for _, row in df_snap.iterrows():
        # ←‌ همین‌جا خطا بود
        ts = pd.to_datetime(row[time_col])

        pos = time_map.get(ts)
        if pos is None or pos + 1 >= len(merged):
            continue                                      # label نداریم

        label = int((merged.iloc[pos + 1][close_col] -
                     merged.iloc[pos    ][close_col]) > 0)

        X = row[all_cols].to_frame().T.astype("float32")
        if X.isna().any().any():                         # پر کردن NaN
            X = X.fillna({col: scaler_means[i]
                          for i, col in enumerate(all_cols)})

        proba = est.predict_proba(X)[:, 1] if isinstance(est, Pipeline) \
                else est.predict_proba(X)[:, 1]
        pred  = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]

        y_true.append(label)
        y_pred.append(int(pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    decided = y_pred != -1
    n_total       = len(y_pred)
    n_decided     = int(decided.sum())
    n_unpredicted = n_total - n_decided
    n_correct     = int((y_pred[decided] == y_true[decided]).sum())
    n_incorrect   = n_decided - n_correct
    acc = accuracy_score(y_true[decided], y_pred[decided]) if n_decided else 0.0
    f1  = f1_score     (y_true[decided], y_pred[decided]) if n_decided else 0.0

    return dict(total=n_total, decided=n_decided, unpred=n_unpredicted,
                correct=n_correct, incorrect=n_incorrect,
                acc=acc, f1=f1, y_pred=y_pred)
# ──────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
#                        Diff features
# ════════════════════════════════════════════════════════════════
def diff_snapshots(ref_csv: Path, live_csv: Path, diff_txt: Path,
                   abs_tol: float = 1e-6, rel_tol: float = 1e-9) -> int:
    df_ref, df_live = pd.read_csv(ref_csv), pd.read_csv(live_csv)
    time_col = next((c for c in df_ref.columns if c.endswith("_time")
                     and c in df_live.columns), None)
    if not time_col:
        LOG.error("no *_time column in common!"); return 0

    for df in (df_ref, df_live):
        df[time_col] = pd.to_datetime(df[time_col])
        df.dropna(subset=[time_col], inplace=True)

    merged = df_ref.merge(df_live, on=time_col, suffixes=("_ref", "_live"))
    diff_cols, total_diff, worst = [], 0, 0.0

    for col in [c for c in df_ref.columns if c != time_col and c in df_live.columns]:
        a, b = merged[f"{col}_ref"], merged[f"{col}_live"]
        if pd.api.types.is_numeric_dtype(a):
            abs_d = (a - b).abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_d = abs_d / np.maximum(a.abs(), b.abs())
            mis = (abs_d > abs_tol) & (rel_d > rel_tol)
            if mis.any():
                diff_cols.append(col); total_diff += int(mis.sum())
                worst = max(worst, float(abs_d[mis].max()))
        else:
            mis = (a.fillna("__") != b.fillna("__"))
            if mis.any():
                diff_cols.append(col); total_diff += int(mis.sum())

    diff_txt.write_text("\n".join(diff_cols) + "\n", encoding="utf-8")
    LOG.info("🔍 diff_cells=%d | worst |Δ|=%.3g | diff_columns=%d → %s",
             total_diff, worst, len(diff_cols), diff_txt.name)
    return len(diff_cols)

# ════════════════════════════════════════════════════════════════
#                          CLI
# ════════════════════════════════════════════════════════════════
def cli():
    p = argparse.ArgumentParser("Chimney vs Live tester (no-SMOTE) + progress")
    p.add_argument("--model", default="best_model.pkl")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--start")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--chimney-snap", default="chimney_snapshot.csv")
    p.add_argument("--live-snap",    default="live_snapshot.csv")
    p.add_argument("--diff-columns", default="diff_columns.txt")
    return p.parse_args()

# ════════════════════════════════════════════════════════════════
#                            MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = cli()

    mdl_path = Path(args.model).resolve()
    if not mdl_path.is_file():
        LOG.error("Model %s not found!", mdl_path); sys.exit(1)

    payload   = joblib.load(mdl_path)
    pipe_fit  : Pipeline = payload["pipeline"]
    window    = int(payload["window_size"])
    neg_thr   = float(payload["neg_thr"])
    pos_thr   = float(payload["pos_thr"])
    feats     = payload["feats"]
    all_cols  = payload["train_window_cols"]

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

    merged_all = prep.load_data()
    # ──────────────────────────────────────────────────────────────
    # تنها ۴۰۰۰ ردیف پایانی را نگه می‌داریم   (window را هم لحاظ کن)
    # ──────────────────────────────────────────────────────────────
    N_LAST = 4_000                           # ← این عدد را هر وقت خواستید عوض کنید
    if len(merged_all) > N_LAST + window:    # window برای داشتن لیبل کندل بعدی
        merged_all = merged_all.tail(N_LAST + window).reset_index(drop=True)
        LOG.info("⚡ Back-test limited to last %d rows (+window)", N_LAST)
    # ──────────────────────────────────────────────────────────────

    # ➊ دودکش
    chimney_snapshot(prep, merged_all, window, feats, all_cols,
                     args.start, args.rows, Path(args.chimney_snap))

    # ➋ لایو
    y_true_live, y_pred_live = live_snapshot(
        prep, merged_all, live_est, window,
        neg_thr, pos_thr, all_cols,
        args.start, args.rows, Path(args.live_snap)
    )

    # ➌ diff
    n_feat_diff = diff_snapshots(Path(args.chimney_snap),
                                 Path(args.live_snap),
                                 Path(args.diff_columns))

    # ➍ متریک‌ها
    df_chim = pd.read_csv(args.chimney_snap)
    df_live = pd.read_csv(args.live_snap)

    met_ch = compute_metrics(df_chim, merged_all, pipe_fit,
                             neg_thr, pos_thr, all_cols)
    met_lv = compute_metrics(df_live, merged_all, live_est,
                             neg_thr, pos_thr, all_cols)

    # برچسب‌های متفاوت
    min_len = min(len(met_ch["y_pred"]), len(met_lv["y_pred"]))
    lab_ch  = met_ch["y_pred"][:min_len]
    lab_lv  = met_lv["y_pred"][:min_len]
    lbl_diff = int((lab_ch != lab_lv).sum()) + abs(len(met_ch["y_pred"]) - len(met_lv["y_pred"]))

    # ─────────── گزارش نهایی ───────────
    LOG.info("══════════════════  FINAL REPORT  ══════════════════")
    for tag, m in [("CHIMNEY", met_ch), ("LIVE", met_lv)]:
        LOG.info("%s ➔ total=%d | decided=%d | correct=%d | wrong=%d | "
                 "unpred=%d | Acc=%.4f | F1=%.4f",
                 tag, m["total"], m["decided"], m["correct"],
                 m["incorrect"], m["unpred"], m["acc"], m["f1"])

    LOG.info("Label differences (Chimney ↔ Live) : %d", lbl_diff)
    LOG.info("Feature columns with any diff      : %d", n_feat_diff)
    LOG.info("🎉 All done.")
