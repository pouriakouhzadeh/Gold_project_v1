#!/usr/bin/env python3
"""
realtime_live_backtest_fixed.py
────────────────────────────────
تست «دودکش» (batch) در مقابل «لایو» بعد از یکسان‑سازی کامل مسیر آماده‑سازی.

تغییراتِ کلیدی نسبت به نسخهٔ قبل
================================
1. **selected_features=None** در فراخوانى `ready_incremental` تا فیلترهاى
   صفر‑ثابت، trailing‑zero و bad_cols مثل دودکش اجرا شوند.
2. بعد از آماده‑سازی، DataFrame با `all_cols` بازآرایى می‌شود تا تعداد ستون‌ها
   دقیقاً با دادهٔ دودکش و مدل برابر بماند.
3. متریک Balanced‑Accuracy اضافه شد تا افت عملکرد در نمونه‌هاى نامتعادل
   به‌وضوح دیده شود.
4. گزارش نهایى خلاصه‌‌تر و صریح‌تر: Acc, BalAcc, F1 و Conf‑Ratio.

استفاده:
────────
```bash
python realtime_live_backtest_fixed.py \
       --model best_model.pkl \
       --data-dir /path/to/csvs \
       --rows 2000
```
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive  # نسخهٔ بدون SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("backtest")

# ════════════════════════════════════════════════════════════════
#           Helper: build live estimator (بدون SMOTE)
# ════════════════════════════════════════════════════════════════

def build_live_estimator(fitted_pipe: Pipeline) -> ModelPipelineLive:
    """Extracts scaler + classifier from GA‑trained pipeline and wraps them
    inside `ModelPipelineLive` so that لایو بتواند همان ضرایب را استفاده کند."""
    scaler = fitted_pipe.named_steps["scaler"]
    trained_clf = fitted_pipe.named_steps["classifier"]  # may be Calibrated

    if isinstance(trained_clf, CalibratedClassifierCV):
        # Calibrator حاوى پایپ‌لاین کامل است؛ هایپر ها را از estimator می‌گیریم
        lr = trained_clf.estimator.named_steps["classifier"]
    else:
        lr = trained_clf

    hp_for_live = {
        k: v for k, v in lr.get_params().items()
        if k in [
            "C", "max_iter", "tol", "penalty", "solver",
            "fit_intercept", "class_weight", "multi_class"
        ]
    }

    live = ModelPipelineLive(hyperparams=hp_for_live, calibrate=False)
    live.base_pipe = Pipeline([("scaler", scaler), ("clf", trained_clf)])
    if isinstance(trained_clf, CalibratedClassifierCV):
        live._calibrator = trained_clf  # type: ignore  # دسترسى مستقیم
    return live

# ════════════════════════════════════════════════════════════════
#                     Snapshot utilities
# ════════════════════════════════════════════════════════════════

def save_snapshot(df_feat: pd.DataFrame, ts: pd.Series, time_col: str,
                  rows_lim: int, out_csv: Path):
    if rows_lim and len(df_feat) > rows_lim:
        df_feat, ts = (df.tail(rows_lim).reset_index(drop=True)
                       for df in (df_feat, ts))
    snap = df_feat.assign(**{time_col: ts.dt.strftime("%Y-%m-%d %H:%M:%S")})
    snap.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved  (%d rows · %d cols)", out_csv.name, *snap.shape)

# ----------------------------------------------------------------

def chimney_snapshot(
    prep: PREPARE_DATA_FOR_TRAIN,
    merged: pd.DataFrame,
    window: int,
    feats: List[str],
    all_cols: List[str],
    start: str | None,
    rows: int,
    out_csv: Path,
):
    LOG.info("▶ Building CHIMNEY snapshot …")
    X_raw, _, _, _ = prep.ready(merged, window=window,
                                selected_features=feats, mode="train")

    # تضمین همهٔ ستون‌ها (برخی ممکن است بعد از فیلتر حذف شده باشند)
    for c in all_cols:
        if c not in X_raw.columns:
            X_raw[c] = np.nan
    X_raw = X_raw.reindex(columns=all_cols).astype("float32")

    time_col = f"{prep.main_timeframe}_time"
    ts = merged[time_col].iloc[window : window + len(X_raw)].reset_index(drop=True)

    if start:
        keep = ts >= pd.Timestamp(start)
        X_raw, ts = X_raw[keep], ts[keep]

    save_snapshot(X_raw, ts, time_col, rows, out_csv)

# ----------------------------------------------------------------

def live_snapshot(
    prep: PREPARE_DATA_FOR_TRAIN,
    merged: pd.DataFrame,
    live_est: ModelPipelineLive,
    window: int,
    neg_thr: float,
    pos_thr: float,
    all_cols: List[str],
    start: str | None,
    rows: int,
    out_csv: Path,
):
    LOG.info("▶ Running LIVE back‑test …")

    time_col, close_col = f"{prep.main_timeframe}_time", f"{prep.main_timeframe}_close"
    merged[time_col] = pd.to_datetime(merged[time_col])
    if start:
        merged = merged[merged[time_col] >= pd.Timestamp(start)]
    merged = merged.reset_index(drop=True)

    scaler_means = live_est.base_pipe.named_steps["scaler"].mean_
    snaps, y_true, y_pred = [], [], []

    loop = tqdm(range(window, len(merged) - 1), desc="LIVE back‑test",
                unit="row", leave=False)
    for idx in loop:
        sub = merged.iloc[idx - window : idx + 1].copy().reset_index(drop=True)

        # ⬇️ فیلترهاى یکسان با دودکش (selected_features=None)
        X_inc, _ = prep.ready_incremental(sub, window=window,
                                          selected_features=None)
        if X_inc.empty:
            continue

        # ستون‌هاى گمشده را اضافه و مرتب می‌کنیم
        for c in all_cols:
            if c not in X_inc.columns:
                X_inc[c] = np.nan
        X_inc = X_inc[all_cols].astype("float32")

        # پرکردن NaN با میانگین اسکیلر (اگر وجود داشت)
        if X_inc.isna().any().any():
            mean_dict = {col: scaler_means[i] for i, col in enumerate(all_cols)}
            X_inc = X_inc.fillna(mean_dict)

        proba = live_est.predict_proba(X_inc)[:, 1]
        pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]

        ts = merged.loc[idx, time_col]
        snaps.append(pd.Series({time_col: ts.strftime("%Y-%m-%d %H:%M:%S"),
                                **{c: X_inc.iloc[0][c] for c in all_cols}}))

        # label آینده
        y_true.append(int((merged.iloc[idx + 1][close_col] -
                           merged.iloc[idx][close_col]) > 0))
        y_pred.append(int(pred))

    snap_df = pd.DataFrame(snaps)
    save_snapshot(snap_df, pd.to_datetime(snap_df[time_col]),
                  time_col, rows, out_csv)

    return pd.Series(y_true), pd.Series(y_pred)

# ════════════════════════════════════════════════════════════════
#                 Metrics for each snapshot
# ════════════════════════════════════════════════════════════════

def compute_metrics(df_snap: pd.DataFrame, merged: pd.DataFrame, est,
                    neg_thr: float, pos_thr: float,
                    all_cols: List[str]) -> Dict[str, float | int | np.ndarray]:
    time_col = next(c for c in df_snap.columns if c.endswith("_time"))
    close_col = time_col.replace("_time", "_close")

    # map time‑stamp → index برای لیبل آینده
    t2idx = dict(zip(merged[time_col], merged.index))

    scaler = (est.named_steps["scaler"]
              if isinstance(est, Pipeline) else est.base_pipe.named_steps["scaler"])
    scaler_means = scaler.mean_

    y_true, y_pred = [], []
    for _, row in df_snap.iterrows():
        ts = pd.to_datetime(row[time_col])
        idx = t2idx.get(ts)
        if idx is None or idx + 1 >= len(merged):
            continue
        label = int((merged.iloc[idx + 1][close_col] - merged.iloc[idx][close_col]) > 0)
        X = row[all_cols].to_frame().T.astype("float32")
        if X.isna().any().any():
            X = X.fillna({col: scaler_means[i] for i, col in enumerate(all_cols)})

        prob = est.predict_proba(X)[:, 1] if isinstance(est, Pipeline) else est.predict_proba(X)[:, 1]
        pred = ModelPipelineLive.apply_thresholds(prob, neg_thr, pos_thr)[0]
        y_true.append(label)
        y_pred.append(int(pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    decided = y_pred != -1
    conf_ratio = decided.mean() if len(decided) else 0.0

    return dict(
        total=len(y_pred),
        decided=int(decided.sum()),
        correct=int((y_pred[decided] == y_true[decided]).sum()),
        incorrect=int((y_pred[decided] != y_true[decided]).sum()),
        unpred=int((~decided).sum()),
        acc=accuracy_score(y_true[decided], y_pred[decided]) if decided.any() else 0.0,
        balacc=balanced_accuracy_score(y_true[decided], y_pred[decided]) if decided.any() else 0.0,
        f1=f1_score(y_true[decided], y_pred[decided]) if decided.any() else 0.0,
        conf=conf_ratio,
        y_pred=y_pred,
    )

# ════════════════════════════════════════════════════════════════
#                           CLI
# ════════════════════════════════════════════════════════════════

def parse_cli():
    p = argparse.ArgumentParser("Chimney ↔ Live tester (filters‑synced)")
    p.add_argument("--model", default="best_model.pkl")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--start", help="اختیاری: شروع بک‑تست (YYYY‑MM‑DD)")
    p.add_argument("--rows", type=int, default=2000, help="رows in snapshots")
    p.add_argument("--chimney-snap", default="chimney_snapshot.csv")
    p.add_argument("--live-snap", default="live_snapshot.csv")
    return p.parse_args()

# ════════════════════════════════════════════════════════════════
#                            MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_cli()

    mdl_path = Path(args.model).resolve()
    if not mdl_path.is_file():
        LOG.error("Model %s not found!", mdl_path)
        sys.exit(1)

    payload = joblib.load(mdl_path)
    pipe_fit: Pipeline = payload["pipeline"]
    window = int(payload["window_size"])
    neg_thr, pos_thr = float(payload["neg_thr"]), float(payload["pos_thr"])
    feats: List[str] = payload["feats"]
    all_cols: List[str] = payload["train_window_cols"]

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

    merged_all = prep.load_data()
    N_LAST = 4000
    if len(merged_all) > N_LAST + window:
        merged_all = merged_all.tail(N_LAST + window).reset_index(drop=True)
        LOG.info("⚡ Back‑test limited to last %d rows (+window)", N_LAST)

    # ➊ دودکش
    chimney_snapshot(prep, merged_all, window, feats, all_cols,
                     args.start, args.rows, Path(args.chimney_snap))

    # ➋ لایو
    _ = live_snapshot(
        prep, merged_all, live_est, window,
        neg_thr, pos_thr, all_cols,
        args.start, args.rows, Path(args.live_snap),
    )

    # ➌ متریک‌ها
    df_chim = pd.read_csv(args.chimney_snap)
    df_live = pd.read_csv(args.live_snap)

    met_ch = compute_metrics(df_chim, merged_all, pipe_fit,
                             neg_thr, pos_thr, all_cols)
    met_lv = compute_metrics(df_live, merged_all, live_est,
                             neg_thr, pos_thr, all_cols)

    # ═════ گزارش نهایى ═════
    LOG.info("══════════════════  FINAL REPORT  ══════════════════")
    for tag, m in [("CHIMNEY", met_ch), ("LIVE", met_lv)]:
        LOG.info(
            "%s ➔ size=%d | conf=%.2f | Acc=%.4f | BalAcc=%.4f | F1=%.4f | correct=%d | wrong=%d | unpred=%d",
            tag, m["total"], m["conf"], m["acc"], m["balacc"], m["f1"],
            m["correct"], m["incorrect"], m["unpred"],
        )

    # اختلاف برچسب‌ها میان دو مسیر
    min_len = min(len(met_ch["y_pred"]), len(met_lv["y_pred"]))
    lbl_diff = int((met_ch["y_pred"][:min_len] != met_lv["y_pred"][:min_len]).sum())
    LOG.info("Label differences (Chimney ↔ Live) : %d", lbl_diff)

    LOG.info("🎉 All done.")
