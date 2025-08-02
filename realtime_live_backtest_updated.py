#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_live_backtest_updated.py
─────────────────────────────────
مقایسهٔ «دودکش» (batch) با «لایو» (incremental) بر روی یک مدل ذخیره شدهٔ
“best_model.pkl” به‌گونه‌ای که پیش‌پردازش و منطق پیش‌بینی در هر دو مسیر
کاملاً یکسان باشد و در پایان، دقّت و سایر معیارها برای هر دو نیز گزارش شود.

تغییرات کلیدی نسبت به نسخه‌های قبلی
-----------------------------------
1. **build_live_estimator** کاملاً مقاوم شده است؛ دیگر به named_steps روی
   LogisticRegression دست نمی‌زند و هر اسکیلری را (حتی اگر نام متفاوتی
   داشته باشد) پیدا می‌کند.
2. در هر دو مسیر ستون‌ها به ترتیبِ «train_window_cols» مرتب می‌شوند تا هیچ
   جابجایی ستونی باعث اختلاف نشود.
3. هرجا مقادیر NaN وجود داشته باشد، با همان میانگینِ اسکیلر آموزش‌دیده
   جایگزین می‌شود؛ دقیقاً مشابه دودکش.
4. گزارش نهایی شامل Acc، F1، Balanced‑Accuracy، نرخ «confident» و تعداد
   label‑diff است.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive  # نسخهٔ بدون SMOTE

# ────────────────── پیکربندی لاگر ──────────────────
LOG = logging.getLogger("tester")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("comparison_tool.log")],
)

# ══════════════════════════════════════════════════
#                ساخت مدل لایو
# ══════════════════════════════════════════════════

def _first_scaler(pipe: Pipeline):
    """Return the first StandardScaler inside *pipe* (depth‑1 search)."""
    for name, step in pipe.named_steps.items():
        if isinstance(step, StandardScaler):
            return step
    return None


def build_live_estimator(
    fitted_pipe: Pipeline, *, keep_calibrator: bool = True
) -> ModelPipelineLive:
    """ساخت نسخهٔ لایو از پایپ‌لاین آموزش‑دیده (بدون SMOTE)."""

    LOG.info("⛏  Building live estimator …")

    scaler = _first_scaler(fitted_pipe)
    if scaler is None:
        LOG.error("❌  No StandardScaler found inside the trained pipeline!")
        raise ValueError("trained pipeline must contain a StandardScaler step")

    trained_cls = fitted_pipe.named_steps["classifier"]

    # ───── کلاس نهایی و هایپرها ─────
    if isinstance(trained_cls, CalibratedClassifierCV):
        final_clf = trained_cls  # کالیبراتور را نگه می‌داریم
        hyperparams = trained_cls.estimator.get_params()
    else:
        final_clf = trained_cls
        hyperparams = final_clf.get_params()

    # ───── پایپ‌لاین لایو ─────
    live = ModelPipelineLive(hyperparams=hyperparams, calibrate=False)

    imputer = live.base_pipe.named_steps["imputer"]  # از مدل لایو اولیه بگیر
    live.base_pipe = Pipeline(
        [
            ("imputer", imputer),
            ("scaler", scaler),
            ("clf", final_clf),
        ]
    )

    if keep_calibrator and isinstance(trained_cls, CalibratedClassifierCV):
        live._calibrator = trained_cls  # pylint: disable=protected-access

    LOG.info("✅  Live estimator ready")
    return live


# ══════════════════════════════════════════════════
#               Utilities for snapshots
# ══════════════════════════════════════════════════

def _ensure_columns(df: pd.DataFrame, cols_order: List[str]) -> pd.DataFrame:
    """Ensure *df* has all columns in *cols_order* (fill missing with NaN) and reorder."""
    for c in cols_order:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols_order]


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
    """Build snapshot in *batch* mode (دودکش)."""

    LOG.info("▶ Building CHIMNEY snapshot …")

    X_raw, _, _, _ = prep.ready(merged, window=window, selected_features=feats, mode="train")
    X_raw = _ensure_columns(X_raw.astype("float32"), all_cols)

    time_col = f"{prep.main_timeframe}_time"
    ts = merged[time_col].iloc[window : window + len(X_raw)].reset_index(drop=True)

    if start:
        keep = ts >= pd.Timestamp(start)
        X_raw, ts = X_raw[keep], ts[keep]

    if rows and len(X_raw) > rows:
        X_raw, ts = X_raw.tail(rows), ts.tail(rows)

    snap = X_raw.copy()
    snap[time_col] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    snap.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved  (%d rows · %d cols)", out_csv.name, *snap.shape)


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
) -> Tuple[pd.Series, pd.Series]:
    """Run incremental back‑test and store snapshot."""

    LOG.info("▶ Running LIVE back-test …")

    time_col = f"{prep.main_timeframe}_time"
    close_col = f"{prep.main_timeframe}_close"

    merged[time_col] = pd.to_datetime(merged[time_col])
    if start:
        merged = merged[merged[time_col] >= pd.Timestamp(start)].reset_index(drop=True)

    # stats for filling NaN exactly like scaler
    scaler_means = live_est.base_pipe.named_steps["scaler"].mean_

    snaps: List[pd.Series] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    loop = tqdm(
        range(window, len(merged) - 1), desc="LIVE", unit="row", leave=False
    )
    for idx in loop:
        sub = merged.iloc[idx - window : idx + 1].reset_index(drop=True)
        X_inc, _ = prep.ready_incremental(
            sub, window=window, selected_features=all_cols
        )
        if X_inc.empty:
            continue

        X_inc = _ensure_columns(X_inc.astype("float32"), all_cols)
        if X_inc.isna().any().any():
            fill = {c: scaler_means[i] for i, c in enumerate(all_cols)}
            X_inc = X_inc.fillna(fill)

        proba = live_est.predict_proba(X_inc)[:, 1]
        pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]
        ts = merged.loc[idx, time_col]

        # snapshot row
        snaps.append(pd.Series({**{time_col: ts.strftime("%Y-%m-%d %H:%M:%S")}, **X_inc.iloc[0].to_dict()}))

        # label for next candle
        y_true.append(int((merged.iloc[idx + 1][close_col] - merged.iloc[idx][close_col]) > 0))
        y_pred.append(int(pred))

        if rows and len(snaps) >= rows:
            break

    snap_df = pd.DataFrame(snaps)
    snap_df.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved  (%d rows · %d cols)", out_csv.name, *snap_df.shape)

    return pd.Series(y_true), pd.Series(y_pred)


# ══════════════════════════════════════════════════
#                Metrics for a snapshot
# ══════════════════════════════════════════════════

def compute_metrics(
    snap_df: pd.DataFrame,
    merged: pd.DataFrame,
    estimator,  # Pipeline یا ModelPipelineLive
    neg_thr: float,
    pos_thr: float,
    all_cols: List[str],
) -> Dict[str, float | int | np.ndarray]:
    """Return accuracy/F1/BalAcc plus raw predictions."""

    time_col = next(c for c in snap_df.columns if c.endswith("_time"))
    close_col = time_col.replace("_time", "_close")

    # map datetime → index for fast lookup
    time_map = dict(zip(merged[time_col], merged.index))

    scaler = (
        estimator.named_steps["scaler"]
        if isinstance(estimator, Pipeline)
        else estimator.base_pipe.named_steps["scaler"]
    )
    scaler_means = scaler.mean_

    y_true: List[int] = []
    y_pred: List[int] = []

    for _, row in snap_df.iterrows():
        ts = pd.to_datetime(row[time_col])
        pos = time_map.get(ts)
        if pos is None or pos + 1 >= len(merged):
            continue  # cannot build label

        label = int((merged.iloc[pos + 1][close_col] - merged.iloc[pos][close_col]) > 0)
        X = _ensure_columns(row[all_cols].to_frame().T.astype("float32"), all_cols)
        if X.isna().any().any():
            X = X.fillna({c: scaler_means[i] for i, c in enumerate(all_cols)})

        proba = (
            estimator.predict_proba(X)[:, 1]
            if not isinstance(estimator, Pipeline)
            else estimator.predict_proba(X)[:, 1]
        )
        pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]

        y_true.append(label)
        y_pred.append(int(pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    decided_mask = y_pred != -1
    decided_ratio = decided_mask.mean() if len(decided_mask) else 0.0

    if decided_mask.any():
        acc = accuracy_score(y_true[decided_mask], y_pred[decided_mask])
        f1 = f1_score(y_true[decided_mask], y_pred[decided_mask])
        balacc = balanced_accuracy_score(y_true[decided_mask], y_pred[decided_mask])
    else:
        acc = f1 = balacc = 0.0

    return {
        "total": len(y_pred),
        "conf": decided_ratio,
        "acc": acc,
        "f1": f1,
        "balacc": balacc,
        "correct": int(((y_pred == y_true) & decided_mask).sum()),
        "incorrect": int((decided_mask & (y_pred != y_true)).sum()),
        "unpred": int((~decided_mask).sum()),
        "y_pred": y_pred,
    }


# ══════════════════════════════════════════════════
#                       CLI & MAIN
# ══════════════════════════════════════════════════

def cli():
    p = argparse.ArgumentParser("Chimney vs Live tester (no‑SMOTE)")
    p.add_argument("--model", default="best_model.pkl")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--start")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--chimney", default="chimney_snapshot.csv")
    p.add_argument("--live", default="live_snapshot.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()

    # ───── 1) Load model ─────
    mdl_path = Path(args.model).resolve()
    if not mdl_path.is_file():
        LOG.error("Model %s not found", mdl_path)
        sys.exit(1)

    payload = joblib.load(mdl_path)
    pipe_fit: Pipeline = payload["pipeline"]
    window = int(payload["window_size"])
    neg_thr = float(payload["neg_thr"])
    pos_thr = float(payload["pos_thr"])
    feats = payload["feats"]
    all_cols = payload["train_window_cols"]

    live_est = build_live_estimator(pipe_fit)

    # ───── 2) Load data ─────
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

    # محدود کردن حجم برای سرعت
    N_LAST = 4000
    if len(merged_all) > N_LAST + window:
        merged_all = merged_all.tail(N_LAST + window).reset_index(drop=True)
        LOG.info("⚡ Back‑test limited to last %d rows (+window)", N_LAST)

    # ───── 3) Build snapshots ─────
    chimney_snapshot(
        prep,
        merged_all,
        window,
        feats,
        all_cols,
        args.start,
        args.rows,
        Path(args.chimney),
    )

    live_snapshot(
        prep,
        merged_all,
        live_est,
        window,
        neg_thr,
        pos_thr,
        all_cols,
        args.start,
        args.rows,
        Path(args.live),
    )

    # ───── 4) Metrics ─────
    df_chim = pd.read_csv(args.chimney)
    df_live = pd.read_csv(args.live)

    met_ch = compute_metrics(df_chim, merged_all, pipe_fit, neg_thr, pos_thr, all_cols)
    met_lv = compute_metrics(df_live, merged_all, live_est, neg_thr, pos_thr, all_cols)

    # اختلاف برچسب‌ها میان دو مسیر
    min_len = min(len(met_ch["y_pred"]), len(met_lv["y_pred"]))
    lbl_diff = int(
        (met_ch["y_pred"][:min_len] != met_lv["y_pred"][:min_len]).sum()
    )

    # ───── Final report ─────
    LOG.info("══════════════════  FINAL REPORT  ══════════════════")
    for tag, m in [("CHIMNEY", met_ch), ("LIVE", met_lv)]:
        LOG.info(
            "%s ➔ size=%d | conf=%.2f | Acc=%.4f | BalAcc=%.4f | F1=%.4f | correct=%d | wrong=%d | unpred=%d",
            tag,
            m["total"],
            m["conf"],
            m["acc"],
            m["balacc"],
            m["f1"],
            m["correct"],
            m["incorrect"],
            m["unpred"],
        )

    LOG.info("Label differences (Chimney ↔ Live): %d", lbl_diff)
    LOG.info("🎉 All done.")
