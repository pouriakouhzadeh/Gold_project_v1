#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_live_backtest_updated.py
─────────────────────────────────
مقایسهٔ «دودکش» (batch) و «لایو» (incremental) روی مدل ذخیره‌شده‌ی
best_model.pkl با تضمین یکسان‌بودن پیش‌پردازش و گزارش کامل متریک‌ها.
"""

from __future__ import annotations

import argparse, logging, sys
from pathlib import Path
from typing import List, Dict, Tuple

import joblib, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive          # نسخهٔ بدون SMOTE

# ─────────────── لاگ ───────────────
LOG = logging.getLogger("tester")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("comparison_tool.log")],
)

# ═════════════════════════════════════════════════════
#            1)  ساخت نسخهٔ لایوِ مدلِ آموزش‌دیده
# ═════════════════════════════════════════════════════
def _first_scaler(pipe: Pipeline) -> StandardScaler | None:
    """اولین StandardScaler عمق-۱ را برمی‌گرداند (اگر باشد)."""
    for step in pipe.named_steps.values():
        if isinstance(step, StandardScaler):
            return step
    return None


def build_live_estimator(fitted_pipe: Pipeline,
                         keep_calibrator: bool = True) -> ModelPipelineLive:
    """
    از Pipeline آموزش‌دیده، نسخهٔ لایو بدون SMOTE و بدون Imputer می‌سازد.
    تمام وزن‌های آموزش‌دیده حفظ می‌شود.
    """
    LOG.info("⛏  Building live estimator …")

    scaler = _first_scaler(fitted_pipe)
    if scaler is None:
        raise ValueError("❌  No StandardScaler found inside trained pipeline.")

    trained_cls = fitted_pipe.named_steps["classifier"]
    if isinstance(trained_cls, CalibratedClassifierCV):
        final_clf   = trained_cls           # کالیبره‌شده بماند
        hyperparams = trained_cls.estimator.get_params()
    else:                                   # LogisticRegression خالص
        final_clf   = trained_cls
        hyperparams = final_clf.get_params()

    # ――― مدل لایو → فقط «اسکیلر + کلاسیفایر» ―――
    live = ModelPipelineLive(hyperparams=hyperparams, calibrate=False)
    live.base_pipe = Pipeline([
        ("scaler", scaler),                 # ← فیت-شده
        ("clf",    final_clf),              # ← فیت-شده
    ])

    if keep_calibrator and isinstance(trained_cls, CalibratedClassifierCV):
        live._calibrator = trained_cls      # pylint: disable=protected-access

    LOG.info("✅  Live estimator ready")
    return live

# ═════════════════════════════════════════════════════
#                2)  توابع کمکی Snapshot
# ═════════════════════════════════════════════════════
def _ensure_columns(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """ستون‌های مفقود را با NaN اضافه و مرتب می‌کند."""
    for c in order:
        if c not in df.columns:
            df[c] = np.nan
    return df[order]

def chimney_snapshot(prep: PREPARE_DATA_FOR_TRAIN, merged: pd.DataFrame,
                     window: int, feats: List[str], all_cols: List[str],
                     start: str | None, rows: int, out_csv: Path) -> None:

    LOG.info("▶ Building CHIMNEY snapshot …")
    X_raw, _, _, _ = prep.ready(merged, window=window,
                                selected_features=feats, mode="train")
    X_raw = _ensure_columns(X_raw.astype("float32"), all_cols)

    tcol = f"{prep.main_timeframe}_time"
    ts = merged[tcol].iloc[window:window+len(X_raw)].reset_index(drop=True)
    if start:
        keep = ts >= pd.Timestamp(start)
        X_raw, ts = X_raw[keep], ts[keep]
    if rows and len(X_raw) > rows:
        X_raw, ts = X_raw.tail(rows), ts.tail(rows)

    X_raw[tcol] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    X_raw.to_csv(out_csv, index=False)
    LOG.info("📄 %s saved  (%d×%d)", out_csv.name, *X_raw.shape)

# -----------------------------------------------------
def live_snapshot(prep: PREPARE_DATA_FOR_TRAIN, merged: pd.DataFrame,
                  live: ModelPipelineLive, window: int,
                  neg: float, pos: float, all_cols: List[str],
                  start: str | None, rows: int, out_csv: Path
                 ) -> Tuple[np.ndarray, np.ndarray]:

    LOG.info("▶ Running LIVE back-test …")

    tcol = f"{prep.main_timeframe}_time"
    ccol = f"{prep.main_timeframe}_close"
    merged[tcol] = pd.to_datetime(merged[tcol])
    if start:
        merged = merged[merged[tcol] >= pd.Timestamp(start)].reset_index(drop=True)

    scaler_means = live.base_pipe.named_steps["scaler"].mean_

    snaps, y_true, y_pred = [], [], []
    rng = tqdm(range(window, len(merged)-1), desc="LIVE", unit="row", leave=False)

    for idx in rng:
        sub = merged.iloc[idx-window: idx+1].reset_index(drop=True)
        X_inc, _ = prep.ready_incremental(sub, window=window,
                                          selected_features=all_cols)
        if X_inc.empty:
            continue

        X_inc = _ensure_columns(X_inc.astype("float32"), all_cols)
        if X_inc.isna().any().any():
            X_inc = X_inc.fillna(
                {c: scaler_means[i] for i, c in enumerate(all_cols)}
            )

        proba = live.predict_proba(X_inc)[:, 1]
        pred  = ModelPipelineLive.apply_thresholds(proba, neg, pos)[0]

        snaps.append(
            pd.Series({tcol: merged.loc[idx, tcol].strftime("%Y-%m-%d %H:%M:%S"),
                       **X_inc.iloc[0].to_dict()})
        )
        # برچسب کندل بعد
        y_true.append(int((merged.loc[idx+1, ccol] - merged.loc[idx, ccol]) > 0))
        y_pred.append(int(pred))

        if rows and len(snaps) >= rows:
            break

    pd.DataFrame(snaps).to_csv(out_csv, index=False)
    LOG.info("📄 %s saved  (%d×%d)", out_csv.name, len(snaps), len(all_cols)+1)
    return np.array(y_true), np.array(y_pred)

# ═════════════════════════════════════════════════════
#                   3)  متریک‌ها
# ═════════════════════════════════════════════════════
def compute_metrics(snap: pd.DataFrame, merged: pd.DataFrame,
                    est, neg: float, pos: float,
                    all_cols: List[str]) -> Dict[str, float | int | np.ndarray]:

    tcol = next(c for c in snap.columns if c.endswith("_time"))
    ccol = tcol.replace("_time", "_close")
    time_map = dict(zip(merged[tcol], merged.index))

    scaler = (est.named_steps["scaler"]
              if isinstance(est, Pipeline)
              else est.base_pipe.named_steps["scaler"])
    means = scaler.mean_

    yt, yp = [], []
    for _, row in snap.iterrows():
        ts = pd.to_datetime(row[tcol]); pos_idx = time_map.get(ts)
        if pos_idx is None or pos_idx+1 >= len(merged): continue

        label = int((merged.loc[pos_idx+1, ccol] - merged.loc[pos_idx, ccol]) > 0)
        X = _ensure_columns(row[all_cols].to_frame().T.astype("float32"), all_cols)
        if X.isna().any().any():
            X = X.fillna({c: means[i] for i, c in enumerate(all_cols)})

        p = est.predict_proba(X)[:, 1] if not isinstance(est, Pipeline) \
            else est.predict_proba(X)[:, 1]
        pr = ModelPipelineLive.apply_thresholds(p, neg, pos)[0]

        yt.append(label); yp.append(int(pr))

    yt, yp = np.array(yt), np.array(yp)
    mask = yp != -1
    return dict(
        total=len(yp),
        conf=float(mask.mean()) if len(mask) else 0.0,
        acc=accuracy_score(yt[mask], yp[mask]) if mask.any() else 0.0,
        balacc=balanced_accuracy_score(yt[mask], yp[mask]) if mask.any() else 0.0,
        f1=f1_score(yt[mask], yp[mask]) if mask.any() else 0.0,
        correct=int(((yp == yt) & mask).sum()),
        incorrect=int((mask & (yp != yt)).sum()),
        unpred=int((~mask).sum()),
        y_pred=yp,
    )

# ═════════════════════════════════════════════════════
#                      4)  CLI
# ═════════════════════════════════════════════════════
def cli():
    p = argparse.ArgumentParser("Chimney vs Live (no-SMOTE)")
    p.add_argument("--model",     default="best_model.pkl")
    p.add_argument("--data-dir",  default=".")
    p.add_argument("--start")
    p.add_argument("--rows",      type=int, default=2000)
    p.add_argument("--chimney",   default="chimney_snapshot.csv")
    p.add_argument("--live",      default="live_snapshot.csv")
    return p.parse_args()

# ═════════════════════════════════════════════════════
#                        5)  MAIN
# ═════════════════════════════════════════════════════
if __name__ == "__main__":
    args = cli()

    # ---------- Load artefacts ----------
    payload = joblib.load(args.model)
    pipe_fit: Pipeline = payload["pipeline"]
    window   = int(payload["window_size"])
    neg_thr  = float(payload["neg_thr"])
    pos_thr  = float(payload["pos_thr"])
    feats    = payload["feats"]
    all_cols = payload["train_window_cols"]

    live_est = build_live_estimator(pipe_fit)

    # ---------- Load data ----------
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
    merged = prep.load_data()

    # محدود کردن برای سرعت
    N_LAST = 4000
    if len(merged) > N_LAST + window:
        merged = merged.tail(N_LAST + window).reset_index(drop=True)
        LOG.info("⚡ Back-test limited to last %d rows (+window)", N_LAST)

    # ---------- Snapshots ----------
    chimney_snapshot(prep, merged, window, feats, all_cols,
                     args.start, args.rows, Path(args.chimney))
    live_snapshot   (prep, merged, live_est, window,
                     neg_thr, pos_thr, all_cols,
                     args.start, args.rows, Path(args.live))

    # ---------- Metrics ----------
    met_ch = compute_metrics(pd.read_csv(args.chimney), merged,
                             pipe_fit, neg_thr, pos_thr, all_cols)
    met_lv = compute_metrics(pd.read_csv(args.live), merged,
                             live_est, neg_thr, pos_thr, all_cols)

    # ---------- Report ----------
    diff_labels = int((met_ch["y_pred"][:min(len(met_ch["y_pred"]),
                                            len(met_lv["y_pred"]))] !=
                       met_lv["y_pred"][:min(len(met_ch["y_pred"]),
                                             len(met_lv["y_pred"]))]).sum())

    LOG.info("══════════════════  FINAL REPORT  ══════════════════")
    for tag, m in [("CHIMNEY", met_ch), ("LIVE", met_lv)]:
        LOG.info("%s ➔ size=%d | conf=%.2f | Acc=%.4f | BalAcc=%.4f | "
                 "F1=%.4f | correct=%d | wrong=%d | unpred=%d",
                 tag, m["total"], m["conf"], m["acc"], m["balacc"],
                 m["f1"], m["correct"], m["incorrect"], m["unpred"])
    LOG.info("Label differences (Chimney ↔ Live): %d", diff_labels)
    LOG.info("🎉 All done.")
