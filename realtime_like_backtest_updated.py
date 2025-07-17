#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_like_backtest_updated.py – 30-minute back-tester (feature snapshot)
────────────────────────────────────────────────────────────────────────────
* Predicts on **every 30-minute close** starting **exactly** where you want.
* روزهای شنبه/یکشنبه و تایم‌استمپ‌های تکراری حذف می‌شوند.
* در هر گام فقط یک سطر برای مدل می‌سازد و آن را در `live_features_tail.csv`
  ذخیره می‌کند تا بعداً با خروجی دودکش مقایسه شود.
* می‌توانید با `--rows 2000` مشخص کنید دقیقاً چند سطر آخر شبیه‌سازی شود؛
  بنابراین زمان اجرا کوتاه است و طول فایل دقیقاً با دودکش برابر می‌شود.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from dynamic_threshold_adjuster import DynamicThresholdAdjuster

# ───────────────────── configuration ─────────────────────
RAW_FILES: Dict[str, str] = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T": "XAUUSD_M5.csv",
    "1H": "XAUUSD_H1.csv",
}
LIVE_CSV = {tf: fp.replace(".csv", f".F_{tf}_live.csv") for tf, fp in RAW_FILES.items()}
MAX_ROWS = 10_000
LOGFILE = "realtime_like_backtest.log"
SNAP_FILE = "live_features_tail.csv"

# ───────────────────── helpers ─────────────────────
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    for col in df.select_dtypes(include=[np.number]):
        if df[col].isna().any():
            df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=["object"]):
        if df[col].isna().any():
            mode = df[col].mode()
            df[col].fillna(mode.iloc[0] if not mode.empty else "", inplace=True)
    return df.reset_index(drop=True)


def load_src(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["time"], low_memory=False).sort_values("time", ignore_index=True)


def dump_until(df: pd.DataFrame, ts: pd.Timestamp, out_fp: str, ffill: bool) -> None:
    """Write rows ≤ ts into *out_fp*. If *ffill* is True, we keep copying the last
    row forward (max 1 h gap) to avoid look-ahead."""
    idx = df["time"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return
    if ts - df.loc[idx, "time"] > pd.Timedelta(hours=1):
        return
    slice_df = clean_df(df.iloc[: idx + 1].tail(MAX_ROWS).copy())
    slice_df.to_csv(out_fp, index=False)


# ───────────────────── CLI ─────────────────────
cli = argparse.ArgumentParser("Realtime back-test (30-minute cadence, snapshot)")
cli.add_argument("--start", default="2024-01-02 00:00", help="ISO start datetime (e.g. 2024-01-02 00:00)")
cli.add_argument("--rows", type=int, default=2000, help="exact rows to keep (matching chimney)")
cli.add_argument("-m", "--model", default="best_model.pkl")
cli.add_argument("-o", "--out", default="realtime_like_report.csv")
cli.add_argument("--dyn-thr", action="store_true", help="enable dynamic threshold adjuster")
cli.add_argument("--verbose", action="store_true")
args = cli.parse_args()

# ───────────────────── logging ─────────────────────
file_hd = logging.FileHandler(LOGFILE, mode="w", encoding="utf-8")
file_hd.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
console_hd = logging.StreamHandler()
console_hd.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
console_hd.setLevel(logging.DEBUG if args.verbose else logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[file_hd, console_hd], force=True)
log = logging.getLogger("rt_backtest")

# ───────────────────── load raw CSVs ─────────────────────
raw_src = {tf: load_src(fp) for tf, fp in RAW_FILES.items()}
for tf in raw_src:
    raw_src[tf] = (
        raw_src[tf]
        .loc[~raw_src[tf]["time"].dt.dayofweek.isin([5, 6])]  # remove weekends
        .drop_duplicates(subset="time", keep="last")
        .reset_index(drop=True)
    )
log.info("✅ Loaded %d CSVs (weekends & duplicates removed)", len(raw_src))

# cleanup residual *_live.csv*
for fp in LIVE_CSV.values():
    if os.path.exists(fp):
        os.remove(fp)
if os.path.exists(SNAP_FILE):
    os.remove(SNAP_FILE)

# ───────────────────── load model ─────────────────────
mdl = joblib.load(args.model)
pipe = mdl["pipeline"]
neg_thr: float = mdl["neg_thr"]
pos_thr: float = mdl["pos_thr"]
win: int = mdl["window_size"]
feats: List[str] = mdl["feats"]
cols: List[str] = mdl["train_window_cols"]
log.info("✅ Model loaded | window=%d | feats=%d | cols=%d", win, len(feats), len(cols))

# ───────────────────── helper objects ─────────────────────
thr_adj = DynamicThresholdAdjuster(atr_high=10.0, vol_low=500.0, shift=0.01)
prep = PREPARE_DATA_FOR_TRAIN(filepaths=LIVE_CSV, main_timeframe="30T", verbose=False)

try:
    start_ts = pd.Timestamp(args.start)
except ValueError:
    log.error("Invalid --start format. Use e.g. '2024-01-02 00:00'.")
    sys.exit(1)

main_times = raw_src["30T"]["time"].loc[~raw_src["30T"]["time"].dt.dayofweek.isin([5, 6])]
main_times = main_times.iloc[main_times.searchsorted(start_ts, side="left") : -1]

# اگر کاربر تعداد ردیف خواست، فقط آخر rows+1 کندل را نگه دار
if args.rows > 0 and len(main_times) - 1 > args.rows:
    main_times = main_times[-(args.rows + 1) :]

main_times_list = main_times.tolist()
log.info("Simulation steps = %d (→ expected snapshot rows = %d)", len(main_times_list) - 1, len(main_times_list) - 1)

# ───────────────────── containers ─────────────────────
y_true: List[int] = []
y_pred: List[int] = []
records: List[dict] = []
pred_acc: List[int] = []

# ───────────────────── main loop ─────────────────────
for step, now in enumerate(main_times_list[:-1], 1):
    nxt_time = main_times_list[step]

    cur_close = raw_src["30T"].loc[raw_src["30T"]["time"] == now, "close"].iloc[0]
    nxt_close = raw_src["30T"].loc[raw_src["30T"]["time"] == nxt_time, "close"].iloc[0]
    lbl = int(nxt_close > cur_close)
    y_true.append(lbl)

    # update *_live.csv* up to `now`
    for tf in LIVE_CSV:
        dump_until(raw_src[tf], now, LIVE_CSV[tf], ffill=(tf == "1H"))
    if not all(os.path.exists(fp) for fp in LIVE_CSV.values()):
        continue  # wait until all TFs ready

    merged = prep.load_data()
    X_live, _ = prep.ready_incremental(merged, window=win, selected_features=feats)
    X_live = (
        X_live.reindex(columns=cols)  # اطمینان از ترتیب و وجود ستون‌ها
        .astype(float)
    )

    # snapshot row
    X_live.assign(**{"30T_time": now}).to_csv(
        SNAP_FILE,
        mode="a",
        header=not os.path.exists(SNAP_FILE),
        index=False,
    )

    proba = float(pipe.predict_proba(X_live)[:, 1][0])

    # نام ستون‌ها مطابق prepare_data_for_train (حروف کوچک)
    last_atr = merged.get("30T_atr_14", pd.Series([np.nan])).shift(1).iloc[-1]
    last_vol = merged.get("volume", pd.Series([np.nan])).shift(1).iloc[-1]
    last_atr = float(last_atr) if np.isfinite(last_atr) else 1.0
    last_vol = float(last_vol) if np.isfinite(last_vol) else 1_000.0

    n_thr, p_thr = (
        thr_adj.adjust(neg_thr, pos_thr, last_atr, last_vol) if args.dyn_thr else (neg_thr, pos_thr)
    )

    dec = -1
    if proba <= n_thr:
        dec = 0
    elif proba >= p_thr:
        dec = 1
    y_pred.append(dec)

    if dec != -1:
        pred_acc.append(int(dec == lbl))
    cum_acc = float(np.mean(pred_acc)) if pred_acc else 0.0

    txt_dec = {1: "BUY", 0: "SEL", -1: "NAN"}[dec]
    log.info(
        "[%d/%d] %s p=%.4f thr=(%.3f,%.3f) → %-3s true=%s Δ=%.2f acc=%.3f",
        step,
        len(main_times_list) - 1,
        now,
        proba,
        n_thr,
        p_thr,
        txt_dec,
        "BUY" if lbl else "SEL",
        nxt_close - cur_close,
        cum_acc,
    )

    records.append(
        {
            "iteration": step,
            "time": str(now),
            "price": cur_close,
            "delta_price": nxt_close - cur_close,
            "proba": proba,
            "decision": txt_dec,
            "true_dir": "BUY" if lbl else "SEL",
            "correct": dec != -1 and dec == lbl,
            "cumulative_acc": cum_acc,
        }
    )

# ───────────────────── metrics & export ─────────────────────
arr_true = np.array(y_true)
arr_pred = np.array(y_pred)

mask = arr_pred != -1  # فقط سطرهایی که مدل BUY/SEL داده است
if mask.any():
    conf_ratio = float(mask.mean())
    acc = accuracy_score(arr_true[mask], arr_pred[mask])
    f1 = f1_score(arr_true[mask], arr_pred[mask])
else:
    conf_ratio = acc = f1 = 0.0

out_path = Path(args.out)
out_path.parent.mkdir(exist_ok=True, parents=True)
pd.DataFrame(records).to_csv(out_path, index=False)

# clean temp *_live.csv
for fp in LIVE_CSV.values():
    if os.path.exists(fp):
        os.remove(fp)

print(
    f"\nConf-ratio          : {conf_ratio:.3f}"
    f"\nF1-score            : {f1:.4f}"
    f"\nAccuracy            : {acc:.4f}"
    f"\nCSV report          : {out_path.resolve()}"
    f"\nRuntime log         : {Path(LOGFILE).resolve()}"
    f"\nSnapshot features   : {Path(SNAP_FILE).resolve()}\n"
)
