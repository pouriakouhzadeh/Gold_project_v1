#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_like_backtest.py – self‑contained, tested 30‑minute back‑tester
────────────────────────────────────────────────────────────────────────
* Predicts on **every 30‑minute close** (main timeframe = 30T)
* 1‑hour timeframe is forward‑filled with its latest closed candle so no
  look‑ahead occurs.
* Cleans each slice (Inf→NaN, fill, dedup) before writing *_live.csv*.
* Uses exactly the same `PREPARE_DATA_FOR_TRAIN` class as training.
* Outputs a CSV report plus concise console metrics.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

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
LIVE_CSV: Dict[str, str] = {tf: fp.replace(".csv", f".F_{tf}_live.csv") for tf, fp in RAW_FILES.items()}
MAX_ROWS = 10_000
LOGFILE = "realtime_like_backtest.log"

# ───────────────────── cleaning helper ─────────────────────

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning similar to training stage."""
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "")
    return df.reset_index(drop=True)

# ───────────────────── load helper ─────────────────────

def load_src(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return (
        pd.read_csv(path, parse_dates=["time"], low_memory=False)
        .sort_values("time", ignore_index=True)
    )

# ───────────────────── slice writer ─────────────────────

def dump_until(df: pd.DataFrame, ts: pd.Timestamp, out_fp: str, ffill: bool) -> None:
    """Write rows ≤ *ts*; if *ffill* and last row < ts, copy last row with time=ts."""
    idx = df["time"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return
    T1 = '0 days + 01:00:00'
    T1 = pd.Timedelta(T1)
    if ts - df.loc[idx,"time"] > T1 :
        return
    slice_df = df.iloc[: idx + 1].tail(MAX_ROWS).copy()
    # if ffill and slice_df.iloc[-1]["time"] > ts:
    #     last_row = slice_df.iloc[-1].copy()
    #     last_row["time"] = ts
    #     slice_df = pd.concat([slice_df, last_row.to_frame().T], ignore_index=True)
    slice_df = clean_df(slice_df)
    slice_df.to_csv(out_fp, index=False)

# ───────────────────── CLI ─────────────────────
cli = argparse.ArgumentParser("Realtime back‑test (30‑minute cadence)")
cli.add_argument("--start", default="2024-01-02", help="start date (YYYY‑MM‑DD)")
cli.add_argument("-m", "--model", default="best_model.pkl")
cli.add_argument("-o", "--out", default="realtime_like_report.csv")
cli.add_argument("--dyn-thr", action="store_true")
cli.add_argument("--verbose", action="store_true")
args = cli.parse_args()

# ───────────────────── logging ─────────────────────
file_hd = logging.FileHandler(LOGFILE, mode="w", encoding="utf-8")
file_hd.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
file_hd.setLevel(logging.DEBUG)
console_hd = logging.StreamHandler()
console_hd.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
console_hd.setLevel(logging.DEBUG if args.verbose else logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[file_hd, console_hd], force=True)
log = logging.getLogger("rt_backtest")

# ───────────────────── load data & model ─────────────────────
raw_src = {tf: load_src(fp) for tf, fp in RAW_FILES.items()}
for tf in raw_src:
    raw_src[tf] = (
        raw_src[tf]
        # 1) حذف کندل‌های شنبه/یکشنبه
        .loc[~raw_src[tf]["time"].dt.dayofweek.isin([5, 6])]
        # 2) حذف هر نوع ردیفِ با تایم‌استمپ تکراری (فقط آخرین ردیف می‌مانَد)
        .drop_duplicates(subset="time", keep="last")
        .reset_index(drop=True)
    )

log.info("✅ Loaded %d source CSVs (weekends & duplicates removed)", len(raw_src))


for fp in LIVE_CSV.values():
    if os.path.exists(fp):
        os.remove(fp)

mdl = joblib.load(args.model)
pipe, neg_thr, pos_thr = mdl["pipeline"], mdl["neg_thr"], mdl["pos_thr"]
win, feats, cols = mdl["window_size"], mdl["feats"], mdl["train_window_cols"]
log.info("✅ Model loaded | window=%d | feats=%d", win, len(cols))

thr_adj = DynamicThresholdAdjuster(atr_high=10.0, vol_low=500.0, shift=0.01)
prep = PREPARE_DATA_FOR_TRAIN(filepaths=LIVE_CSV, main_timeframe="30T", verbose=False)

main_times = raw_src["30T"]["time"]
main_times = main_times[~main_times.dt.dayofweek.isin([5, 6])] 
start_idx = main_times.searchsorted(pd.Timestamp(args.start), side="left")
main_times = main_times.iloc[start_idx:-1]
main_times_list = main_times.tolist()   
log.info("Total 30‑minute steps: %d", len(main_times))

# ───────────────────── main loop ─────────────────────
y_true: List[int] = []
y_pred: List[int] = []
records: List[dict] = []
pred_acc: List[int] = []

for step, now in enumerate(main_times_list[:-1], 1):      # ❶
    print(f"\n\n\n")
    try :
        print(f"Iteration : {len(y_pred)} , wins = {np.sum(pred_acc)} loses = {len(pred_acc) - np.sum(pred_acc)} , ACC = {(np.round(cum_acc*100))/100}\n\n\n")
    except:
        print("Some variables not initial yet . wait until fill all variables...")
        print(f"\n\n\n")
    cur_time = now
    nxt_time = main_times_list[step]          # همیشه یک گام جلوتر

    curC = raw_src["30T"].loc[raw_src["30T"]["time"] == cur_time, "close"].iloc[0]
    nxtC = raw_src["30T"].loc[raw_src["30T"]["time"] == nxt_time, "close"].iloc[0]
    lbl  = int(nxtC > curC)
    y_true.append(lbl)
    
    if args.verbose:
        print(f"[{step}/{len(main_times)}] {now}")

    # write live CSVs: forward‑fill only for 1H
    for tf in LIVE_CSV:
        dump_until(raw_src[tf], now, LIVE_CSV[tf], ffill=(tf == "1H"))
    Flag = True
    for fp in LIVE_CSV.values():
        if not(os.path.exists(fp)):
            Flag = False
    if Flag == True :        
        merged = prep.load_data()
        prep_out = prep.ready_incremental(merged, window=win, selected_features=feats)
        X_live = (prep_out[0] if isinstance(prep_out, tuple) else prep_out)[cols].astype(float).tail(1)

        proba = float(pipe.predict_proba(X_live)[:, 1][0])

        last_atr = merged["30T_ATR_14"].shift(1).iloc[-1] if "30T_ATR_14" in merged.columns else np.nan
        last_vol = merged["30T_volume"].shift(1).iloc[-1] if "30T_volume" in merged.columns else np.nan
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
            "[%d/%d] %s p=%.4f thr=(%.3f,%.3f) → %-3s true=%s close=%.2f Δ=%.2f acc=%.3f",
            step,
            len(main_times),
            now,
            proba,
            n_thr,
            p_thr,
            txt_dec,
            "BUY" if lbl else "SEL",
            curC,
            nxtC - curC,
            cum_acc,
        )

        records.append(
            {
                "iteration": step,
                "time": str(now),
                "price": curC,
                "delta_price": nxtC - curC,
                "proba": proba,
                "decision": txt_dec,
                "true_dir": "BUY" if lbl else "SEL",
                "correct": dec != -1 and dec == lbl,
                "cumulative_acc": cum_acc,
            }
        )
    else :
        print("One or more CSV files not created")

# ───────────────────── metrics & export ─────────────────────
arr_true = np.array(y_true)
arr_pred = np.array(y_pred)
mask = arr_pred != -1
conf_ratio = float(mask.mean()) if mask.any() else 0.0
acc = accuracy_score(arr_true[mask], arr_pred[mask]) if mask.any() else 0.0
f1 = f1_score(arr_true[mask], arr_pred[mask]) if mask.any() else 0.0

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(out_path, index=False)

for fp in LIVE_CSV.values():
    if os.path.exists(fp):
        os.remove(fp)

print(
    f"\nConf‑ratio: {conf_ratio:.3f}   F1: {f1:.4f}   Acc: {acc:.4f}\n"
    f"CSV  → {out_path.resolve()}\n"
    f"LOG  → {Path(LOGFILE).resolve()}"
)
