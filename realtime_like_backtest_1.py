#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_like_backtest.py  –  FINAL (intersection‑sync, 2025‑06‑12)
────────────────────────────────────────────────────────────────────────
Back‑test the model candle‑by‑candle *only* at timestamps that exist in
**all** four source time‑frames (30T, 15T, 5T, 1H). This guarantees that
feature preparation never sees partially‑formed candles and eliminates
skip/NAN events caused by misaligned data.
"""
from __future__ import annotations

import argparse
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Dict, List, Set

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from dynamic_threshold_adjuster import DynamicThresholdAdjuster

# ───────────────────────── configuration ─────────────────────────
RAW_FILES: Dict[str, str] = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T" : "XAUUSD_M5.csv",
    "1H" : "XAUUSD_H1.csv",
}
LIVE_CSV: Dict[str, str] = {tf: fp.replace(".csv", f".F_{tf}_live.csv") for tf, fp in RAW_FILES.items()}
MAX_ROWS = 5_000
LOGFILE  = "realtime_like_backtest.log"

# ───────────────────────── cleaning helper ─────────────────────────

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "")
    return df.reset_index(drop=True)

# ───────────────────────── load helper ─────────────────────────

def load_src(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["time"]).sort_values("time", ignore_index=True)

# ───────────────────────── slice writer ─────────────────────────

def dump_until(df: pd.DataFrame, ts: pd.Timestamp, out_fp: str) -> None:
    idx = df["time"].searchsorted(ts, side="right") - 1
    slice_df = clean_df(df.iloc[: idx + 1].tail(MAX_ROWS).copy())
    slice_df.to_csv(out_fp, index=False)

# ───────────────────────── CLI ─────────────────────────
cli = argparse.ArgumentParser("Realtime‑like back‑test (intersection mode)")
cli.add_argument("--start", default="2024-01-02", help="start date (YYYY‑MM‑DD)")
cli.add_argument("-m", "--model", default="best_model.pkl", help="trained model pickle")
cli.add_argument("-o", "--out", default="realtime_like_report.csv", help="CSV report path")
cli.add_argument("--dyn-thr", action="store_true", help="enable dynamic thresholding")
cli.add_argument("--verbose", action="store_true", help="verbose console output")
args = cli.parse_args()

# ───────────────────────── logging ─────────────────────────
file_hd = logging.FileHandler(LOGFILE, mode="w", encoding="utf-8")
file_hd.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
file_hd.setLevel(logging.DEBUG)

console_hd = logging.StreamHandler()
console_hd.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
console_hd.setLevel(logging.DEBUG if args.verbose else logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[file_hd, console_hd], force=True)
log = logging.getLogger("rt_backtest")

# ───────────────────────── load raw csvs ─────────────────────────
raw_src = {tf: load_src(fp) for tf, fp in RAW_FILES.items()}
log.info("✅ Loaded %d source CSVs", len(raw_src))

# remove old live csvs
for fp in LIVE_CSV.values():
    if os.path.exists(fp):
        os.remove(fp)

# ───────────────────────── build intersection timeline ─────────────────────────
common_times: Set[pd.Timestamp] = set(raw_src["30T"]["time"])
for tf in ("15T", "5T", "1H"):
    common_times &= set(raw_src[tf]["time"])
common_times = sorted(t for t in common_times if t >= pd.Timestamp(args.start))

if len(common_times) < 2:
    raise RuntimeError("Not enough overlapping timestamps after start date.")

log.info("Timeline length after intersection: %d", len(common_times))

# ───────────────────────── model & helpers ─────────────────────────
mdl      = joblib.load(args.model)
pipe     = mdl["pipeline"]
neg_thr  = mdl["neg_thr"]
pos_thr  = mdl["pos_thr"]
win      = mdl["window_size"]
feats    = mdl["feats"]
cols     = mdl["train_window_cols"]
log.info("✅ Model loaded | window=%d | feats=%d", win, len(cols))

thr_adj = DynamicThresholdAdjuster(atr_high=10.0, vol_low=500.0, shift=0.01)

prep = PREPARE_DATA_FOR_TRAIN(filepaths=LIVE_CSV, main_timeframe="30T", verbose=False)

prep = PREPARE_DATA_FOR_TRAIN(filepaths=LIVE_CSV, main_timeframe="30T", verbose=False)

# ───────────────────────── main loop ─────────────────────────
y_true: List[int] = []
y_pred: List[int] = []
records: List[dict] = []
pred_acc: List[int] = []

for step, now in enumerate(common_times[:-1], 1):  # exclude the very last (needs next close)
    # indices for price lookup
    idx = raw_src["30T"]["time"].searchsorted(now)
    curC = raw_src["30T"].at[idx, "close"]
    nxtC = raw_src["30T"].at[idx + 1, "close"]
    lbl  = int(nxtC > curC)
    y_true.append(lbl)

    if args.verbose:
        print(f"[{step}/{len(common_times)-1}] {now}", flush=True)

    # write synced live CSVs
    for tf in LIVE_CSV:
        dump_until(raw_src[tf], now, LIVE_CSV[tf])

    merged = prep.load_data()
    prep_out = prep.ready_incremental(merged, window=win, selected_features=feats)
    X_live = (prep_out[0] if isinstance(prep_out, tuple) else prep_out)[cols].astype(float).tail(1)

    proba = float(pipe.predict_proba(X_live)[:, 1][0])

    last_atr = merged["30T_ATR_14"].shift(1).iloc[-1] if "30T_ATR_14" in merged.columns else np.nan
    last_vol = merged["30T_volume"].shift(1).iloc[-1] if "30T_volume" in merged.columns else np.nan
    last_atr = float(last_atr) if pd.notna(last_atr) else 1.0
    last_vol = float(last_vol) if pd.notna(last_vol) else 1_000.0

    n_thr, p_thr = thr_adj.adjust(neg_thr, pos_thr, last_atr, last_vol) if args.dyn_thr else (neg_thr, pos_thr)

    dec = -1
    if proba <= n_thr:
        dec = 0
    elif proba >= p_thr:
        dec = 1
    y_pred.append(dec)

    if dec != -1:
        pred_acc.append(int(dec == lbl))

    cum_acc  = np.mean(pred_acc) if pred_acc else 0.0
    roll_acc = np.mean(pred_acc[-50:]) if len(pred_acc) >= 50 else np.nan

    txt_dec = {1: "BUY", 0: "SEL", -1: "NAN"}[dec]
    log.info("[%d/%d] %s p=%.4f thr=(%.3f,%.3f) → %-3s true=%s close=%.2f Δ=%.2f acc=%.3f roll50=%.3f",
             step, len(common_times)-1, now, proba, n_thr, p_thr, txt_dec,
             "BUY" if lbl else "SEL", curC, nxtC-curC, cum_acc, roll_acc)

    records.append({
        "iteration": step,
        "time": str(now),
        "price": curC,
        "delta_price": nxtC - curC,
        "proba": proba,
        "neg_thr": n_thr,
        "pos_thr": p_thr,
        "decision": txt_dec,
        "true_dir": "BUY" if lbl else "SEL",
        "correct": dec != -1 and dec == lbl,
        "cumulative_acc": cum_acc,
        "rolling_acc50": roll_acc,
    })

# ───────────────────────── metrics & export ─────────────────────────
arr_true, arr_pred = np.array(y_true), np.array(y_pred)
mask = arr_pred != -1
conf_ratio = mask.mean() if mask.size else 0.0
acc = accuracy_score(arr_true[mask], arr_pred[mask]) if mask.any() else 0.0
f1  = f1_score(arr_true[mask], arr_pred[mask]) if mask.any() else 0.0

out_path = Path(args.out)
out_path.parent.mkdir(exist_ok=True, parents=True)
pd.DataFrame(records).to_csv(out_path, index=False)

# cleanup live CSVs
for fp in LIVE_CSV.values():
    if os.path.exists(fp):
        os.remove(fp)

print(f"\nConf‑ratio: {conf_ratio:.3f}   F1: {f1:.4f}   Acc: {acc:.4f}")
print(f"CSV → {out_path.resolve()}")
print(f"LOG → {Path(LOGFILE).resolve()}")
