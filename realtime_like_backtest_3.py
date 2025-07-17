#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_like_backtest.py – FINAL (30‑minute cadence, forward‑fill 1H)
────────────────────────────────────────────────────────────────────────
Predicts on every 30‑minute close; 1‑hour timeframe is forward‑filled
with its latest closed candle to avoid peeking ahead.
"""
from __future__ import annotations

import argparse, logging, os
from pathlib import Path
from typing import Dict, List

import joblib, numpy as np, pandas as pd
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
LIVE_CSV: Dict[str, str] = {tf: fp.replace(".csv", f".F_{tf}_live.csv")
                            for tf, fp in RAW_FILES.items()}
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
    """Write rows ≤ *ts*; forward‑fill 1H if last row < ts."""
    idx = df["time"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return
    slice_df = df.iloc[: idx + 1].tail(MAX_ROWS).copy()
    # forward‑fill candle time if necessary
    if slice_df.iloc[-1]["time"] < ts:
        last_row = slice_df.iloc[-1].copy()
        last_row["time"] = ts
        slice_df = pd.concat([slice_df, last_row.to_frame().T], ignore_index=True)
    slice_df = clean_df(slice_df)
    slice_df.to_csv(out_fp, index=False)

# ───────────────────────── CLI ─────────────────────────
cli = argparse.ArgumentParser("Realtime back‑test (30‑minute cadence)")
cli.add_argument("--start", default="2024-01-02", help="start date (YYYY‑MM‑DD)")
cli.add_argument("-m", "--model", default="best_model.pkl")
cli.add_argument("-o", "--out", default="realtime_like_report.csv")
cli.add_argument("--dyn-thr", action="store_true")
cli.add_argument("--verbose", action="store_true")
args = cli.parse_args()

# ───────────────────────── logging ─────────────────────────
file_hd = logging.FileHandler(LOGFILE, mode="w", encoding="utf-8")
file_hd.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       "%Y-%m-%d %H:%M:%S"))
file_hd.setLevel(logging.DEBUG)
console_hd = logging.StreamHandler()
console_hd.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
console_hd.setLevel(logging.DEBUG if args.verbose else logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[file_hd, console_hd], force=True)
log = logging.getLogger("rt_backtest")

# ───────────────────────── data & model ─────────────────────────
raw_src = {tf: load_src(fp) for tf, fp in RAW_FILES.items()}
log.info("✅ Loaded %d source CSVs", len(raw_src))
for fp in LIVE_CSV.values():
    if os.path.exists(fp):
        os.remove(fp)

mdl = joblib.load(args.model)
pipe, neg_thr, pos_thr = mdl["pipeline"], mdl["neg_thr"], mdl["pos_thr"]
win, feats, cols = mdl["window_size"], mdl["feats"], mdl["train_window_cols"]
log.info("✅ Model loaded | window=%d | feats=%d", win, len(cols))

thr_adj = DynamicThresholdAdjuster(atr_high=10.0, vol_low=500.0, shift=0.01)
prep = PREPARE_DATA_FOR_TRAIN(filepaths=LIVE_CSV, main_timeframe="30T", verbose=False)

# 30‑minute timeline
main_times = raw_src["30T"]["time"]
start_idx  = main_times.searchsorted(pd.Timestamp(args.start), side="left")
main_times = main_times.iloc[start_idx:-1]
log.info("Total 30‑minute steps: %d", len(main_times))

# ───────────────────────── main loop ─────────────────────────
y_true: List[int] = []
y_pred: List[int] = []
records: List[dict] = []
pred_acc: List[int] = []

for step, now in enumerate(main_times, 1):
    print(f"\n\n\n")
    try :
        print(f"Iteration : {len(y_pred)} , wins = {np.sum(pred_acc)} loses = {len(pred_acc) - np.sum(pred_acc)} , ACC = {(np.round(cum_acc*100))/100}\n\n\n")
    except:
        print("Some variables not initial yet . wait until fill all variables...")
        print(f"\n\n\n")
    idx   = raw_src["30T"]["time"].searchsorted(now)
    curC  = raw_src["30T"].iat[idx, raw_src["30T"].columns.get_loc("close")]
    nxtC  = raw_src["30T"].iat[idx + 1, raw_src["30T"].columns.get_loc("close")]
    lbl   = int(nxtC > curC)
    y_true.append(lbl)

    if args.verbose:
        print(f"[{step}/{len(main_times)}] {now}", flush=True)

    # build live CSVs
    for tf in LIVE_CSV:
        dump_until(raw_src[tf], now, LIVE_CSV[tf])

    merged   = prep.load_data()
    prep_out = prep.ready_incremental(merged, window=win, selected_features=feats)
    X_live   = (prep_out[0] if isinstance(prep_out, tuple) else prep_out)[cols].astype(float).tail(1)

    proba = float(pipe.predict_proba(X_live)[:, 1][0])

    last_atr = merged["30T_ATR_14"].shift(1).iloc[-1] if "30T_ATR_14" in merged.columns else np.nan
    last_vol = merged["30T_volume"].shift(1).iloc[-1] if "30T_volume" in merged.columns else np.nan
    last_atr = float(last_atr) if np.isfinite(last_atr) else 1.0
    last_vol = float(last_vol) if np.isfinite(last_vol) else 1_000.0

    n_thr, p_thr = (thr_adj.adjust(neg_thr, pos_thr, last_atr, last_vol)
                    if args.dyn_thr else (neg_thr, pos_thr))

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
    log.info(
        f"[{step}/{len(main_times)}] {now} p={proba:.4f} thr=({n_thr:.3f},{p_thr:.3f}) → {txt_dec:3} true={'BUY' if lbl else 'SEL'} "
        f"close={curC:.2f} Δ={nxtC-curC:.2f} acc={cum_acc:.3f} roll50={roll_acc:.3f}")

    records.append({
        "iteration": step, "time": str(now), "price": curC, "delta_price": nxtC - curC,
        "proba": proba, "neg_thr": n_thr, "pos_thr": p_thr, "decision": txt_dec,
        "true_dir": "BUY" if lbl else "SEL", "correct": dec != -1 and dec == lbl,
        "cumulative_acc": cum_acc})
