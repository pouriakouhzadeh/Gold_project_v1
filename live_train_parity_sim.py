#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_train_parity_sim.py
هدف: شبیه‌سازی دقیق لایو روی ۴۰۰۰ (یا هر N) نمونه‌ی آخر با نوشتن tail-CSV در هر تکرار،
مقایسه‌ی آن با baseline-by-tail (همان مسیر)، و هم‌زمان سنجش دقت TRAIN (با پچ آفست ۲).
خروجی‌ها داخل --model-dir:
- deploy_live_tail.csv           ← خروجی لایو/بیس‌لاینِ بر اساس tail
- deploy_train_fixed.csv         ← خروجی TRAIN با آفستِ اصلاح‌شده
- deploy_parity.log              ← لاگ کامل + READY/NOT-READY
Exit codes: 0=READY, 2=NOT READY
"""
from __future__ import annotations
import os, sys, math, json, shutil, tempfile, argparse, logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np, pandas as pd, joblib

# ==== project imports ====
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}
ALL_TFS = ("30T","1H","15T","5T")

# ---------- Logging ----------
def setup_logger(path: str, verbose: bool) -> logging.Logger:
    lg = logging.getLogger("deploy-parity")
    lg.setLevel(logging.DEBUG if verbose else logging.INFO)
    lg.propagate = False
    for h in list(lg.handlers): lg.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO); lg.addHandler(sh)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); lg.addHandler(fh)
    return lg

# ---------- IO ----------
def load_model(model_dir: str):
    payload = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    pipeline   = payload["pipeline"]
    window     = int(payload.get("window_size", 1))
    neg_thr    = float(payload.get("neg_thr", 0.5))
    pos_thr    = float(payload.get("pos_thr", 0.5))
    final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    if not isinstance(final_cols, list): final_cols = list(final_cols)
    return payload, pipeline, window, neg_thr, pos_thr, final_cols

def load_raw_csvs(args) -> Dict[str, pd.DataFrame]:
    paths = {tf: os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP[tf]}.csv") for tf in TF_MAP}
    raw: Dict[str, pd.DataFrame] = {}
    for tf, p in paths.items():
        if os.path.isfile(p):
            df = pd.read_csv(p)
            if "time" not in df.columns: raise KeyError(f"{p} missing 'time'")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            raw[tf] = df
        elif tf == "30T":
            raise FileNotFoundError(f"Main TF (30T) missing: {p}")
    if "30T" not in raw: raise FileNotFoundError("30T missing")
    return raw

def write_tail_csvs(raw: Dict[str, pd.DataFrame], symbol: str, cutoff: pd.Timestamp,
                    out_dir: str, hist_rows: Dict[str,int]) -> Dict[str,str]:
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str,str] = {}
    for tf, df in raw.items():
        sub = df[df["time"] <= cutoff]
        if sub.empty: continue
        need = int(hist_rows.get(tf, 200))
        if need>0 and len(sub)>need: sub = sub.tail(need)
        mapped = TF_MAP.get(tf, tf)
        p = os.path.join(out_dir, f"{symbol}_{mapped}.csv")
        cols = ["time"] + [c for c in sub.columns if c!="time"]
        sub[cols].to_csv(p, index=False)
        out[tf] = p
    return out

# ---------- Tail policy ----------
def auto_tail_like(args, raw_30_len: int) -> Dict[str,int]:
    """ساده: برای ۳۰T از args.hist_30t استفاده کن؛ و سایر TF با ضرایب."""
    b30 = int(args.hist_30t)
    return {"30T": b30,
            "1H":  int(max(64,  math.ceil(b30*args.mult_1h))),
            "15T": int(max(128, math.ceil(b30*args.mult_15t))),
            "5T":  int(max(256, math.ceil(b30*args.mult_5t)))}

# ---------- Decide ----------
def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

# ---------- Core ----------
def run_tail_live_and_baseline(args, log, payload, pipe, window, neg_thr, pos_thr, final_cols):
    raw = load_raw_csvs(args)
    main = raw["30T"]
    N = len(main)
    end_i = N-2
    start_i = max(0, end_i - int(args.tail_iters) + 1)
    total = end_i - start_i + 1
    tails = auto_tail_like(args, N)
    log.info("[tails] %s", tails)

    rows = []
    tmp_root = tempfile.mkdtemp(prefix="deploy_tail_")
    try:
        for k in range(total):
            i = start_i + k
            cutoff = pd.to_datetime(main.loc[i,"time"])
            itdir = os.path.join(tmp_root, f"iter_{k:06d}")
            fps = write_tail_csvs(raw, args.symbol, cutoff, itdir, tails)
            if "30T" not in fps: shutil.rmtree(itdir, ignore_errors=True); continue

            prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                          verbose=False, fast_mode=True, strict_disk_feed=True)
            merged = prep.load_data()
            X, _, _, _, t_idx = prep.ready(
                merged, window=window, selected_features=final_cols,
                mode="predict", with_times=True, predict_drop_last=True
            )
            if X.empty: shutil.rmtree(itdir, ignore_errors=True); continue
            X = X.reindex(columns=final_cols, fill_value=0.0)
            prob = float(pipe.predict_proba(X.tail(1))[:,1])
            pred = decide(prob, neg_thr, pos_thr)
            # true از 30T
            c0 = float(main.loc[i,"close"]); c1 = float(main.loc[i+1,"close"])
            true = 1 if (c1-c0)>0 else 0
            rows.append({"iter":k+1,"cutoff":cutoff,"feat_time": (pd.to_datetime(t_idx.iloc[-1]) if t_idx is not None and len(t_idx) else pd.NaT),
                         "prob":prob,"pred":int(pred),"true":int(true)})
            shutil.rmtree(itdir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    live_df = pd.DataFrame(rows)
    live_out = os.path.join(args.model_dir, "deploy_live_tail.csv")
    live_df.to_csv(live_out, index=False)
    mask = live_df["pred"].values != -1
    acc = (float((live_df.loc[mask,"pred"].values == live_df.loc[mask,"true"].values).sum())/max(1, int(mask.sum())))*100.0
    log.info("[LIVE-by-tail] total=%d predicted=%d acc=%.4f%%",
             len(live_df), int(mask.sum()), acc)
    return live_df, acc

def run_train_fixed(args, log, payload, pipe, window, neg_thr, pos_thr, final_cols):
    """TRAIN با پچ آفست۲ (بدون تغییر کلاس اصلی، اینجا آفست را اعمال می‌کنیم)."""
    filepaths = {tf: f"{args.data_dir}/{args.symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()

    # خروجی آماده‌ی TRAIN
    X_tr, y_tr, _, _ = prep.ready(
        merged, window=window, selected_features=final_cols,
        mode="train", predict_drop_last=False, train_drop_last=False
    )
    # --- پچ آفست ۲ برای y (هم‌ترازی با shift(1).diff()) ---
    # اگر آماده‌سازی داخلی تصحیح نشده باشد، همین‌جا اصلاح می‌کنیم:
    if len(y_tr) == len(X_tr) + 2:
        y_tr = y_tr.iloc[2:].reset_index(drop=True)
    elif len(y_tr) >= len(X_tr):
        # حالت عمومی: y را به اندازه‌ی X_tr از ابتدای سری می‌بُریم
        y_tr = y_tr.iloc[:len(X_tr)].reset_index(drop=True)

    if (not X_tr.empty) and all(c in X_tr.columns for c in final_cols):
        X_tr = X_tr[final_cols]

    probs = pipe.predict_proba(X_tr)[:,1] if len(X_tr) else np.array([], float)
    preds = np.full(len(probs), -1, dtype=int)
    preds[probs <= neg_thr] = 0
    preds[probs >= pos_thr] = 1

    df = pd.DataFrame({"prob":probs, "pred":preds, "true": y_tr.values if len(y_tr) else []})
    out = os.path.join(args.model_dir, "deploy_train_fixed.csv")
    df.to_csv(out, index=False)
    mask = df["pred"].values != -1
    acc = (float((df.loc[mask,"pred"].values == df.loc[mask,"true"].values).sum())/max(1,int(mask.sum())))*100.0 if len(df) else 0.0
    log.info("[TRAIN-fixed] total=%d predicted=%d acc=%.4f%%", len(df), int(mask.sum()), acc)
    return df, acc

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy parity simulator: LIVE-by-tail vs TRAIN-fixed.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tail-iters", type=int, default=4000)
    p.add_argument("--hist-30t", type=int, default=6000)  # tail بزرگ تا همگرایی
    p.add_argument("--mult-1h", type=float, default=0.5)
    p.add_argument("--mult-15t", type=float, default=2.0)
    p.add_argument("--mult-5t", type=float, default=6.0)
    p.add_argument("--acc-diff-tol", type=float, default=0.25, help="حداکثر اختلاف قابل‌قبول بین TRAIN و LIVE")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", default="deploy_parity.log")
    return p.parse_args()

def main():
    args = parse_args()
    log = setup_logger(os.path.join(args.model_dir, args.log_file), args.verbose)
    log.info("=== deploy parity starting ===")
    payload, pipe, window, neg_thr, pos_thr, final_cols = load_model(args.model_dir)
    log.info("Model window=%d thr(neg=%.3f,pos=%.3f) cols=%d", window, neg_thr, pos_thr, len(final_cols))

    live_df, live_acc = run_tail_live_and_baseline(args, log, payload, pipe, window, neg_thr, pos_thr, final_cols)
    tr_df, tr_acc     = run_train_fixed(args, log, payload, pipe, window, neg_thr, pos_thr, final_cols)

    # مقایسه‌ی اختلاف دقت
    diff = abs(live_acc - tr_acc)
    log.info("[COMPARE] train=%.4f%%  live=%.4f%%  | Δ=%.4f%% (tol=%.4f%%)", tr_acc, live_acc, diff, args.acc_diff_tol)

    ready = (diff <= args.acc_diff_tol)
    if ready:
        log.info("READY ✅  → دقت TRAIN و LIVE هم‌تراز هستند.")
        sys.exit(0)
    else:
        log.error("NOT-READY ❌  → اختلاف دقت بیش از tol است.")
        sys.exit(2)

if __name__ == "__main__":
    main()


# python3 -u live_train_parity_sim.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 4000 \
#   --hist-30t 6000 \
#   --verbose

