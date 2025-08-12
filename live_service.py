#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_service.py — سرویس لایو برای اتصال به متاتریدر (MQL4)
- منتظر می‌ماند تا هر چهار CSV (M5/M15/M30/H1) برای یک کندل 30دقیقه جدید آماده شوند
- PREPARE_DATA_FOR_TRAIN با mode="predict" فیچر آخرین ردیف پایدار را می‌سازد
- با مدل آموزش‌دیده (همان pipeline + کالیبراسیون) پیش‌بینی می‌زند
- آستانه‌ها را اعمال می‌کند و نتیجه را در signal.json (یا CSV) می‌نویسد
"""

from __future__ import annotations
import argparse, json, time, sys, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import joblib

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("Live service for MT4 CSVs")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--data-dir", default=".", help="Folder with live CSVs from MT4")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix")
    p.add_argument("--poll-sec", type=float, default=2.0, help="Polling interval (seconds)")
    p.add_argument("--ctx-5t", type=int, default=3000)
    p.add_argument("--ctx-15t", type=int, default=1200)
    p.add_argument("--ctx-30t", type=int, default=500)
    p.add_argument("--ctx-1h", type=int, default=300)
    p.add_argument("--out-json", default="signal.json", help="Where to write the latest signal")
    p.add_argument("--log-file", default="live_service.log")
    return p.parse_args()

# ---------- Logging ----------
def setup_logger(path: str):
    log = logging.getLogger("live_service")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers): log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log

# ---------- Paths ----------
BASE = {"5T":"{sym}_M5_live.csv","15T":"{sym}_M15_live.csv","30T":"{sym}_M30_live.csv","1H":"{sym}_H1_live.csv"}

def csv_paths(data_dir: Path, sym: str) -> Dict[str, Path]:
    return {tf: data_dir / patt.format(sym=sym) for tf, patt in BASE.items()}

# ---------- IO ----------
def read_csv_tail(fp: Path, ctx: int) -> pd.DataFrame:
    # خواندن ساده؛ اگر فایل خیلی بزرگ است می‌توانید از خواندن chunkی استفاده کنید
    df = pd.read_csv(fp)
    df = df.tail(max(2, ctx)).copy()
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def last_time_of(df: pd.DataFrame) -> pd.Timestamp | None:
    return None if df.empty else pd.to_datetime(df["time"].iloc[-1])

def write_signal_atomic(obj: dict, out_path: Path):
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(out_path)

# ---------- PREP ----------
def build_prep(live_files: Dict[str, str]):
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
    try:
        return PREPARE_DATA_FOR_TRAIN(main_timeframe="30T", filepaths=live_files, verbose=False, fast_mode=True)
    except TypeError:
        return PREPARE_DATA_FOR_TRAIN(main_timeframe="30T", filepaths=live_files, verbose=False)

# ---------- MAIN ----------
def main():
    args = parse_args()
    log  = setup_logger(args.log_file)

    # مدل
    payload = joblib.load(args.model)
    pipe    = payload["pipeline"]
    feats   = list(payload["train_window_cols"])
    window  = int(payload["window_size"])
    neg_thr = float(payload["neg_thr"])
    pos_thr = float(payload["pos_thr"])

    log.info("Model loaded: %s | window=%d | neg=%.3f | pos=%.3f | #feats=%d",
             Path(args.model).name, window, neg_thr, pos_thr, len(feats))

    data_dir = Path(args.data_dir).resolve()
    paths = csv_paths(data_dir, args.symbol)
    ctx   = {"5T":args.ctx_5t, "15T":args.ctx_15t, "30T":args.ctx_30t, "1H":args.ctx_1h}
    out_json = Path(args.out_json).resolve()

    # PREP دائمی (هر بار همان مسیر فایل‌ها، محتوا آپدیت می‌شود)
    prep = build_prep({tf: str(p) for tf, p in paths.items()})

    last_processed: pd.Timestamp | None = None
    log.info("Service started. Watching folder: %s", data_dir)

    while True:
        try:
            # 1) بخوان: هر چهار CSV با کانتکست مورد نیاز
            if not all(p.is_file() for p in paths.values()):
                time.sleep(args.poll_sec); continue

            dfs = {tf: read_csv_tail(fp, ctx[tf]) for tf, fp in paths.items()}
            lt  = {tf: last_time_of(df) for tf, df in dfs.items()}

            # 2) هم‌زمانی ساده: همه باید آخرین «time» را داشته باشند و M30 زمان مرجع است
            t30 = lt["30T"]
            if t30 is None:
                time.sleep(args.poll_sec); continue

            # اگر CSVهای دیگر عقب‌ترند، صبر کن
            if any(lt[tf] is None or lt[tf] < t30 for tf in ("5T","15T","1H")):
                time.sleep(args.poll_sec); continue

            # دابل‌پردازش نکن
            if (last_processed is not None) and (t30 <= last_processed):
                time.sleep(args.poll_sec); continue

            # 3) آماده‌سازی: PREP از همین فایل‌ها می‌خواند (mode="predict")
            merged = prep.load_data()
            tcol = "30T_time" if "30T_time" in merged.columns else "time"
            if merged.empty or merged[tcol].isna().all():
                time.sleep(args.poll_sec); continue

            # آخرین رکورد پایدار (ممکن است به هر دلیل t_last <= t30 باشد)
            t_last = pd.to_datetime(merged[tcol].dropna().iloc[-1])

            X, _, _, _ = prep.ready(merged, window=window, selected_features=feats, mode="predict")
            if X.empty:
                time.sleep(args.poll_sec); continue

            # کنترل ستون‌ها + تمیزکاری
            miss = [c for c in feats if c not in X.columns]
            if miss:
                log.warning("Missing %d features, skip. ex: %s", len(miss), ", ".join(miss[:6]))
                time.sleep(args.poll_sec); continue

            x_last = X.iloc[[-1]][feats].replace([np.inf, -np.inf], np.nan)
            if x_last.isna().any().any():
                med = X[feats].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
                x_last = x_last.fillna(med).fillna(0.0)
            if not np.all(np.isfinite(x_last.values)):
                log.warning("Non-finite after fill. Skip.")
                time.sleep(args.poll_sec); continue
            x_last = x_last.astype("float32")

            # 4) پیش‌بینی + آستانه
            proba = float(pipe.predict_proba(x_last)[:,1][0])
            if proba <= neg_thr: signal = 0
            elif proba >= pos_thr: signal = 1
            else: signal = -1

            # 5) خروجی برای MT4 (اتمیک)
            msg = {
                "time": t_last.strftime("%Y-%m-%d %H:%M"),
                "proba": round(proba, 6),
                "signal": int(signal),   # -1=no-trade, 0=sell/flat, 1=buy
                "neg_thr": neg_thr,
                "pos_thr": pos_thr,
                "window": window,
                "n_features": len(feats),
            }
            write_signal_atomic(msg, out_json)
            log.info("Signal @ %s → %s (p=%.4f)", msg["time"], msg["signal"], msg["proba"])

            last_processed = t30
            time.sleep(args.poll_sec)

        except KeyboardInterrupt:
            log.info("Interrupted. Bye.")
            break
        except Exception as e:
            logging.getLogger("live_service").exception("Error in loop: %s", e)
            time.sleep(max(2.0, args.poll_sec))
            

if __name__ == "__main__":
    main()
