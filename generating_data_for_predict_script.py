#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generating_data_for_predict_script.py
Author: Pouria + Assistant

وظایف:
  - از CSVهای اصلی (M5,M15,M30,H1) می‌خواند.
  - با PREPARE_DATA_FOR_TRAIN داده خام ادغام‌شده می‌سازد و مرتب می‌کند.
  - برای steps قدم آخر سری، در هر قدم ۴ فایل *_live.csv تا زمان t می‌نویسد.
  - منتظر answer.txt از اسکریپت دپلوی می‌ماند (حداکثر timeout-sec).
  - تارگت را با prep.ready(...) روی همان برش تا t استخراج و با answer مقایسه می‌کند.
  - آمار لحظه‌ای را روی کنسول چاپ می‌کند.
  - سپس answer.txt را حذف می‌کند و به قدم بعدی می‌رود.

پارامترها:
  --raw-dir
  --watch-dir
  --steps (default: 200)
  --timeout-sec (default: 40)
  --positive-class (default: 1)  => BUY ↔️ این کلاس
  --cleanup-answer               => اگر از قبل answer.txt هست، قبل از شروع قدم حذف کند
  --log-path
"""

from __future__ import annotations
import os, sys, time, argparse, logging
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "generator_parity.log"

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("generator_parity")
    logger.handlers.clear(); logger.propagate = False
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def resolve_timeframe_paths(base_dir: Path, symbol: str="XAUUSD") -> Dict[str, Path]:
    cands = {
        "30T": [f"{symbol}_M30.csv", f"{symbol}_30T.csv"],
        "15T": [f"{symbol}_M15.csv", f"{symbol}_15T.csv"],
        "5T" : [f"{symbol}_M5.csv",  f"{symbol}_5T.csv" ],
        "1H" : [f"{symbol}_H1.csv",  f"{symbol}_1H.csv" ],
    }
    out: Dict[str, Path] = {}
    for tf, names in cands.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found = p; break
        if not found:
            for child in base_dir.iterdir():
                if child.is_file() and any(child.name.lower()==nm.lower() for nm in names):
                    found = child; break
        if found:
            out[tf] = found
    if "30T" not in out:
        raise FileNotFoundError("Main timeframe file (M30/30T) not found.")
    return out

def read_csv_mql4(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    tc = pick("time","datetime","date")
    oc = pick("open"); hc = pick("high"); lc = pick("low"); cc = pick("close")
    vc = pick("volume","tick_volume","tickvolume")
    df = df.rename(columns={tc:"time", oc:"open", hc:"high", lc:"low", cc:"close", vc:"volume"})
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def write_live_subset(df: pd.DataFrame, ts: pd.Timestamp, out_path: Path):
    sub = df.loc[:ts].copy()
    sub = sub.reset_index().rename(columns={"time":"time"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--watch-dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--timeout-sec", type=int, default=40)
    ap.add_argument("--positive-class", type=int, default=1)
    ap.add_argument("--cleanup-answer", action="store_true")
    ap.add_argument("--log-path", default=str(Path.home() / "generator_parity.log"))
    args = ap.parse_args()

    logger = setup_logger(args.log_path)
    logger.info("=== generator_parity started ===")

    raw_dir = Path(args.raw_dir).resolve()
    watch_dir = Path(args.watch_dir).resolve()
    files = resolve_timeframe_paths(raw_dir, "XAUUSD")
    for tf in ("30T","15T","5T","1H"):
        if tf in files:
            logger.info("[raw] %s  -> %s", tf, files[tf])

    # Load raw timeframe dfs once
    df_30 = read_csv_mql4(files["30T"])
    df_15 = read_csv_mql4(files["15T"])
    df_05 = read_csv_mql4(files["5T"])
    df_1H = read_csv_mql4(files["1H"])

    # PREP for targets
    prep = PREPARE_DATA_FOR_TRAIN(filepaths={k:str(v) for k,v in files.items()},
                                  main_timeframe="30T", verbose=True,
                                  fast_mode=False, strict_disk_feed=False)
    raw_merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    raw_merged[tcol] = pd.to_datetime(raw_merged[tcol], utc=False)
    raw_merged.sort_values(tcol, inplace=True)
    raw_merged.reset_index(drop=True, inplace=True)
    total = len(raw_merged)

    # انتخاب بازه: 200 ردیف آخر 30T (با کمی حاشیه)
    last_n = max(200, args.steps)
    start_idx = max(0, total - last_n)
    times = list(raw_merged[tcol].iloc[start_idx : start_idx + args.steps])

    # مسیر فایل‌های live
    m5_live  = watch_dir / "XAUUSD_M5_live.csv"
    m15_live = watch_dir / "XAUUSD_M15_live.csv"
    m30_live = watch_dir / "XAUUSD_M30_live.csv"
    h1_live  = watch_dir / "XAUUSD_H1_live.csv"
    answer_fp = watch_dir / "answer.txt"

    # اگر پاسخ از قبل مانده، حذف کنیم
    if args.cleanup_answer and answer_fp.exists():
        try: answer_fp.unlink()
        except Exception: pass

    decided = correct = incorrect = none = timeout = 0

    logger.info("[INIT] main_timeframe=30T")

    for i, ts in enumerate(times, start=1):
        # ۱) ۴ فایل live تا لحظه ts
        write_live_subset(df_30, ts, m30_live)
        write_live_subset(df_15, ts, m15_live)
        write_live_subset(df_05, ts, m5_live)
        write_live_subset(df_1H, ts, h1_live)
        logger.info("[Step %d/%d] wrote *_live.csv @ %s — waiting for answer.txt ...",
                    i, args.steps, ts)

        # ۲) منتظر answer از دپلوی
        t0 = time.time()
        got = False
        while time.time() - t0 < args.timeout_sec:
            if answer_fp.exists():
                got = True; break
            time.sleep(1.0)

        if not got:
            timeout += 1
            logger.info("[Result %d] ts=%s answer=TIMEOUT | target=? | —", i, ts)
            logger.info("[Running] decided=%d correct=%d incorrect=%d none=%d timeout=%d | acc(decidable)=%.2f%% | coverage=%.2f%%",
                        decided, correct, incorrect, none, timeout,
                        (100.0*correct/decided if decided else 0.0),
                        (100.0*decided/(decided+none+timeout) if (decided+none+timeout)>0 else 0.0))
            # فایل‌های live را دست نمی‌زنیم؛ دپلوی خودش در صورت پاسخ پاک می‌کند.
            continue

        # ۳) پاسخ را بخوانیم
        try:
            ans_text = (answer_fp.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans_text = "NONE"

        # ۴) تارگت را با همان مسیر آماده‌سازی آموزش از برش تا ts بسازیم
        df_cut = raw_merged.loc[raw_merged[tcol] <= ts].copy()
        Xc, yc, _, _ = prep.ready(
            df_cut,
            window=2,  # window از متادیتا شما 2 است؛ اگر نیاز داشتید پارامتریک کنید.
            selected_features=[],  # برای استخراج y نیازی به ستون انتخابی نیست
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        # y ممکن است لیست/ndarray باشد
        y_series = pd.Series(yc).reset_index(drop=True) if yc is not None else pd.Series(dtype=int)
        y_true = int(y_series.iloc[-1]) if len(y_series)>0 else -1  # -1=نامشخص

        # ۵) مقایسه پاسخ با تارگت
        # نگاشت: کلاس مثبت (مثلاً 1) ↔ BUY ؛ کلاس منفی (0) ↔ SELL
        if ans_text == "NONE":
            none += 1
            decided_now = False
        else:
            decided += 1
            decided_now = True

        correct_now = False
        if decided_now and y_true in (0,1):
            if (ans_text == "BUY"  and y_true == args.positive_class) or \
               (ans_text == "SELL" and y_true != args.positive_class):
                correct += 1; correct_now = True
            else:
                incorrect += 1

        logger.info("[Result %d] ts=%s answer=%s | target=%s | %s",
                    i, ts, ans_text, ("BUY" if y_true==args.positive_class else "SELL" if y_true in (0,1) else "?"),
                    "OK" if correct_now else ("—" if not decided_now else "WRONG"))
        logger.info("[Running] decided=%d correct=%d incorrect=%d none=%d timeout=%d | acc(decidable)=%.2f%% | coverage=%.2f%%",
                    decided, correct, incorrect, none, timeout,
                    (100.0*correct/decided if decided else 0.0),
                    (100.0*decided/(decided+none+timeout) if (decided+none+timeout)>0 else 0.0))

        # ۶) answer.txt را حذف کنیم تا قدم بعدی آزاد شود
        try: answer_fp.unlink()
        except Exception: pass

        # نوبت بعدی...
        time.sleep(1.0)

    logger.info("[Final] decided=%d correct=%d incorrect=%d none=%d timeout=%d | acc(decidable)=%.2f%% | coverage=%.2f%%",
                decided, correct, incorrect, none, timeout,
                (100.0*correct/decided if decided else 0.0),
                (100.0*decided/(decided+none+timeout) if (decided+none+timeout)>0 else 0.0))

if __name__ == "__main__":
    main()
