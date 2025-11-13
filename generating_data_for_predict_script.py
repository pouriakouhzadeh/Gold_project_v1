#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generating_data_for_predict_script.py
Author: Pouria + Assistant (final handshake)

وظایف:
  - از CSVهای خام (M5/M15/M30/H1) دیتافریم ادغام‌شده آموزش را با PREP می‌سازد.
  - ۲۰۰ قدم آخر (قابل تغییر با --steps) را روی تایم‌استَمپ‌های ۳۰ دقیقه‌ای پیمایش می‌کند.
  - در هر قدم ۴ فایل *_live.csv را تا آن لحظه می‌نویسد، سپس منتظر answer.txt می‌ماند.
  - answer را با تارگت واقعی همان لحظه مقایسه می‌کند و آمار لحظه‌ای و تجمعی را روی کنسول چاپ می‌کند.
  - طبق سناریو شما، **پس از خواندن** answer.txt آن را حذف می‌کند (تا قدم بعدی آغاز شود).
  - اگر تا timeout پاسخی نیاید، نتیجه‌ی آن قدم TIMEOUT ثبت می‌شود.

نیازمندی‌ها:
  - pandas, numpy
  - PREPARE_DATA_FOR_TRAIN از پروژه
"""

from __future__ import annotations
import os, sys, time, argparse, logging
from logging import Formatter, StreamHandler
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
except Exception as e:
    PREPARE_DATA_FOR_TRAIN = None
    raise ImportError("Cannot import PREPARE_DATA_FOR_TRAIN. Put it on PYTHONPATH.") from e

APP = "generator_parity"
LOG_FILE = "generator_parity.log"

def setup_logger(path: str) -> logging.Logger:
    log = logging.getLogger(APP)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = Formatter("%(asctime)s INFO: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    ch = StreamHandler(sys.stdout); ch.setFormatter(fmt); log.addHandler(ch)
    return log

def resolve_timeframe_paths(base_dir: str, symbol: str="XAUUSD"):
    cand = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    resolved = {}
    for tf, names in cand.items():
        found = None
        for nm in names:
            p = os.path.join(base_dir, nm)
            if os.path.exists(p): found = p; break
        if not found:
            low = [n.lower() for n in names]
            for fn in os.listdir(base_dir):
                if fn.lower() in low:
                    found = os.path.join(base_dir, fn); break
        if found: resolved[tf] = found
    if "30T" not in resolved:
        raise FileNotFoundError("Main timeframe '30T' CSV not found in raw-dir.")
    return resolved

def read_tf_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time") or cols.get("datetime") or cols.get("date")
    if not tcol: raise ValueError(f"{path}: cannot find time column")
    df = df.rename(columns={
        tcol: "time",
        cols.get("open","open"): "open",
        cols.get("high","high"): "high",
        cols.get("low","low"): "low",
        cols.get("close","close"): "close",
        cols.get("volume", cols.get("tick_volume","volume")): "volume",
    })
    df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df

def write_live_csv(df: pd.DataFrame, upto_ts: pd.Timestamp, out_path: str):
    sub = df[df["time"] <= pd.to_datetime(upto_ts)].copy()
    sub.to_csv(out_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--watch-dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--timeout-sec", type=int, default=40)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--log-path", default=LOG_FILE)
    ap.add_argument("--cleanup-answer", action="store_true", help="delete answer.txt after reading")
    args = ap.parse_args()

    log = setup_logger(args.log_path)
    log.info("=== generator_parity started ===")

    # 1) مسیر فایل‌های خام و بارگذاری
    files = resolve_timeframe_paths(args.raw_dir, args.symbol)
    for k in ("30T","15T","5T","1H"):
        if k in files: log.info("[raw] %s  -> %s", k, files[k])

    # برای تولید تارگت و همسان سازی ویژگی‌ها
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=files, main_timeframe="30T",
                                  verbose=True, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged[tcol] = pd.to_datetime(merged[tcol], utc=False)
    merged = merged.sort_values(tcol).reset_index(drop=True)
    log.info("[INIT] main_timeframe=%s", prep.main_timeframe)

    # داده‌های تایم‌فریم‌ها برای نوشتن live
    df30 = read_tf_csv(files["30T"])
    df15 = read_tf_csv(files["15T"])
    df5  = read_tf_csv(files["5T"])
    df1h = read_tf_csv(files["1H"])

    # ناحیه آخر برای شبیه‌سازی
    total = len(merged)
    if total < args.steps + 10:
        start = 10
    else:
        start = total - args.steps
    stamps = merged[tcol].iloc[start : start + args.steps].tolist()

    # مسیر فایل‌های live و answer
    m5_live  = os.path.join(args.watch_dir, "XAUUSD_M5_live.csv")
    m15_live = os.path.join(args.watch_dir, "XAUUSD_M15_live.csv")
    m30_live = os.path.join(args.watch_dir, "XAUUSD_M30_live.csv")
    h1_live  = os.path.join(args.watch_dir, "XAUUSD_H1_live.csv")
    answer_fp = os.path.join(args.watch_dir, "answer.txt")

    # شمارنده‌ها
    decided = correct = incorrect = none = timeout = 0

    for i, ts in enumerate(stamps, start=1):
        # 2) نوشتن ۴ فایل live تا لحظه ts
        write_live_csv(df5,  ts, m5_live)
        write_live_csv(df15, ts, m15_live)
        write_live_csv(df30, ts, m30_live)
        write_live_csv(df1h, ts, h1_live)
        log.info(f"[Step {i}/{args.steps}] wrote *_live.csv @ {ts} — waiting for answer.txt ...")

        # 3) انتظار برای answer
        t0 = time.time()
        got = False
        while time.time() - t0 < args.timeout_sec:
            if os.path.exists(answer_fp):
                # برای جلوگیری از خواندن ناقص، کمی صبر کن
                time.sleep(0.1)
                with open(answer_fp, "r", encoding="utf-8") as f:
                    ans = f.read().strip().upper()
                got = True
                break
            time.sleep(0.5)

        if not got:
            timeout += 1
            log.info(f"[Result {i}] ts={ts} answer=TIMEOUT | target=? | —")
            continue

        # 4) محاسبه تارگت واقعی همان لحظه
        cut = merged[merged[tcol] <= pd.to_datetime(ts)].copy()
        X, y, _, _ = prep.ready(
            cut,
            window=2,  # window در متا 2 است؛ اگر در پروژه شما متغیر است، می‌توانید هم از meta بخوانید.
            selected_features=[],  # برای هدف نیازی به ستون‌ها نداریم
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        # y آخرین مقدار، تارگت لحظه ts
        y_true = int(np.ravel(y)[-1]) if len(y) else -1
        target_label = "BUY" if y_true == 1 else ("SELL" if y_true == 0 else "NONE")

        # 5) ارزیابی و گزارش
        if ans == "NONE":
            none += 1
        else:
            decided += 1
            if ans == target_label:
                correct += 1
            else:
                incorrect += 1

        acc = (correct / decided)*100.0 if decided > 0 else 0.0
        cover = (decided / (i - timeout))*100.0 if (i - timeout) > 0 else 0.0
        log.info(f"[Result {i}] ts={ts} answer={ans} | target={target_label} | —")
        log.info(f"[Running] decided={decided} correct={correct} incorrect={incorrect} none={none} timeout={timeout} | acc(decidable)={acc:.2f}% | coverage={cover:.2f}%")

        # 6) پاک کردن answer و ادامه
        if args.cleanup_answer and os.path.exists(answer_fp):
            try: os.remove(answer_fp)
            except Exception: pass

    log.info("[Final] decided=%d correct=%d incorrect=%d none=%d timeout=%d", decided, correct, incorrect, none, timeout)
    acc = (correct / decided)*100.0 if decided>0 else 0.0
    log.info(f"[Final] acc(decidable)={acc:.2f}% | steps={args.steps}")

if __name__ == "__main__":
    main()
