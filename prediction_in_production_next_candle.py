#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_in_production_next_candle.py (نسخه‌ی بازنویسی شده)

دپلوی آفلاین/لایو با منطق «پیش‌بینی کندل بعدی (t → t+1)»:

- ژنراتور (یا MT4) فایل‌های *_live.csv را تولید می‌کند.
- این اسکریپت:
    1) از XAUUSD_M30_live.csv زمان آخرین کندل را می‌گیرد (ts_now).
    2) روی CSVهای خام (XAUUSD_M30/M15/M5/H1) تا ts_now فیچر می‌سازد.
    3) با predict_drop_last=True، آخرین سطر ناپایدار را حذف می‌کند.
    4) آخرین سطر پایدار فیچر را به مدل می‌دهد و جهت کندل بعدی را پیش‌بینی می‌کند.
    5) جواب را در answer.txt می‌نویسد تا MT4 بخواند.

در لایو:
- هیچ Feature Selection اجرا نمی‌شود.
- فقط از train_window_cols ذخیره‌شده در best_model.pkl استفاده می‌شود.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from ModelSaver import ModelSaver

LOG = logging.getLogger("deploy_next")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def decide_action_single(p: float, neg_thr: float, pos_thr: float) -> str:
    if p >= pos_thr:
        return "BUY"
    if p <= neg_thr:
        return "SELL"
    return "NONE"

def read_last_time(csv_path: Path) -> pd.Timestamp | None:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df.dropna(subset=["time"], inplace=True)
    if df.empty:
        return None
    return df["time"].iloc[-1]

def collect_live_paths(base: Path, symbol: str) -> Dict[str, Path]:
    return {
        "30T": base / f"{symbol}_M30_live.csv",
        "15T": base / f"{symbol}_M15_live.csv",
        "5T":  base / f"{symbol}_M5_live.csv",
        "1H":  base / f"{symbol}_H1_live.csv",
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str,
                    help="پوشه‌ی حاوی CSVهای خام متاتریدر و best_model.pkl")
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--poll-sec", default=0.5, type=float,
                    help="فاصله‌ی چک‌کردن فایل‌های *_live (ثانیه)")
    ap.add_argument("--max-steps", default=10_000, type=int,
                    help="حداکثر تعداد استپ برای شبیه‌سازی")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    symbol = args.symbol

    # ---------- 1) بارگذاری مدل ----------
    saver = ModelSaver(model_dir=str(base))
    payload = saver.load_full()      # dict
    model = payload["pipeline"]      # EnsembleModel یا مدل تکی
    meta = payload

    window = int(meta["window_size"])
    neg_thr = float(meta["neg_thr"])
    pos_thr = float(meta["pos_thr"])
    train_cols = list(
        meta.get("train_window_cols")
        or meta.get("feats")
        or []
    )

    if not train_cols:
        raise ValueError("train_window_cols/feats در مدل ذخیره نشده است.")

    # آستانه‌های جداگانه‌ی هر مدل در Ensemble (اگر موجود باشد)
    hyper = meta.get("hyperparams", {}) or {}
    neg_thrs = hyper.get("neg_thrs")
    pos_thrs = hyper.get("pos_thrs")

    if not neg_thrs or not pos_thrs:
        # سازگاری با مدل تک‌مدلی
        neg_thrs = [neg_thr]
        pos_thrs = [pos_thr]

    LOG.info(
        "Model loaded: window=%d · neg_thr=%.3f · pos_thr=%.3f · n_cols=%d",
        window,
        neg_thr,
        pos_thr,
        len(train_cols),
    )

    # ---------- 2) PREPARE روی CSVهای خام ----------
    filepaths_raw = {
        "30T": base / f"{symbol}_M30.csv",
        "15T": base / f"{symbol}_M15.csv",
        "5T":  base / f"{symbol}_M5.csv",
        "1H":  base / f"{symbol}_H1.csv",
    }
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={tf: str(p) for tf, p in filepaths_raw.items()},
        main_timeframe="30T",
        verbose=False,
        fast_mode=True,       # لایو → drift-scan کامل لازم نیست
        strict_disk_feed=False,
    )

    merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged[tcol] = pd.to_datetime(merged[tcol], errors="coerce")
    merged.sort_values(tcol, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # ---------- 3) مسیرهای لاگ و فایل answer ----------
    ans_path = base / "answer.txt"
    feed_log_path = base / "deploy_X_feed_log.csv"
    pred_path = base / "deploy_predictions.csv"
    feat_tail_path = base / "deploy_X_feed_tail200.csv"

    feed_log_path.unlink(missing_ok=True)
    pred_path.unlink(missing_ok=True)
    feat_tail_path.unlink(missing_ok=True)

    live_paths = collect_live_paths(base, symbol)
    last_ts_seen: pd.Timestamp | None = None
    cover_cum = 0.0
    total_steps = 0
    traded = 0

    LOG.info("=== Deploy started (next-candle logic, safe-last-row) ===")

    for step in range(1, args.max_steps + 1):
        # --- 1) منتظر *_M30_live از ژنراتور / MT4 ---
        ts_now = read_last_time(live_paths["30T"])
        if ts_now is None:
            time.sleep(args.poll_sec)
            continue

        if last_ts_seen is not None and ts_now <= last_ts_seen:
            time.sleep(args.poll_sec)
            continue

        # --- 2) ساب‌ست دیتا تا ts_now از merged آماده‌شده ---
        sub = merged[merged[tcol] <= ts_now].copy()
        if sub.empty:
            LOG.warning("No data up to %s", ts_now)
            time.sleep(args.poll_sec)
            continue

        # --- 3) فیچرها برای تمام کندل‌ها تا ts_now ---
        # در mode="predict":
        # - پنجره‌بندی کامل (مثل TRAIN)
        # - با predict_drop_last=True، آخرین کندل (احتمالاً ناپایدار) حذف می‌شود.
        #   پس آخرین سطر X_all مربوط به آخرین کندل «کاملاً بسته شده» است.
        X_all, _, _, price_ser, t_idx = prep.ready(
            sub,
            window=window,
            selected_features=train_cols,   # همان ستون‌های TRAIN (بدون FS جدید)
            mode="predict",
            with_times=True,
            predict_drop_last=True,         # 🔴 سطر آخر را حذف کن
            train_drop_last=False,
        )

        if X_all.empty:
            LOG.warning("ready() returned empty at %s (after dropping last row)", ts_now)
            time.sleep(args.poll_sec)
            continue

        # --- 4) آخرین کندل پایدار (t_feat) ---
        X_last = X_all.tail(1).reset_index(drop=True)
        ts_feat = pd.to_datetime(t_idx.iloc[-1])

        if ts_feat > ts_now:
            LOG.warning(
                "Time mismatch (unexpected): ts_feat=%s > ts_now=%s (after drop_last)",
                ts_feat,
                ts_now,
            )

        # --- 5) پیش‌بینی احتمال لانگ ---
        prob = float(model.predict_proba(X_last)[:, 1][0])

        # Ensemble با آستانه‌های جداگانه
        if hasattr(model, "predict_actions"):
            actions_int = model.predict_actions(X_last, neg_thrs, pos_thrs)
            a_int = int(actions_int[0])
            if a_int == 1:
                action = "BUY"
            elif a_int == 0:
                action = "SELL"
            else:
                action = "NONE"
        else:
            # مدل تک‌مدلی
            action = decide_action_single(prob, neg_thr, pos_thr)

        total_steps += 1
        if action != "NONE":
            traded += 1
        cover_cum = traded / float(total_steps) if total_steps > 0 else 0.0

        # --- 6) نوشتن answer.txt برای ژنراتور/MT4 ---
        try:
            ans_path.write_text(action, encoding="utf-8")
        except Exception as e:
            LOG.error("Could not write answer.txt: %s", e)

        # --- 7) لاگ feed (فیچرهای استفاده‌شده) ---
        row_feed = {
            "timestamp": ts_feat,          # زمان فیچر (کندل t)
            "timestamp_trigger": ts_now,   # زمان تریگر (کندل t یا t+1، طبق طراحی ژنراتور)
            "y_prob": prob,
            "action": action,
            "neg_thr": neg_thr,
            "pos_thr": pos_thr,
            "cover_cum": cover_cum,
        }
        hdr = not feed_log_path.is_file()
        pd.DataFrame([row_feed]).to_csv(
            feed_log_path,
            mode="a",
            header=hdr,
            index=False,
        )

        # --- 8) ذخیره‌ی فیچرهای همان سطر برای مقایسه ---
        df_feat_row = X_last.copy()
        df_feat_row.insert(0, "timestamp_trigger", ts_now)
        df_feat_row.insert(0, "timestamp", ts_feat)
        hdr_feat = not feat_tail_path.is_file()
        pd.DataFrame(df_feat_row).to_csv(
            feat_tail_path,
            mode="a",
            header=hdr_feat,
            index=False,
        )

        # --- 9) لاگ predictions High-level ---
        row_pred = {
            "timestamp": ts_feat,
            "timestamp_trigger": ts_now,
            "y_prob": prob,
            "action": action,
            "cover_cum": cover_cum,
        }
        hdr2 = not pred_path.is_file()
        pd.DataFrame([row_pred]).to_csv(
            pred_path,
            mode="a",
            header=hdr2,
            index=False,
        )

        last_ts_seen = ts_now

        LOG.info(
            "[Deploy] step=%d ts_now=%s ts_feat=%s action=%s prob=%.3f cover_cum=%.3f",
            step,
            ts_now,
            ts_feat,
            action,
            prob,
            cover_cum,
        )

        time.sleep(args.poll_sec)

    LOG.info("=== Deploy finished ===")


if __name__ == "__main__":
    main()
