#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_in_production_next_candle.py

دپلوی آفلاین/لایو:
- در هر استپ، با دیدن فایل‌های *_live.csv و زمان آخرین کندل M30 (ts_now)،
  روی تمام دیتا تا ts_now فیچر می‌سازد،
  آخرین سطر X (زمان ts_now) را به مدل می‌دهد،
  و جهت کندل بعدی (ts_now→ts_next) را پیش‌بینی می‌کند.

هم‌راستا با منطق شماره ۴:
- در مسیر PREDICT، هیچ «سطر آخری» حذف نمی‌شود (predict_drop_last=False).
- پایداری از طریق safe_agg_group + detect_bad_cols_tf در PREPARE_DATA_FOR_TRAIN
  و حذف ستون‌های ناپایدار تامین می‌شود.
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


def decide_action(p: float, neg_thr: float, pos_thr: float) -> str:
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
        "5T": base / f"{symbol}_M5_live.csv",
        # 1H در شبیه‌سازی استفاده مستقیم نمی‌شود؛ این مسیر صرفاً جهت کامل بودن است
        "1H": base / f"{symbol}_H1_live.csv",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--poll-sec", default=0.5, type=float)
    ap.add_argument("--max-steps", default=10_000, type=int)
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    symbol = args.symbol

    # ---------- مدل ----------
    # ---------- مدل ----------
    saver = ModelSaver(model_dir=str(base))
    payload = saver.load_full()      # خروجی یک dict است
    pipeline = payload["pipeline"]
    meta = payload
    model = pipeline
    window = int(meta["window_size"])
    neg_thr = float(meta["neg_thr"])
    pos_thr = float(meta["pos_thr"])
    train_cols = list(meta["train_window_cols"])


    if not train_cols:
        raise ValueError("train_window_cols/feats در مدل ذخیره نشده است.")

    LOG.info(
        "Model loaded: window=%d neg_thr=%.3f pos_thr=%.3f cols=%d",
        window,
        neg_thr,
        pos_thr,
        len(train_cols),
    )

    # ---------- PREPARE روی CSV خام (نه *_live) ----------
    filepaths_raw = {
        "30T": base / f"{symbol}_M30.csv",
        "15T": base / f"{symbol}_M15.csv",
        "5T": base / f"{symbol}_M5.csv",
        "1H": base / f"{symbol}_H1.csv",
    }
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={tf: str(p) for tf, p in filepaths_raw.items()},
        main_timeframe="30T",
        verbose=False,
        fast_mode=True,   # در دپلوی، drift-scan کامل لازم نیست
        strict_disk_feed=False,
    )
    merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged[tcol] = pd.to_datetime(merged[tcol], errors="coerce")
    merged.sort_values(tcol, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # ---------- مسیرهای لاگ ----------
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

    for _ in range(args.max_steps):
        # منتظر *_live از ژنراتور / MT4
        ts_now = read_last_time(live_paths["30T"])
        if ts_now is None:
            time.sleep(args.poll_sec)
            continue

        if last_ts_seen is not None and ts_now <= last_ts_seen:
            time.sleep(args.poll_sec)
            continue

        # ساب‌ست تا ts_now از mergedِ آماده‌شده (safe_agg_group + حذف bad_cols)
        sub = merged[merged[tcol] <= ts_now].copy()
        if sub.empty:
            LOG.warning("No data up to %s", ts_now)
            time.sleep(args.poll_sec)
            continue

        # فیچرها برای تمام کندل‌ها تا ts_now
        # استفاده از سطر پایدارِ ماقبل آخر:
        # در mode="predict" اگر predict_drop_last=True باشد، سطر آخر حذف می‌شود
        # و آخرین سطر X_all مربوط به آخرین کندل «کاملاً پایدار» خواهد بود.
        X_all, _, _, price_ser, t_idx = prep.ready(
            sub,
            window=window,
            selected_features=train_cols,
            mode="predict",
            with_times=True,
            predict_drop_last=True,   # ⬅️ سطر آخر را حذف کن (کندل ناپایدار فعلی)
            train_drop_last=False,
        )

        if X_all.empty:
            LOG.warning("ready() returned empty at %s (after dropping last row)", ts_now)
            time.sleep(args.poll_sec)
            continue

        # آخرین کندل پایدار (مثلاً کندلی که قبل از ts_now بسته شده)
        X_last = X_all.tail(1).reset_index(drop=True)
        ts_feat = pd.to_datetime(t_idx.iloc[-1])

        if ts_feat > ts_now:
            LOG.warning(
                "Time mismatch (unexpected): ts_feat=%s > ts_now=%s (after drop_last)",
                ts_feat,
                ts_now,
            )

        prob = float(model.predict_proba(X_last)[:, 1][0])
        action = decide_action(prob, neg_thr, pos_thr)

        total_steps += 1
        if action != "NONE":
            traded += 1
        cover_cum = traded / float(total_steps) if total_steps > 0 else 0.0

        # ---------- نوشتن answer.txt برای ژنراتور / MT4 ----------
        try:
            ans_path.write_text(action, encoding="utf-8")
        except Exception as e:
            LOG.error("Could not write answer.txt: %s", e)

        # ---------- لاگ feed ----------
        row_feed = {
            "timestamp": ts_feat,
            "timestamp_trigger": ts_now,
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

        # ---------- لاگ فیچرهای استفاده‌شده برای مقایسه ----------
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

        # ---------- لاگ predictions_highlevel ----------
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
            "[Deploy] ts_now=%s ts_feat=%s action=%s prob=%.3f cover_cum=%.3f",
            ts_now,
            ts_feat,
            action,
            prob,
            cover_cum,
        )

        # ژنراتور/MT4 معمولاً فایل‌های *_live را برای استپ بعدی overwrite می‌کند.
        time.sleep(args.poll_sec)

    LOG.info("=== Deploy finished ===")


if __name__ == "__main__":
    main()
