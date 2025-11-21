#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_like_sim_v3_next_candle.py

SIM آفلاین با منطق «در زمان t، پیش‌بینی جهت کندل بعدی (t→t+1)».
"""

from __future__ import annotations
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from ModelSaver import ModelSaver


LOG = logging.getLogger("live_like_sim")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--last-n", default=200, type=int, help="تعداد آخرین کندل‌هایی که برای SIM استفاده می‌شوند")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    symbol = args.symbol

    # ---------- مدل و متا ----------
    saver = ModelSaver(model_dir=str(base))
    pipeline, meta = saver.load_full()      # این تابع را با امضای واقعی ModelSaver خودت هماهنگ کن

    model = pipeline        # wrapper که predict_proba دارد
    window = int(meta["window_size"])
    neg_thr = float(meta["neg_thr"])
    pos_thr = float(meta["pos_thr"])
    train_cols = list(meta["train_window_cols"])

    LOG.info("Loaded model: window=%d neg_thr=%.3f pos_thr=%.3f cols=%d",
             window, neg_thr, pos_thr, len(train_cols))

    # ---------- PREPARE ----------
    filepaths = {
        "30T": base / f"{symbol}_M30.csv",
        "15T": base / f"{symbol}_M15.csv",
        "5T":  base / f"{symbol}_M5.csv",
        "1H":  base / f"{symbol}_H1.csv",
    }
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={tf: str(p) for tf, p in filepaths.items()},
        main_timeframe="30T",
        verbose=True
    )

    merged = prep.load_data()
    LOG.info("Merged data loaded: shape=%s", merged.shape)

    # ---------- X,y برای کل دیتاست (کندل بعدی) ----------
    X_all, y_all, _, price_all, t_idx_all = prep.ready(
        merged,
        window=window,
        selected_features=train_cols,
        mode="train",           # چون می‌خواهیم y_true داشته باشیم
        with_times=True,
        predict_drop_last=False,
        train_drop_last=False,
    )

    if len(X_all) == 0:
        LOG.error("No rows after ready()")
        return

    # tail
    n = min(args.last_n, len(X_all))
    X_tail = X_all.tail(n).reset_index(drop=True)
    y_tail = pd.Series(y_all).tail(n).reset_index(drop=True)
    t_tail = pd.to_datetime(t_idx_all).tail(n).reset_index(drop=True)

    LOG.info("SIM on last %d rows (from %s to %s)", n, t_tail.iloc[0], t_tail.iloc[-1])

    # ---------- پیش‌بینی ----------
    y_prob = model.predict_proba(X_tail)[:, 1]
    actions = np.array([decide_action(p, neg_thr, pos_thr) for p in y_prob])

    # ---------- محاسبه‌ی دقت و کاور ----------
    mask_trade = actions != "NONE"
    traded = int(mask_trade.sum())
    total = len(actions)

    if traded > 0:
        y_pred = np.where(actions[mask_trade] == "BUY", 1, 0)
        y_true = y_tail[mask_trade].to_numpy()
        acc = (y_pred == y_true).mean()
    else:
        acc = 0.0

    cover = traded / float(total)
    LOG.info("SIM result: acc=%.3f cover=%.3f traded=%d total=%d", acc, cover, traded, total)

    # ---------- ذخیره‌ی خروجی برای مقایسه با دپلوی/ژنراتور ----------
    out_sim = base / "sim_predictions.csv"
    df_out = pd.DataFrame({
        "timestamp": t_tail,
        "y_true": y_tail,
        "y_prob": y_prob,
        "action": actions,
        "neg_thr": neg_thr,
        "pos_thr": pos_thr,
    })
    df_out.to_csv(out_sim, index=False)
    LOG.info("sim_predictions.csv written: %s", out_sim)


if __name__ == "__main__":
    main()
