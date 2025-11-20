#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generating_data_for_predict_script.py  —  ژنراتور نقش MT4 (هماهنگ با دپلوی)

نقش:
- روی CSVهای خام مَتی‌تریدر (XAUUSD_M5/M15/M30/H1.csv) کار می‌کند
- در هر استپ:
    • XAUUSD_*_live.csv را تا ts_now می‌سازد
    • منتظر answer.txt از دپلوی می‌ماند
    • از deploy_X_feed_log.csv، ts_feat و y_prob و cover_cum را می‌گیرد
    • y_true را از دیتای خام M30 (close[t+1] > close[t]) محاسبه می‌کند
    • دقت و کاور تجمعی را می‌سازد و در generator_predictions.csv می‌نویسد
"""

from __future__ import annotations

import time
import logging
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

LOG = logging.getLogger("generator_mt4")


# ---------- Logging ----------
def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity > 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------- Helpers ----------
def resolve_raw_paths(base: Path, symbol: str) -> Dict[str, Path]:
    return {
        "30T": base / f"{symbol}_M30.csv",
        "15T": base / f"{symbol}_M15.csv",
        "5T": base / f"{symbol}_M5.csv",
        "1H": base / f"{symbol}_M1H.csv" if (base / f"{symbol}_M1H.csv").is_file() else base / f"{symbol}_H1.csv",
    }


def live_name(path: Path) -> Path:
    """XAUUSD_M30.csv → XAUUSD_M30_live.csv"""
    return path.with_name(path.stem + "_live" + path.suffix)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--last-n", default=200, type=int)
    ap.add_argument("--sleep", default=0.5, type=float)
    ap.add_argument("--verbosity", default=1, type=int)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    base = Path(args.base_dir).resolve()
    symbol = args.symbol

    LOG.info("=== Generator (MT4-like) started ===")
    LOG.info("Base dir=%s | Symbol=%s | last-n=%d", base, symbol, args.last_n)

    raw_paths = resolve_raw_paths(base, symbol)
    for tf, p in raw_paths.items():
        if not p.is_file():
            LOG.error("Raw CSV for %s not found: %s", tf, p)
            return
        LOG.info("[raw] %s -> %s", tf, p)

    # --- Load raw CSVs (بدون PREPARE) ---
    raw: Dict[str, pd.DataFrame] = {}
    for tf, p in raw_paths.items():
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise ValueError(f"'time' column missing in {p}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        raw[tf] = df

    df30 = raw["30T"]
    if len(df30) < 2:
        LOG.error("Not enough M30 data.")
        return

    total = len(df30)
    # حداکثر ایندکسی که می‌توانیم y_true (close[t+1]>close[t]) برایش بسازیم
    max_idx_for_label = total - 2
    if max_idx_for_label <= 0:
        LOG.error("Not enough M30 rows for labels.")
        return

    n_steps = min(args.last_n, max_idx_for_label)
    start_idx = max(0, max_idx_for_label - n_steps + 1)
    idx_range = range(start_idx, start_idx + n_steps)

    LOG.info("Using %d steps from M30 index [%d .. %d]", n_steps, start_idx, start_idx + n_steps - 1)

    # نگه‌داشتن آخرین N ردیف هر TF (برای سبک شدن فایل‌های live)
    SL = {"30T": 1000, "15T": 2000, "5T": 5000, "1H": 1000}

    ans_path = base / "answer.txt"
    feed_log_path = base / "deploy_X_feed_log.csv"
    gen_pred_path = base / "generator_predictions.csv"

    # پاک کردن نتیجهٔ قبلی ژنراتور، اگر وجود دارد
    gen_pred_path.unlink(missing_ok=True)

    wins = loses = none = 0
    acc_gen = cover_gen = 0.0

    for step, idx in enumerate(idx_range, start=1):
        ts_now = df30.loc[idx, "time"]

        # --- 1) ساخت CSVهای زنده تا ts_now ---
        for tf, df in raw.items():
            cut = df[df["time"] <= ts_now].tail(SL.get(tf, 500)).copy()
            out = live_name(raw_paths[tf])
            cut.to_csv(out, index=False)

        LOG.info(
            "[Step %d/%d] Live CSVs written at %s — waiting for answer.txt …",
            step,
            n_steps,
            ts_now,
        )

        # --- 2) انتظار برای answer.txt از دپلوی ---
        while not ans_path.exists():
            time.sleep(args.sleep)

        try:
            ans = ans_path.read_text(encoding="utf-8").strip().upper() or "NONE"
        except Exception:
            ans = "NONE"

        # برای استپ بعدی پاک شود
        try:
            ans_path.unlink(missing_ok=True)
        except Exception:
            pass

        # --- 3) خواندن آخرین ردیف از deploy_X_feed_log.csv (ts_feat, y_prob, cover_cum_deploy) ---
        ts_feat = ts_now
        y_prob = np.nan
        cover_dep = np.nan
        try:
            dflog = pd.read_csv(feed_log_path)
            if "timestamp" in dflog.columns:
                dflog["timestamp"] = pd.to_datetime(dflog["timestamp"], errors="coerce")
                last = dflog.iloc[-1]
                ts_feat = pd.to_datetime(last["timestamp"])
                if "y_prob" in last:
                    y_prob = float(last["y_prob"])
                if "cover_cum" in last:
                    cover_dep = float(last["cover_cum"])
        except Exception as e:
            LOG.warning("Could not read deploy_X_feed_log.csv (using ts_now as ts_feat): %s", e)

        # --- 4) محاسبه‌ی y_true از دیتای خام M30 با ts_feat ---
        try:
            ridx_list = df30.index[df30["time"] == ts_feat].tolist()
            if ridx_list and ridx_list[0] < len(df30) - 1:
                ridx = ridx_list[0]
                c0 = float(df30.loc[ridx, "close"])
                c1 = float(df30.loc[ridx + 1, "close"])
                real_up: Optional[bool] = c1 > c0
            else:
                real_up = None
        except Exception as e:
            LOG.warning("Failed to compute y_true for %s: %s", ts_feat, e)
            real_up = None

        # --- 5) نگاشت اکشن به pred_up ---
        if ans == "BUY":
            pred_up: Optional[bool] = True
        elif ans == "SELL":
            pred_up = False
        else:
            pred_up = None

        if pred_up is None or real_up is None:
            none += 1
            y_true_int = np.nan
        else:
            y_true_int = int(real_up)
            if pred_up == real_up:
                wins += 1
            else:
                loses += 1

        traded = wins + loses
        cover_gen = traded / float(step)
        acc_gen = (wins / traded) if traded > 0 else 0.0

        LOG.info(
            "[Step %d] ts_now=%s ts_feat=%s action=%s y_true=%s | "
            "acc_gen=%.3f cover_gen=%.3f (wins=%d loses=%d none=%d) | cover_dep=%.3f",
            step,
            ts_now,
            ts_feat,
            ans,
            str(real_up),
            acc_gen,
            cover_gen,
            wins,
            loses,
            none,
            cover_dep,
        )

        # --- 6) ذخیره‌ی رکورد این استپ در generator_predictions.csv ---
        row = {
            "timestamp": ts_feat,
            "timestamp_trigger": ts_now,
            "action": ans,
            "y_true": y_true_int,
            "y_prob": y_prob,
            "cover_cum_deploy": cover_dep,
            "cover_cum_gen": cover_gen,
            "acc_cum_gen": acc_gen,
        }
        hdr = not gen_pred_path.is_file()
        pd.DataFrame([row]).to_csv(
            gen_pred_path,
            mode="a",
            header=hdr,
            index=False,
        )

    # --- گزارش نهایی ---
    LOG.info(
        "[Final] acc_gen=%.3f cover_gen=%.3f wins=%d loses=%d none=%d",
        acc_gen,
        cover_gen,
        wins,
        loses,
        none,
    )


if __name__ == "__main__":
    main()
