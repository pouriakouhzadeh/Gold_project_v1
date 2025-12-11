#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generation_data_for_predict_script.py (بازنویسی شده)

ژنراتور آفلاین که نقش MT4 را بازی می‌کند:

- روی CSVهای خام متاتریدر (XAUUSD_M30/M15/M5/H1) کار می‌کند.
- در هر استپ روی یک کندل M30 با زمان ts_now:
    1) برای هر تایم‌فریم تا زمان ts_now یک فایل *_live.csv می‌سازد.
    2) منتظر می‌ماند دپلوی روی *_live مدل را اجرا کند و answer.txt را
       با یکی از BUY/SELL/NONE بنویسد.
    3) آخرین سطر deploy_X_feed_log.csv را می‌خواند (y_prob و cover_cum_deploy).
    4) برچسب واقعی y_true را از روی M30 خام به صورت
           y_true(t_feat) = 1{ close(t_feat+1) > close(t_feat) }
       حساب می‌کند؛ ts_feat همان ستون 'timestamp' در لاگ دپلوی است.
    5) دقت و کاور تجمعی ژنراتور را به‌روز کرده و در generator_predictions.csv
       ذخیره می‌کند.
"""

from __future__ import annotations

import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

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
    p_1h = base / f"{symbol}_M1H.csv"
    if not p_1h.is_file():
        p_1h = base / f"{symbol}_H1.csv"

    return {
        "30T": base / f"{symbol}_M30.csv",
        "15T": base / f"{symbol}_M15.csv",
        "5T":  base / f"{symbol}_M5.csv",
        "1H":  p_1h,
    }

def live_name(path: Path) -> Path:
    """XAUUSD_M30.csv → XAUUSD_M30_live.csv"""
    return path.with_name(path.stem + "_live" + path.suffix)

def _detect_close_col(df: pd.DataFrame) -> str:
    candidates = ["close", "Close", "CLOSE"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No close column found in M30 data. Available columns: {list(df.columns)}")

# ---------- MAIN ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str,
                    help="پوشه‌ی حاوی CSVهای خام متاتریدر و اسکریپت دپلوی")
    ap.add_argument("--symbol", default="XAUUSD", type=str,
                    help="نام سیمبل (مثلاً XAUUSD)")
    ap.add_argument(
        "--last-n",
        default=200,
        type=int,
        help="تعداد استپ‌هایی که می‌خواهی شبیه‌سازی شود (از انتهای M30)",
    )
    ap.add_argument("--sleep", default=0.5, type=float,
                    help="تاخیر بین چک‌کردن answer.txt (ثانیه)")
    ap.add_argument("--verbosity", default=1, type=int,
                    help="1 = INFO, 0 = WARNING")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    base = Path(args.base_dir).resolve()
    symbol = args.symbol

    LOG.info("=== Generator (MT4-like, next-candle target) started ===")
    LOG.info("Base dir=%s | Symbol=%s | last-n=%d", base, symbol, args.last_n)

    # ---------- 1) مسیر CSVهای خام ----------
    raw_paths = resolve_raw_paths(base, symbol)
    for tf, p in raw_paths.items():
        if not p.is_file():
            LOG.error("Raw CSV for %s not found: %s", tf, p)
            return
        LOG.info("[raw] %s -> %s", tf, p)

    # ---------- 2) بارگذاری CSVهای خام ----------
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

    close_col_30 = _detect_close_col(df30)

    total = len(df30)
    max_idx_for_label = total - 2  # نیاز به t+1 داریم
    if max_idx_for_label <= 0:
        LOG.error("Not enough M30 rows for labels (need at least 2).")
        return

    n_steps = min(args.last_n, max_idx_for_label)
    start_idx = max(0, max_idx_for_label - n_steps + 1)
    idx_range = range(start_idx, start_idx + n_steps)

    LOG.info(
        "Using %d steps from M30 index [%d .. %d]",
        n_steps,
        start_idx,
        start_idx + n_steps - 1,
    )

    # تعداد ردیف‌هایی که در *_live نگه می‌داریم
    SL = {"30T": 1000, "15T": 2000, "5T": 5000, "1H": 1000}

    ans_path = base / "answer.txt"
    feed_log_path = base / "deploy_X_feed_log.csv"
    gen_pred_path = base / "generator_predictions.csv"

    gen_pred_path.unlink(missing_ok=True)

    wins = 0
    loses = 0
    none = 0
    acc_gen = 0.0
    cover_gen = 0.0

    # ---------- 3) حلقه‌ی اصلی روی کندل‌های M30 ----------
    for step, idx in enumerate(idx_range, start=1):
        ts_now = df30.loc[idx, "time"]

        # --- 3.1) ساخت CSVهای زنده تا ts_now ---
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

        # --- 3.2) انتظار برای answer.txt از دپلوی ---
        while not ans_path.exists():
            time.sleep(args.sleep)

        try:
            ans_raw = ans_path.read_text(encoding="utf-8").strip().upper()
        except Exception:
            ans_raw = ""
        finally:
            try:
                ans_path.unlink(missing_ok=True)
            except Exception:
                pass

        if ans_raw not in ("BUY", "SELL", "NONE"):
            ans = "NONE"
        else:
            ans = ans_raw

        # --- 3.3) خواندن آخرین رکورد deploy_X_feed_log.csv ---
        ts_feat = ts_now
        y_prob = np.nan
        cover_dep = np.nan

        try:
            dflog = pd.read_csv(feed_log_path)
            if dflog.empty:
                raise ValueError("deploy_X_feed_log.csv is empty")

            if "timestamp" in dflog.columns:
                dflog["timestamp"] = pd.to_datetime(dflog["timestamp"], errors="coerce")
            if "timestamp_trigger" in dflog.columns:
                dflog["timestamp_trigger"] = pd.to_datetime(
                    dflog["timestamp_trigger"], errors="coerce"
                )

            last = dflog.iloc[-1]
            if "timestamp" in last:
                ts_feat = pd.to_datetime(last["timestamp"])

            if "y_prob" in last:
                try:
                    y_prob = float(last["y_prob"])
                except Exception:
                    y_prob = np.nan

            if "cover_cum" in last:
                try:
                    cover_dep = float(last["cover_cum"])
                except Exception:
                    cover_dep = np.nan

            if "timestamp_trigger" in last and not pd.isna(last["timestamp_trigger"]):
                ts_trig = pd.to_datetime(last["timestamp_trigger"])
                if ts_trig != ts_now:
                    LOG.warning(
                        "timestamp_trigger mismatch at step %d: deploy=%s generator=%s",
                        step,
                        ts_trig,
                        ts_now,
                    )

        except Exception as e:
            LOG.warning(
                "Could not read deploy_X_feed_log.csv (using ts_now as ts_feat): %s", e
            )

        # --- 3.4) محاسبه‌ی y_true بر اساس ts_feat (زمان فیچر) ---
        y_true_dep: Optional[float] = None
        y_true_int = np.nan

        try:
            ts_feat_norm = pd.to_datetime(ts_feat)
            idx_feat_arr = np.where(df30["time"].values == ts_feat_norm)[0]
            if len(idx_feat_arr) == 0:
                idx_feat = idx
            else:
                idx_feat = int(idx_feat_arr[0])

            if idx_feat + 1 < len(df30):
                c_now = float(df30.loc[idx_feat, close_col_30])
                c_next = float(df30.loc[idx_feat + 1, close_col_30])
                y_true_dep = 1.0 if (c_next > c_now) else 0.0
                y_true_int = int(y_true_dep)
            else:
                y_true_dep = None
                y_true_int = np.nan

        except Exception as e:
            LOG.warning("Could not compute y_true for ts_feat=%s: %s", ts_feat, e)
            y_true_dep = None
            y_true_int = np.nan

        # --- 3.5) به‌روزرسانی آمار ژنراتور ---
        if ans == "BUY":
            pred_label: Optional[int] = 1
        elif ans == "SELL":
            pred_label = 0
        else:
            pred_label = None

        if pred_label is None or y_true_dep is None:
            none += 1
        else:
            y_true_int = int(y_true_dep)
            if pred_label == y_true_int:
                wins += 1
            else:
                loses += 1

        traded = wins + loses
        cover_gen = traded / float(step) if step > 0 else 0.0
        acc_gen = (wins / traded) if traded > 0 else 0.0

        LOG.info(
            "[Step %d] ts_now=%s ts_feat=%s action=%s y_true=%s | "
            "acc_gen=%.3f cover_gen=%.3f (wins=%d loses=%d none=%d) | cover_dep=%.3f",
            step,
            ts_now,
            ts_feat,
            ans,
            str(y_true_dep),
            acc_gen,
            cover_gen,
            wins,
            loses,
            none,
            cover_dep,
        )

        # --- 3.6) ذخیره‌ی رکورد در generator_predictions.csv ---
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

    # ---------- 4) گزارش نهایی ----------
    LOG.info(
        "[Final] acc_gen=%.3f cover_gen=%.3f wins=%d loses=%d none=%d",
        acc_gen,
        cover_gen,
        wins,
        loses,
        none,
    )
    LOG.info("=== Generator finished ===")


if __name__ == "__main__":
    main()
