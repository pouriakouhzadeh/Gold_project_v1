#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_like_sim.py  (aligned GT = y.iloc[-1])
شبیه‌ساز «واقعی» تولید CSV در هر کندل 30 دقیقه:
 - در هر iteration فقط tail لازم هر TF تا زمان t_cur نوشته می‌شود (context قابل تنظیم)
 - PREPARE_DATA_FOR_TRAIN (fast_mode) همان CSVهای موقت را می‌خواند و X,y می‌سازد
 - پیش‌بینی روی آخرین سطر X انجام می‌شود؛ برچسب با y.iloc[-1] مقایسه می‌شود (هم‌تراز)
 - آستانه‌ها اعمال می‌شود؛ آمار لحظه‌ای چاپ + لاگ؛ CSVهای موقت پاک می‌شوند
"""

from __future__ import annotations
import argparse, os, shutil, logging, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("Live-like CSV simulation over last N candles")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--base-data-dir", default=".", help="Folder containing full-history CSVs (XAUUSD_M*.csv)")
    p.add_argument("--tmp-dir", default="./_sim_csv", help="Folder to write per-iteration live CSVs")
    p.add_argument("--n-test", type=int, default=4000, help="How many last M30 candles to simulate")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix for CSV filenames")
    p.add_argument("--log-file", default="live_like_sim.log", help="Log file path")

    # tail context sizes (adjust for your indicators)
    p.add_argument("--ctx-5t", type=int, default=3000, help="5M context bars (tail)")
    p.add_argument("--ctx-15t", type=int, default=1200, help="15M context bars (tail)")
    p.add_argument("--ctx-30t", type=int, default=500, help="30M context bars (tail)")
    p.add_argument("--ctx-1h", type=int, default=300, help="1H context bars (tail)")
    return p.parse_args()

# ---------- Logging ----------
def setup_logger(log_path: str):
    log = logging.getLogger("live_sim")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log

# ---------- File name maps ----------
BASE_FILENAMES = {"5T":"{sym}_M5.csv","15T":"{sym}_M15.csv","30T":"{sym}_M30.csv","1H":"{sym}_H1.csv"}
LIVE_FILENAMES = {"5T":"{sym}_M5_live.csv","15T":"{sym}_M15_live.csv","30T":"{sym}_M30_live.csv","1H":"{sym}_H1_live.csv"}

# ---------- IO helpers ----------
def load_base_csvs(base_dir: Path, symbol: str) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for tf, patt in BASE_FILENAMES.items():
        f = base_dir / patt.format(sym=symbol)
        if not f.is_file():
            raise FileNotFoundError(f"Missing base CSV: {f}")
        df = pd.read_csv(f)
        expected = ["time","open","high","low","close","volume"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing columns: {missing}")
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        dfs[tf] = df
    return dfs

def build_live_paths(tmp_dir: Path, symbol: str) -> Dict[str, str]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return {tf: str(tmp_dir / patt.format(sym=symbol)) for tf, patt in LIVE_FILENAMES.items()}

def write_live_csvs_until(
    dfs_full: Dict[str, pd.DataFrame],
    t_until: pd.Timestamp,
    live_paths: Dict[str, str],
    ctx_map: Dict[str, int],
) -> None:
    """برای هر TF، رکوردهای time<=t_until را گرفته و فقط tail=context می‌نویسد."""
    for tf, out_path in live_paths.items():
        df = dfs_full[tf]
        df_cut = df[df["time"] <= t_until]
        ctx = int(ctx_map.get(tf, 500))
        df_cut = df_cut.tail(max(2, ctx)).copy()  # حداقل ۲ سطر برای diff/agg
        if df_cut.empty:
            df_cut = df.head(2).copy()
        df_cut["time"] = df_cut["time"].dt.strftime("%Y-%m-%d %H:%M")
        df_cut.to_csv(out_path, index=False)

def delete_live_csvs(tmp_dir: Path):
    if tmp_dir.is_dir():
        for p in tmp_dir.glob("*_live.csv"):
            try: p.unlink()
            except Exception: pass

# ---------- MAIN ----------
def main():
    args = parse_args()
    log  = setup_logger(args.log_file)

    mdl_path = Path(args.model).expanduser().resolve()
    if not mdl_path.is_file():
        raise FileNotFoundError(f"{mdl_path} not found")

    payload   = joblib.load(mdl_path)
    pipe_fit  : Pipeline = payload["pipeline"]
    window    : int      = int(payload["window_size"])
    neg_thr   : float    = float(payload["neg_thr"])
    pos_thr   : float    = float(payload["pos_thr"])
    feat_cols : List[str]= list(payload["train_window_cols"])

    log.info("Loaded model: %s | window=%d | neg_thr=%.3f | pos_thr=%.3f | #feats=%d",
             mdl_path.name, window, neg_thr, pos_thr, len(feat_cols))

    base_dir   = Path(args.base_data_dir).resolve()
    tmp_dir    = Path(args.tmp_dir).resolve()
    symbol     = args.symbol
    ctx_map    = {"5T": args.ctx_5t, "15T": args.ctx_15t, "30T": args.ctx_30t, "1H": args.ctx_1h}
    live_paths = build_live_paths(tmp_dir, symbol)

    # CSVهای کامل برای تعیین نقاط تصمیم (فقط M30)
    dfs_full = load_base_csvs(base_dir, symbol)
    m30 = dfs_full["30T"]
    if len(m30) < args.n_test + 1:
        raise RuntimeError(f"Not enough M30 rows ({len(m30)}) for n-test={args.n_test}")

    anchor_times = m30["time"].tail(args.n_test).reset_index(drop=True)

    # یک‌بار PREPARE_DATA_FOR_TRAIN بساز (same paths, هر بار CSV overwrite می‌شود)
    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        filepaths={"30T": live_paths["30T"], "15T": live_paths["15T"], "5T": live_paths["5T"], "1H": live_paths["1H"]},
        verbose=False,
        fast_mode=True,  # اگر پارامتر fast_mode ندارید، این را حذف کنید
    )

    wins = loses = undecided = decided = total_pred = 0

    try:
        for k, t_cur in enumerate(anchor_times, start=1):
            t0 = time.perf_counter()
            write_live_csvs_until(dfs_full, t_cur, live_paths, ctx_map)
            t_csv = time.perf_counter() - t0

            t1 = time.perf_counter()
            merged = prep.load_data()
            t_load = time.perf_counter() - t1

            # ساخت X,y برای همین iteration
            X, y, _, _ = prep.ready(
                merged, window=window, selected_features=feat_cols, mode="train"
            )

            # اگر warm-up کامل نشده
            if X.empty or len(y) == 0:
                total_pred += 1; undecided += 1
                acc = (wins / decided) if decided else 0.0
                log.info("[%-4d] %s  →  X empty (warm-up).  cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                         k, t_cur.strftime("%Y-%m-%d %H:%M"), wins, loses, undecided, decided, acc, t_csv, t_load)
                delete_live_csvs(tmp_dir)
                continue

            # اطمینان از حضور تمام ستون‌ها
            missing_cols = [c for c in feat_cols if c not in X.columns]
            if missing_cols:
                total_pred += 1; undecided += 1
                log.warning("[%-4d] %s  → missing %d feature columns; skipping. e.g. %s",
                            k, t_cur.strftime("%Y-%m-%d %H:%M"),
                            len(missing_cols), ", ".join(missing_cols[:5]))
                delete_live_csvs(tmp_dir)
                continue

            # آخرین سطرِ X + پاک‌سازی NaN/Inf
            x_last = X.iloc[[-1]][feat_cols]
            x_last = x_last.replace([np.inf, -np.inf], np.nan)
            if x_last.isna().any().any():
                med = X[feat_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
                x_last = x_last.fillna(med).fillna(0.0)
            if not np.all(np.isfinite(x_last.values)):
                total_pred += 1; undecided += 1
                log.warning("[%-4d] %s  → still non-finite after fill; skipping.",
                            k, t_cur.strftime("%Y-%m-%d %H:%M"))
                delete_live_csvs(tmp_dir)
                continue
            x_last = x_last.astype("float32")

            # پیش‌بینی + آستانه‌ها
            proba = float(pipe_fit.predict_proba(x_last)[:, 1][0])
            pred  = -1
            if proba <= neg_thr: pred = 0
            elif proba >= pos_thr: pred = 1

            # ⚠️ هم‌تراز با X: برچسب درست همان y.iloc[-1] است (نه compute_gt_next)
            gt = int(y.iloc[-1])

            total_pred += 1
            if pred == -1:
                undecided += 1
                status = "∅"
            else:
                decided += 1
                if pred == gt:
                    wins += 1; status = "WIN"
                else:
                    loses += 1; status = "LOSE"

            acc = (wins / decided) if decided else 0.0
            # برای شفافیت: زمانِ رکورد پیش‌بینی‌شده احتمالاً کندلِ قبلِ t_cur است؛ اگر خواستی می‌تونی از merged، دومین last time را لاگ کنی.
            log.info(
                "[%-4d] %s  proba=%.4f  pred=%s  gt=%s  → %-5s | cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                k,
                t_cur.strftime("%Y-%m-%d %H:%M"),
                proba,
                { -1:"-1", 0:"0", 1:"1" }[pred],
                str(gt),
                status,
                wins, loses, undecided, decided, acc,
                t_csv, t_load
            )

            delete_live_csvs(tmp_dir)

    finally:
        final_acc = (wins / decided) if decided else 0.0
        log.info(
            "DONE. total=%d | decided=%d | wins=%d | loses=%d | undecided=%d | acc=%.4f",
            total_pred, decided, wins, loses, undecided, final_acc
        )
        try:
            if tmp_dir.is_dir() and not any(tmp_dir.glob("*")):
                shutil.rmtree(tmp_dir)
        except Exception:
            pass

if __name__ == "__main__":
    main()
