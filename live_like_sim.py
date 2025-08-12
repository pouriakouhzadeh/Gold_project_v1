#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_like_sim.py  — Realistic live-backtest simulator (predict-mode, stable-last-row)

چی کار می‌کند؟
- برای هر t از آخر n_test کندل 30 دقیقه، چهار CSV زنده تا «کندل بسته‌شده‌ی t» می‌نویسد.
- PREPARE_DATA_FOR_TRAIN دقیقاً همین CSVها را می‌خواند، با mode="predict" فیچر می‌سازد.
- پیش‌بینی روی آخرین ردیف X انجام می‌شود؛ برچسب حقیقی = جهت کندل (t→t+1) از M30 کامل.
- آستانه‌ها اعمال می‌شود؛ آمار لحظه‌ای لاگ + کنسول؛ CSVهای موقت همان لحظه حذف می‌شوند.

پیش‌نیاز هم‌ترازی:
- در آموزش، X_t باید فقط از اطلاعات تا t-1 ساخته شود و y_t = 1{close_{t+1} > close_t}.
- ستون‌های ورودی inference باید دقیقاً با payload["train_window_cols"] یکسان باشند.
"""

from __future__ import annotations
import argparse, os, shutil, logging, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline


# ===== CLI ==================================================================
def parse_args():
    p = argparse.ArgumentParser("Live-like CSV simulation over last N M30 candles")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--base-data-dir", default=".", help="Folder containing full-history CSVs (XAUUSD_M*.csv)")
    p.add_argument("--tmp-dir", default="./_sim_csv", help="Folder to write per-iteration live CSVs")
    p.add_argument("--n-test", type=int, default=4000, help="Number of last M30 candles to simulate")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix for CSV filenames")
    p.add_argument("--log-file", default="live_like_sim.log", help="Log file path")

    # tail context sizes (تنظیم بر حسب اندیکاتورها)
    p.add_argument("--ctx-5t", type=int, default=3000, help="5M context (tail rows)")
    p.add_argument("--ctx-15t", type=int, default=1200, help="15M context (tail rows)")
    p.add_argument("--ctx-30t", type=int, default=500,  help="30M context (tail rows)")
    p.add_argument("--ctx-1h", type=int, default=300,  help="1H context (tail rows)")

    # اگر خواستید CSVها را برای بررسی نگه دارید
    p.add_argument("--keep-csv", action="store_true", help="Do not delete per-iteration live CSVs")
    return p.parse_args()


# ===== Logging ===============================================================
def setup_logger(log_path: str):
    log = logging.getLogger("live_sim")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log


# ===== File name maps ========================================================
BASE_FILENAMES = {"5T":"{sym}_M5.csv","15T":"{sym}_M15.csv","30T":"{sym}_M30.csv","1H":"{sym}_H1.csv"}
LIVE_FILENAMES = {"5T":"{sym}_M5_live.csv","15T":"{sym}_M15_live.csv","30T":"{sym}_M30_live.csv","1H":"{sym}_H1_live.csv"}


# ===== IO helpers ============================================================
def load_base_csvs(base_dir: Path, symbol: str) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for tf, patt in BASE_FILENAMES.items():
        fp = base_dir / patt.format(sym=symbol)
        if not fp.is_file():
            raise FileNotFoundError(f"Missing base CSV: {fp}")
        df = pd.read_csv(fp)
        expected = ["time","open","high","low","close","volume"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} missing columns: {missing}")
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
    """
    برای هر TF، رکوردهای time<=t_until را گرفته و فقط tail=context می‌نویسد (حداقل 2 ردیف).
    """
    for tf, out_path in live_paths.items():
        df = dfs_full[tf]
        df_cut = df[df["time"] <= t_until]
        ctx = int(ctx_map.get(tf, 500))
        df_cut = df_cut.tail(max(2, ctx)).copy()
        if df_cut.empty:
            df_cut = df.head(2).copy()
        df_cut["time"] = df_cut["time"].dt.strftime("%Y-%m-%d %H:%M")
        df_cut.to_csv(out_path, index=False)


def delete_live_csvs(tmp_dir: Path):
    if not tmp_dir.is_dir():
        return
    for p in tmp_dir.glob("*_live.csv"):
        try:
            p.unlink()
        except Exception:
            pass


def compute_gt_next(m30_df: pd.DataFrame, t_ref: pd.Timestamp) -> Optional[int]:
    """
    GT = جهت کندل بعدی در M30 کامل، نسبت به زمان مرجع t_ref.
    اگر t_ref آخرین کندل باشد، None برمی‌گرداند.
    """
    idx = m30_df.index[m30_df["time"] == t_ref]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i + 1 >= len(m30_df):
        return None
    c0 = float(m30_df.loc[i, "close"])
    c1 = float(m30_df.loc[i + 1, "close"])
    return 1 if (c1 - c0) > 0 else 0


# ===== PREPARE wrapper (robust to different signatures) ======================
def build_prep(filepaths: Dict[str, str]):
    """
    بعضی نسخه‌های PREPARE_DATA_FOR_TRAIN آرگومان fast_mode ندارند.
    این سازنده را طوری نوشته‌ام که هر دو حالت را پشتیبانی کند.
    """
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
    try:
        return PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T",
            filepaths=filepaths,
            verbose=False,
            fast_mode=True,       # اگر نبود، except → بدون fast_mode
        )
    except TypeError:
        return PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T",
            filepaths=filepaths,
            verbose=False,
        )


# ===== MAIN =================================================================
def main():
    args = parse_args()
    log  = setup_logger(args.log_file)

    # ---- load model payload ----
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

    # ---- base data ----
    base_dir   = Path(args.base_data_dir).resolve()
    tmp_dir    = Path(args.tmp_dir).resolve()
    symbol     = args.symbol
    ctx_map    = {"5T": args.ctx_5t, "15T": args.ctx_15t, "30T": args.ctx_30t, "1H": args.ctx_1h}
    live_paths = build_live_paths(tmp_dir, symbol)

    dfs_full = load_base_csvs(base_dir, symbol)
    m30 = dfs_full["30T"]
    log.info("Base CSV sizes → 5T=%d | 15T=%d | 30T=%d | 1H=%d",
             len(dfs_full["5T"]), len(dfs_full["15T"]), len(dfs_full["30T"]), len(dfs_full["1H"]))

    if len(m30) < args.n_test + 1:
        raise RuntimeError(f"Not enough M30 rows ({len(m30)}) for n-test={args.n_test}")

    # نقاط تصمیم: آخرین n_test کندل‌های بسته‌شده‌ی 30T
    anchor_times = m30["time"].tail(args.n_test).reset_index(drop=True)
    log.info("Prepared %d anchor times. Starting simulation …", len(anchor_times))

    # یک PREP ثابت؛ هر iteration همان مسیر CSVها را overwrite می‌کند
    prep = build_prep(
        {"30T": live_paths["30T"], "15T": live_paths["15T"], "5T": live_paths["5T"], "1H": live_paths["1H"]}
    )

    # ---- counters ----
    wins = loses = undecided = decided = total_pred = 0

    try:
        for k, t_cur in enumerate(anchor_times, start=1):
            t_csv0 = time.perf_counter()
            write_live_csvs_until(dfs_full, t_cur, live_paths, ctx_map)
            t_csv = time.perf_counter() - t_csv0

            # لود و فیچر—ساخت با mode="predict" (آخرین ردیف، پایدار و بدون نگاه به آینده)
            t_load0 = time.perf_counter()
            merged = prep.load_data()
            t_load = time.perf_counter() - t_load0

            # اگر ادغام، کندل آخر را نتواند بسازد، ممکن است last_time < t_cur شود؛ GT را روی last_time می‌گیریم
            tcol = "30T_time" if "30T_time" in merged.columns else "time"
            if merged.empty or merged[tcol].isna().all():
                total_pred += 1; undecided += 1
                acc = (wins / decided) if decided else 0.0
                log.info("[%-4d] %s  → merged empty. cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                         k, pd.to_datetime(t_cur).strftime("%Y-%m-%d %H:%M"),
                         wins, loses, undecided, decided, acc, t_csv, t_load)
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            last_time = pd.to_datetime(merged[tcol].dropna().iloc[-1])

            # ساخت X با mode="predict": y دامی است ولی X آخرین ردیف، همان ردیف پایدارِ t=last_time
            X, _, _, _ = prep.ready(
                merged, window=window, selected_features=feat_cols, mode="predict"
            )

            if X.empty:
                total_pred += 1; undecided += 1
                acc = (wins / decided) if decided else 0.0
                log.info("[%-4d] %s  → X empty (warm-up). cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                         k, last_time.strftime("%Y-%m-%d %H:%M"),
                         wins, loses, undecided, decided, acc, t_csv, t_load)
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            # اطمینان از یکسانی ستون‌ها
            missing = [c for c in feat_cols if c not in X.columns]
            if missing:
                total_pred += 1; undecided += 1
                log.warning("[%-4d] %s  → missing %d feature columns; skip. ex: %s",
                            k, last_time.strftime("%Y-%m-%d %H:%M"),
                            len(missing), ", ".join(missing[:6]))
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            # آخرین ردیف پایدار
            x_last = X.iloc[[-1]][feat_cols]
            x_last = x_last.replace([np.inf, -np.inf], np.nan)
            if x_last.isna().any().any():
                med = X[feat_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
                x_last = x_last.fillna(med).fillna(0.0)
            vals = x_last.values
            if not np.all(np.isfinite(vals)):
                total_pred += 1; undecided += 1
                log.warning("[%-4d] %s  → still non-finite after fill; skip.",
                            k, last_time.strftime("%Y-%m-%d %H:%M"))
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue
            x_last = x_last.astype("float32")

            # پیش‌بینی
            proba = float(pipe_fit.predict_proba(x_last)[:, 1][0])
            if proba <= neg_thr: pred = 0
            elif proba >= pos_thr: pred = 1
            else: pred = -1

            # GT دقیقاً نسبت به last_time (هم‌راستا با X آخر)
            gt = compute_gt_next(dfs_full["30T"], last_time)

            total_pred += 1
            if pred == -1 or gt is None:
                undecided += 1; status = "∅"
            else:
                decided += 1
                if pred == gt: wins += 1; status = "WIN"
                else:          loses += 1; status = "LOSE"

            acc = (wins / decided) if decided else 0.0
            misalign_note = "" if last_time == pd.to_datetime(t_cur) else " (note: merged last_time ≠ anchor)"
            log.info(
                "[%-4d] %s  proba=%.4f  pred=%s  gt=%s  → %-5s | cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f%s  [csv %.3fs | load %.3fs]",
                k, last_time.strftime("%Y-%m-%d %H:%M"),
                proba, { -1:"-1", 0:"0", 1:"1" }[pred],
                "-" if gt is None else str(gt),
                status, wins, loses, undecided, decided, acc, misalign_note,
                t_csv, t_load
            )

            if not args.keep_csv:
                delete_live_csvs(tmp_dir)

        # summary
        final_acc = (wins / decided) if decided else 0.0
        log.info("DONE. total=%d | decided=%d | wins=%d | loses=%d | undecided=%d | acc=%.4f",
                 total_pred, decided, wins, loses, undecided, final_acc)

    finally:
        # اگر فولدر خالی است، حذف
        if not args.keep_csv:
            try:
                if tmp_dir.is_dir() and not any(tmp_dir.glob("*")):
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
