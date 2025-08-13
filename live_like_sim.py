#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_like_sim.py — Live-like backtest (REAL vs TRAIN-aligned)

REAL  mode (default):  predict on X_t  → GT(t→t+1)   [مثل محیط واقعی]
TRAIN mode (--mode train): predict on X_{t-1} → GT(t-1→t) [مطابق بچ/آموزش]

نکته مهم: چون در شبیه‌سازی از CSVهای کوتاه (tail) استفاده می‌کنیم،
drift-scan و کشف ستون‌های ناپایدار روی این tail معنی‌دار نیست.
بنابراین پیش‌فرض: fast_mode=True (بدون drift-scan). اگر در TRAIN
با fast_mode=False بوده‌اید و تاریخ شروع مشترک را دارید، فقط در آن صورت
--fast-mode 0 را فعال کنید.
"""

from __future__ import annotations
import argparse, os, shutil, logging, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser("Live-like simulation (real vs train-aligned)")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--base-data-dir", default=".", help="Folder containing full-history CSVs")
    p.add_argument("--tmp-dir", default="./_sim_csv", help="Folder to write per-iteration live CSVs")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix (XAUUSD)")
    p.add_argument("--n-test", type=int, default=4000, help="# of last M30 bars to simulate")
    p.add_argument("--log-file", default="live_like_sim.log", help="Path to rotating log file")

    # Context tails (چند ردیف آخر هر TF را نگه داریم که warm-up اندیکاتورها پوشش داده شود)
    p.add_argument("--ctx-5t", type=int, default=3000)
    p.add_argument("--ctx-15t", type=int, default=1200)
    p.add_argument("--ctx-30t", type=int, default=500)
    p.add_argument("--ctx-1h", type=int, default=300)

    p.add_argument("--keep-csv", action="store_true", help="Keep generated live CSVs per step")

    # Decision thresholds override
    p.add_argument("--neg-thr", type=float, default=None)
    p.add_argument("--pos-thr", type=float, default=None)

    # Alignment: real | train
    p.add_argument("--mode", choices=["real", "train"], default="real",
                   help="real: X_t→(t→t+1) | train: X_{t-1}→(t-1→t)")

    # PREP fast_mode (default True توصیه می‌شود چون drift-scan روی tail خراب می‌کند)
    p.add_argument("--fast-mode", type=int, default=1, choices=[0,1],
                   help="1: disable drift-scan/bad-cols scan (recommended for tail CSVs)")

    return p.parse_args()

# -------------------- logging --------------------
def setup_logger(path: str):
    log = logging.getLogger("live_sim")
    log.setLevel(logging.INFO); log.propagate = False
    for h in list(log.handlers): log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log

# -------------------- file maps --------------------
BASE_FILENAMES = {
    "5T" : "{sym}_M5.csv",
    "15T": "{sym}_M15.csv",
    "30T": "{sym}_M30.csv",
    "1H" : "{sym}_H1.csv",
}
LIVE_FILENAMES = {
    "5T" : "{sym}_M5_live.csv",
    "15T": "{sym}_M15_live.csv",
    "30T": "{sym}_M30_live.csv",
    "1H" : "{sym}_H1_live.csv",
}

# -------------------- IO helpers --------------------
def load_base_csvs(base_dir: Path, symbol: str) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for tf, patt in BASE_FILENAMES.items():
        fp = base_dir / patt.format(sym=symbol)
        if not fp.is_file():
            raise FileNotFoundError(f"Missing base CSV: {fp}")
        df = pd.read_csv(fp)
        for c in ["time","open","high","low","close","volume"]:
            if c not in df.columns:
                raise ValueError(f"{fp} missing column: {c}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
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
    """Write 'live' CSVs as if MT exported up to and including t_until."""
    for tf, out_path in live_paths.items():
        df = dfs_full[tf]
        df_cut = df[df["time"] <= t_until]
        ctx = int(ctx_map.get(tf, 500))
        # حداقل دو ردیف برای ایمنی diff/shift
        df_cut = df_cut.tail(max(2, ctx)).copy()
        if df_cut.empty:
            df_cut = df.head(2).copy()
        # شبیه CSV خروجی MT
        df_cut["time"] = df_cut["time"].dt.strftime("%Y-%m-%d %H:%M")
        df_cut.to_csv(out_path, index=False)

def delete_live_csvs(tmp_dir: Path):
    if not tmp_dir.is_dir(): return
    for p in tmp_dir.glob("*_live.csv"):
        try: p.unlink()
        except Exception: pass

# -------------------- GT helper --------------------
def compute_gt_next(m30_df: pd.DataFrame, t_ref: pd.Timestamp) -> Optional[int]:
    """
    GT_real(t) = 1{ close(t+1) > close(t) } from full stable M30 data.
    """
    idx = m30_df.index[m30_df["time"] == t_ref]
    if len(idx) == 0: return None
    i = int(idx[0])
    if i + 1 >= len(m30_df): return None
    return int((float(m30_df.loc[i+1, "close"]) - float(m30_df.loc[i, "close"])) > 0)

# -------------------- PREP wrapper --------------------
def build_prep(filepaths: Dict[str, str], fast_mode: bool):
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
    # drift-scan فقط وقتی fast_mode=False است (پیش‌فرض این اسکریپت: True)
    try:
        return PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T",
            filepaths=filepaths,
            verbose=False,
            fast_mode=bool(fast_mode),
        )
    except TypeError:
        # نسخه قدیمی بدون fast_mode
        return PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T", filepaths=filepaths, verbose=False
        )

# -------------------- MAIN --------------------
def main():
    args = parse_args()
    log = setup_logger(args.log_file)

    # Load model bundle
    mdl_path = Path(args.model).expanduser().resolve()
    if not mdl_path.is_file():
        raise FileNotFoundError(f"{mdl_path} not found")
    payload   = joblib.load(mdl_path)
    pipe_fit  : Pipeline = payload["pipeline"]
    window    : int      = int(payload["window_size"])
    neg_thr_p : float    = float(payload["neg_thr"])
    pos_thr_p : float    = float(payload["pos_thr"])
    feat_cols : List[str]= list(payload["train_window_cols"])

    neg_thr   = float(args.neg_thr) if args.neg_thr is not None else neg_thr_p
    pos_thr   = float(args.pos_thr) if args.pos_thr is not None else pos_thr_p

    log.info("Loaded model: %s | window=%d | neg_thr=%.3f | pos_thr=%.3f | #feats=%d",
             mdl_path.name, window, neg_thr, pos_thr, len(feat_cols))

    # Base data & paths
    base_dir   = Path(args.base_data_dir).resolve()
    tmp_dir    = Path(args.tmp_dir).resolve()
    symbol     = args.symbol
    ctx_map    = {"5T": args.ctx_5t, "15T": args.ctx_15t, "30T": args.ctx_30t, "1H": args.ctx_1h}
    live_paths = build_live_paths(tmp_dir, symbol)
    dfs_full   = load_base_csvs(base_dir, symbol)
    m30        = dfs_full["30T"]

    log.info("Base CSV sizes → 5T=%d | 15T=%d | 30T=%d | 1H=%d",
             len(dfs_full["5T"]), len(dfs_full["15T"]), len(dfs_full["30T"]), len(dfs_full["1H"]))

    if len(m30) < args.n_test + 1:
        raise RuntimeError(f"Not enough M30 rows ({len(m30)}) for n-test={args.n_test}")

    # Decision points = last n_test closed bars on 30T
    anchor_times = m30["time"].tail(args.n_test).reset_index(drop=True)
    mode_name = "TRAIN-ALIGNED" if (args.mode == "train") else "REAL"
    log.info("Prepared %d anchor times. Mode=%s. Starting simulation …", len(anchor_times), mode_name)

    # PREP روی CSVهای «لایو» tail: پیش‌فرض fast_mode=True تا drift/bad_cols اجرا نشود
    prep = build_prep({
        "30T": live_paths["30T"],
        "15T": live_paths["15T"],
        "5T" : live_paths["5T"],
        "1H" : live_paths["1H"],
    }, fast_mode=bool(args.fast_mode))

    wins = loses = undecided = decided = total_pred = 0
    first_warn_miss = True

    try:
        for k, t_cur in enumerate(anchor_times, start=1):
            t_cur = pd.to_datetime(t_cur)

            # 1) Write live CSVs (up to and including t_cur)
            t0 = time.perf_counter()
            write_live_csvs_until(dfs_full, t_cur, live_paths, ctx_map)
            t_csv = time.perf_counter() - t0

            # 2) Merge & engineer via PREP (same path as training)
            t1 = time.perf_counter()
            merged = prep.load_data()
            t_load = time.perf_counter() - t1

            tcol = "30T_time" if "30T_time" in merged.columns else "time"
            if merged.empty or merged[tcol].isna().all():
                total_pred += 1; undecided += 1
                acc = (wins / decided) if decided else 0.0
                log.info("[%-4d] %s  → merged empty. cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                         k, t_cur.strftime("%Y-%m-%d %H:%M"),
                         wins, loses, undecided, decided, acc, t_csv, t_load)
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            # 3) Build X (+times) according to mode
            if args.mode == "train":
                # مثل TRAIN: ردیف آخر بعد از پنجره‌بندی حذف می‌شود → X آخر معادل t-1
                out = prep.ready(
                    merged.copy(),
                    window=window,
                    selected_features=feat_cols,
                    mode="train",
                    with_times=True,
                )
                # unpack
                X, y, _, _, t_idx = out
                if X.empty or len(y) == 0:
                    total_pred += 1; undecided += 1
                    acc = (wins / decided) if decided else 0.0
                    log.info("[%-4d] %s  → X empty (warm-up). cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f [csv %.3fs | load %.3fs]",
                             k, t_cur.strftime("%Y-%m-%d %H:%M"),
                             wins, loses, undecided, decided, acc, t_csv, t_load)
                    if not args.keep_csv: delete_live_csvs(tmp_dir)
                    continue

                t_feat = pd.to_datetime(t_idx.iloc[-1])  # t-1
                gt = int(y.iloc[-1])                     # حرکت t-1→t
            else:
                # REAL: در PREDICT هیچ حذف آخری نداریم → آخرین X = t
                out = prep.ready(
                    merged.copy(),
                    window=window,
                    selected_features=feat_cols,
                    mode="predict",
                    with_times=True,
                )
                X, _, _, _, t_idx = out
                if X.empty or t_idx.empty:
                    total_pred += 1; undecided += 1
                    acc = (wins / decided) if decided else 0.0
                    log.info("[%-4d] %s  → X empty (warm-up). cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f [csv %.3fs | load %.3fs]",
                             k, t_cur.strftime("%Y-%m-%d %H:%M"),
                             wins, loses, undecided, decided, acc, t_csv, t_load)
                    if not args.keep_csv: delete_live_csvs(tmp_dir)
                    continue

                t_feat = pd.to_datetime(t_idx.iloc[-1])          # t
                gt = compute_gt_next(dfs_full["30T"], t_feat)    # GT(t→t+1)

            # 4) Feature columns must match training (order + presence)
            missing = [c for c in feat_cols if c not in X.columns]
            if missing:
                total_pred += 1; undecided += 1
                if first_warn_miss:
                    log.warning("Feature mismatch with training. Showing once… "
                                "(e.g. missing: %s)", ", ".join(missing[:10]))
                    first_warn_miss = False
                log.warning("[%-4d] %s  → missing %d features; skip.",
                            k, t_feat.strftime("%Y-%m-%d %H:%M"), len(missing))
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            # 5) Prepare last row (fill NaNs like training)
            x_last = X.iloc[[-1]][feat_cols].replace([np.inf, -np.inf], np.nan)
            if x_last.isna().any().any():
                med = X[feat_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
                x_last = x_last.fillna(med).fillna(0.0)
            if not np.all(np.isfinite(x_last.values)):
                total_pred += 1; undecided += 1
                log.warning("[%-4d] %s  → non-finite after fill; skip.",
                            k, t_feat.strftime("%Y-%m-%d %H:%M"))
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue
            x_last = x_last.astype("float32")

            # 6) Predict
            proba = float(pipe_fit.predict_proba(x_last)[:, 1][0])
            if proba <= neg_thr:
                pred = 0
            elif proba >= pos_thr:
                pred = 1
            else:
                pred = -1

            # 7) Score & log
            total_pred += 1
            if (gt is None) or (pred == -1):
                undecided += 1
                status = "∅"
            else:
                decided += 1
                if pred == gt:
                    wins += 1; status = "WIN"
                else:
                    loses += 1; status = "LOSE"

            acc = (wins / decided) if decided else 0.0
            log.info(
                "[%-4d] %s  proba=%.4f  pred=%s  gt=%s  → %-5s | "
                "cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                k, t_feat.strftime("%Y-%m-%d %H:%M"),
                proba, { -1:"-1", 0:"0", 1:"1" }[pred],
                "-" if gt is None else str(gt),
                status, wins, loses, undecided, decided, acc, t_csv, t_load
            )

            if not args.keep_csv:
                delete_live_csvs(tmp_dir)

        final_acc = (wins / decided) if decided else 0.0
        log.info("DONE. total=%d | decided=%d | wins=%d | loses=%d | undecided=%d | acc=%.4f | mode=%s | fast_mode=%s",
                 total_pred, decided, wins, loses, undecided, final_acc, mode_name, bool(args.fast_mode))

    finally:
        if not args.keep_csv:
            try:
                if tmp_dir.is_dir() and not any(tmp_dir.glob("*")):
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass

if __name__ == "__main__":
    main()



# python3 live_like_sim.py --mode real --fast-mode 1
# python3 live_like_sim.py --mode train --fast-mode 1
# python3 live_like_sim.py --mode real --fast-mode 0
# ولی برای CSVهای کوتاه «لایو»، --fast-mode 1 توصیه می‌شود.