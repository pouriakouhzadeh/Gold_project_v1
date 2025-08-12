#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_like_sim.py  (predict-mode + warm-up aware + stability check)
شبیه‌ساز واقعی اجرای مدل:
 - Anchor و GT از M30.csv استخراج می‌شود (سریع و شفاف)
 - در هر iteration فقط tail لازم هر TF تا t_cur نوشته می‌شود (ctx قابل تنظیم)
 - PREPARE_DATA_FOR_TRAIN همان CSVهای موقت را می‌خواند (mode="predict")
 - GT نسبت به آخرین زمان merge محاسبه می‌شود (بدون misalign)
 - Warm-up مدیریت می‌شود: burn-in، min-x-rows، و چک پایداری اختیاری
 - لاگ لحظه‌ای و گزارش نهایی؛ CSVهای موقت پاک می‌شوند
"""

from __future__ import annotations
import argparse, shutil, logging, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("Live-like CSV simulation over last N candles (warm-up aware)")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--base-data-dir", default=".", help="Folder containing full-history CSVs (XAUUSD_M*.csv)")
    p.add_argument("--tmp-dir", default="./_sim_csv", help="Folder to write per-iteration live CSVs")
    p.add_argument("--n-test", type=int, default=4000, help="How many last M30 candles to simulate")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix for CSV filenames")
    p.add_argument("--log-file", default="live_like_sim.log", help="Log file path")

    # Tail context sizes per TF
    p.add_argument("--ctx-5t", type=int, default=3000, help="5M context bars (tail)")
    p.add_argument("--ctx-15t", type=int, default=1200, help="15M context bars (tail)")
    p.add_argument("--ctx-30t", type=int, default=500,  help="30M context bars (tail)")
    p.add_argument("--ctx-1h", type=int, default=300,   help="1H context bars (tail)")

    # Warm-up controls
    p.add_argument("--burn-in", type=int, default=50, help="Initial iterations to skip from scoring")
    p.add_argument("--min-x-rows", type=int, default=200, help="Require at least this many X rows before scoring")

    # Optional stability check (heavier)
    p.add_argument("--stability-check", action="store_true", help="Double-check x_last with larger context")
    p.add_argument("--stab-ctx-mult", type=float, default=2.0, help="Multiplier for context during stability check")
    p.add_argument("--stab-atol", type=float, default=1e-10, help="abs tolerance for x_last match")
    p.add_argument("--stab-rtol", type=float, default=1e-6, help="rel tolerance for x_last match")
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
    """برای هر TF، رکوردهای time<=t_until را گرفته و فقط tail=context می‌نویسد (حداقل ۲ ردیف)."""
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
    if tmp_dir.is_dir():
        for p in tmp_dir.glob("*_live.csv"):
            try: p.unlink()
            except Exception: pass

# ---------- GT from M30 ----------
def build_anchor_and_gt_from_m30(m30: pd.DataFrame, n_test: int) -> Tuple[pd.Series, Dict[pd.Timestamp, int]]:
    """
    بر اساس 30T:
      - anchor_times = آخر n_test زمان‌ها
      - gt_map[time] = جهت کندل بعدی (sign(close(t+1)-close(t)))
    """
    m30 = m30.copy()
    m30["time"] = pd.to_datetime(m30["time"])
    m30.sort_values("time", inplace=True)
    m30.reset_index(drop=True, inplace=True)

    if len(m30) < n_test + 1:
        raise RuntimeError(f"Not enough M30 rows ({len(m30)}) for n-test={n_test}")

    anchor_times = m30["time"].tail(n_test).reset_index(drop=True)

    gt_map: Dict[pd.Timestamp, int] = {}
    for i in range(len(m30) - 1):
        t = m30.loc[i, "time"]
        gt = int((float(m30.loc[i+1, "close"]) - float(m30.loc[i, "close"])) > 0.0)
        gt_map[t] = gt
    return anchor_times, gt_map

# ---------- Stability check (optional) ----------
def compute_x_last(prep: PREPARE_DATA_FOR_TRAIN,
                   merged_live: pd.DataFrame,
                   feat_cols: List[str],
                   window: int) -> pd.DataFrame:
    X, _, _, _ = prep.ready(merged_live, window=window, selected_features=feat_cols, mode="predict")
    if X.empty:
        return X
    x_last = X.iloc[[-1]][feat_cols]
    x_last = x_last.replace([np.inf, -np.inf], np.nan)
    if x_last.isna().any().any():
        med = X[feat_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        x_last = x_last.fillna(med).fillna(0.0)
    return x_last.astype("float32")

def xlast_matches(a: pd.DataFrame, b: pd.DataFrame, rtol: float, atol: float) -> bool:
    if a.empty or b.empty:
        return False
    va, vb = a.values.ravel(), b.values.ravel()
    if va.shape != vb.shape:
        return False
    return np.allclose(va, vb, rtol=rtol, atol=atol, equal_nan=False)

# ---------- MAIN ----------
def main():
    args = parse_args()
    log  = setup_logger(args.log_file)

    # -- 0) Load model artefacts
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

    # -- 1) Load base CSVs & anchors/GT
    dfs_full = load_base_csvs(base_dir, symbol)
    log.info("Base CSV sizes → 5T=%d | 15T=%d | 30T=%d | 1H=%d",
             len(dfs_full["5T"]), len(dfs_full["15T"]), len(dfs_full["30T"]), len(dfs_full["1H"]))

    m30 = dfs_full["30T"].copy()
    anchor_times, gt_map = build_anchor_and_gt_from_m30(m30, args.n_test)
    log.info("Prepared %d anchor times. Starting simulation …", len(anchor_times))

    # -- 2) Persistent PREP
    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        filepaths={"30T": live_paths["30T"], "15T": live_paths["15T"], "5T": live_paths["5T"], "1H": live_paths["1H"]},
        verbose=False,
        fast_mode=True,   # اگر پارامتری با این نام ندارید، حذف کنید
    )

    wins = loses = undecided = decided = total_pred = warmup_skipped = 0

    try:
        for k, t_cur in enumerate(anchor_times, start=1):
            t0 = time.perf_counter()
            write_live_csvs_until(dfs_full, t_cur, live_paths, ctx_map)
            t_csv = time.perf_counter() - t0

            t1 = time.perf_counter()
            merged_live = prep.load_data()
            t_load = time.perf_counter() - t1

            tcol = "30T_time"
            if tcol not in merged_live.columns or merged_live.empty:
                warmup_skipped += 1
                log.info("[%-4d] %s  → merged empty or missing %s (warm-up skip). [csv %.3fs | load %.3fs]",
                         k, t_cur.strftime("%Y-%m-%d %H:%M"), tcol, t_csv, t_load)
                delete_live_csvs(tmp_dir)
                continue

            last_time = pd.to_datetime(merged_live[tcol].dropna().iloc[-1])

            # --- X (predict) ---
            X, _, _, _ = prep.ready(merged_live, window=window, selected_features=feat_cols, mode="predict")
            if (X.empty) or (len(X) < args.min_x_rows) or (X[feat_cols].isna().any().any()):
                warmup_skipped += 1
                log.info("[%-4d] %s  → warm-up (X rows=%d, NaN=%s). Skipped.  [csv %.3fs | load %.3fs]",
                         k, last_time.strftime("%Y-%m-%d %H:%M"),
                         len(X), bool(X[feat_cols].isna().any().any()) if not X.empty else True,
                         t_csv, t_load)
                delete_live_csvs(tmp_dir)
                continue

            # --- Stability check (اختیاری) ---
            if args.stability_check:
                # بزرگ‌تر کردن context موقتاً
                big_ctx = {
                    kf: int(max(2, int(v * args.stab_ctx_mult)))
                    for kf, v in ctx_map.items()
                }
                write_live_csvs_until(dfs_full, t_cur, live_paths, big_ctx)
                merged_big = prep.load_data()
                x_main = compute_x_last(prep, merged_live, feat_cols, window)
                x_big  = compute_x_last(prep, merged_big , feat_cols, window)
                # بازگردانی فایل‌ها به context اصلی برای iteration بعد
                write_live_csvs_until(dfs_full, t_cur, live_paths, ctx_map)

                if (x_main.empty) or (x_big.empty) or (not xlast_matches(x_main, x_big, args.stab_rtol, args.stab_atol)):
                    warmup_skipped += 1
                    log.info("[%-4d] %s  → stability-check not passed (warm-up). Skipped.  [csv %.3fs | load %.3fs]",
                             k, last_time.strftime("%Y-%m-%d %H:%M"), t_csv, t_load)
                    delete_live_csvs(tmp_dir)
                    continue

                x_last = x_main
            else:
                # x_last بدون stability check
                x_last = compute_x_last(prep, merged_live, feat_cols, window)
                if x_last.empty or (not np.all(np.isfinite(x_last.values))):
                    warmup_skipped += 1
                    log.info("[%-4d] %s  → x_last not finite (warm-up). Skipped. [csv %.3fs | load %.3fs]",
                             k, last_time.strftime("%Y-%m-%d %H:%M"), t_csv, t_load)
                    delete_live_csvs(tmp_dir)
                    continue

            # --- Burn-in iterations ---
            if k <= args.burn_in:
                warmup_skipped += 1
                log.info("[%-4d] %s  → burn-in (%d/%d). Skipped.",
                         k, last_time.strftime("%Y-%m-%d %H:%M"), k, args.burn_in)
                delete_live_csvs(tmp_dir)
                continue

            # --- Predict + thresholds ---
            proba = float(pipe_fit.predict_proba(x_last)[:, 1][0])
            pred  = -1
            if proba <= neg_thr: pred = 0
            elif proba >= pos_thr: pred = 1

            # --- GT نسبت به last_time ---
            gt = gt_map.get(last_time, None)

            total_pred += 1
            if (pred == -1) or (gt is None):
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
                "[%-4d] %s  proba=%.4f  pred=%s  gt=%s  → %-5s | cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                k,
                last_time.strftime("%Y-%m-%d %H:%M"),
                proba,
                { -1:"-1", 0:"0", 1:"1" }[pred],
                ("-" if gt is None else str(gt)),
                status,
                wins, loses, undecided, decided, acc,
                t_csv, t_load
            )

            delete_live_csvs(tmp_dir)

    finally:
        eff_total = total_pred
        final_acc = (wins / decided) if decided else 0.0
        logging.getLogger("live_sim").info(
            "DONE. scored_total=%d | decided=%d | wins=%d | loses=%d | undecided=%d | acc=%.4f | warmup_skipped=%d",
            eff_total, decided, wins, loses, undecided, final_acc, warmup_skipped
        )
        try:
            if tmp_dir.is_dir() and not any(tmp_dir.glob("*")):
                shutil.rmtree(tmp_dir)
        except Exception:
            pass

if __name__ == "__main__":
    main()
