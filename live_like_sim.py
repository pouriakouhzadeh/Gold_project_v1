#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_like_sim.py — Realistic live-backtest simulator (predict-mode, stable-last-row + guardrails)

چه می‌کند؟
- برای هر t از آخر n_test کندل 30 دقیقه، چهار CSV «زنده» تا آخرین کندلِ بسته‌شده و مشترک بین همه‌ی تایم‌فریم‌ها می‌نویسد.
- PREPARE_DATA_FOR_TRAIN همان CSVها را با mode="predict" می‌خواند و فیچر می‌سازد (بدون نگاه به آینده، سطر آخر پایدار).
- پیش‌بینی روی آخرین ردیف X انجام می‌شود؛ GT = جهت کندل بعدی نسبت به همان last_time در M30 کامل.
- آستانه‌ها اعمال می‌شود؛ آمار لحظه‌ای در لاگ چاپ می‌شود؛ CSVهای موقت در صورت عدم نیاز حذف می‌گردند.

گاردریل‌ها:
- هم‌ترازی تایم‌استمپ‌ها بین TFها (take_last_closed_rows)
- warm-up کافی، نبود NaN/Inf، تطابق و ترتیب دقیق ستون‌ها با train_window_cols، و dtype=float32 (guard_and_prepare_for_predict)
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



from gp_guardrails import (
    guard_and_prepare_for_predict,
    take_last_closed_rows,
    latest_common_timestamp,
    WarmupNotEnough, ColumnsMismatch, BadValuesFound, TimestampNotAligned
)

# ===== CLI ==================================================================
def parse_args():
    p = argparse.ArgumentParser("Live-like CSV simulation over last N M30 candles")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--base-data-dir", default=".", help="Folder containing full-history CSVs (XAUUSD_M*.csv)")
    p.add_argument("--tmp-dir", default="./_sim_csv", help="Folder to write per-iteration live CSVs")
    p.add_argument("--n-test", type=int, default=4000, help="Number of last M30 candles to simulate")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix for CSV filenames")
    p.add_argument("--log-file", default="live_like_sim.log", help="Log file path")

    # tail context sizes (تنظیم بر حسب اندیکاتورها / warm-up)
    p.add_argument("--ctx-5t", type=int, default=3000, help="5M context (tail rows)")
    p.add_argument("--ctx-15t", type=int, default=1200, help="15M context (tail rows)")
    p.add_argument("--ctx-30t", type=int, default=500,  help="30M context (tail rows)")
    p.add_argument("--ctx-1h", type=int, default=300,  help="1H context (tail rows)")

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
    """برای هر TF، رکوردهای time<=t_until را گرفته و فقط tail=context می‌نویسد (حداقل 2 ردیف)."""
    for tf, out_path in live_paths.items():
        df = dfs_full[tf]
        df_cut = df[df["time"] <= t_until]
        ctx = int(ctx_map.get(tf, 500))
        df_cut = df_cut.tail(max(2, ctx)).copy()
        if df_cut.empty:
            df_cut = df.head(2).copy()
        df_cut["time"] = df_cut["time"].dt.strftime("%Y-%m-%d %H:%M")
        df_cut.to_csv(out_path, index=False)

def read_live_csvs(live_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for tf, p in live_paths.items():
        df = pd.read_csv(p)
        df["time"] = pd.to_datetime(df["time"])
        out[tf] = df
    return out

def rewrite_from_frames(frames: Dict[str, pd.DataFrame], live_paths: Dict[str, str]) -> None:
    for tf, df in frames.items():
        df2 = df.copy()
        df2["time"] = pd.to_datetime(df2["time"]).dt.strftime("%Y-%m-%d %H:%M")
        df2.to_csv(live_paths[tf], index=False)

def delete_live_csvs(tmp_dir: Path):
    if not tmp_dir.is_dir():
        return
    for p in tmp_dir.glob("*_live.csv"):
        try:
            p.unlink()
        except Exception:
            pass

def compute_gt_next(m30_df: pd.DataFrame, t_ref: pd.Timestamp) -> Optional[int]:
    """GT = جهت کندل بعدی در M30 کامل نسبت به t_ref. اگر t_ref آخرین کندل باشد → None."""
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
    """برخی نسخه‌های PREPARE_DATA_FOR_TRAIN آرگومان fast_mode ندارند؛ هر دو را پشتیبانی می‌کنیم."""
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
    try:
        return PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T",
            filepaths=filepaths,
            verbose=False,
            fast_mode=True,
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
            # 1) ساخت CSVهای «زنده» تا t_cur
            t0 = time.perf_counter()
            write_live_csvs_until(dfs_full, t_cur, live_paths, ctx_map)

            # 2) هم‌ترازی تایم‌استمپ‌ها: trim همه‌ی TFها تا آخرین تایم مشترک (کندل‌های بسته‌شده)
            try:
                live_frames = read_live_csvs(live_paths)
                live_frames_aligned = take_last_closed_rows(live_frames)
                rewrite_from_frames(live_frames_aligned, live_paths)
                ts_common = latest_common_timestamp(live_frames_aligned)
            except (TimestampNotAligned, Exception) as e:
                total_pred += 1; undecided += 1
                dt = pd.to_datetime(t_cur).strftime("%Y-%m-%d %H:%M")
                log.warning("[%-4d] %s  → timestamp alignment failed: %s  → skip (undecided).",
                            k, dt, str(e))
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            t_csv = time.perf_counter() - t0

            # 3) لود و ادغام و ساخت فیچر (mode='predict': آخرین سطر پایدار)
            t_load0 = time.perf_counter()
            merged = prep.load_data()
            t_load = time.perf_counter() - t_load0

            tcol = "30T_time" if "30T_time" in merged.columns else "time"
            if merged.empty or merged[tcol].isna().all():
                # warm-up یا عدم ادغام موفق
                total_pred += 1; undecided += 1
                acc = (wins / decided) if decided else 0.0
                log.info("[%-4d] %s  → merged empty. cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                         k, pd.to_datetime(t_cur).strftime("%Y-%m-%d %H:%M"),
                         wins, loses, undecided, decided, acc, t_csv, t_load)
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            last_time = pd.to_datetime(merged[tcol].dropna().iloc[-1])

            # 4) ساخت X (انتخاب عین ستون‌های آموزش) + گاردریل‌ها
            X, _, _, _ = prep.ready(merged, window=window, selected_features=feat_cols, mode="predict")

            # حداقل تاریخچه‌ی لازم (بر اساس ctx های تعریف‌شده)
            min_required_history = {"5T": args.ctx_5t, "15T": args.ctx_15t, "30T": args.ctx_30t, "1H": args.ctx_1h}
            ctx_lengths = {}
            try:
                # طول تاریخچه‌های فعلی را از همان live_frames_aligned (بعد از trim) می‌گیریم
                ctx_lengths = {tf: len(df) for tf, df in live_frames_aligned.items()}
                # اعمال گاردریل‌ها (warm-up/NaN-Inf/columns-order/dtype)
                X = guard_and_prepare_for_predict(
                    X=X,
                    train_window_cols=feat_cols,
                    min_required_history=min_required_history,
                    ctx_history_lengths=ctx_lengths,
                    where="X_last"
                )
            except (WarmupNotEnough, ColumnsMismatch, BadValuesFound) as e:
                total_pred += 1; undecided += 1
                acc = (wins / decided) if decided else 0.0
                log.warning("[%-4d] %s  → guard failed: %s  → skip (undecided). cum acc=%.4f  [csv %.3fs | load %.3fs]",
                            k, last_time.strftime("%Y-%m-%d %H:%M"), str(e), acc, t_csv, t_load)
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            if X.empty:
                total_pred += 1; undecided += 1
                acc = (wins / decided) if decided else 0.0
                log.info("[%-4d] %s  → X empty after prepare (warm-up). cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f  [csv %.3fs | load %.3fs]",
                         k, last_time.strftime("%Y-%m-%d %H:%M"),
                         wins, loses, undecided, decided, acc, t_csv, t_load)
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            # 5) پیش‌بینی روی آخرین ردیف (پایدار و align شده با last_time)
            x_last = X.iloc[[-1]][feat_cols].astype("float32")
            try:
                proba = float(pipe_fit.predict_proba(x_last)[:, 1][0])
            except Exception as e:
                total_pred += 1; undecided += 1
                log.warning("[%-4d] %s  → predict_proba failed: %s → skip.",
                            k, last_time.strftime("%Y-%m-%d %H:%M"), str(e))
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            if proba <= neg_thr: pred = 0
            elif proba >= pos_thr: pred = 1
            else: pred = -1

            # GT دقیقاً نسبت به last_time (هم‌راستا با X آخر)
            gt = compute_gt_next(dfs_full["30T"], last_time)

            # 6) آمار
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
                "[%-4d] %s  proba=%.4f  pred=%s  gt=%s  → %-5s | cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f%s  [csv %.3fs | load %.3fs | ctx %s]",
                k, last_time.strftime("%Y-%m-%d %H:%M"),
                proba, { -1:"-1", 0:"0", 1:"1" }[pred],
                "-" if gt is None else str(gt),
                status, wins, loses, undecided, decided, acc, misalign_note,
                t_csv, t_load, str({k2: ctx_lengths.get(k2, 0) for k2 in ["5T","15T","30T","1H"]})
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
