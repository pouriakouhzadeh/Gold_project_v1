#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_like_sim.py — شبیه‌سازی «شبیه-لایو» برای دو پروتکل:
  • TRAIN-ALIGNED:  X_{t-1}  →  GT(t-1→t)   (عینِ batch)
  • REAL:
      - اگر --predict-drop-last=0:  X_t  →  GT(t→t+1)
      - اگر --predict-drop-last=1:  پس از ساخت فیچرها، آخرین ردیف حذف می‌شود،
        و با بریدن CSVها تا t_cur (که همان t+1 است)، X آخر می‌شود t، ولی
        به‌دلیل drop-last، X آخر عملاً t-1 است؛ در نتیجه GT روی t_feat (=t-1)
        محاسبه می‌شود که همان هدف t→t+1 را می‌سنجد، بدون استفاده از مقادیر
        ناپایدار t.

نکتهٔ مهم:
- برای مقایسه دقیق با batch، از حالت TRAIN استفاده کنید:
    --mode train --fast-mode 1
- برای شبیه‌سازی واقعیِ t→t+1 با حذفِ «بعد از فیچر»، اجرا کنید:
    --mode real --predict-drop-last 1 --fast-mode 1

نیازمندی: کلاس PREPARE_DATA_FOR_TRAIN باید پارامتر
predict_drop_last را در ready(...) بپذیرد و «بعد از ساخت فیچرها»
آخرین ردیف را حذف کند (طبق راهنمایی‌ای که اعمال کردید).
"""

from __future__ import annotations
import argparse, os, shutil, logging, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser("Live-like simulation (TRAIN vs REAL; drop-last-after-features supported)")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--base-data-dir", default=".", help="Folder containing full-history CSVs")
    p.add_argument("--tmp-dir", default="./_sim_csv", help="Folder to write per-iteration live CSVs")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol prefix (XAUUSD)")
    p.add_argument("--n-test", type=int, default=4000, help="# of last M30 bars to simulate")
    p.add_argument("--log-file", default="live_like_sim.log", help="Path to rotating log file")

    # Context tails (برای warm-up اندیکاتورها روی هر TF)
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
                   help="real: predict on X_t or X_{t-1} (با drop-last) | train: X_{t-1} → (t-1→t)")

    # AFTER-feature drop of last row (برای REAL)
    p.add_argument("--predict-drop-last", type=int, choices=[0,1], default=1,
                   help="In REAL mode: drop last row AFTER feature creation (default=1)")

    # PREP fast_mode (توصیه برای tail CSV = 1)
    p.add_argument("--fast-mode", type=int, default=1, choices=[0,1],
                   help="1: disable drift-scan/bad-cols scan (recommended for tail CSVs)")

    # ممیزی
    p.add_argument("--audit", type=int, default=0, help="Every N steps, print CSV last times & t_cur/t_feat")

    # اجبار اجرای سریالی joblib (در صورت نیاز)
    p.add_argument("--no-parallel", type=int, default=0, choices=[0,1],
                   help="Try to force serial joblib backend (debug)")

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

def maybe_force_serial_joblib(flag: bool, log: logging.Logger):
    if not flag:
        return
    try:
        import joblib
        joblib.parallel_backend('threading')
        os.environ["JOBLIB_START_METHOD"] = "threading"
        log.info("Joblib backend forced to 'threading' for debugging.")
    except Exception as e:
        log.warning("Could not force joblib backend: %s", e)

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
    """Write 'live' CSVs as if platform exported up to and including t_until."""
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
    if not tmp_dir.is_dir(): return
    for p in tmp_dir.glob("*_live.csv"):
        try: p.unlink()
        except Exception: pass

def read_last_times_from_live(live_paths: Dict[str,str]) -> Dict[str, Optional[pd.Timestamp]]:
    out: Dict[str, Optional[pd.Timestamp]] = {}
    for tf, path in live_paths.items():
        try:
            df = pd.read_csv(path)
            t = pd.to_datetime(df["time"].iloc[-1])
        except Exception:
            t = None
        out[tf] = t
    return out

# -------------------- GT helper --------------------
def compute_gt_next(m30_df: pd.DataFrame, t_ref: pd.Timestamp) -> Optional[int]:
    """
    GT(t_ref) = 1{ close(t_ref+1) > close(t_ref) } from full stable M30 data.
    """
    idx = m30_df.index[m30_df["time"] == t_ref]
    if len(idx) == 0: return None
    i = int(idx[0])
    if i + 1 >= len(m30_df): return None
    return int((float(m30_df.loc[i+1, "close"]) - float(m30_df.loc[i, "close"])) > 0)

def prev_m30_time(m30_df: pd.DataFrame, t: pd.Timestamp) -> Optional[pd.Timestamp]:
    idx = m30_df.index[m30_df["time"] == t]
    if len(idx) == 0: return None
    i = int(idx[0])
    if i - 1 < 0: return None
    return pd.to_datetime(m30_df.loc[i-1, "time"])

# -------------------- PREP wrapper --------------------
def build_prep(filepaths: Dict[str, str], fast_mode: bool):
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
    try:
        return PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T",
            filepaths=filepaths,
            verbose=False,
            fast_mode=bool(fast_mode),
        )
    except TypeError:
        return PREPARE_DATA_FOR_TRAIN(
            main_timeframe="30T", filepaths=filepaths, verbose=False
        )

# -------------------- MAIN --------------------
def main():
    args = parse_args()
    log = setup_logger(args.log_file)
    maybe_force_serial_joblib(bool(args.no_parallel), log)

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

    # انتخاب نقاط تصمیم: آخرین n_test کندلِ بسته‌شده‌ی 30T
    anchor_times = m30["time"].tail(args.n_test).reset_index(drop=True)

    mode_name = "TRAIN-ALIGNED" if (args.mode == "train") else ("REAL (drop-last)" if args.predict_drop_last else "REAL")
    log.info("Prepared %d anchor times. Mode=%s. Starting simulation …", len(anchor_times), mode_name)

    # PREP روی CSVهای «لایو»: fast_mode=True توصیه می‌شود
    prep = build_prep({
        "30T": live_paths["30T"],
        "15T": live_paths["15T"],
        "5T" : live_paths["5T"],
        "1H" : live_paths["1H"],
    }, fast_mode=bool(args.fast_mode))

    wins = loses = undecided = decided = total_pred = 0

    try:
        for k, t_cur in enumerate(anchor_times, start=1):
            t_cur = pd.to_datetime(t_cur)

            # 1) بنویس تا و شامل t_cur
            t0 = time.perf_counter()
            write_live_csvs_until(dfs_full, t_cur, live_paths, ctx_map)
            t_csv = time.perf_counter() - t0

            # 2) Merge & engineer via PREP (همان مسیر TRAIN)
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

            # 3) آماده‌سازی X (+times)
            if args.mode == "train":
                out = prep.ready(
                    merged.copy(),
                    window=window,
                    selected_features=feat_cols,
                    mode="train",
                    with_times=True,
                )
                X, y, _, price_ser, t_idx = out
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
                # REAL: با یا بدون drop-last بعد از فیچر
                out = prep.ready(
                    merged.copy(),
                    window=window,
                    selected_features=feat_cols,
                    mode="predict",
                    with_times=True,
                    predict_drop_last=bool(args.predict_drop_last),
                )
                # نسخه جدید ۵-خروجی می‌دهد: X, _, feats, price_ser, t_idx
                if len(out) == 5:
                    X, _, _, price_ser, t_idx = out
                else:
                    # پشتیبانی از نسخه‌های قدیمی‌تر
                    X, _, _, t_idx = out
                    price_ser = None

                if X.empty or t_idx is None or len(t_idx) == 0:
                    total_pred += 1; undecided += 1
                    acc = (wins / decided) if decided else 0.0
                    log.info("[%-4d] %s  → X empty (warm-up). cum: wins=%d loses=%d undecided=%d decided=%d acc=%.4f [csv %.3fs | load %.3fs]",
                             k, t_cur.strftime("%Y-%m-%d %H:%M"),
                             wins, loses, undecided, decided, acc, t_csv, t_load)
                    if not args.keep_csv: delete_live_csvs(tmp_dir)
                    continue

                t_feat = pd.to_datetime(t_idx.iloc[-1])

                # GT همیشه از CSV کامل محاسبه می‌شود
                gt = compute_gt_next(dfs_full["30T"], t_feat)

            # 4) ممیزی (اختیاری)
            if args.audit and (k % int(args.audit) == 0):
                last_times = read_last_times_from_live(live_paths)
                msg = (f"[AUDIT {k}] last_csv: 5T={last_times.get('5T')} | "
                       f"15T={last_times.get('15T')} | 30T={last_times.get('30T')} | 1H={last_times.get('1H')}")
                log.info(msg)
                # اگر drop-last فعال است، انتظار داریم t_feat == prev_m30(t_cur)
                if args.mode == "real" and bool(args.predict_drop_last):
                    t_prev = prev_m30_time(dfs_full["30T"], t_cur)
                    log.info("[AUDIT %d] t_cur=%s | expected t_feat=%s | actual t_feat=%s | GT=%s",
                             k, t_cur, t_prev, t_feat, "-" if gt is None else gt)
                else:
                    log.info("[AUDIT %d] t_cur=%s | t_feat=%s | GT=%s",
                             k, t_cur, t_feat, "-" if gt is None else gt)

            # 5) Feature columns must match training (order + presence)
            missing = [c for c in feat_cols if c not in X.columns]
            if missing:
                total_pred += 1; undecided += 1
                log.warning("[%-4d] %s  → missing %d features; e.g. %s … skip.",
                            k, t_feat.strftime("%Y-%m-%d %H:%M"),
                            len(missing), ", ".join(missing[:10]))
                if not args.keep_csv: delete_live_csvs(tmp_dir)
                continue

            # 6) Prepare last row (fill NaNs like training)
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

            # 7) Predict
            proba = float(pipe_fit.predict_proba(x_last)[:, 1][0])
            if proba <= neg_thr:
                pred = 0
            elif proba >= pos_thr:
                pred = 1
            else:
                pred = -1

            # 8) Score & log
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
        log.info("DONE. total=%d | decided=%d | wins=%d | loses=%d | undecided=%d | acc=%.4f | mode=%s | fast_mode=%s | predict_drop_last=%s",
                 total_pred, decided, wins, loses, undecided, final_acc, mode_name, bool(args.fast_mode), bool(args.predict_drop_last))

    finally:
        if not args.keep_csv:
            try:
                if tmp_dir.is_dir() and not any(tmp_dir.glob("*")):
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass

if __name__ == "__main__":
    main()


# like train :
#     python3 live_like_sim.py \
#   --mode train \
#   --fast-mode 1 \
#   --base-data-dir . \
#   --symbol XAUUSD \
#   --n-test 4000 \
#   --log-file live_like_train.log


# real stable :
# python3 live_like_sim.py \
#   --mode real \
#   --predict-drop-last 1 \
#   --fast-mode 1 \
#   --audit 50 \
#   --base-data-dir . \
#   --symbol XAUUSD \
#   --n-test 600 \
#   --log-file live_like_real.log

# برای شبیه‌سازی واقعیِ t→t+1 با حذف «بعد از فیچر» (پروتکل REAL-Stable):
# python3 live_like_sim.py \
#   --mode real \
#   --predict-drop-last 0 \
#   --fast-mode 1 \
#   --base-data-dir . \
#   --symbol XAUUSD \
#   --n-test 600 \
#   --log-file live_like_real_xt.log
