#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_in_production_parity.py
Author: Pouria + Assistant

وظایف:
  - هر ثانیه پوشه watch را چک می‌کند:
      اگر answer.txt نبود و چهار فایل *_live.csv (M5,M15,M30,H1) حاضر و پایدار بودند
      => بر اساس آخرین زمان 30دقیقه‌ای، داده را تا همان لحظه کات می‌کند،
         فیچرها را دقیقاً مثل آموزش می‌سازد، predict_proba می‌گیرد،
         آستانه‌ها را اعمال می‌کند، BUY/SELL/NONE را در answer.txt می‌نویسد،
         سپس هر چهار فایل *_live.csv را حذف می‌کند.
  - کش پارکتی از دیتافریم خامِ ادغام‌شده می‌سازد/می‌خواند (سرعت).
  - برابری ستون‌ها با train_window_cols از متادیتای مدل را enforce می‌کند (report+fill).

پارامترهای مهم:
  --watch-dir          مسیر فایل‌های live و answer.txt
  --raw-dir            مسیر CSVهای کامل تاریخی (XAUUSD_M5.csv, M15, M30, H1)
  --model-path         مسیر best_model.pkl
  --meta-path          مسیر best_model.meta.json (default: همین پوشه مدل)
  --answer-path        نام فایل پاسخ (default: answer.txt)
  --cache-dir          مسیر کش پارکتی (default: ./cache_parity)
  --poll-sec           فاصله بررسی (ثانیه)
  --main-tf            تایم‌فریم اصلی (default: 30T)
  --min-context_bars   حداقل کندل‌های لازم روی تایم‌فریم اصلی (default: 3)
  --positive-class     برچسب کلاس مثبت مدل (default: 1)
  --thr-low / --thr-high  override آستانه‌ها (اختیاری)
  --reset-state        بازسازی اجباری کش
  --log-path           مسیر لاگ

یادداشت:
  متادیتا (window_size, train_window_cols, neg_thr, pos_thr) از best_model.meta.json خوانده می‌شود
  و در صورت وجود train_distribution.json آستانه‌ها override می‌شوند.  (متادیتا: window_size=2, ...).
"""

from __future__ import annotations
import os, sys, time, json, gc, argparse, logging, hashlib
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import multiprocessing as mp

# Limit BLAS threads (safe defaults)
for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","TBB_NUM_THREADS"):
    os.environ.setdefault(var, str(max(1, mp.cpu_count())))

# Optional: oneAPI accel
try:
    from sklearnex import patch_sklearn  # type: ignore
    patch_sklearn()
except Exception:
    pass

import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ---- Project prep class
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # must exist in PYTHONPATH

APP_NAME = "prediction_in_production_parity"

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.handlers.clear(); logger.propagate = False
    logger.setLevel(logging.INFO)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_path, maxBytes=20_000_000, backupCount=3, encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def md5_of_list(cols: List[str]) -> str:
    return hashlib.md5(",".join(cols).encode("utf-8")).hexdigest()

def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(text).strip() + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def stable_file(p: Path, stable_ms: int = 150) -> bool:
    if not p.exists(): return False
    m1 = p.stat().st_mtime_ns
    time.sleep(stable_ms/1000.0)
    return p.exists() and p.stat().st_mtime_ns == m1

def resolve_timeframe_paths(base_dir: Path, symbol: str="XAUUSD") -> Dict[str, str]:
    cands = {
        "30T": [f"{symbol}_M30.csv", f"{symbol}_30T.csv"],
        "15T": [f"{symbol}_M15.csv", f"{symbol}_15T.csv"],
        "5T" : [f"{symbol}_M5.csv",  f"{symbol}_5T.csv" ],
        "1H" : [f"{symbol}_H1.csv",  f"{symbol}_1H.csv" ],
    }
    out: Dict[str,str] = {}
    for tf, names in cands.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found = str(p); break
        if not found:
            # case-insensitive fallback
            for child in base_dir.iterdir():
                if child.is_file() and any(child.name.lower()==nm.lower() for nm in names):
                    found = str(child); break
        if found:
            out[tf] = found
    if "30T" not in out:
        raise FileNotFoundError("Main timeframe file (M30/30T) not found in raw-dir.")
    return out

def read_csv_mql4(path: Path) -> pd.DataFrame:
    """Normalize a timeframe CSV to (time, open, high, low, close, volume) with time index."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    tc = pick("time","datetime","date")
    oc = pick("open"); hc = pick("high"); lc = pick("low"); cc = pick("close")
    vc = pick("volume","tick_volume","tickvolume")
    need = [tc,oc,hc,lc,cc,vc]
    if any(c is None for c in need):
        raise ValueError(f"CSV columns not recognized in {path}")
    df = df.rename(columns={tc:"time", oc:"open", hc:"high", lc:"low", cc:"close", vc:"volume"})
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df

def ensure_columns(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols: return X
    X2 = X.copy()
    missing = [c for c in cols if c not in X2.columns]
    for c in missing: X2[c] = 0.0
    return X2[cols]

def load_meta(meta_path: Path, td_path: Optional[Path]) -> dict:
    """Load window_size, train_window_cols, thresholds from meta and train_distribution if exists."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    payload = {
        "window_size": int(meta.get("window_size", meta.get("window", 1))),
        "train_window_cols": list(meta.get("train_window_cols", meta.get("feats", []))),
        "neg_thr": float(meta.get("neg_thr", 0.005)),
        "pos_thr": float(meta.get("pos_thr", 0.995)),
    }
    # Optional override by train_distribution.json if available
    if td_path and td_path.exists():
        try:
            with open(td_path, "r", encoding="utf-8") as jf:
                td = json.load(jf)
            if "neg_thr" in td: payload["neg_thr"] = float(td["neg_thr"])
            if "pos_thr" in td: payload["pos_thr"] = float(td["pos_thr"])
        except Exception:
            pass
    return payload

def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        model = joblib_load(f)
    return model

def build_or_load_cache(prep: PREPARE_DATA_FOR_TRAIN, cache_dir: Path, reset: bool, logger: logging.Logger,
                        tcol: str) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "merged_raw.parquet"
    if reset and cache_path.exists():
        cache_path.unlink()
    merged: Optional[pd.DataFrame] = None
    if cache_path.exists():
        try:
            merged = pd.read_parquet(cache_path)
            # ensure time column
            if tcol not in merged.columns:
                if isinstance(merged.index, pd.DatetimeIndex):
                    merged[tcol] = merged.index
                else:
                    # try to find a time-like column
                    for cand in ("time","datetime","Date","Timestamp","30T_time","m30_time"):
                        if cand in merged.columns:
                            merged[tcol] = pd.to_datetime(merged[cand], utc=False); break
            merged[tcol] = pd.to_datetime(merged[tcol], utc=False)
            merged.sort_values(tcol, inplace=True)
            merged.reset_index(drop=True, inplace=True)
            logger.info("[cache] loading merged_raw from %s", str(cache_path))
            return merged
        except Exception as e:
            logger.warning("[cache] failed to load cache (%s). Rebuilding...", e)
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass

    logger.info("[prep] building merged_raw via PREPARE_DATA_FOR_TRAIN.load_data()")
    merged = prep.load_data()
    # force proper time column
    if tcol not in merged.columns:
        # try to locate a time column from prep
        for c in merged.columns:
            if "time" in c.lower():
                merged[tcol] = pd.to_datetime(merged[c], utc=False)
                break
    if tcol not in merged.columns:
        raise KeyError(f"Cannot locate '{tcol}' in merged data; check PREPARE_DATA_FOR_TRAIN.load_data()")
    merged[tcol] = pd.to_datetime(merged[tcol], utc=False)
    merged.sort_values(tcol, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    # save
    merged.to_parquet(cache_path, index=False)
    logger.info("[cache] saved merged_raw to %s", str(cache_path))
    return merged

def positive_index_of(model, positive_class: int = 1) -> int:
    """Return index of positive_class in predict_proba output."""
    try:
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            # pipeline
            est = list(model.named_steps.values())[-1]
            classes = getattr(est, "classes_", None)
        if classes is None:
            # last resort: assume classes=[0,1]
            return 1 if positive_class == 1 else 0
        classes = list(classes)
        return classes.index(positive_class)
    except Exception:
        return 1 if positive_class == 1 else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--meta-path", default="best_model.meta.json")
    ap.add_argument("--train-dist-path", default="train_distribution.json")
    ap.add_argument("--answer-path", default="answer.txt")
    ap.add_argument("--cache-dir", default="./cache_parity")
    ap.add_argument("--poll-sec", type=float, default=1.0)
    ap.add_argument("--main-tf", default="30T")
    ap.add_argument("--min-context_bars", type=int, default=3)
    ap.add_argument("--positive-class", type=int, default=1)
    ap.add_argument("--thr-low", type=float, default=None)
    ap.add_argument("--thr-high", type=float, default=None)
    ap.add_argument("--reset-state", action="store_true")
    ap.add_argument("--log-path", default=str(Path.home() / "prediction_in_production_parity.log"))
    args = ap.parse_args()

    logger = setup_logger(args.log_path)
    logger.info("=== %s started ===", APP_NAME)

    watch_dir = Path(args.watch_dir).resolve()
    raw_dir   = Path(args.raw_dir).resolve()
    model_p   = Path(args.model_path).resolve()
    meta_p    = Path(args.meta_path).resolve() if Path(args.meta_path).exists() else (model_p.parent / "best_model.meta.json")
    train_dist_p = Path(args.train_dist_path).resolve() if Path(args.train_dist_path).exists() else (model_p.parent / "train_distribution.json")

    # --- Load model + meta ---
    model = load_model(model_p)
    meta_payload = load_meta(meta_p, train_dist_p)
    window = int(meta_payload["window_size"])
    train_cols: List[str] = list(meta_payload.get("train_window_cols", []))
    neg_thr = float(args.thr_low if args.thr_low is not None else meta_payload["neg_thr"])
    pos_thr = float(args.thr_high if args.thr_high is not None else meta_payload["pos_thr"])
    task = "classifier"
    logger.info("Model loaded | window=%d | feats=%d | thr=(%.3f, %.3f) | task=%s",
                window, len(train_cols), neg_thr, pos_thr, task)
    if train_cols:
        logger.info("Train columns md5=%s", md5_of_list(train_cols))

    # --- Resolve raw file paths (for reference/log) ---
    filepaths = resolve_timeframe_paths(raw_dir, symbol="XAUUSD")
    for tf in ("5T","15T","30T","1H"):
        if tf in filepaths:
            logger.info("[raw] %s -> %s", tf, filepaths[tf])

    # --- PREP & cache of merged raw ---
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=args.main_tf,
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    tcol = f"{prep.main_timeframe}_time"
    merged_raw = build_or_load_cache(prep, Path(args.cache_dir), args.reset_state, logger, tcol=tcol)
    logger.info("[init] merged_raw shape=%s | first=%s | last=%s",
                merged_raw.shape, merged_raw[tcol].min(), merged_raw[tcol].max())

    # --- Live files in watch dir ---
    m5_live  = watch_dir / "XAUUSD_M5_live.csv"
    m15_live = watch_dir / "XAUUSD_M15_live.csv"
    m30_live = watch_dir / "XAUUSD_M30_live.csv"
    h1_live  = watch_dir / "XAUUSD_H1_live.csv"
    answer_fp = watch_dir / args.answer_path

    last_ts_done: Optional[pd.Timestamp] = None
    pos_idx = positive_index_of(model, args.positive_class)
    logger.info("Using positive_class=%s (proba column index=%s) for BUY mapping", args.positive_class, pos_idx)

    while True:
        try:
            # اگر پاسخ هنوز هست، صبر کن تا ژنراتور آن را بردارد
            if answer_fp.exists():
                time.sleep(args.poll_sec); continue

            live_present = all(p.exists() for p in [m5_live, m15_live, m30_live, h1_live])
            if not live_present:
                time.sleep(args.poll_sec); continue
            if not all(stable_file(p, 150) for p in [m5_live, m15_live, m30_live, h1_live]):
                time.sleep(args.poll_sec); continue

            # زمان مرجع از M30_live
            df_m30 = read_csv_mql4(m30_live)
            if df_m30.empty:
                time.sleep(args.poll_sec); continue
            ts_now = df_m30.index.max()

            # جلوگیری از پردازش تکراری
            if (last_ts_done is not None) and (ts_now <= last_ts_done):
                time.sleep(args.poll_sec); continue

            # اگر ts_now جلوتر از کش است، کش را بازسازی کن
            if ts_now > merged_raw[tcol].max():
                logger.info("[refresh] ts_now beyond cached last -> rebuilding merged_raw")
                merged_raw = build_or_load_cache(prep, Path(args.cache_dir), True, logger, tcol=tcol)

            # برش تا ts_now
            df_cut = merged_raw.loc[merged_raw[tcol] <= ts_now].copy()
            if df_cut.shape[0] < max(args.min_context_bars, window+1):
                logger.warning("[skip] not enough context on main TF at %s (rows=%d)", ts_now, df_cut.shape[0])
                time.sleep(args.poll_sec); continue

            # ساخت فیچرها دقیقا مثل آموزش
            X, y, _, _ = prep.ready(
                df_cut,
                window=window,
                selected_features=train_cols,
                mode="train",
                predict_drop_last=False,
                train_drop_last=False
            )
            if X is None or len(X)==0:
                logger.warning("[skip] empty X at %s", ts_now)
                time.sleep(args.poll_sec); continue

            # هم‌قدسازی و امن‌سازی ستون‌ها
            L = len(X) if y is None else min(len(X), len(y))
            X = pd.DataFrame(X).iloc[:L].reset_index(drop=True)
            X = ensure_columns(X, train_cols)
            X_last = X.tail(1)

            # proba
            prob_all = model.predict_proba(X_last)
            prob_buy = float(prob_all[0, pos_idx])

            # آستانه‌ها → برچسب
            if prob_buy >= pos_thr:
                label = "BUY"
            elif prob_buy <= neg_thr:
                label = "SELL"
            else:
                label = "NONE"

            atomic_write_text(answer_fp, label)
            logger.info("[Predict] ts=%s | prob=%.6f | thr=(%.3f,%.3f) → %s | X_cols=%d md5=%s | wrote=%s",
                        ts_now, prob_buy, neg_thr, pos_thr, label, X_last.shape[1],
                        md5_of_list(list(X_last.columns)), str(answer_fp))

            # پس از نوشتن پاسخ، ۴ فایل live حذف شوند تا چرخه بعدی آماده شود
            for p in [m5_live, m15_live, m30_live, h1_live]:
                try:
                    p.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("Failed to remove %s: %s", p, e)

            last_ts_done = ts_now
            gc.collect()
            time.sleep(args.poll_sec)

        except Exception as e:
            logger.exception("Unhandled error in loop: %s", e)
            time.sleep(max(1.0, args.poll_sec))

if __name__ == "__main__":
    main()
