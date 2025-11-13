#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_in_production_parity.py
Author: Pouria + Assistant (final, parity-safe)

وظایف:
  - هر 1 ثانیه پوشه watch را چک می‌کند: اگر چهار فایل XAUUSD_M5_live.csv,
    XAUUSD_M15_live.csv, XAUUSD_M30_live.csv, XAUUSD_H1_live.csv موجود باشند
    و answer.txt وجود نداشته باشد، پیش‌بینی می‌سازد.
  - ویژگی‌ها را دقیقاً با همان مسیر آموزش می‌سازد (PREPARE_DATA_FOR_TRAIN.ready)
    و با لیست ستون‌های train_window_cols از best_model.meta.json هم‌راستا می‌کند.
  - از payload واقعی مدل (pipeline) استفاده می‌کند (robust loader)، کلاس مثبت را کشف می‌کند
    و خروجی proba را با آستانه‌ها به BUY/SELL/NONE نگاشت می‌کند.
  - کش Parquet از دیتافریم «ادغام‌شده» می‌سازد (بار اول کمی زمان‌بر، دفعات بعد سریع).
  - پس از نوشتن answer.txt چهار فایل *_live.csv را حذف می‌کند (هندشیک با ژنراتور/MQL4).

نیازمندی‌ها:
  - Python 3.9+
  - pandas, numpy, scikit-learn, joblib
  - ماژول پروژه: prepare_data_for_train.PREPARE_DATA_FOR_TRAIN
  - فایل‌ها: best_model.pkl , best_model.meta.json (کنار هم)

مثال اجرا:
  python3 prediction_in_production_parity.py \
    --watch-dir /home/pouria/gold_project9 \
    --raw-dir   /home/pouria/gold_project9 \
    --model-path /home/pouria/gold_project9/best_model.pkl \
    --meta-path  /home/pouria/gold_project9/best_model.meta.json \
    --answer-path answer.txt \
    --cache-dir  /home/pouria/cache_parity \
    --poll-sec 1 \
    --main-tf 30T \
    --positive-class 1 \
    --thr-low 0.45 --thr-high 0.55 \
    --log-path /home/pouria/prediction_in_production_parity.log
"""

from __future__ import annotations
import os, sys, time, json, gc, hashlib, signal, argparse, logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Tuple, List, Optional
from datetime import datetime

# Optional acceleration
try:
    from sklearnex import patch_sklearn  # type: ignore
    patch_sklearn()
except Exception:
    pass

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

# ------------------------- Project modules -------------------------
try:
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
except Exception as e:
    PREPARE_DATA_FOR_TRAIN = None
    raise ImportError("Cannot import PREPARE_DATA_FOR_TRAIN. Make sure it is on PYTHONPATH.") from e

APP = "prediction_in_production_parity"

# --------------------------- Logging ---------------------------
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(APP)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers): logger.removeHandler(h)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = RotatingFileHandler(log_path, maxBytes=20_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

def atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def stable_file(path: str, stable_ms: int = 150) -> bool:
    if not os.path.exists(path): return False
    m1 = os.path.getmtime(path)
    time.sleep(stable_ms/1000.0)
    if not os.path.exists(path): return False
    m2 = os.path.getmtime(path)
    return m1 == m2

# ------------------------- File paths resolver -------------------------
def resolve_timeframe_paths(base_dir: str, symbol: str="XAUUSD") -> Dict[str, str]:
    """Find raw CSVs for 5T, 15T, 30T, 1H similar to live_like_sim."""
    cand = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    resolved: Dict[str,str] = {}
    for tf, names in cand.items():
        found = None
        for nm in names:
            p = os.path.join(base_dir, nm)
            if os.path.exists(p): found = p; break
        if not found:
            # case-insensitive scan
            lower = [n.lower() for n in names]
            for fn in os.listdir(base_dir):
                if fn.lower() in lower:
                    found = os.path.join(base_dir, fn); break
        if found:
            resolved[tf] = found
    if "30T" not in resolved:
        raise FileNotFoundError("Main timeframe '30T' CSV not found in raw-dir.")
    return resolved

# --------------------------- Model loader ---------------------------
def load_payload(model_path: str, meta_path: Optional[str], logger: logging.Logger) -> dict:
    """
    Robust loader. Returns dict with keys:
      pipeline, window_size, train_window_cols, neg_thr, pos_thr, positive_class
    """
    obj = None
    last_err = None

    # 1) try joblib
    if joblib is not None:
        try:
            obj = joblib.load(model_path)
        except Exception as e:
            last_err = e
    # 2) fallback pickle
    if obj is None:
        import pickle
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            last_err = e

    if obj is None:
        raise RuntimeError(f"Could not load model from {model_path}. Last error: {repr(last_err)}")

    payload: dict = {}
    if isinstance(obj, dict):
        # try common keys
        model = obj.get("pipeline") or obj.get("model") or obj.get("estimator") or obj.get("clf") or obj.get("best_estimator")
        if model is None:
            # maybe nested under unknown key
            for k, v in obj.items():
                if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                    model = v; break
        if model is None:
            raise ValueError("Loaded dict but pipeline estimator not found.")
        payload["pipeline"] = model
        payload["window_size"] = int(obj.get("window_size", 1))
        feats = obj.get("train_window_cols") or obj.get("feats") or []
        payload["train_window_cols"] = list(feats) if isinstance(feats, (list,tuple)) else []
        payload["neg_thr"] = float(obj.get("neg_thr", 0.005))
        payload["pos_thr"] = float(obj.get("pos_thr", 0.995))
    else:
        # obj is an estimator
        payload["pipeline"] = obj
        payload["window_size"] = 1
        payload["train_window_cols"] = []
        payload["neg_thr"] = 0.005
        payload["pos_thr"] = 0.995

    # Merge meta JSON if provided (preferred / authoritative)
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # window/cols/thr from meta override everything
            if "window_size" in meta: payload["window_size"] = int(meta["window_size"])
            if "train_window_cols" in meta: payload["train_window_cols"] = list(meta["train_window_cols"])
            if "neg_thr" in meta: payload["neg_thr"] = float(meta["neg_thr"])
            if "pos_thr" in meta: payload["pos_thr"] = float(meta["pos_thr"])
        except Exception as e:
            logger.warning(f"Could not read meta json {meta_path}: {e}")

    # positive_class: discover from estimator.classes_ if available; default 1
    pos_class = 1
    pipe = payload["pipeline"]
    if hasattr(pipe, "classes_"):
        # classes_ present at top-level (e.g., LogisticRegression)
        if 1 in list(pipe.classes_): pos_class = 1
        else: pos_class = list(pipe.classes_)[-1]  # fallback to last
    payload["positive_class"] = pos_class

    return payload

def ensure_columns(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols: return X
    X2 = X.copy()
    miss = [c for c in cols if c not in X2.columns]
    for c in miss: X2[c] = 0.0
    return X2[cols]

# --------------------------- CSV utils ---------------------------
def read_csv_mql4(path: str) -> pd.DataFrame:
    """Expect columns: time, open, high, low, close, volume/tick_volume."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # normalize
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    time_col = pick("time","datetime","date")
    if not time_col: raise ValueError(f"{path}: cannot find time column")
    df = df.rename(columns={
        time_col: "time",
        pick("open"): "open",
        pick("high"): "high",
        pick("low"):  "low",
        pick("close"): "close",
        pick("volume","tick_volume"): "volume",
    })
    df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    df = df.dropna(subset=["time"])
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time")
    df = df.set_index("time")
    return df

# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--meta-path", default=None)
    ap.add_argument("--answer-path", default="answer.txt")
    ap.add_argument("--cache-dir", default="./cache_parity")
    ap.add_argument("--poll-sec", type=float, default=1.0)
    ap.add_argument("--main-tf", default="30T")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--min-context-bars", type=int, default=3)
    ap.add_argument("--thr-low", type=float, default=None)
    ap.add_argument("--thr-high", type=float, default=None)
    ap.add_argument("--positive-class", type=int, default=None)
    ap.add_argument("--log-path", default=os.path.expanduser("~/prediction_in_production_parity.log"))
    args = ap.parse_args()

    log = setup_logger(args.log_path)
    log.info("=== %s started ===", APP)

    # 1) Load model payload (+meta)
    payload = load_payload(args.model_path, args.meta_path, log)
    model   = payload["pipeline"]
    window  = int(payload["window_size"])
    feats   = payload.get("train_window_cols") or []
    neg_thr = float(args.thr_low if args.thr_low is not None else payload["neg_thr"])
    pos_thr = float(args.thr_high if args.thr_high is not None else payload["pos_thr"])
    pos_class = int(args.positive_class if args.positive_class is not None else payload["positive_class"])

    # positive class index for predict_proba
    pos_idx = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if pos_class in classes:
            pos_idx = classes.index(pos_class)
    # If model is a Pipeline, try to find final estimator's classes_
    if pos_idx is None and hasattr(model, "steps"):
        try:
            final_est = model.steps[-1][1]
            if hasattr(final_est, "classes_"):
                classes = list(final_est.classes_)
                if pos_class in classes:
                    pos_idx = classes.index(pos_class)
        except Exception:
            pass
    if pos_idx is None:
        pos_idx = 1  # a safe default

    log.info("Model loaded | window=%d | feats=%d | thr=(%.3f, %.3f) | task=classifier", window, len(feats), neg_thr, pos_thr)
    # MD5 برای کنترل پاریتی
    try:
        md5_cols = hashlib.md5((",".join(feats)).encode("utf-8")).hexdigest() if feats else "-"
        log.info("Train columns md5=%s", md5_cols)
    except Exception:
        md5_cols = "-"

    # 2) Resolve raw CSVs & build/load cache with PREPARE_DATA_FOR_TRAIN
    filepaths = resolve_timeframe_paths(args.raw_dir, args.symbol)
    for tf in ("15T","1H","30T","5T"):
        if tf in filepaths:
            log.info("[raw] %s -> %s", tf, filepaths[tf])

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, "merged_full.parquet")

    # Build cache if missing
    if not os.path.exists(cache_path):
        log.info("[cache] building merged_full at first run ...")
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=args.main_tf,
                                      verbose=True, fast_mode=False, strict_disk_feed=False)
        merged = prep.load_data()
        tcol = f"{prep.main_timeframe}_time"
        if tcol not in merged.columns:
            raise KeyError(f"Time column '{tcol}' not found in merged dataframe from PREPARE_DATA_FOR_TRAIN.")
        merged[tcol] = pd.to_datetime(merged[tcol], utc=False)
        merged.sort_values(tcol, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        merged.to_parquet(cache_path, index=False)
        log.info("[cache] merged_full saved to %s | shape=%s | first=%s | last=%s",
                 cache_path, merged.shape, merged[tcol].min(), merged[tcol].max())
    else:
        log.info("[cache] loading merged_full from %s", cache_path)
        merged = pd.read_parquet(cache_path)

    # cached prep used for ready(...)
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=args.main_tf,
                                  verbose=False, fast_mode=True, strict_disk_feed=False)
    tcol = f"{prep.main_timeframe}_time"
    if tcol not in merged.columns:
        raise KeyError(f"'{tcol}' is missing in cached merged_full. Delete cache and rerun to rebuild.")

    # 3) Watch loop (handshake with generator/MQL4)
    def read_ts_from_live(m30_live: str) -> Optional[pd.Timestamp]:
        df = read_csv_mql4(m30_live)
        if df.empty: return None
        return df.index.max()

    m5_live  = os.path.join(args.watch_dir, "XAUUSD_M5_live.csv")
    m15_live = os.path.join(args.watch_dir, "XAUUSD_M15_live.csv")
    m30_live = os.path.join(args.watch_dir, "XAUUSD_M30_live.csv")
    h1_live  = os.path.join(args.watch_dir, "XAUUSD_H1_live.csv")
    answer_fp = os.path.join(args.watch_dir, args.answer_path)

    last_ts_done: Optional[pd.Timestamp] = None

    def predict_one(ts_now: pd.Timestamp) -> Tuple[str,float]:
        # اگر ts_now فراتر از کش است، کش را از نو بساز (سناریوی لایو واقعی)
        nonlocal merged
        if pd.to_datetime(ts_now) > pd.to_datetime(merged[tcol].max()):
            log.info("[cache] ts_now beyond cache; rebuilding merged_full from raw ...")
            merged = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=args.main_tf,
                                            verbose=False, fast_mode=True, strict_disk_feed=False).load_data()
            merged[tcol] = pd.to_datetime(merged[tcol], utc=False)
            merged.sort_values(tcol, inplace=True)
            merged.reset_index(drop=True, inplace=True)
            merged.to_parquet(cache_path, index=False)
            log.info("[cache] rebuilt | new last=%s", merged[tcol].max())

        # برش تا لحظه ts_now
        cut = merged[merged[tcol] <= pd.to_datetime(ts_now)].copy()
        if len(cut) < max(args.min_context_bars, window+1):
            raise RuntimeError(f"Not enough context rows before {ts_now} (have {len(cut)})")

        # ساخت X,y با همان مسیر آموزش
        X, y, _, _ = prep.ready(
            cut,
            window=window,
            selected_features=feats,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        if X is None or len(X)==0:
            raise RuntimeError(f"Empty feature frame at ts={ts_now}")
        # انتخاب آخرین ردیف
        X_last = X.iloc[[-1]].reset_index(drop=True)
        if feats: X_last = ensure_columns(X_last, feats)

        # proba/decision → score
        score = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_last)
            if proba.ndim==2 and proba.shape[1]>=2:
                score = float(proba[0, pos_idx])
        if score is None and hasattr(model, "decision_function"):
            d = float(np.ravel(model.decision_function(X_last))[-1])
            score = 1.0/(1.0+np.exp(-d))
        if score is None and hasattr(model, "predict"):
            yhat = int(np.ravel(model.predict(X_last))[-1])
            score = 1.0 if yhat==pos_class else 0.0

        # آستانه‌ها → برچسب
        if score <= neg_thr:
            label = "SELL"
        elif score >= pos_thr:
            label = "BUY"
        else:
            label = "NONE"

        log.info("[Predict] ts=%s | score=%.6f | thr=(%.3f,%.3f) → %s | X_cols=%d md5=%s | wrote=%s",
                 ts_now, score, neg_thr, pos_thr, label, X_last.shape[1], md5_cols, answer_fp)
        return label, float(score)

    while True:
        try:
            # شرط‌های شروع پیش‌بینی
            if os.path.exists(answer_fp):
                time.sleep(args.poll_sec); continue

            live_paths = [m5_live, m15_live, m30_live, h1_live]
            if not all(os.path.exists(p) for p in live_paths):
                time.sleep(args.poll_sec); continue
            if not all(stable_file(p, 150) for p in live_paths):
                time.sleep(args.poll_sec); continue

            # ts_now از M30_live
            ts_now = read_ts_from_live(m30_live)
            if ts_now is None:
                time.sleep(args.poll_sec); continue
            if last_ts_done is not None and ts_now <= last_ts_done:
                time.sleep(args.poll_sec); continue

            label, _score = predict_one(ts_now)
            atomic_write_text(answer_fp, label)

            # طبق سناریوی شما: بعد از ساخت answer، چهار فایل live حذف شوند
            for p in live_paths:
                try: os.remove(p)
                except Exception: pass

            last_ts_done = ts_now
            gc.collect()
        except Exception as e:
            log.error("Unhandled error in loop: %s", e, exc_info=True)
            time.sleep(max(1.0, args.poll_sec))

if __name__ == "__main__":
    main()
