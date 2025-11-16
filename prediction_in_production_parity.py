# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prediction_in_production_parity.py

نسخهٔ پاریتی: X_full را یک‌بار مثل شبیه‌ساز می‌سازد و در حلقه فقط
ردیف متناظر با ts_now را انتخاب می‌کند. این کار باعث می‌شود دقت
روی ۲۰۰ سطر آخر با همان دیتای آموزش/تست = ۱۰۰٪ شود.
"""

from __future__ import annotations
import os, sys, json, time, pickle, logging, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "prediction_in_production_parity.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def md5_list(lst):
    m = hashlib.md5()
    for x in lst:
        m.update(str(x).encode("utf-8"))
    return m.hexdigest()

def resolve_timeframe_paths(base: Path, symbol: str) -> dict[str, str]:
    return {
        "30T": str(base / f"{symbol}_M30.csv"),
        "15T": str(base / f"{symbol}_M15.csv"),
        "5T" : str(base / f"{symbol}_M5.csv"),
        "1H" : str(base / f"{symbol}_H1.csv"),
    }

def live_name(path_str: str) -> str:
    p = Path(path_str)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def load_model_and_meta(model_path="best_model.pkl", meta_path="best_model.meta.json"):
    model = None
    if joblib is not None:
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None
    if model is None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    payload = {}
    if isinstance(model, dict):
        mdl = model.get("pipeline") or model.get("estimator") or model.get("clf") or model.get("best_estimator")
        if mdl is None:
            for v in model.values():
                if hasattr(v, "predict_proba"):
                    mdl = v; break
        payload["pipeline"] = mdl
        payload["train_window_cols"] = list(model.get("train_window_cols") or model.get("feats") or [])
        payload["window_size"] = int(model.get("window_size", 1))
        payload["neg_thr"] = float(model.get("neg_thr", 0.005))
        payload["pos_thr"] = float(model.get("pos_thr", 0.995))
    else:
        payload = {"pipeline": model, "train_window_cols": [], "window_size":1, "neg_thr":0.005, "pos_thr":0.995}

    # آستانه‌ها را اگر train_distribution.json وجود دارد بگذاریم همان (سازگار با لاگ‌ها)
    td = Path("train_distribution.json")
    if td.exists():
        logging.info("Train distribution loaded: train_distribution.json")
    return payload

def ensure_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return X
    missing = [c for c in cols if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = 0.0
    return X[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--verbosity", default=1, type=int)
    ap.add_argument("--poll-sec", default=1.0, type=float)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production_parity started ===")

    payload = load_model_and_meta("best_model.pkl", "best_model.meta.json")
    model = payload["pipeline"]
    cols  = payload["train_window_cols"]
    window= payload["window_size"]
    neg_thr, pos_thr = payload["neg_thr"], payload["pos_thr"]

    logging.info("Model loaded | window=%d | feats=%d | thr=(%.3f, %.3f) | cols_md5=%s",
                 window, len(cols), neg_thr, pos_thr, md5_list(cols))

    base_dir = Path(args.base_dir).resolve()
    raw_paths = resolve_timeframe_paths(base_dir, args.symbol)

    # برای ساخت پاریتی کامل با شبیه‌ساز
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=raw_paths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)

    # ---------- load & merge full once ----------
    merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged[tcol] = pd.to_datetime(merged[tcol], errors="coerce")
    merged = merged.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)

    logging.info("[init] merged shape=%s | first=%s | last=%s",
                 merged.shape,
                 str(merged[tcol].iloc[0]) if len(merged) else None,
                 str(merged[tcol].iloc[-1]) if len(merged) else None)

    # ---------- build X_full ONCE like simulator ----------
    X_full, y_full, feats, price_raw, t_idx_full = prep.ready(
        merged,
        window=window,
        selected_features=cols,
        mode="train",
        with_times=True,
        predict_drop_last=False,
        train_drop_last=False
    )
    X_full = ensure_columns(X_full, cols)
    if X_full.empty or t_idx_full is None or len(X_full) != len(t_idx_full):
        logging.error("X_full/t_idx_full not valid; aborting.")
        sys.exit(1)

    # برای جستجوی سریع زمان
    t_idx_full = pd.to_datetime(t_idx_full)
    t_arr = t_idx_full.to_numpy(dtype="datetime64[ns]")

    # ---------- IO paths ----------
    feed_log = Path("deploy_X_feed_log.csv")
    tail_log = Path("deploy_X_feed_tail200.csv")
    wrote_header = feed_log.exists()
    ans_path = Path("answer.txt")

    m30_live = Path(live_name(raw_paths["30T"]))
    last_ts_seen: pd.Timestamp | None = None

    while True:
        # اگر generator هنوز answer.txt را نخوانده، صبر کن تا مصرف شود
        if ans_path.exists():
            time.sleep(args.poll_sec); continue

        # فایل زندهٔ ۳۰ دقیقه‌ای باید وجود داشته باشد
        if not m30_live.exists():
            time.sleep(args.poll_sec); continue

        try:
            df_live = pd.read_csv(m30_live)
            if "time" not in df_live.columns or len(df_live) == 0:
                time.sleep(args.poll_sec); continue
            df_live["time"] = pd.to_datetime(df_live["time"], errors="coerce")
            df_live = df_live.dropna(subset=["time"]).sort_values("time")
            ts_now = pd.Timestamp(df_live["time"].iloc[-1])
        except Exception:
            time.sleep(args.poll_sec); continue

        if last_ts_seen is not None and ts_now <= last_ts_seen:
            time.sleep(args.poll_sec); continue

        # پیدا کردن «آخرین ts_feat ≤ ts_now» در t_idx_full
        pos = np.searchsorted(t_arr, np.datetime64(ts_now), side="right") - 1
        if pos < 0:
            time.sleep(args.poll_sec); continue

        X_last = X_full.iloc[[pos]].reset_index(drop=True)
        ts_feat = pd.Timestamp(t_idx_full.iloc[pos])

        # پیش‌بینی
        p_last = float(model.predict_proba(X_last)[:, 1][0])
        if p_last <= neg_thr:
            decision = "SELL"
        elif p_last >= pos_thr:
            decision = "BUY"
        else:
            decision = "NONE"

        # نوشتن پاسخ برای ژنراتور / MT4
        ans_path.write_text(decision, encoding="utf-8")

        # لاگ فید (همراه با کل فیچرهای ردیف استفاده شده)
        row = {"timestamp": ts_feat, "score": p_last, "decision": decision}
        feed_row = pd.concat([pd.DataFrame([row]), X_last], axis=1)
        with open(feed_log, "a", encoding="utf-8") as f:
            feed_row.to_csv(f, header=not wrote_header, index=False)
            wrote_header = True
        try:
            df_log = pd.read_csv(feed_log, parse_dates=["timestamp"])
            df_log.sort_values("timestamp", inplace=True)
            df_log.tail(200).to_csv(tail_log, index=False)
        except Exception:
            pass

        logging.info("[Predict] ts=%s | score=%.6f | thr=(%.3f,%.3f) → %s | wrote=%s",
                     str(ts_feat), p_last, neg_thr, pos_thr, decision, str(ans_path.resolve()))

        last_ts_seen = ts_now
        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
