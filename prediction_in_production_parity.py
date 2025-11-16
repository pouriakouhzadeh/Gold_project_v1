# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prediction_in_production_parity.py

این اسکریپت مدل آموزش‌دیده را بارگذاری می‌کند، داده‌های خام را شبیه‌سازی کرده
و در هر بار آپدیتِ فایل زنده، پیش‌بینی جدیدی می‌دهد. برای حفظ پاریتی با
شبیه‌ساز، حالت `train` استفاده می‌شود تا آخرین سطر بدون برچسب حذف گردد و
ویژگی‌ها دقیقاً با زمان برچسب‌گذاری شده مطابقت کنند.
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

# --- logger ---
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

# --- helper to load model & meta ---
def load_model_and_meta(model_path: str="best_model.pkl",
                        meta_path: str="best_model.meta.json") -> dict:
    """Loads model and meta; applies threshold overrides if train_distribution.json exists."""
    model = None
    if joblib is not None:
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None
    if model is None:
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Cannot load model: {e}")
    payload = {}
    if isinstance(model, dict):
        mdl = model.get("pipeline") or model.get("estimator") or model.get("clf") or model.get("best_estimator")
        if mdl is None:
            for v in model.values():
                if hasattr(v, "predict_proba"): mdl = v; break
        payload["pipeline"] = mdl
        cols = model.get("train_window_cols") or model.get("feats") or []
        payload["train_window_cols"] = list(cols)
        payload["window_size"] = int(model.get("window_size", 1))
        payload["neg_thr"] = float(model.get("neg_thr", 0.005))
        payload["pos_thr"] = float(model.get("pos_thr", 0.995))
    else:
        payload = {"pipeline": model, "train_window_cols": [],
                   "window_size":1, "neg_thr":0.005, "pos_thr":0.995}

    # meta file for column list & thresholds
    if os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            payload["train_window_cols"] = meta.get("train_window_cols") or meta.get("feats") or payload["train_window_cols"]
            payload["window_size"] = int(meta.get("window_size", payload["window_size"]))
            payload["neg_thr"] = float(meta.get("neg_thr", payload["neg_thr"]))
            payload["pos_thr"] = float(meta.get("pos_thr", payload["pos_thr"]))
        except Exception:
            pass

    # optional override from train_distribution.json
    if os.path.exists("train_distribution.json"):
        try:
            td = json.loads(Path("train_distribution.json").read_text())
            if "neg_thr" in td: payload["neg_thr"] = float(td["neg_thr"])
            if "pos_thr" in td: payload["pos_thr"] = float(td["pos_thr"])
            logging.info("Train distribution loaded: train_distribution.json")
        except Exception:
            pass
    return payload

# --- ensure column order identical to training ---
def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure df has exactly the columns in cols, in order; fill missing with 0.0"""
    df = df.copy()
    # cast numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[cols]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.fillna(0.0)

# --- resolve raw paths ---
def resolve_timeframe_paths(base: Path, symbol: str) -> dict[str,str]:
    cands = {
        "30T":[f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T":[f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" :[f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" :[f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    out = {}
    for tf,names in cands.items():
        found = None
        for nm in names:
            p = base / nm
            if p.exists(): found = str(p); break
        if found is None:
            # fallback case insensitive
            for ch in base.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            logging.info("[raw] %s -> %s", tf, found)
            out[tf] = found
    if "30T" not in out:
        raise FileNotFoundError("Main timeframe '30T' not found in directory.")
    return out

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

# --- main loop ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--poll-sec", type=float, default=0.2)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production_parity started ===")

    payload = load_model_and_meta("best_model.pkl", "best_model.meta.json")
    model = payload["pipeline"]
    cols = payload.get("train_window_cols") or []
    window = int(payload.get("window_size", 1))
    neg_thr = float(payload.get("neg_thr", 0.005))
    pos_thr = float(payload.get("pos_thr", 0.995))
    logging.info("Model loaded | window=%d | feats=%d | thr=(%.3f,%.3f) | cols_md5=%s",
                 window, len(cols), neg_thr, pos_thr, hashlib.md5("".join(cols).encode()).hexdigest())

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba()")

    base_dir = Path(args.base_dir).resolve()
    raw_paths = resolve_timeframe_paths(base_dir, args.symbol)
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=raw_paths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    # load & merge full data once
    merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged[tcol] = pd.to_datetime(merged[tcol], errors="coerce")
    merged = merged.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    logging.info("[init] merged shape=%s | first=%s | last=%s",
                 merged.shape,
                 str(merged[tcol].iloc[0]) if len(merged) else None,
                 str(merged[tcol].iloc[-1]) if len(merged) else None)

    # feed log
    feed_log = Path("deploy_X_feed_log.csv")
    tail_log = Path("deploy_X_feed_tail200.csv")
    wrote_header = feed_log.exists()
    ans_path = Path("answer.txt")

    # file produced by MT4/Generator for 30T
    m30_live = Path(live_name(raw_paths["30T"]))

    last_ts_seen: pd.Timestamp|None = None
    while True:
        # avoid overwriting answer.txt if generator hasn't consumed
        if ans_path.exists():
            time.sleep(args.poll_sec)
            continue
        if not m30_live.exists():
            time.sleep(args.poll_sec)
            continue
        try:
            df_live = pd.read_csv(m30_live)
            if "time" not in df_live.columns or len(df_live)==0:
                time.sleep(args.poll_sec); continue
            df_live["time"] = pd.to_datetime(df_live["time"], errors="coerce")
            df_live = df_live.dropna(subset=["time"]).sort_values("time")
            ts_now = pd.Timestamp(df_live["time"].iloc[-1])
        except Exception:
            time.sleep(args.poll_sec)
            continue
        if last_ts_seen is not None and ts_now <= last_ts_seen:
            time.sleep(args.poll_sec)
            continue

        # cut merged up to ts_now (drop future)
        sub = merged[merged[tcol] <= ts_now].copy()
        if sub.empty:
            time.sleep(args.poll_sec)
            continue

        # build X_train & y_train with times; mode='train' removes last row (no y)
        X_train, y_train, feats, price_raw, t_idx = prep.ready(
            sub,
            window=window,
            selected_features=cols,
            mode="train",
            with_times=True,
            predict_drop_last=False,
            train_drop_last=False
        )
        if X_train.empty:
            time.sleep(args.poll_sec)
            continue

        # align columns
        X_train = ensure_columns(X_train, cols)
        # last row of X_train corresponds to last available label
        X_last = X_train.tail(1).reset_index(drop=True)
        ts_feat = pd.to_datetime(t_idx).iloc[-1]

        p_last = float(model.predict_proba(X_last)[:,1][0])
        if p_last <= neg_thr:
            decision = "SELL"
        elif p_last >= pos_thr:
            decision = "BUY"
        else:
            decision = "NONE"

        # write answer
        ans_path.write_text(decision, encoding="utf-8")

        # log feed row
        row = {"timestamp": ts_feat, "score": p_last, "decision": decision}
        feed_row = pd.concat([pd.DataFrame([row]), pd.DataFrame(X_last)], axis=1)
        with open(feed_log, "a", encoding="utf-8") as f:
            feed_row.to_csv(f, header=not wrote_header, index=False)
            wrote_header = True
        try:
            df_log = pd.read_csv(feed_log, parse_dates=["timestamp"])
            df_log.sort_values("timestamp", inplace=True)
            df_log.tail(200).to_csv(tail_log, index=False)
        except Exception:
            pass
        logging.info("[Predict] ts_train=%s | p=%.6f → %s | wrote=%s",
                     str(ts_feat), p_last, decision, str(ans_path.resolve()))
        last_ts_seen = ts_now
        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
