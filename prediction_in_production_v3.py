# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prediction_in_production_v5.py
- All files live in the SAME folder as this script.
- Waits for XAUUSD_M30_live.csv (main), tolerates late arrivals (no crash).
- Cut BEFORE resample using PREPARE_DATA_FOR_TRAIN_PRODUCTION, same pipeline.
- Robust model loader (pickle/joblib/zip/gzip/bz2/lzma/json).
- Align columns; write answer.txt atomically.
"""

from __future__ import annotations
import os, sys, time, json, pickle, gzip, bz2, lzma, zipfile, logging, argparse
from pathlib import Path
import numpy as np
import pandas as pd

from prepare_data_for_train_production import PREPARE_DATA_FOR_TRAIN_PRODUCTION

LOG_FILE = "prediction_in_production_v5.log"

# ---------------- Logging ----------------
def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

# ---------------- Robust model loader ----------------
def _try_pickle(fp: str):
    with open(fp, "rb") as f:
        return pickle.load(f)

def _try_joblib(fp: str):
    import joblib
    return joblib.load(fp)

def _try_gzip(fp: str):
    with gzip.open(fp, "rb") as f:
        return pickle.load(f)

def _try_bz2(fp: str):
    with bz2.open(fp, "rb") as f:
        return pickle.load(f)

def _try_lzma(fp: str):
    with lzma.open(fp, "rb") as f:
        return pickle.load(f)

def _try_zip(fp: str):
    with zipfile.ZipFile(fp, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            with zf.open(name, "r") as f:
                data = f.read()
                try:
                    return pickle.loads(data)
                except Exception:
                    try:
                        return json.loads(data.decode("utf-8"))
                    except Exception:
                        pass
    raise ValueError("No loadable member found in ZIP.")

def _raw_head(fp: str, n=8) -> bytes:
    with open(fp, "rb") as f:
        return f.read(n)

def load_payload_best_model(pkl_path: str="best_model.pkl") -> dict:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"'best_model.pkl' not found at {p.resolve()}")

    head = _raw_head(pkl_path, 8)
    loaders = []
    if head.startswith(b"\x80"):
        loaders = [_try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"\x1f\x8b"):
        loaders = [_try_gzip, _try_pickle, _try_joblib, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"BZh"):
        loaders = [_try_bz2, _try_pickle, _try_joblib, _try_gzip, _try_lzma, _try_zip]
    elif head.startswith(b"\xfd7zXZ\x00"):
        loaders = [_try_lzma, _try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_zip]
    elif head.startswith(b"PK\x03\x04"):
        loaders = [_try_zip, _try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_lzma]
    else:
        loaders = [_try_joblib, _try_pickle, _try_gzip, _try_bz2, _try_lzma, _try_zip]

    last_err = None
    raw = None
    for loader in loaders:
        try:
            raw = loader(pkl_path)
            break
        except Exception as e:
            last_err = e

    if raw is None:
        try:
            with open(pkl_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            pass

    if raw is None:
        raise RuntimeError(
            f"Could not load best_model.pkl. First bytes: {head!r}. Last error: {repr(last_err)}"
        )

    payload = {}
    if isinstance(raw, dict):
        model = raw.get("pipeline") or raw.get("model") or raw.get("estimator") \
                or raw.get("clf") or raw.get("best_estimator")
        if model is None:
            for k, v in raw.items():
                if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                    model = v; break
        if model is None:
            raise ValueError("Loaded dict but no estimator found.")
        payload["pipeline"] = model
        payload["window_size"] = int(raw.get("window_size", 1))
        feats = raw.get("train_window_cols") or raw.get("feats") or []
        payload["train_window_cols"] = list(feats) if isinstance(feats,(list,tuple)) else []
        payload["neg_thr"] = float(raw.get("neg_thr", 0.005))
        payload["pos_thr"] = float(raw.get("pos_thr", 0.995))
        if "scaler" in raw: payload["scaler"] = raw["scaler"]
    else:
        payload["pipeline"] = raw
        payload["window_size"] = 1
        payload["train_window_cols"] = []
        payload["neg_thr"] = 0.005
        payload["pos_thr"] = 0.995

    if os.path.exists("train_distribution.json"):
        try:
            with open("train_distribution.json", "r", encoding="utf-8") as jf:
                td = json.load(jf)
            if "neg_thr" in td: payload["neg_thr"] = float(td["neg_thr"])
            if "pos_thr" in td: payload["pos_thr"] = float(td["pos_thr"])
            logging.info("Train distribution loaded from train_distribution.json")
        except Exception:
            pass

    return payload

# ---------------- helpers ----------------
def ensure_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return X
    X2 = X.copy()
    missing = [c for c in cols if c not in X2.columns]
    for c in missing:
        X2[c] = 0.0
    return X2[cols]

def find_live_paths_here(symbol: str) -> dict[str,str]:
    """
    Looks for *_live.csv only in current folder.
    """
    here = Path(".").resolve()
    mapping = {
        "30T": f"{symbol}_M30_live.csv",
        "15T": f"{symbol}_M15_live.csv",
        "5T" : f"{symbol}_M5_live.csv",
        "1H" : f"{symbol}_H1_live.csv",
    }
    res = {}
    for tf, nm in mapping.items():
        p = here / nm
        if p.exists():
            res[tf] = str(p)
    return res

def find_raw_paths_here(symbol: str) -> dict[str,str]:
    here = Path(".").resolve()
    candidates = {
        "30T": [f"{symbol}_M30.csv", f"{symbol}_30T.csv"],
        "15T": [f"{symbol}_M15.csv", f"{symbol}_15T.csv"],
        "5T" : [f"{symbol}_M5.csv",  f"{symbol}_5T.csv" ],
        "1H" : [f"{symbol}_H1.csv",  f"{symbol}_1H.csv" ],
    }
    out = {}
    for tf, names in candidates.items():
        for nm in names:
            p = here / nm
            if p.exists():
                out[tf] = str(p); break
    return out

def atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--best-model", type=str, default="best_model.pkl")
    ap.add_argument("--poll-sec", type=float, default=2.0)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production_v5 started ===")

    # 1) model
    payload = load_payload_best_model(args.best_model)
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or []
    window  = int(payload.get("window_size", 1))
    neg_thr = float(payload.get("neg_thr", 0.005))
    pos_thr = float(payload.get("pos_thr", 0.995))
    logging.info("Model loaded | feats=%d | neg_thr=%.3f | pos_thr=%.3f", len(cols), neg_thr, pos_thr)

    # 2) wait for LIVE 30T in current folder
    last_seen_ts = None
    backoff = args.poll_sec

    while True:
        live_paths = find_live_paths_here(args.symbol)
        if "30T" not in live_paths:
            logging.warning("[wait] 30T live CSV not found in current folder. Waiting...")
            time.sleep(backoff)
            backoff = min(backoff * 1.3, 8.0)
            continue

        # combine with any available others
        filepaths = {**live_paths}
        # (اختیاری) اگر M15/M5/H1 لایو نبودند، وجود خام را هم در همین پوشه چک کن (برای تکمیل فیچرها)
        raw_paths = find_raw_paths_here(args.symbol)
        for tf in ("15T","5T","1H"):
            if tf not in filepaths and tf in raw_paths:
                filepaths[tf] = raw_paths[tf]

        # 3) prep
        prep = PREPARE_DATA_FOR_TRAIN_PRODUCTION(filepaths=filepaths, main_timeframe="30T", verbose=False)
        # لود دیتای تا انتهای فایل لایو اصلی:
        try:
            m30 = pd.read_csv(filepaths["30T"])
            if "time" not in m30.columns: raise ValueError("M30_live has no 'time' column")
            m30["time"] = pd.to_datetime(m30["time"], errors="coerce")
            m30 = m30.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            if m30.empty:
                time.sleep(backoff); continue
            ts_now = m30["time"].iloc[-1]
        except Exception as e:
            logging.error("Failed reading main live CSV: %s", e)
            time.sleep(backoff); continue

        if last_seen_ts is not None and pd.Timestamp(ts_now) <= pd.Timestamp(last_seen_ts):
            # nothing new
            time.sleep(args.poll-sec if hasattr(args, 'poll-sec') else backoff)
            continue

        # 4) Build merged up to ts_now
        try:
            merged = prep.load_data_up_to(ts_now)
            tcol = "30T_time"
            merged[tcol] = pd.to_datetime(merged[tcol], errors="coerce")
            merged = merged.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
            if merged.empty:
                logging.warning("Merged is empty up to %s; retrying...", ts_now)
                time.sleep(backoff); continue
        except Exception as e:
            logging.error("prep.load_data_up_to error: %s", e)
            time.sleep(backoff); continue

        # 5) Ready() to produce X/y (train-mode to keep labels rule)
        try:
            X, y, feats = prep.ready(merged, window=window, selected_features=cols, mode="train", with_times=False)
            if X.empty:
                logging.warning("X is empty at %s; retrying...", ts_now)
                time.sleep(backoff); continue
        except Exception as e:
            logging.error("prep.ready error: %s", e)
            time.sleep(backoff); continue

        X_last = ensure_columns(X.tail(1).reset_index(drop=True), cols)

        # 6) Predict → BUY/SELL/NONE
        try:
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(X_last)[:,1][0])
                if p <= neg_thr:
                    yhat = "SELL"
                elif p >= pos_thr:
                    yhat = "BUY"
                else:
                    yhat = "NONE"
            else:
                pred = int(model.predict(X_last)[0])
                yhat = "BUY" if pred==1 else "SELL"
                p = np.nan
        except Exception as e:
            logging.error("Model inference error: %s", e)
            time.sleep(backoff); continue

        # 7) Write answer.txt atomically
        try:
            out = Path("answer.txt")
            atomic_write(out, yhat)
            logging.info("answer=%s (p=%.6f) @ %s", yhat, p if not np.isnan(p) else -1, ts_now)
        except Exception as e:
            logging.error("answer.txt write error: %s", e)

        last_seen_ts = ts_now
        backoff = args.poll_sec
        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
