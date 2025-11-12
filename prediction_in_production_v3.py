# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
pridiction_in_production.py
---------------------------
اسکریپت پیش‌بینی لایو: فایل‌های *_live.csv را می‌خواند، با کلاس Production پردازش می‌کند
و نتیجه را به صورت BUY/SELL/NONE در answer.txt می‌نویسد.
"""

from __future__ import annotations
import os, sys, time, json, gzip, bz2, lzma, zipfile, pickle, logging, argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from prepare_data_for_train_production import PREPARE_DATA_FOR_TRAIN_PRODUCTION

LOG_FILE = "pridiction_in_production.log"

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
            raw = loader(pkl_path); break
        except Exception as e:
            last_err = e

    if raw is None:
        try:
            with open(pkl_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            pass

    if raw is None:
        raise RuntimeError(f"Could not load best_model.pkl. First bytes: {head!r}. Last error: {repr(last_err)}")

    payload = {}
    if isinstance(raw, dict):
        model = raw.get("pipeline") or raw.get("model") or raw.get("estimator") or raw.get("clf") or raw.get("best_estimator")
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
        payload["neg_thr"] = float(raw.get("neg_thr", 0.495))
        payload["pos_thr"] = float(raw.get("pos_thr", 0.505))
        if "scaler" in raw:
            payload["scaler"] = raw["scaler"]
    else:
        payload["pipeline"] = raw
        payload["window_size"] = 1
        payload["train_window_cols"] = []
        payload["neg_thr"] = 0.495
        payload["pos_thr"] = 0.505

    # override اختیاری
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

# ---------------- سایر کمک‌متدها ----------------
def resolve_live_paths(base_dir: Path, symbol: str) -> Dict[str, str]:
    candidates = {
        "30T": [f"{symbol}_30T_live.csv", f"{symbol}_M30_live.csv"],
        "15T": [f"{symbol}_15T_live.csv", f"{symbol}_M15_live.csv"],
        "5T" : [f"{symbol}_5T_live.csv",  f"{symbol}_M5_live.csv" ],
        "1H" : [f"{symbol}_1H_live.csv",  f"{symbol}_H1_live.csv" ],
    }
    resolved = {}
    for tf, names in candidates.items():
        for nm in names:
            p = base_dir / nm
            if p.exists():
                resolved[tf] = str(p); break
    if "30T" not in resolved:
        raise FileNotFoundError(f"Main timeframe '30T' live csv not found under {base_dir}")
    return resolved

def ensure_columns(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return X
    X2 = X.copy()
    missing = [c for c in cols if c not in X2.columns]
    for c in missing:
        X2[c] = 0.0
    # ترتیب دقیق
    return X2[cols]

def prob_to_signal(p: float, lo: float, hi: float) -> str:
    if np.isnan(p):
        return "NONE"
    if p <= lo:
        return "SELL"
    if p >= hi:
        return "BUY"
    return "NONE"

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--poll", type=float, default=0.15, help="seconds between polls")
    ap.add_argument("--best-model", type=str, default="best_model.pkl")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== pridiction_in_production started ===")

    payload = load_payload_best_model(args.best_model)
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or []
    lo_thr  = float(payload["neg_thr"])
    hi_thr  = float(payload["pos_thr"])
    logging.info(f"Model loaded | feats={len(cols)} | neg_thr={lo_thr:.3f} | pos_thr={hi_thr:.3f}")

    base_dir = Path(args.base_dir).resolve()
    ans_path = base_dir / "answer.txt"
    last_mtime: float = 0.0

    while True:
        try:
            filepaths = resolve_live_paths(base_dir, args.symbol)
            # تشخیص آپدیت بر اساس mtime فایل 30T
            main_live = None
            for nm in (f"{args.symbol}_30T_live.csv", f"{args.symbol}_M30_live.csv"):
                p = base_dir / nm
                if p.exists():
                    main_live = p; break
            if main_live is None:
                time.sleep(args.poll); continue
            mt = main_live.stat().st_mtime
            if mt <= last_mtime:
                time.sleep(args.poll); continue
            last_mtime = mt

            # پردازش با کلاس Production (Cut-Before-Resample با خود فایل‌های live)
            prep = PREPARE_DATA_FOR_TRAIN_PRODUCTION(
                filepaths=filepaths, main_timeframe="30T",
                verbose=False, fast_mode=True, strict_disk_feed=True, disable_drift=True
            )
            merged = prep.load_data_up_to(pd.Timestamp.max)
            if merged.empty:
                time.sleep(args.poll); continue

            # فقط فیچرهایی که مدل انتظار دارد (Feature-Freeze)
            X_all, _, _ = prep.ready(merged, selected_features=cols, make_labels=False)
            if X_all.empty:
                time.sleep(args.poll); continue

            X_last = ensure_columns(X_all.tail(1).reset_index(drop=True), cols)
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(X_last)[:, 1][0])
                sig = prob_to_signal(p, lo_thr, hi_thr)
            else:
                yhat = int(model.predict(X_last)[0])
                p = np.nan
                sig = "BUY" if yhat == 1 else ("SELL" if yhat == 0 else "NONE")

            ans_path.write_text(sig, encoding="utf-8")
            logging.info(f"answer.txt -> {sig} (p={p:.4f})")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception(e)
            time.sleep(max(0.25, args.poll))

if __name__ == "__main__":
    main()
