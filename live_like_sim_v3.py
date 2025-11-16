# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, time, pickle, logging, argparse, gzip, bz2, lzma, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

try:
    import joblib
except Exception:
    joblib = None

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "live_like_sim_v3.log"

# ---------------- Logging ----------------
def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

# ---------------- Model loader (robust) ----------------
def _try_pickle(fp: str):
    with open(fp, "rb") as f: return pickle.load(f)
def _try_joblib(fp: str):
    if joblib is None: raise RuntimeError("joblib not available")
    return joblib.load(fp)
def _try_gzip(fp: str):
    with gzip.open(fp, "rb") as f: return pickle.load(f)
def _try_bz2(fp: str):
    with bz2.open(fp, "rb") as f: return pickle.load(f)
def _try_lzma(fp: str):
    with lzma.open(fp, "rb") as f: return pickle.load(f)
def _try_zip(fp: str):
    with zipfile.ZipFile(fp, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"): continue
            with zf.open(name, "r") as f:
                data = f.read()
                try: return pickle.loads(data)
                except Exception:
                    try: return json.loads(data.decode("utf-8"))
                    except Exception: pass
    raise ValueError("No loadable member found in ZIP.")

def _raw_head(fp: str, n=8) -> bytes:
    with open(fp, "rb") as f: return f.read(n)

def load_payload_best_model(pkl_path: str="best_model.pkl") -> dict:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"'best_model.pkl' not found at {p.resolve()}")

    head = _raw_head(pkl_path, 8)
    if   head.startswith(b"\x80"): loaders = [_try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"\x1f\x8b"): loaders = [_try_gzip, _try_pickle, _try_joblib, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"BZh"): loaders = [_try_bz2, _try_pickle, _try_joblib, _try_gzip, _try_lzma, _try_zip]
    elif head.startswith(b"\xfd7zXZ\x00"): loaders = [_try_lzma, _try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_zip]
    elif head.startswith(b"PK\x03\x04"): loaders = [_try_zip, _try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_lzma]
    else: loaders = [_try_joblib, _try_pickle, _try_gzip, _try_bz2, _try_lzma, _try_zip]

    raw = None; last_err = None
    for L in loaders:
        try:
            raw = L(pkl_path); break
        except Exception as e:
            last_err = e
    if raw is None:
        try:
            with open(pkl_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            pass
    if raw is None:
        raise RuntimeError(f"Could not load best_model.pkl; first bytes: {head!r}; last err: {repr(last_err)}")

    payload: dict = {}
    if isinstance(raw, dict):
        model = raw.get("pipeline") or raw.get("model") or raw.get("estimator") \
                or raw.get("clf") or raw.get("best_estimator")
        if model is None:
            for k,v in raw.items():
                if hasattr(v,"predict") or hasattr(v,"predict_proba"):
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
        payload = {"pipeline": raw, "window_size": 1, "train_window_cols": [], "neg_thr":0.005, "pos_thr":0.995}

    # thresholds override (optional)
    td_path = "train_distribution.json"
    if os.path.exists(td_path):
        try:
            with open(td_path, "r", encoding="utf-8") as f:
                td = json.load(f)
            if "neg_thr" in td: payload["neg_thr"] = float(td["neg_thr"])
            if "pos_thr" in td: payload["pos_thr"] = float(td["pos_thr"])
            logging.info("Train distribution loaded: train_distribution.json")
        except Exception:
            pass
    return payload

# ---------------- path resolver ----------------
def resolve_timeframe_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    candidates = {
        "30T":[f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T":[f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" :[f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" :[f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    resolved = {}
    for tf,names in candidates.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found = str(p); break
        if found is None:
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            logging.info(f"[paths] {tf} -> {found}")
            resolved[tf] = found
    if "30T" not in resolved:
        raise FileNotFoundError(f"Main timeframe '30T' not found under {base_dir}.")
    return resolved

# --------------- column guard ---------------
def ensure_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols: return X
    X2 = X.copy()
    for c in cols:
        if c not in X2.columns: X2[c] = 0.0
    return X2[cols].astype(float)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=200)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== live_like_sim_v3 starting ===")

    payload = load_payload_best_model("best_model.pkl")
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or []
    window  = int(payload["window_size"])
    neg_thr, pos_thr = float(payload["neg_thr"]), float(payload["pos_thr"])
    logging.info(f"Model loaded | window={window} | feats={len(cols)} | thr=({neg_thr:.3f},{pos_thr:.3f})")

    base_dir = Path(args.base_dir).resolve()
    filepaths = resolve_timeframe_paths(base_dir, args.symbol)

    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=True, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged[tcol] = pd.to_datetime(merged[tcol])
    merged.sort_values(tcol, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    logging.info(f"[load_data] merged shape={merged.shape}")

    # Build X,y with exact train semantics (to have ground-truth y) and with_times for timestamps
    X_all, y_all, feats, _, t_idx = prep.ready(
        merged,
        window=window,
        selected_features=cols,
        mode="train",
        with_times=True,
        predict_drop_last=False,
        train_drop_last=False
    )
    if X_all.empty:
        logging.error("Empty features after ready(); abort.")
        return

    # Align to training columns
    X_all = ensure_columns(X_all, cols)
    y_all = pd.Series(y_all).astype(int).reset_index(drop=True)
    t_idx = pd.to_datetime(t_idx).reset_index(drop=True)

    # last-N slice
    L = len(X_all)
    n  = min(args.last_n, L)
    X_tail = X_all.iloc[-n:].reset_index(drop=True)
    y_tail = y_all.iloc[-n:].reset_index(drop=True)
    t_tail = t_idx.iloc[-n:].reset_index(drop=True)

    # predict with thresholds
    prob = model.predict_proba(X_tail)[:, 1]
    y_pred = np.full(n, -1, dtype=int)
    y_pred[prob <= neg_thr] = 0
    y_pred[prob >= pos_thr] = 1

    mask = (y_pred != -1)
    acc = accuracy_score(y_tail[mask], y_pred[mask]) if mask.any() else np.nan
    bacc = balanced_accuracy_score(y_tail[mask], y_pred[mask]) if mask.any() else np.nan
    cover = float(mask.mean()) if n else 0.0
    logging.info(f"[Final] acc={acc:.3f} bAcc={bacc:.3f} cover={cover:.3f} "
                 f"Correct={int((y_pred[mask]==y_tail[mask]).sum())} "
                 f"Incorrect={int(mask.sum() - (y_pred[mask]==y_tail[mask]).sum())} "
                 f"Unpred={int((~mask).sum())}")

    # save feed snapshot as seen by the model (exact columns+order)
    df_feed = pd.DataFrame(X_tail, columns=cols)
    df_feed.insert(0, "timestamp", t_tail.values)
    df_feed.to_csv("sim_X_feed_tail200.csv", index=False)

    # save predictions compare
    pd.DataFrame({
        "timestamp": t_tail,
        "y_true": y_tail,
        "prob": prob,
        "pred": y_pred
    }).to_csv("predictions_compare.csv", index=False)

    logging.info("=== live_like_sim_v3 completed ===")

if __name__ == "__main__":
    main()
