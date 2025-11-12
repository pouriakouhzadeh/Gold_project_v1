# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prediction_in_production_parity.py
----------------------------------
Production predictor with EXACT parity to live_like_sim_v3:
- Uses only XAUUSD_M30_live.csv as a trigger to get ts_now.
- Builds features from the SAME full raw CSVs used in training:
    PREPARE_DATA_FOR_TRAIN(..., fast_mode=False, strict_disk_feed=False)
- Resample/feature → THEN cut (identical to the simulator).
- Uses saved thresholds from best_model.pkl exactly as-is.
- Logs rich debug info (MD5 of column order, last-row timestamps, etc.).

How to run (two terminals):
  1) Terminal A (generator):
     python generator_parity.py --last-n 200 --symbol XAUUSD

  2) Terminal B (predictor):
     python prediction_in_production_parity.py --base-dir . --symbol XAUUSD --best-model best_model.pkl

Both must be in the directory that contains your original raw CSVs
(e.g. XAUUSD_M30.csv, XAUUSD_M15.csv, XAUUSD_M5.csv, XAUUSD_H1.csv).
"""

from __future__ import annotations
import os, sys, time, json, pickle, gzip, bz2, lzma, zipfile, logging, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "prediction_in_production_parity.log"

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
    if head.startswith(b"\x80"):      # pickle
        loaders = [_try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"\x1f\x8b"): # gzip
        loaders = [_try_gzip, _try_pickle, _try_joblib, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"BZh"):      # bz2
        loaders = [_try_bz2, _try_pickle, _try_joblib, _try_gzip, _try_lzma, _try_zip]
    elif head.startswith(b"\xfd7zXZ\x00"): # xz/lzma
        loaders = [_try_lzma, _try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_zip]
    elif head.startswith(b"PK\x03\x04"):   # zip
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
            f"Could not load best_model.pkl with multiple strategies. "
            f"First bytes: {head!r}. Last error: {repr(last_err)}"
        )

    payload: dict = {}
    if isinstance(raw, dict):
        model = raw.get("pipeline") or raw.get("model") or raw.get("estimator") \
                or raw.get("clf") or raw.get("best_estimator")
        if model is None:
            for k, v in raw.items():
                if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                    model = v; break
        if model is None:
            raise ValueError("Loaded dict but no estimator found (keys tried: pipeline/model/estimator/clf/best_estimator).")

        payload["pipeline"] = model
        payload["window_size"] = int(raw.get("window_size", 1))
        feats = raw.get("train_window_cols") or raw.get("feats") or []
        payload["train_window_cols"] = list(feats) if isinstance(feats, (list,tuple)) else []
        payload["neg_thr"] = float(raw.get("neg_thr", 0.5))   # defaults will be overwritten by saved ones
        payload["pos_thr"] = float(raw.get("pos_thr", 0.5))
        if "scaler" in raw: payload["scaler"] = raw["scaler"]
    else:
        payload["pipeline"] = raw
        payload["window_size"] = 1
        payload["train_window_cols"] = []
        payload["neg_thr"] = 0.5
        payload["pos_thr"] = 0.5

    # no override from train_distribution.json here; use exactly what's saved
    return payload

# ---------------- helpers ----------------
def resolve_timeframe_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    candidates = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    resolved: dict[str,str] = {}
    for tf, names in candidates.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists():
                found = str(p); break
        if found:
            logging.info(f"[raw] {tf} -> {found}")
            resolved[tf] = found
    if "30T" not in resolved:
        raise FileNotFoundError(f"Main timeframe '30T' not found under {base_dir}")
    return resolved

def find_m30_live_path(here: Path, symbol: str) -> Path | None:
    for nm in (f"{symbol}_M30_live.csv", f"{symbol}_30T_live.csv"):
        p = here / nm
        if p.exists():
            return p
    return None

def ensure_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return X
    X2 = X.copy()
    miss = [c for c in cols if c not in X2.columns]
    for c in miss: X2[c] = 0.0
    return X2[cols]

def md5_of_list(xs:list[str]) -> str:
    h = hashlib.md5()
    for s in xs: h.update(str(s).encode("utf-8"))
    return h.hexdigest()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--best-model", type=str, default="best_model.pkl")
    ap.add_argument("--poll-sec", type=float, default=0.5)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production_parity started ===")

    # 1) model
    payload = load_payload_best_model(args.best_model)
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or []
    window  = int(payload.get("window_size", 1))
    neg_thr = float(payload.get("neg_thr", 0.5))
    pos_thr = float(payload.get("pos_thr", 0.5))

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    logging.info("Model loaded | window=%d | feats=%d | thr=(%.3f, %.3f) | cols_md5=%s",
                 window, len(cols), neg_thr, pos_thr, md5_of_list(cols))

    # 2) full raw → merged exactly like simulator
    base_dir = Path(args.base_dir).resolve()
    raw_paths = resolve_timeframe_paths(base_dir, args.symbol)
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=raw_paths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    merged_full = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged_full[tcol] = pd.to_datetime(merged_full[tcol])
    merged_full.sort_values(tcol, inplace=True)
    merged_full.reset_index(drop=True, inplace=True)
    logging.info("[init] merged_full shape=%s | first=%s | last=%s",
                 merged_full.shape,
                 str(merged_full[tcol].iloc[0]) if len(merged_full) else None,
                 str(merged_full[tcol].iloc[-1]) if len(merged_full) else None)

    # 3) loop: wait for M30_live trigger, then Resample→Cut parity
    here = Path(".").resolve()
    last_ts = None
    ans_path = here / "answer.txt"

    while True:
        if ans_path.exists():
            time.sleep(args.poll_sec)
            continue

        p_m30 = find_m30_live_path(here, args.symbol)
        if p_m30 is None:
            time.sleep(args.poll_sec)
            continue

        # read trigger to extract ts_now
        try:
            df = pd.read_csv(p_m30)
            if "time" not in df.columns: 
                time.sleep(args.poll_sec); 
                continue
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df.dropna(subset=["time"], inplace=True)
            if df.empty:
                time.sleep(args.poll_sec); 
                continue
            ts_now = pd.to_datetime(df["time"].iloc[-1])
        except Exception as e:
            logging.error("[read live] %s", e)
            time.sleep(args.poll_sec)
            continue

        # skip if not new
        if (last_ts is not None) and (ts_now <= last_ts):
            time.sleep(args.poll_sec)
            continue

        # Resample was already done on merged_full → now Cut up to ts_now
        df_cut = merged_full[merged_full[tcol] <= ts_now].copy()
        if df_cut.empty:
            logging.warning("df_cut empty at %s", ts_now)
            time.sleep(args.poll_sec)
            continue

        # ready(...) exactly like simulator (mode=train, no drop)
        X, y, _, _ = prep.ready(
            df_cut,
            window=window,
            selected_features=cols,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        if X.empty:
            ans_path.write_text("NONE", encoding="utf-8")
            logging.info("[Predict] X empty @ %s → answer=NONE", ts_now)
        else:
            X = ensure_columns(X, cols)
            x_last = X.tail(1)
            prob = float(model.predict_proba(x_last)[:, 1][0])

            if   prob >= pos_thr: ans = "BUY"
            elif prob <= neg_thr: ans = "SELL"
            else:                 ans = "NONE"

            ans_path.write_text(ans, encoding="utf-8")
            logging.info("[Predict] ts=%s | prob=%.6f | thr=(%.3f,%.3f) → %s | X_cols=%d md5=%s",
                         ts_now, prob, neg_thr, pos_thr, ans, X.shape[1], md5_of_list(list(X.columns)))

        # cleanup live files (optional – به ژنراتور اجازه ادامه می‌دهد)
        try: p_m30.unlink(missing_ok=True)
        except Exception: pass
        for nm in (f"{args.symbol}_M15_live.csv", f"{args.symbol}_M5_live.csv", f"{args.symbol}_H1_live.csv",
                   f"{args.symbol}_15T_live.csv", f"{args.symbol}_5T_live.csv", f"{args.symbol}_1H_live.csv"):
            p = here / nm
            try: p.unlink(missing_ok=True)
            except Exception: pass

        last_ts = ts_now
        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
