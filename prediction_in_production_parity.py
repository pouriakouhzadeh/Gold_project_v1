# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prediction_in_production_parity.py
----------------------------------
Exact-parity online predictor for 30-minute timeframe.

- Trigger: the presence/update of XAUUSD_M30_live.csv (or XAUUSD_30T_live.csv / XAUUSD.F_M30_live.csv).
- Build features from the SAME full raw CSVs used in training (M5/M15/M30/H1).
- Resample/feature → THEN cut up to ts_now (identical to simulator).
- Use thresholds saved with the model; optionally override from train_distribution.json (like v3).
- Write BUY/SELL/NONE to ./answer.txt for MT4 to consume.
"""

from __future__ import annotations
import os, sys, time, json, pickle, gzip, bz2, lzma, zipfile, logging, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # same pipeline as trainer/simulator

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

# ---------------- Robust model loader (pickle/joblib/gzip/bz2/xz/zip/json) ----------------
def _try_pickle(fp: str):
    with open(fp, "rb") as f: return pickle.load(f)
def _try_joblib(fp: str):
    import joblib; return joblib.load(fp)
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

    last_err = None; raw = None
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
        raise RuntimeError(f"Could not load best_model.pkl; first bytes: {head!r}; last err: {repr(last_err)}")

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
        payload["neg_thr"] = float(raw.get("neg_thr", 0.005))
        payload["pos_thr"] = float(raw.get("pos_thr", 0.995))
        if "scaler" in raw: payload["scaler"] = raw["scaler"]
    else:
        payload["pipeline"] = raw
        payload["window_size"] = 1
        payload["train_window_cols"] = []
        payload["neg_thr"] = 0.005
        payload["pos_thr"] = 0.995

    # Optional override to mirror live_like_sim_v3 behavior
    td_path = "train_distribution.json"
    if os.path.exists(td_path):
        try:
            with open(td_path, "r", encoding="utf-8") as jf:
                td = json.load(jf)
            if "neg_thr" in td: payload["neg_thr"] = float(td["neg_thr"])
            if "pos_thr" in td: payload["pos_thr"] = float(td["pos_thr"])
            logging.info("Train distribution loaded: train_distribution.json")
        except Exception:
            pass

    return payload

# ---------------- path helpers ----------------
def resolve_timeframe_paths(base_dir: Path, symbol: str) -> dict[str, str]:
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
            if p.exists(): found = str(p); break
        if found is None:
            # case-insensitive scan
            for child in base_dir.iterdir():
                if child.is_file() and any(child.name.lower()==nm.lower() for nm in names):
                    found = str(child); break
        if found:
            logging.info(f"[raw] {tf} -> {found}")
            resolved[tf] = found
    if "30T" not in resolved:
        raise FileNotFoundError(f"Main timeframe '30T' not found under {base_dir}.")
    return resolved

def find_m30_live_path(here: Path, symbol: str) -> Path | None:
    for nm in (f"{symbol}_M30_live.csv", f"{symbol}_30T_live.csv", f"{symbol}.F_M30_live.csv", f"{symbol}_F_M30_live.csv"):
        p = here / nm
        if p.exists(): return p
    return None

def ensure_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols: return X
    X2 = X.copy()
    miss = [c for c in cols if c not in X2.columns]
    for c in miss: X2[c] = 0.0
    return X2[cols].astype(float)

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
    ap.add_argument("--poll-sec", type=float, default=0.25)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production_parity started ===")

    # 1) model
    payload = load_payload_best_model(args.best_model)
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or []
    window  = int(payload.get("window_size", 1))
    neg_thr = float(payload.get("neg_thr", 0.005))
    pos_thr = float(payload.get("pos_thr", 0.995))
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")
    logging.info(
        "Model loaded | window=%d | feats=%d | thr=(%.3f, %.3f) | cols_md5=%s",
        window, len(cols), neg_thr, pos_thr, md5_of_list(cols)
    )

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

    # 3) loop: wait for 30T trigger, then Resample→Cut parity
    here = Path(".").resolve()
    last_ts = None
    ans_path = here / "answer.txt"

    while True:
        # MT4/Generator will remove answer.txt after reading; اگر هنوز هست یعنی نوبت جدیدی نیامده
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
                time.sleep(args.poll_sec); continue
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df.dropna(subset=["time"], inplace=True)
            if df.empty:
                time.sleep(args.poll_sec); continue
            ts_now = pd.to_datetime(df["time"].iloc[-1])
        except Exception as e:
            logging.error("[read live] %s", e)
            time.sleep(args.poll_sec)
            continue

        if (last_ts is not None) and (ts_now <= last_ts):
            time.sleep(args.poll_sec)
            continue

        # Resample already done on merged_full → now Cut up to ts_now
        df_cut = merged_full[merged_full[tcol] <= ts_now].copy()
        if df_cut.empty:
            logging.warning("df_cut empty at %s", ts_now)
            time.sleep(args.poll_sec); continue

        # same path as simulator (mode='train', no drop_last)
        X, y, _, _ = prep.ready(
            df_cut,
            window=window,
            selected_features=cols,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        if X.empty:
            logging.warning("Empty X after ready() at %s", ts_now)
            time.sleep(args.poll_sec); continue

        L = min(len(X), len(y))
        X = ensure_columns(X.iloc[:L].reset_index(drop=True), cols)
        y = pd.Series(y).iloc[:L].reset_index(drop=True)

        probs = model.predict_proba(X)[:, 1]
        p_last = float(probs[-1])
        if p_last <= neg_thr: decision = "SELL"
        elif p_last >= pos_thr: decision = "BUY"
        else: decision = "NONE"

        try:
            ans_path.write_text(decision, encoding="utf-8")
        except Exception as e:
            logging.error("Could not write answer.txt: %s", e)
            time.sleep(args.poll_sec); continue

        logging.info("[Predict] ts=%s | score=%.6f | thr=(%.3f,%.3f) → %s | wrote=%s",
                     ts_now, p_last, neg_thr, pos_thr, decision, str(ans_path.resolve()))
        last_ts = ts_now
        # small sleep to avoid tight loop
        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
