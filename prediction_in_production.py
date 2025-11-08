# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, pickle, logging, argparse, gzip, bz2, lzma, zipfile, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "prediction_in_production.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

# ---------------- robust model loader ----------------
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

def load_payload_best_model(pkl_path: str="best_model.pkl") -> tuple[dict, Path]:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"'best_model.pkl' not found at {p.resolve()}")
    head = _raw_head(pkl_path, 8)
    loaders = []
    if head.startswith(b"\x80"):      loaders=[_try_pickle,_try_joblib,_try_gzip,_try_bz2,_try_lzma,_try_zip]
    elif head.startswith(b"\x1f\x8b"): loaders=[_try_gzip,_try_pickle,_try_joblib,_try_bz2,_try_lzma,_try_zip]
    elif head.startswith(b"BZh"):      loaders=[_try_bz2,_try_pickle,_try_joblib,_try_gzip,_try_lzma,_try_zip]
    elif head.startswith(b"\xfd7zXZ\x00"): loaders=[_try_lzma,_try_pickle,_try_joblib,_try_gzip,_try_bz2,_try_zip]
    elif head.startswith(b"PK\x03\x04"):   loaders=[_try_zip,_try_pickle,_try_joblib,_try_gzip,_try_bz2,_try_lzma]
    else: loaders=[_try_joblib,_try_pickle,_try_gzip,_try_bz2,_try_lzma,_try_zip]

    raw=None; last_err=None
    for loader in loaders:
        try:
            raw = loader(pkl_path); break
        except Exception as e:
            last_err=e
    if raw is None:
        try:
            with open(pkl_path,"r",encoding="utf-8") as f:
                raw=json.load(f)
        except Exception:
            raise RuntimeError(f"Could not load best_model.pkl. First bytes={head!r}, last_err={repr(last_err)}")

    payload={}
    if isinstance(raw, dict):
        model = raw.get("pipeline") or raw.get("model") or raw.get("estimator") or raw.get("clf") or raw.get("best_estimator")
        if model is None:
            for k,v in raw.items():
                if hasattr(v,"predict") or hasattr(v,"predict_proba"):
                    model=v; break
        if model is None:
            raise ValueError("Loaded dict but no estimator found.")
        payload["pipeline"]=model
        payload["window_size"]=int(raw.get("window_size",1))
        feats = raw.get("train_window_cols") or raw.get("feats") or []
        payload["train_window_cols"]= list(feats) if isinstance(feats,(list,tuple)) else []
        payload["neg_thr"]=float(raw.get("neg_thr",0.005))
        payload["pos_thr"]=float(raw.get("pos_thr",0.995))
        # مسیر train_distribution (نسبی) اگر موجود است
        payload["train_distribution"] = raw.get("train_distribution")
        if "scaler" in raw: payload["scaler"]=raw["scaler"]
    else:
        payload["pipeline"]=raw
        payload["window_size"]=1
        payload["train_window_cols"]=[]
        payload["neg_thr"]=0.005
        payload["pos_thr"]=0.995
        payload["train_distribution"]=None

    # ⚠️ فقط اگر فایل «کنار مدل» وجود دارد override کن
    model_dir = p.parent
    td_name   = payload.get("train_distribution")
    if td_name:
        td_path = model_dir / td_name
        if td_path.exists():
            try:
                with open(td_path,"r",encoding="utf-8") as jf:
                    td = json.load(jf)
                payload["neg_thr"] = float(td.get("neg_thr", payload["neg_thr"]))
                payload["pos_thr"] = float(td.get("pos_thr", payload["pos_thr"]))
                logging.info(f"Train distribution loaded from model dir: {td_path.name}")
            except Exception:
                pass
    return payload, model_dir

def resolve_live_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    patterns = {
        "30T": [f"{symbol}_30T_live.csv", f"{symbol}_M30_live.csv"],
        "15T": [f"{symbol}_15T_live.csv", f"{symbol}_M15_live.csv"],
        "5T" : [f"{symbol}_5T_live.csv",  f"{symbol}_M5_live.csv" ],
        "1H" : [f"{symbol}_1H_live.csv",  f"{symbol}_H1_live.csv" ],
    }
    resolved={}
    for tf, names in patterns.items():
        found=None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found=str(p); break
        if not found:
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found=str(ch); break
        if found:
            resolved[tf]=found
    return resolved

def reindex_strict(X: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """به ترتیب دقیق train_window_cols ست کن؛ ستون‌های غایب با 0 پر شوند (اما شمارش شود)."""
    if not cols:
        return X.copy(), 0
    miss = [c for c in cols if c not in X.columns]
    X2 = X.reindex(columns=cols, fill_value=0.0)
    return X2, len(miss)

def sha1_of_array_lastrow(X: pd.DataFrame) -> str:
    if X.empty: return "-"
    row = X.tail(1).to_numpy(dtype=np.float64, copy=False)
    b = row.tobytes(order="C")
    return hashlib.sha1(b).hexdigest()[:12]

def should_skip_holiday(df_main: pd.DataFrame, main_tf="30T") -> bool:
    tcol=f"{main_tf}_time"
    if tcol not in df_main.columns: tcol="time" if "time" in df_main.columns else None
    if not tcol: return False
    idx=pd.to_datetime(df_main[tcol], errors="coerce")
    if idx.empty: return False
    last=idx.iloc[-1]
    return last.dayofweek in (5,6)  # شنبه/یکشنبه

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--audit", action="store_true", help="Dump audit_pred.csv with per-step feature hashes.")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production started ===")

    payload, model_dir = load_payload_best_model("best_model.pkl")
    model   = payload["pipeline"]
    window  = int(payload["window_size"])
    cols    = payload.get("train_window_cols") or []
    neg_thr = float(payload["neg_thr"]); pos_thr = float(payload["pos_thr"])
    logging.info(f"Model loaded | window={window} | feats={len(cols)} | neg_thr={neg_thr:.3f} | pos_thr={pos_thr:.3f}")

    base = Path(args.base_dir).resolve()
    ans_path = base / "answer.txt"

    # audit init
    audit_path = base / "audit_pred.csv"
    if args.audit and (not audit_path.exists()):
        audit_path.write_text("time,cols,missing,sha1,prob,decision\n", encoding="utf-8")

    while True:
        if ans_path.exists():
            time.sleep(args.sleep); continue

        live_paths = resolve_live_paths(base, args.symbol)
        # ⛔ بدون 30T اصلاً پیش‌بینی نکن
        if "30T" not in live_paths:
            time.sleep(args.sleep); continue

        filepaths = {tf: live_paths[tf] for tf in live_paths.keys()}

        try:
            prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                          verbose=False, fast_mode=True, strict_disk_feed=True)
            merged = prep.load_data()

            if should_skip_holiday(merged, "30T"):
                ans_path.write_text("NONE", encoding="utf-8")
                logging.info("[Skip] Weekend detected → answer=NONE")
                for p in list(live_paths.values()):
                    try: Path(p).unlink(missing_ok=True)
                    except Exception: pass
                time.sleep(args.sleep); continue

            X, _, _, _ = prep.ready(
                merged,
                window=window,
                selected_features=cols,     # ← دقیقاً همان ستون‌های train بعد از window
                mode="predict",
                with_times=False,
                predict_drop_last=True
            )

            if X.empty:
                ans="NONE"
                ans_path.write_text(ans, encoding="utf-8")
                logging.info("[Predict] X empty → answer=NONE")
            else:
                X, miss_n = reindex_strict(X, cols)
                sha = sha1_of_array_lastrow(X)

                if hasattr(model,"predict_proba"):
                    prob = float(model.predict_proba(X.tail(1))[:,1][0])
                    if   prob >= pos_thr: ans="BUY"
                    elif prob <= neg_thr: ans="SELL"
                    else: ans="NONE"
                    ans_path.write_text(ans, encoding="utf-8")
                    logging.info(f"[Predict] cols={X.shape[1]} miss={miss_n} sha={sha} prob={prob:.6f} → {ans}")
                    if args.audit:
                        # زمان آخرین ردیف merge را برداریم اگر هست
                        tcol = "30T_time" if "30T_time" in merged.columns else ("time" if "time" in merged.columns else "")
                        last_t = str(pd.to_datetime(merged[tcol].iloc[-1])) if tcol else ""
                        audit_path.open("a", encoding="utf-8").write(f"{last_t},{X.shape[1]},{miss_n},{sha},{prob:.8f},{ans}\n")
                else:
                    cls = int(model.predict(X.tail(1))[0])
                    ans = "BUY" if cls==1 else "SELL"
                    ans_path.write_text(ans, encoding="utf-8")
                    logging.info(f"[Predict] cols={X.shape[1]} miss={miss_n} sha={sha} cls={cls} → {ans}")
                    if args.audit:
                        tcol = "30T_time" if "30T_time" in merged.columns else ("time" if "time" in merged.columns else "")
                        last_t = str(pd.to_datetime(merged[tcol].iloc[-1])) if tcol else ""
                        audit_path.open("a", encoding="utf-8").write(f"{last_t},{X.shape[1]},{miss_n},{sha},,\n")

            # پاک‌سازی فایل‌های _live
            for p in list(live_paths.values()):
                try: Path(p).unlink(missing_ok=True)
                except Exception: pass

        except Exception as e:
            logging.exception(f"[ERROR] {e}")
            try: ans_path.write_text("NONE", encoding="utf-8")
            except Exception: pass

        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
