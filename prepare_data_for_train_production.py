# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, time, pickle, logging, argparse, gzip, bz2, lzma, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from prepare_data_for_train_production import PREPARE_DATA_FOR_TRAIN_PRODUCTION

LOG_FILE = "live_like_sim_v4.log"

# ---------------- Logging ----------------
def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

# ---------------- Robust model loader (مثل v3) ----------------
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

    # thresholds override از train_distribution.json (اختیاری)
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

# ---------------- path resolver ----------------
def resolve_timeframe_paths(base_dir: Path, symbol: str) -> dict:
    candidates = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    resolved = {}
    for tf, names in candidates.items():
        for nm in names:
            p = base_dir / nm
            if p.exists():
                resolved[tf] = str(p)
                break
    if "30T" not in resolved:
        raise FileNotFoundError(f"Main timeframe '30T' not found under {base_dir}")
    return resolved

# ---------------- ensure columns order ----------------
def ensure_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return X
    X2 = X.copy()
    missing = [c for c in cols if c not in X2.columns]
    for c in missing:
        X2[c] = 0.0
    return X2[cols]

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== live_like_sim_v4 starting (cut BEFORE resample) ===")

    # 1) مدل
    payload = load_payload_best_model("best_model.pkl")
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or []
    window  = int(payload["window_size"])
    logging.info(f"Model loaded | window={window} | feats={len(cols)} | "
                 f"neg_thr={payload['neg_thr']:.3f} | pos_thr={payload['pos_thr']:.3f}")

    # 2) CSV paths
    base_dir = Path(args.base_dir).resolve()
    filepaths = resolve_timeframe_paths(base_dir, args.symbol)

    # 3) PREP (Production)
    prep = PREPARE_DATA_FOR_TRAIN_PRODUCTION(filepaths=filepaths, main_timeframe="30T", verbose=True)
    prep.load_all()

    # برای تعیین بازه‌ی tail و timestamps:
    merged_full = prep.load_data_up_to(pd.Timestamp.max)  # تا انتهای دیتاست
    main_t = f"{prep.main_timeframe}_time"
    if main_t not in merged_full.columns or merged_full.empty:
        logging.info("No data available after processing.")
        return

    merged_full[main_t] = pd.to_datetime(merged_full[main_t])
    merged_full.sort_values(main_t, inplace=True)
    merged_full.reset_index(drop=True, inplace=True)

    total = len(merged_full)
    start_idx = max(0, total - args.last_n)
    logging.info(f"Main rows={total} | last_n={args.last_n} | start_idx={start_idx}")

    timestamps = merged_full.loc[start_idx:, main_t].tolist()
    records = []

    # --- Live-like: تکرار روی هر timestamp (cut → resample → FE → predict) ---
    for i, ts_end in enumerate(timestamps, start=1):
        if i % 200 == 0:
            logging.info(f"[Live] step {i}/{len(timestamps)} @ {ts_end}")

        merged_end = prep.load_data_up_to(pd.Timestamp(ts_end))
        if merged_end.empty:
            continue

        X, y, feats = prep.ready(merged_end, window=window, selected_features=cols, mode="train", with_times=False)
        if X.empty:
            continue

        # آخرین رکورد برای پیش‌بینی لایو
        X_last = ensure_columns(X.tail(1).reset_index(drop=True), cols)

        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X_last)[:, 1][0])
            # آستانه‌ی دوسویه
            if p <= float(payload["neg_thr"]): yhat = 0
            elif p >= float(payload["pos_thr"]): yhat = 1
            else: yhat = -1
        else:
            yhat = int(model.predict(X_last)[0])
            p = np.nan

        # y_true = آخرین برچسب موجود
        y_true = int(y.iloc[-1]) if len(y)>0 else -1

        records.append({
            "timestamp": ts_end,
            "pred_live": yhat,
            "prob_live": p,
            "y_true": y_true
        })

    live_df = pd.DataFrame.from_records(records)
    if live_df.empty:
        logging.info("No live records produced.")
        return

    # --- Chimney روی همان پنجره‌ی tail (جهت تطبیق) ---
    # یک‌بار تا انتهای دیتاست:
    block_tail = merged_full.iloc[start_idx:].copy()
    Xc, yc, _ = prep.ready(block_tail, window=window, selected_features=cols, mode="train", with_times=False)
    if not Xc.empty:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(ensure_columns(Xc, cols))[:, 1]
            yp = np.full(len(prob), -1, dtype=int)
            yp[prob <= float(payload["neg_thr"])] = 0
            yp[prob >= float(payload["pos_thr"])] = 1
        else:
            yp = model.predict(ensure_columns(Xc, cols))
            prob = np.full(len(yp), np.nan)
        chim_ts = block_tail[f"{prep.main_timeframe}_time"].iloc[-len(yp):].reset_index(drop=True)
        chim_df = pd.DataFrame({"timestamp": chim_ts, "pred_chimney": yp})
        live_df = live_df.merge(chim_df, on="timestamp", how="left")
        live_df["agree_pred"] = (live_df["pred_live"] == live_df["pred_chimney"]).astype(int)
    else:
        live_df["pred_chimney"] = -1
        live_df["agree_pred"] = 0

    # --- متریک‌ها ---
    def _metrics(df: pd.DataFrame, col: str):
        sub = df[df[col] != -1]
        if sub.empty:
            return (np.nan, np.nan, 0.0)
        acc = accuracy_score(sub["y_true"], sub[col])
        bacc = balanced_accuracy_score(sub["y_true"], sub[col])
        cover = len(sub) / len(df)
        return (acc, bacc, cover)

    acc_l, bacc_l, cov_l = _metrics(live_df, "pred_live")
    acc_c, bacc_c, cov_c = _metrics(live_df, "pred_chimney")

    logging.info(f"[Final] Live    : acc={acc_l:.3f} bAcc={bacc_l:.3f} cover={cov_l:.3f}")
    logging.info(f"[Final] Chimney : acc={acc_c:.3f} bAcc={bacc_c:.3f} cover={cov_c:.3f}")
    if "agree_pred" in live_df.columns:
        logging.info(f"[Final] Pred agreement (chimney vs live): {live_df['agree_pred'].mean():.3f}")

    live_df.to_csv("predictions_compare_v4.csv", index=False)
    logging.info("=== live_like_sim_v4 completed ===")

if __name__ == "__main__":
    main()
