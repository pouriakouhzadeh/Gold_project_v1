# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, time, pickle, logging, argparse, gzip, bz2, lzma, zipfile
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

# ------------------------- Logging -------------------------
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

# ------------------- Robust model loader -------------------
def _try_pickle(fp: str):
    with open(fp, "rb") as f:
        return pickle.load(f)

def _try_joblib(fp: str):
    if joblib is None: raise RuntimeError("joblib not available")
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
        # pick first member that isn't a directory
        for name in zf.namelist():
            if name.endswith("/"): 
                continue
            with zf.open(name, "r") as f:
                data = f.read()
                # try pickle then json
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
    """
    Tries multiple strategies to load best_model.pkl:
    pickle / joblib / gzip / bz2 / lzma(xz) / zip / json
    Returns normalized payload dict:
      - pipeline (Estimator with predict/proba)
      - window_size (int)
      - neg_thr, pos_thr (float)
      - train_window_cols (list[str])
      - scaler (optional)
    """
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"'best_model.pkl' not found at {p.resolve()}")

    # quick format detection by magic
    head = _raw_head(pkl_path, 8)
    # Pickle magic usually starts with 0x80
    # gzip: 1f 8b ; bz2: 42 5a 68 ; lzma/xz: fd 37 7a 58 5a 00 ; zip: 50 4b 03 04
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
        # Could be text JSON or some custom—try joblib/pickle first, then JSON
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
        # maybe pure JSON text
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
            # maybe nested under a single key
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
        # assume raw is estimator
        payload["pipeline"] = raw
        payload["window_size"] = 1
        payload["train_window_cols"] = []
        payload["neg_thr"] = 0.005
        payload["pos_thr"] = 0.995

    # Optional: thresholds override from train_distribution.json
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

# ---------------- path resolver for raw CSVs ----------------
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
            if p.exists():
                found = str(p); break
        if found is None:
            for child in base_dir.iterdir():
                if child.is_file() and any(child.name.lower()==nm.lower() for nm in names):
                    found = str(child); break
        if found:
            logging.info(f"[paths] {tf} -> {found}")
            resolved[tf] = found

    if "30T" not in resolved:
        raise FileNotFoundError(
            f"Main timeframe '30T' not found under {base_dir}. "
            f"Tried: {', '.join(candidates['30T'])}"
        )
    return resolved

# --------------------- column guard -----------------------
def ensure_columns(X:pd.DataFrame, cols:list[str]) -> pd.DataFrame:
    if not cols:
        return X
    X2 = X.copy()
    missing = [c for c in cols if c not in X2.columns]
    for c in missing: X2[c] = 0.0
    return X2[cols]

# ------------------- evaluation helper -------------------
def evaluate_block(df_part: pd.DataFrame, payload: dict, prep: PREPARE_DATA_FOR_TRAIN,
                   use_thresholds=True, label="Eval"):
    window = int(payload["window_size"])
    cols   = payload.get("train_window_cols") or payload.get("feats") or []
    model  = payload["pipeline"]

    X, y, _, _ = prep.ready(
        df_part,
        window=window,
        selected_features=cols,
        mode="train",
        predict_drop_last=False,
        train_drop_last=False
    )
    if X.empty:
        logging.info(f"[{label}] empty block")
        return dict(size=0, acc=np.nan, bacc=np.nan, cover=np.nan,
                    correct=0, incorrect=0, unpred=0)

    L = min(len(X), len(y))
    X = ensure_columns(X.iloc[:L].reset_index(drop=True), cols)
    y = pd.Series(y).iloc[:L].reset_index(drop=True)

    if hasattr(model, "predict_proba") and use_thresholds:
        probs = model.predict_proba(X)[:, 1]
        yp = np.full(L, -1, dtype=int)
        yp[probs <= float(payload["neg_thr"])] = 0
        yp[probs >= float(payload["pos_thr"])] = 1
    else:
        yp = model.predict(X)

    mask = (yp != -1)
    if mask.any():
        acc  = accuracy_score(y[mask], yp[mask])
        bacc = balanced_accuracy_score(y[mask], yp[mask])
        correct = int(((yp==y)&mask).sum())
        pred_n  = int(mask.sum())
        incorrect = pred_n - correct
        unpred = int((~mask).sum())
        cover = pred_n/len(yp)
    else:
        acc=bacc=cover=np.nan; correct=incorrect=unpred=0

    logging.info(f"[{label}] size={len(yp)} cover={cover:.3f} acc={acc:.3f} "
                 f"bAcc={bacc:.3f} Correct={correct} Incorrect={incorrect} Unpred={unpred}")
    return dict(size=len(yp), acc=acc, bacc=bacc, cover=cover,
                correct=correct, incorrect=incorrect, unpred=unpred)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== live_like_sim_v3 starting ===")

    # 1) load model bundle (robust)
    payload = load_payload_best_model("best_model.pkl")
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or []
    window  = int(payload["window_size"])
    logging.info(f"Model loaded | window={window} | feats={len(cols)} | "
                 f"neg_thr={payload['neg_thr']:.3f} | pos_thr={payload['pos_thr']:.3f}")

    # 2) resolve raw csv paths
    base_dir = Path(args.base_dir).resolve()
    filepaths = resolve_timeframe_paths(base_dir, args.symbol)

    # 3) PREP (batch-like)
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=True, fast_mode=False, strict_disk_feed=False)
    raw = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    raw[tcol] = pd.to_datetime(raw[tcol])
    raw.sort_values(tcol, inplace=True)
    raw.reset_index(drop=True, inplace=True)

    total = len(raw)
    logging.info(f"[load_data] merged shape={raw.shape}")
    logging.info(f"Using last_n={args.last_n} rows for both modes")

    # 4) chimney on tail block
    cut_start = max(0, total - args.last_n - 2*window)
    block_for_batch = raw.iloc[cut_start:].copy()
    _ = evaluate_block(block_for_batch, payload, prep, True, "Chimney-Batch")

    # 5) live-like step-by-step
    logging.info("[Step] Live-like step-by-step simulation")
    records, warm_rows = [], []
    start_for_live = max(window+2, total - args.last_n)

    for cut in range(start_for_live, total):
        if (cut - start_for_live) % 100 == 0:
            logging.info(f"  live cut @ {cut}/{total}")
        df_cut = raw.iloc[:cut].copy()

        Xc, yc, _, _ = prep.ready(
            df_cut,
            window=window,
            selected_features=cols,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        if Xc.empty: continue
        y_series = pd.Series(yc)
        L = min(len(Xc), len(y_series))
        if L == 0: continue

        Xc = ensure_columns(Xc.iloc[:L].reset_index(drop=True), cols)
        y_series = y_series.iloc[:L].reset_index(drop=True)

        if hasattr(model, "predict_proba"):
            prob_all = model.predict_proba(Xc)[:, 1]
            y_pred_all = np.full(L, -1, dtype=int)
            y_pred_all[prob_all <= float(payload["neg_thr"])] = 0
            y_pred_all[prob_all >= float(payload["pos_thr"])] = 1
        else:
            prob_all = np.zeros(L)
            y_pred_all = model.predict(Xc)

        y_hat = int(y_pred_all[-1])
        p_last = float(prob_all[-1])
        y_true = int(y_series.iloc[-1])
        ts = raw.iloc[cut-1][tcol]

        records.append({
            "timestamp": ts, "pred_live": y_hat, "y_true": y_true, "prob_live": p_last
        })

        # single vs batch sanity
        X_last = Xc.tail(1)
        if hasattr(model, "predict_proba"):
            prob_single = float(model.predict_proba(X_last)[:, 1][0])
            warm_rows.append({
                "timestamp": ts,
                "prob_single": prob_single,
                "prob_batch": p_last,
                "diff_prob_batch_minus_single": p_last - prob_single,
                "agree_pred": int(
                    ((prob_single >= float(payload["pos_thr"])) or (prob_single <= float(payload["neg_thr"]))) ==
                    ((p_last      >= float(payload["pos_thr"])) or (p_last      <= float(payload["neg_thr"])))
                )
            })

    logging.info("[Step] Build chimney-vs-live compare table")
    chim_preds = []
    for i, _r in enumerate(records):
        cut = start_for_live + i
        df_cut = raw.iloc[:cut].copy()

        Xc, yc, _, _ = prep.ready(
            df_cut,
            window=window,
            selected_features=cols,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        if Xc.empty: chim_preds.append(-1); continue
        y_series = pd.Series(yc)
        L = min(len(Xc), len(y_series))
        if L == 0: chim_preds.append(-1); continue

        Xc = ensure_columns(Xc.iloc[:L].reset_index(drop=True), cols)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(Xc)[:, 1]
            yp = np.full(L, -1, dtype=int)
            yp[prob <= float(payload["neg_thr"])] = 0
            yp[prob >= float(payload["pos_thr"])] = 1
            chim_preds.append(int(yp[-1]))
        else:
            chim_preds.append(int(model.predict(Xc.tail(1))[0]))

    live_df = pd.DataFrame.from_records(records)
    live_df["pred_chimney"] = chim_preds
    live_df["agree_pred"] = (live_df["pred_chimney"] == live_df["pred_live"]).astype(int)

    def metrics(df:pd.DataFrame, col:str):
        if col not in df: return (np.nan, np.nan, 0, 0, 0, 0.0)
        sub = df[df[col] != -1]
        if sub.empty:
            return (np.nan, np.nan, 0, 0, int((df[col]==-1).sum()), 0.0)
        acc  = accuracy_score(sub["y_true"], sub[col])
        bacc = balanced_accuracy_score(sub["y_true"], sub[col])
        correct = int((sub[col]==sub["y_true"]).sum())
        incorrect = int(len(sub) - correct)
        unpred = int((df[col]==-1).sum())
        cover = len(sub)/len(df)
        return (acc, bacc, correct, incorrect, unpred, cover)

    acc_c, bacc_c, c1, i1, u1, cov_c = metrics(live_df, "pred_chimney")
    acc_l, bacc_l, c2, i2, u2, cov_l = metrics(live_df, "pred_live")

    logging.info(f"[Final] Chimney: acc={acc_c:.3f} bAcc={bacc_c:.3f} cover={cov_c:.3f} "
                 f"Correct={c1} Incorrect={i1} Unpred={u1}")
    logging.info(f"[Final] Live    : acc={acc_l:.3f} bAcc={bacc_l:.3f} cover={cov_l:.3f} "
                 f"Correct={c2} Incorrect={i2} Unpred={u2}")
    logging.info(f"[Final] Pred agreement (chimney vs live): {live_df['agree_pred'].mean():.3f}")

    live_df.to_csv("predictions_compare.csv", index=False)
    pd.DataFrame(warm_rows).to_csv("live_single_vs_batch_preds.csv", index=False)
    pd.DataFrame(columns=["timestamp","feature","chimney_value","live_value","abs_diff","mismatch_flag"])\
      .to_csv("features_compare_detailed.csv", index=False)
    pd.DataFrame(columns=["feature","mismatch_percent","max_abs_diff","count"])\
      .to_csv("features_compare_summary.csv", index=False)

    scaler_note = "scaler_not_present"
    cols = payload.get("train_window_cols") or []
    pd.DataFrame([{"column": c, "note": scaler_note} for c in cols[:max(1, min(5000, len(cols)))]])\
      .to_csv("scaler_check.csv", index=False)

    logging.info("=== live_like_sim_v3 completed ===")

if __name__ == "__main__":
    main()
