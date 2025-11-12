# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations

import os, sys, time, json, joblib, logging, argparse, hashlib
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "prediction_in_production.log"

# ---------- Logging ----------
def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

# ---------- Utils ----------
def md5_list(xs:list[str]) -> str:
    h = hashlib.md5()
    for s in xs:
        h.update(str(s).encode("utf-8"))
    return h.hexdigest()

def load_payload_best_model(pkl_path: str="best_model.pkl") -> dict:
    payload = joblib.load(pkl_path)
    if isinstance(payload, dict):
        model = payload.get("pipeline") or payload.get("model") or payload.get("estimator") or payload.get("clf") or payload.get("best_estimator")
        if model is None:
            raise ValueError("Could not find model inside best_model.pkl dictionary.")
        out = {
            "pipeline": model,
            "window_size": int(payload.get("window_size", 1)),
            "train_window_cols": list(payload.get("train_window_cols") or payload.get("feats") or []),
            "neg_thr": float(payload.get("neg_thr", 0.005)),
            "pos_thr": float(payload.get("pos_thr", 0.995)),
        }
        # optional scaler (legacy)
        if "scaler" in payload: out["scaler"] = payload["scaler"]
        return out
    # payload was a raw estimator
    return {"pipeline": payload, "window_size": 1, "train_window_cols": [], "neg_thr": 0.005, "pos_thr": 0.995}

def resolve_live_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    patterns = {
        "30T": [f"{symbol}_30T_live.csv", f"{symbol}_M30_live.csv"],
        "15T": [f"{symbol}_15T_live.csv", f"{symbol}_M15_live.csv"],
        "5T" : [f"{symbol}_5T_live.csv",  f"{symbol}_M5_live.csv" ],
        "1H" : [f"{symbol}_1H_live.csv",  f"{symbol}_H1_live.csv" ],
    }
    resolved = {}
    for tf, names in patterns.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found = str(p); break
        if not found:
            # case-insensitive fallback
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            resolved[tf] = found
    return resolved

def ensure_columns(X:pd.DataFrame, cols:list[str]) -> pd.DataFrame:
    if not cols: return X
    X2 = X.copy()
    miss = [c for c in cols if c not in X2.columns]
    for c in miss: X2[c] = 0.0
    return X2[cols]

def should_skip_weekend(df_main: pd.DataFrame, main_tf="30T") -> bool:
    tcol = f"{main_tf}_time" if f"{main_tf}_time" in df_main.columns else ("time" if "time" in df_main.columns else None)
    if not tcol: return False
    idx = pd.to_datetime(df_main[tcol], errors="coerce")
    if idx.empty or idx.isna().all(): return False
    last = idx.iloc[-1]
    return last.dayofweek in (5,6)

# ---------- Adaptive Thresholds ----------
class AdaptiveThresholds:
    def __init__(self, base_neg: float, base_pos: float, buf_len:int=1000, q_low=0.10, q_high=0.90, warm_min:int=500, fallback_soft=(0.30, 0.70)):
        self.base_neg = float(base_neg)
        self.base_pos = float(base_pos)
        self.buf = deque(maxlen=buf_len)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.warm_min = int(warm_min)
        self.fallback_soft = (float(fallback_soft[0]), float(fallback_soft[1]))

    def update_and_get(self, p: float) -> tuple[float,float,str]:
        self.buf.append(float(p))
        if len(self.buf) >= self.warm_min:
            arr = np.asarray(self.buf, dtype=float)
            ql = float(np.nanquantile(arr, self.q_low))
            qh = float(np.nanquantile(arr, self.q_high))
            return ql, qh, "adaptive"
        # warmup: اگر آستانه‌های ذخیره‌شده خیلی تند باشند، نرم‌تر کنیم
        if (self.base_neg <= 0.01 and self.base_pos >= 0.99):
            return self.fallback_soft[0], self.fallback_soft[1], "soft-fallback"
        return self.base_neg, self.base_pos, "model"

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production started ===")

    # Load model
    payload = load_payload_best_model("best_model.pkl")
    model   = payload["pipeline"]
    window  = int(payload["window_size"])
    cols    = payload.get("train_window_cols") or []
    neg0    = float(payload["neg_thr"]); pos0 = float(payload["pos_thr"])
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    logging.info(f"Model loaded | window={window} | feats={len(cols)} | neg_thr0={neg0:.3f} | pos_thr0={pos0:.3f}")

    base = Path(args.base_dir).resolve()
    ans_path = base / "answer.txt"
    # adaptive thresholds
    ath = AdaptiveThresholds(neg0, pos0)

    # simple hist buckets for prob1 monitoring
    bucket_edges = np.linspace(0,1,11)
    bucket_cnts = np.zeros(len(bucket_edges)-1, dtype=int)
    step = 0

    while True:
        if ans_path.exists():
            time.sleep(args.sleep); continue

        live_paths = resolve_live_paths(base, args.symbol)
        if "30T" not in live_paths:  # main TF is mandatory
            time.sleep(args.sleep); continue

        filepaths = {tf: live_paths[tf] for tf in live_paths.keys()}

        try:
            # Use fast_mode=False to enable full detection of bad columns and match simulation behavior.
            # strict_disk_feed=True to treat input CSVs as-is (no automatic trimming).
            prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                          verbose=False, fast_mode=False, strict_disk_feed=True)
            merged = prep.load_data()

            # Weekend skip (safety)
            if should_skip_weekend(merged, "30T"):
                ans_path.write_text("NONE", encoding="utf-8")
                logging.info("[Skip] Weekend detected → answer=NONE")
                # cleanup
                for p in list(live_paths.values()):
                    try: Path(p).unlink(missing_ok=True)
                    except Exception: pass
                time.sleep(args.sleep)
                continue

            # Build features exactly like live_like_sim_v3
            X, y_dummy, _, _ = prep.ready(
                merged,
                window=window,
                selected_features=cols,    # enforce exact train window columns
                mode="predict",
                with_times=False,
                predict_drop_last=True     # one safe drop at the end
            )
            if X.empty:
                ans_path.write_text("NONE", encoding="utf-8")
                logging.info("[Predict] X empty → answer=NONE")
            else:
                X = ensure_columns(X, cols)
                x1 = X.tail(1)
                prob = float(model.predict_proba(x1)[:, 1][0])

                # thresholds (possibly adaptive)
                neg_thr, pos_thr, tmode = ath.update_and_get(prob)

                # histogram monitor
                hist_idx = np.digitize([prob], bucket_edges) - 1
                if 0 <= hist_idx[0] < len(bucket_cnts): bucket_cnts[hist_idx[0]] += 1
                step += 1

                if   prob >= pos_thr: ans="BUY"
                elif prob <= neg_thr: ans="SELL"
                else:                 ans="NONE"
                ans_path.write_text(ans, encoding="utf-8")

                if step % 100 == 1:
                    col_hash = md5_list(list(X.columns))
                    # compact histogram to log
                    hist_txt = ",".join(str(int(c)) for c in bucket_cnts.tolist())
                    logging.info(f"[Predict] prob={prob:.6f} → {ans} | thr=({neg_thr:.4f},{pos_thr:.4f})[{tmode}] | "
                                 f"cols={len(cols)} md5={col_hash} | hist(0..1,10bins)={hist_txt}")

            # cleanup live CSVs so generator can proceed
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