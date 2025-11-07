# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, pickle, logging, argparse
from pathlib import Path
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "prediction_in_production.log"

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

def load_payload_best_model(pkl_path: str="best_model.pkl") -> dict:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    payload: dict = {}
    if isinstance(raw, dict):
        model = raw.get("pipeline") or raw.get("model") or raw.get("estimator") or raw.get("clf") or raw.get("best_estimator")
        if model is None:
            raise ValueError("Could not find model inside best_model.pkl dictionary.")
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

    # thresholds override from json if available
    if os.path.exists("train_distribution.json"):
        try:
            with open("train_distribution.json","r",encoding="utf-8") as jf:
                td = json.load(jf)
            payload["neg_thr"] = float(td.get("neg_thr", payload["neg_thr"]))
            payload["pos_thr"] = float(td.get("pos_thr", payload["pos_thr"]))
            logging.info("Train distribution loaded: train_distribution.json")
        except Exception:
            pass

    return payload

def resolve_live_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    """
    Finds *_live.csv files. Requires at least 30T live file.
    """
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

def should_skip_holiday(df_main: pd.DataFrame, main_tf="30T") -> bool:
    tcol = f"{main_tf}_time"
    if tcol not in df_main.columns:
        # try 'time'
        tcol = "time"
    if tcol not in df_main.columns:
        return False
    idx = pd.to_datetime(df_main[tcol])
    if idx.empty: return False
    last = idx.iloc[-1]
    # شنبه/یکشنبه (5,6)
    return last.dayofweek in (5,6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production started ===")

    # Load model once
    payload = load_payload_best_model("best_model.pkl")
    model   = payload["pipeline"]
    window  = int(payload["window_size"])
    cols    = payload.get("train_window_cols") or []
    neg_thr = float(payload["neg_thr"]); pos_thr = float(payload["pos_thr"])
    logging.info(f"Model loaded | window={window} | feats={len(cols)} | "
                 f"neg_thr={neg_thr:.3f} | pos_thr={pos_thr:.3f}")

    base = Path(args.base_dir).resolve()
    ans_path = base / "answer.txt"

    while True:
        # 1) منتظر بمان تا answer.txt وجود نداشته باشد و 4 فایل _live آماده باشند
        if ans_path.exists():
            time.sleep(args.sleep); continue

        live_paths = resolve_live_paths(base, args.symbol)
        if "30T" not in live_paths:
            time.sleep(args.sleep); continue

        # مطمئن شو همهٔ فایل‌هایی که وجود دارند، حداقل 30T را دارند؛ بقیه اختیاری‌اند
        filepaths = {tf: live_paths[tf] for tf in live_paths.keys()}

        try:
            # PREP را با strict_disk_feed=True بساز (بدون drift/trim و دقیقاً همان برش‌های دیسکی)
            prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                          verbose=False, fast_mode=True, strict_disk_feed=True)
            merged = prep.load_data()

            # حذف تعطیلات: اگر آخرین timestamp تعطیل است، NONE و پاکسازی
            if should_skip_holiday(merged, "30T"):
                ans_path.write_text("NONE", encoding="utf-8")
                logging.info("[Skip] Weekend detected → answer=NONE")
                # پاک‌سازی فایل‌های زنده
                for p in list(live_paths.values()):
                    try: Path(p).unlink(missing_ok=True)
                    except Exception: pass
                time.sleep(args.sleep)
                continue

            # آماده‌سازی فیچرها (مثل live_like_sim_v3)
            X, y_dummy, _, _ = prep.ready(
                merged,
                window=window,
                selected_features=cols,
                mode="predict",
                with_times=False,
                predict_drop_last=True  # یک ردیف حفاظتی
            )
            if X.empty:
                ans_path.write_text("NONE", encoding="utf-8")
                logging.info("[Predict] X empty → answer=NONE")
            else:
                X = ensure_columns(X, cols)
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X.tail(1))[:, 1][0])
                    if   prob >= pos_thr: ans="BUY"
                    elif prob <= neg_thr: ans="SELL"
                    else:                 ans="NONE"
                    ans_path.write_text(ans, encoding="utf-8")
                    logging.info(f"[Predict] prob={prob:.6f} → {ans}")
                else:
                    yhat = int(model.predict(X.tail(1))[0])
                    ans = "BUY" if yhat==1 else "SELL"
                    ans_path.write_text(ans, encoding="utf-8")
                    logging.info(f"[Predict] cls={yhat} → {ans}")

            # پاک‌سازی 4 فایل _live (جهت گام بعدی ژنراتور)
            for p in list(live_paths.values()):
                try: Path(p).unlink(missing_ok=True)
                except Exception: pass

        except Exception as e:
            logging.exception(f"[ERROR] {e}")
            # در صورت خطا، NONE بده تا چرخه قفل نشود
            try:
                ans_path.write_text("NONE", encoding="utf-8")
            except Exception:
                pass

        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
