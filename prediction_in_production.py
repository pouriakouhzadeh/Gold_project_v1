#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, logging, warnings
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

LOG_FILE = "prediction.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")]
)

ANSWER_FILE = "answer.txt"

LIVE_FILEPATHS = {
    "30T": "XAUUSD_M30_live.csv",
    "15T": "XAUUSD_M15_live.csv",
    "5T" : "XAUUSD_M5_live.csv",
    "1H" : "XAUUSD_M1H_live.csv" if os.path.exists("XAUUSD_M1H_live.csv") else "XAUUSD_H1_live.csv",
}

MODEL_PATH = "best_model.pkl"
SCALER_CANDIDATES     = ["scaler.pkl", "best_scaler.pkl"]
PAYLOAD_CANDIDATES    = ["payload.json", "best_payload.json"]
TRAIN_DIST_CANDIDATES = ["train_distribution.json", "best_train_distribution.json"]

def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_bundle() -> Dict[str, Any]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    bundle: Dict[str, Any] = {"model": model}
    spath = _first_existing(SCALER_CANDIDATES)
    if spath:
        try:
            bundle["scaler"] = joblib.load(spath)
            logging.info(f"Scaler loaded: {spath}")
        except Exception as e:
            logging.warning(f"Failed to load scaler from {spath}: {e}")

    ppath = _first_existing(PAYLOAD_CANDIDATES)
    if ppath:
        with open(ppath, "r") as f:
            payload = json.load(f)
        bundle.update(payload)
        logging.info(f"Payload loaded: {ppath}")

    tpath = _first_existing(TRAIN_DIST_CANDIDATES)
    if tpath:
        with open(tpath, "r") as f:
            bundle["train_distribution"] = json.load(f)
        logging.info(f"Train distribution loaded: {tpath}")

    bundle["window_size"] = int(bundle.get("window_size", 1))
    bundle["pos_thr"] = float(bundle.get("pos_thr", 0.995))
    bundle["neg_thr"] = float(bundle.get("neg_thr", 0.005))

    feats = bundle.get("selected_features", [])
    if not feats and hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
    bundle["selected_features"] = feats

    return bundle

def live_files_exist() -> bool:
    return all(os.path.exists(p) for p in LIVE_FILEPATHS.values())

def remove_live_files():
    for p in LIVE_FILEPATHS.values():
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

def write_answer(ans: str):
    with open(ANSWER_FILE, "w") as f:
        f.write(ans)

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

def align_X_to_features(X_df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    if not feats:
        return X_df
    for c in feats:
        if c not in X_df.columns:
            X_df[c] = 0.0
    return X_df[feats].copy()

def decide_signal(prob: float, pos_thr: float, neg_thr: float) -> str:
    if prob >= pos_thr: return "BUY"
    if prob <= neg_thr: return "SELL"
    return "NONE"

def main():
    logging.info("=== prediction_in_production started ===")
    bundle = load_bundle()
    model   = bundle["model"]
    scaler  = bundle.get("scaler", None)
    feats   = bundle.get("selected_features", [])
    window  = int(bundle.get("window_size", 1))
    pos_thr = float(bundle.get("pos_thr", 0.995))
    neg_thr = float(bundle.get("neg_thr", 0.005))

    logging.info(f"Model ready | window={window} | feats={len(feats)} | "
                 f"neg_thr={neg_thr} | pos_thr={pos_thr}")

    # آماده‌سازی کلاس روی فایل‌های لایو؛ بدون drift-scan
    prep_live = PREPARE_DATA_FOR_TRAIN(
        filepaths=LIVE_FILEPATHS,
        main_timeframe="30T",
        verbose=False,
        fast_mode=True,         # سریع و بدون drift scan
        strict_disk_feed=True,  # فقط همین فایل‌ها، بدون برشِ تاریخ
    )

    while True:
        # منتظر باش تا answer.txt وجود نداشته باشد و هر 4 CSV حاضر باشند
        if os.path.exists(ANSWER_FILE) or not live_files_exist():
            time.sleep(1.0)
            continue

        try:
            # همان خطِ آموزش: بخوان، فیچر بساز، ادغام کن
            merged_live = prep_live.load_data()
            if merged_live is None or len(merged_live) == 0:
                write_answer("NONE"); remove_live_files(); time.sleep(1.0); continue

            # آماده‌سازی برای پیش‌بینی (y لازم نیست)
            X_all, _, _, _, _ = prep_live.ready(
                merged_live, window=window, selected_features=feats if feats else [],
                mode="predict", with_times=True, predict_drop_last=True
            )
            X_df = pd.DataFrame(X_all)
            X_df = align_X_to_features(X_df, feats)

            if len(X_df) == 0:
                write_answer("NONE"); remove_live_files(); time.sleep(1.0); continue

            x_last = X_df.tail(1).values
            if scaler is not None:
                x_last = scaler.transform(x_last)

            prob = float(model.predict_proba(x_last)[:, 1])
            ans = decide_signal(prob, pos_thr, neg_thr)
            write_answer(ans)
            logging.info(f"[LIVE PRED] prob={prob:.6f} → {ans}")

        except Exception as e:
            logging.exception(f"Error in prediction loop: {e}")
            try:
                write_answer("NONE")
            except Exception:
                pass

        remove_live_files()
        time.sleep(1.0)

if __name__ == "__main__":
    main()
