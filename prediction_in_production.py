#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, logging, warnings
from typing import Dict, Any, List
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
CSV_MAP_LIVE = {
    "5T":  "XAUUSD_M5_live.csv",
    "15T": "XAUUSD_M15_live.csv",
    "30T": "XAUUSD_M30_live.csv",
    "1H":  "XAUUSD_H1_live.csv",
}

# ===== Flexible loader (best_model.pkl first) =====
MODEL_PKL_CANDIDATES  = ["best_model.pkl", "model.pkl"]
SCALER_CANDIDATES     = ["scaler.pkl", "best_scaler.pkl"]
PAYLOAD_CANDIDATES    = ["payload.json", "best_payload.json"]
TRAIN_DIST_CANDIDATES = ["train_distribution.json", "best_train_distribution.json"]

def _first_existing(paths: List[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_bundle_flexible() -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}
    mpath = _first_existing(MODEL_PKL_CANDIDATES)
    if not mpath:
        raise FileNotFoundError(f"Model file not found. Tried: {MODEL_PKL_CANDIDATES}")
    bundle["model"] = joblib.load(mpath)

    spath = _first_existing(SCALER_CANDIDATES)
    if spath:
        try:
            bundle["scaler"] = joblib.load(spath)
            logging.info(f"Scaler loaded: {spath}")
        except Exception as e:
            logging.warning(f"Could not load scaler from {spath}: {e}")

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

    bundle.setdefault("selected_features", bundle.get("feats", []))
    bundle["window_size"] = int(bundle.get("window_size", 1))
    bundle["pos_thr"] = float(bundle.get("pos_thr", 0.995))
    bundle["neg_thr"] = float(bundle.get("neg_thr", 0.005))
    return bundle

# ===== Data prep (use your existing pipeline) =====
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

def decide_signal(prob: float, pos_thr: float, neg_thr: float) -> str:
    if prob >= pos_thr:
        return "BUY"
    if prob <= neg_thr:
        return "SELL"
    return "NONE"

def live_files_exist() -> bool:
    return all(os.path.exists(v) for v in CSV_MAP_LIVE.values())

def remove_live_files():
    for v in CSV_MAP_LIVE.values():
        try:
            os.remove(v)
        except FileNotFoundError:
            pass

def write_answer(ans: str):
    with open(ANSWER_FILE, "w") as f:
        f.write(ans)

def main():
    logging.info("=== prediction_in_production started ===")
    bundle = load_bundle_flexible()
    model  = bundle["model"]
    scaler = bundle.get("scaler", None)
    feats: List[str] = bundle.get("selected_features", [])
    window = int(bundle.get("window_size", 1))
    pos_thr = float(bundle.get("pos_thr", 0.995))
    neg_thr = float(bundle.get("neg_thr", 0.005))

    logging.info(f"Model ready | window={window} | feats={len(feats)} | "
                 f"neg_thr={neg_thr} | pos_thr={pos_thr}")

    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        csv_map=None,             # در این اسکریپت از فایل‌های _live می‌خوانیم
        drift_guard=False,
        remove_holidays=True,
        log_progress=False
    )

    while True:
        # شرط اجرا: answer.txt وجود نداشته باشد و هر ۴ CSV لایو وجود داشته باشد
        if os.path.exists(ANSWER_FILE) or not live_files_exist():
            time.sleep(1.0)
            continue

        try:
            # 1) خواندن ۴ CSV لایو
            dfs = {}
            for tf, fname in CSV_MAP_LIVE.items():
                df = pd.read_csv(fname)
                if df.empty:
                    raise ValueError(f"{fname} is empty")
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time").sort_index()
                dfs[tf] = df

            # 2) ادغام و ری‌سمپل مثل pipeline اصلی
            raw = prep.merge_live_frames(dfs)     # <- باید در کلاس شما پیاده باشد (هم‌ارز merge/align قبلی)
            raw = prep.drop_holidays(raw)         # اگر روز تعطیل بود None برگردان
            if raw is None or len(raw) == 0:
                write_answer("NONE")
                remove_live_files()
                time.sleep(1.0)
                continue

            # 3) ساخت فیچرها
            feat_df = prep.build_features(raw)
            if feats:
                missing = [c for c in feats if c not in feat_df.columns]
                if missing:
                    logging.warning(f"Missing features in live batch: {missing}")
                cols = [c for c in feats if c in feat_df.columns]
                feat_df = feat_df[cols].copy()
            # در لایو هدف را نداریم

            # 4) اسکِیل و پنجره‌ی آخر
            X_all = feat_df.values
            if scaler is not None:
                X_all = scaler.transform(X_all)

            if len(X_all) < window:
                # با پنجره‌ی ناکافی نمی‌توان پیش‌بینی کرد
                write_answer("NONE")
                remove_live_files()
                time.sleep(1.0)
                continue

            X_last = X_all[-window:].reshape(1, -1)
            prob = float(model.predict_proba(X_last)[:, 1])
            ans = decide_signal(prob, pos_thr, neg_thr)

            write_answer(ans)
            logging.info(f"[LIVE PRED] prob={prob:.6f} → {ans}")

        except Exception as e:
            logging.exception(f"Error in prediction loop: {e}")
            # برای ایمن‌سازی، NONE خروجی بده تا ژنراتور متوقف نشود
            try:
                write_answer("NONE")
            except Exception:
                pass

        # 5) پاک کردن CSVهای لایو (قدم تمام شد)
        remove_live_files()
        # 6) اندکی تاخیر برای کاهش فشار CPU
        time.sleep(1.0)

if __name__ == "__main__":
    main()
