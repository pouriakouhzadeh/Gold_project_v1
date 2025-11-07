#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, logging, warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

LOG_FILE = "live_like_sim_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")]
)

# ---------- I/O BUNDLE ----------
MODEL_PATH = "best_model.pkl"
SCALER_CANDIDATES     = ["scaler.pkl", "best_scaler.pkl"]
PAYLOAD_CANDIDATES    = ["payload.json", "best_payload.json"]
TRAIN_DIST_CANDIDATES = ["train_distribution.json", "best_train_distribution.json"]

# ---------- DATA FILES ----------
FILEPATHS = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T" : "XAUUSD_M5.csv",
    "1H" : "XAUUSD_H1.csv",
}

LAST_N = 2000  # تعداد ردیف‌های آخر برای تست

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

    # thresholds & window defaults
    bundle["window_size"] = int(bundle.get("window_size", 1))
    bundle["pos_thr"] = float(bundle.get("pos_thr", 0.995))
    bundle["neg_thr"] = float(bundle.get("neg_thr", 0.005))

    # use model's feature_names_in_ if present
    feats = bundle.get("selected_features", [])
    if not feats and hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
    bundle["selected_features"] = feats

    return bundle

def decide_signal(prob: float, pos_thr: float, neg_thr: float) -> str:
    if prob >= pos_thr: return "BUY"
    if prob <= neg_thr: return "SELL"
    return "NONE"

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

def align_X_to_features(X_df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    """
    اگر لیست فیچرها داده شده باشد، X را دقیقاً به همان ترتیب هم‌تراز می‌کنیم.
    ستون‌هایِ کم‌بود = 0.0؛ ستون‌های اضافه حذف می‌شوند.
    """
    if not feats:
        return X_df
    for c in feats:
        if c not in X_df.columns:
            X_df[c] = 0.0
    X_df = X_df[feats].copy()
    return X_df

def main():
    logging.info("=== live_like_sim_v3 starting ===")
    bundle = load_bundle()
    model   = bundle["model"]
    scaler  = bundle.get("scaler", None)
    feats   = bundle.get("selected_features", [])
    window  = int(bundle.get("window_size", 1))
    pos_thr = float(bundle.get("pos_thr", 0.995))
    neg_thr = float(bundle.get("neg_thr", 0.005))

    logging.info(f"Model loaded | window={window} | feats={len(feats)} | "
                 f"neg_thr={neg_thr} | pos_thr={pos_thr}")

    # آماده‌سازی با نام‌گذاری فایل‌های شما
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths=FILEPATHS,
        main_timeframe="30T",
        verbose=True,
        fast_mode=False,          # در سیم‌لوشن: اجازهٔ drift-scan و … طبق کلاس
        strict_disk_feed=False,
    )

    merged = prep.load_data()  # شامل مهندسی ویژگی‌ها و ادغام 4 تایم‌فریم
    logging.info(f"[load_data] merged shape={merged.shape}")

    # آخرین LAST_N ردیفِ تایم‌فریم اصلی را معیاری می‌گیریم
    if len(merged) > LAST_N:
        merged = merged.tail(LAST_N).copy()
    logging.info(f"Using last_n={len(merged)} rows for both modes")

    # ---------- Chimney (Batch) : mode='train' تا y واقعی داشته باشیم ----------
    Xc, yc, feats_c, price_c, t_c = prep.ready(
        merged, window=window, selected_features=feats if feats else [], mode="train", with_times=True
    )

    # هم‌ترازی ستونی با مدل (اگر feature_names_in_ داشت)
    Xc = pd.DataFrame(Xc)
    Xc = align_X_to_features(Xc, feats)

    # اسکیل (در صورت وجود)
    if scaler is not None:
        Xc_vals = scaler.transform(Xc.values)
    else:
        Xc_vals = Xc.values

    probs_c = model.predict_proba(Xc_vals)[:, 1]
    preds_c = np.array([decide_signal(p, pos_thr, neg_thr) for p in probs_c])

    ydir = np.where(yc > 0, "BUY", "SELL")
    correct = np.sum((preds_c != "NONE") & (preds_c == ydir))
    incorrect = np.sum((preds_c != "NONE") & (preds_c != ydir))
    unpred = np.sum(preds_c == "NONE")
    cover = 1.0 - (unpred / len(preds_c)) if len(preds_c) else 0.0
    acc = correct / max(1, (correct + incorrect))
    logging.info(f"[Chimney-Batch] size={len(preds_c)} cover={cover:.3f} acc={acc:.3f} "
                 f"Correct={correct} Incorrect={incorrect} Unpred={unpred}")

    # ---------- Live-like (گام به گام با همان داده) ----------
    logging.info("[Step] Live-like step-by-step")
    preds_l = []
    probs_l = []
    for i in range(len(Xc)):
        x_last = Xc.iloc[i:i+1].values
        if scaler is not None:
            x_last = scaler.transform(x_last)
        prob = float(model.predict_proba(x_last)[:, 1])
        probs_l.append(prob)
        preds_l.append(decide_signal(prob, pos_thr, neg_thr))
        if (i + 1) % 200 == 0:
            logging.info(f"  live step {i+1}/{len(Xc)}")

    preds_l = np.array(preds_l)
    correct2 = np.sum((preds_l != "NONE") & (preds_l == ydir[:len(preds_l)]))
    incorrect2 = np.sum((preds_l != "NONE") & (preds_l != ydir[:len(preds_l)]))
    unpred2 = np.sum(preds_l == "NONE")
    cover2 = 1.0 - (unpred2 / len(preds_l)) if len(preds_l) else 0.0
    acc2 = correct2 / max(1, (correct2 + incorrect2))
    logging.info(f"[Final] Live : acc={acc2:.3f} cover={cover2:.3f} "
                 f"Correct={correct2} Incorrect={incorrect2} Unpred={unpred2}")

    agree = np.mean(preds_c[:len(preds_l)] == preds_l) if len(preds_l) else 0.0
    logging.info(f"[Final] Pred agreement (chimney vs live): {agree:.3f}")

    out = pd.DataFrame({
        "time": pd.to_datetime(t_c) if t_c is not None else pd.NaT,
        "dir": ydir[:len(preds_c)],
        "chimney_prob": probs_c,
        "chimney_pred": preds_c,
        "live_prob": probs_l + [np.nan]*(len(preds_c)-len(probs_l)),
        "live_pred": list(preds_l) + ["NONE"]*(len(preds_c)-len(preds_l)),
    })
    out.to_csv("predictions_compare_v3.csv", index=False)
    logging.info("[CSV] predictions_compare_v3.csv written")
    logging.info("=== live_like_sim_v3 completed ===")

if __name__ == "__main__":
    main()
