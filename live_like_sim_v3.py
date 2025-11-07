#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, logging, warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

LOG_FILE = "live_like_sim_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)

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

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

def decide_signal(prob: float, pos_thr: float, neg_thr: float) -> str:
    if prob >= pos_thr:
        return "BUY"
    if prob <= neg_thr:
        return "SELL"
    return "NONE"

def main():
    logging.info("=== live_like_sim_v3 starting ===")
    bundle = load_bundle_flexible()
    model  = bundle["model"]
    scaler = bundle.get("scaler", None)
    feats: List[str] = bundle.get("selected_features", [])
    window = int(bundle.get("window_size", 1))
    pos_thr = float(bundle.get("pos_thr", 0.995))
    neg_thr = float(bundle.get("neg_thr", 0.005))

    logging.info(f"Model loaded | window={window} | feats={len(feats)} | "
                 f"neg_thr={neg_thr} | pos_thr={pos_thr}")

    # بدون csv_map (طبق امضای کلاس شما)
    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        drift_guard=True,
        remove_holidays=True,
        log_progress=True,
    )

    raw = prep.load_data()
    raw = prep.cut_last_n(raw, last_n=2000)
    logging.info(f"Raw loaded: shape={raw.shape} | Using last_n=2000")

    feat_df = prep.build_features(raw)
    # اگر feats خالی بود، همه را نگاه می‌داریم
    if feats:
        missing = [c for c in feats if c not in feat_df.columns]
        if missing:
            logging.warning(f"Missing features in batch set: {missing}")
        cols = [c for c in feats if c in feat_df.columns]
        feat_df = feat_df[cols + ["target"]].copy()

    X_all = feat_df.drop(columns=["target"]).values
    if scaler is not None:
        X_all = scaler.transform(X_all)
    y_all = feat_df["target"].values

    Xc, yc, idxc = prep.make_windowed_arrays(X_all, y_all, window=window, return_index=True)
    logging.info("[Step] Evaluate Chimney (batch) on the last-N segment")

    probs_c = bundle["model"].predict_proba(Xc)[:, 1]
    preds_c = np.array([decide_signal(p, pos_thr, neg_thr) for p in probs_c])

    ydir = np.where(yc > 0, "BUY", "SELL")
    correct = np.sum((preds_c != "NONE") & (preds_c == ydir))
    incorrect = np.sum((preds_c != "NONE") & (preds_c != ydir))
    unpred = np.sum(preds_c == "NONE")
    cover = 1.0 - (unpred / len(preds_c))
    acc = correct / max(1, (correct + incorrect))
    logging.info(f"[Chimney-Batch] size={len(preds_c)} cover={cover:.3f} acc={acc:.3f} "
                 f"Correct={correct} Incorrect={incorrect} Unpred={unpred}")

    logging.info("[Step] Live-like step-by-step simulation")
    preds_l, probs_l = [], []
    for k, row_idx in enumerate(idxc):
        start = row_idx - window + 1
        end   = row_idx + 1
        X_last = X_all[start:end]
        if len(X_last) != window:
            continue
        prob = float(model.predict_proba(X_last.reshape(1, -1))[:, 1])
        preds_l.append(decide_signal(prob, pos_thr, neg_thr))
        probs_l.append(prob)
        if (k + 1) % 100 == 0:
            logging.info(f"  live step @{k+1}/{len(idxc)}")

    preds_l = np.array(preds_l)
    y_l = ydir[:len(preds_l)]
    correct2 = np.sum((preds_l != "NONE") & (preds_l == y_l))
    incorrect2 = np.sum((preds_l != "NONE") & (preds_l != y_l))
    unpred2 = np.sum(preds_l == "NONE")
    cover2 = 1.0 - (unpred2 / len(preds_l))
    acc2 = correct2 / max(1, (correct2 + incorrect2))
    logging.info(f"[Final] Live    : acc={acc2:.3f} cover={cover2:.3f} "
                 f"Correct={correct2} Incorrect={incorrect2} Unpred={unpred2}")

    min_len = min(len(preds_c), len(preds_l))
    agree = np.mean(preds_c[:min_len] == preds_l[:min_len])
    logging.info(f"[Final] Pred agreement (chimney vs live): {agree:.3f}")

    out_pred = pd.DataFrame({
        "idx": idxc[:min_len],
        "chimney_pred": preds_c[:min_len],
        "chimney_prob": probs_c[:min_len],
        "live_pred": preds_l[:min_len],
        "live_prob": probs_l[:min_len],
        "dir": ydir[:min_len],
    })
    out_pred.to_csv("predictions_compare_v3.csv", index=False)
    logging.info("[CSV] predictions_compare_v3.csv written")
    logging.info("=== live_like_sim_v3 completed ===")

if __name__ == "__main__":
    main()
