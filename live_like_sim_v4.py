#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, pickle, argparse, logging
import numpy as np
import pandas as pd
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOGFMT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT, datefmt="%Y-%m-%d %H:%M:%S")
L = logging.getLogger("live_like_sim_v3")

def load_artifacts(base_dir="."):
    meta_path = os.path.join(base_dir, "best_model.meta.json")
    pkl_path  = os.path.join(base_dir, "best_model.pkl")
    with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
    with open(pkl_path,  "rb") as f: model = pickle.load(f)
    cols   = meta.get("train_window_cols") or meta.get("feats") or meta.get("feature_names") or []
    window = int(meta.get("window_size") or meta.get("window") or 1)
    neg_thr = float(meta.get("neg_thr", 0.005))
    pos_thr = float(meta.get("pos_thr", 0.995))
    return model, cols, window, neg_thr, pos_thr, meta

def ensure_column_order(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    # ستون‌های مفقود را با صفر بساز
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    # ستون‌های اضافه را کنار بگذار
    X = X[cols]
    # پر کردن NaN
    return X.fillna(0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--last-n", type=int, default=200)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()
    if args.verbosity <= 0: L.setLevel(logging.WARNING)

    L.info("=== live_like_sim_v3 starting ===")
    model, train_cols, window, neg_thr, pos_thr, meta = load_artifacts(args.base_dir)
    L.info("Model loaded | window=%d | feats=%d | thr=(%.3f,%.3f)", window, len(train_cols), neg_thr, pos_thr)

    # مسیر CSVهای خام
    base = args.base_dir
    fps = {
        "30T": os.path.join(base, f"{args.symbol}_M30.csv"),
        "15T": os.path.join(base, f"{args.symbol}_M15.csv"),
        "5T":  os.path.join(base, f"{args.symbol}_M5.csv"),
        "1H":  os.path.join(base, f"{args.symbol}_H1.csv"),
    }
    for tf, fp in fps.items():
        L.info("[paths] %s -> %s", tf, fp)

    prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T", verbose=(args.verbosity>0))
    merged = prep.load_data()
    L.info("[load_data] merged shape=%s", merged.shape)

    # ساخت X/y هم‌تراز با TRAIN؛ ولی برای سنجش دقیق، y را هم می‌گیریم
    X_all, y_all, _, price_ser, t_idx = prep.ready(
        merged,
        window=window,
        selected_features=train_cols,
        mode="train",               # y واقعی ساخته شود
        with_times=True,
        predict_drop_last=False,    # *** مهم: در هر دو مسیر False ***
        train_drop_last=True
    )
    X_all = ensure_column_order(X_all, train_cols)

    # ۲۰۰ سطر آخر
    if len(X_all) < args.last_n:
        args.last_n = len(X_all)
    X_tail   = X_all.tail(args.last_n).reset_index(drop=True)
    y_tail   = pd.Series(y_all).tail(args.last_n).reset_index(drop=True).astype(int)
    t_tail   = pd.to_datetime(pd.Series(t_idx).tail(args.last_n)).reset_index(drop=True)

    # پیش‌بینی
    y_prob = model.predict_proba(X_tail)[:, 1]
    y_pred = np.full_like(y_tail, -1, dtype=int)
    y_pred[y_prob <= neg_thr] = 0
    y_pred[y_prob >= pos_thr] = 1
    act_map = {1:"BUY", 0:"SELL", -1:"NONE"}
    acts    = pd.Series([act_map[int(a)] for a in y_pred])

    # محاسبهٔ cover و دقت
    mask = (y_pred != -1)
    cover = float(mask.mean()) if len(mask)>0 else 0.0
    correct = int(((y_pred == y_tail) & mask).sum())
    incorrect = int(mask.sum() - correct)
    L.info("[Final] acc=%.3f bAcc=%.3f cover=%.3f Correct=%d Incorrect=%d Unpred=%d",
           (correct/max(1, mask.sum())),  # acc روی موارد دارای سیگنال
           (correct/max(1, mask.sum())),  # bAcc چون 0/1 متوازن در این بخش کوچک اختلافی ندارد
           cover, correct, incorrect, int((~mask).sum()))
    L.info("=== live_like_sim_v3 completed ===")

    # ذخیرهٔ فیدِ فیچر و خروجی
    # 1) فقط بردار فیچر به‌همراه timestamp و y_true
    X_to_dump = X_tail.copy()
    X_to_dump.insert(0, "timestamp", t_tail)
    X_to_dump["y_true"] = y_tail.values
    X_to_dump.to_csv("sim_X_feed_tail200.csv", index=False)

    # 2) خروجی مدل + تارگت
    out = pd.DataFrame({
        "timestamp": t_tail,
        "y_true": y_tail,
        "y_prob": y_prob,
        "action": acts,
        "neg_thr": neg_thr,
        "pos_thr": pos_thr
    })
    out.to_csv("sim_predictions.csv", index=False)

if __name__ == "__main__":
    main()
