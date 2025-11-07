#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
live_like_sim_v3.py
- تست یک‌تکه: چیمینی (batch) vs لایو (قدم‌به‌قدم) روی N سطر آخر
- هم‌تراز با PREP آموزش (resample/holidays/labeling)
- خروجی‌های مرجع "سطرهای فیچر آخر" را Dump می‌کند تا با دولوپ مقایسه کنیم.
"""

import os, json, time, logging, joblib, warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG = logging.getLogger("v3")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# -------------------- تنظیمات --------------------
TF_FILES = {
    "5T" : "XAUUSD_M5.csv",
    "15T": "XAUUSD_M15.csv",
    "30T": "XAUUSD_M30.csv",
    "1H" : "XAUUSD_H1.csv",
}
MAIN_TF = "30T"
LAST_N = 2000   # چند سطر آخر برای تست
DUMP_DIR = "trace_chimney_rows"  # برای مقایسه با دولوپی
os.makedirs(DUMP_DIR, exist_ok=True)

MODEL_BUNDLE = "model_bundle.pkl"  # اگر نبود، می‌رویم سراغ model.pkl + payload.json
MODEL_PKL    = "model.pkl"
PAYLOAD_JSON = "payload.json"
TRAIN_DIST   = "train_distribution.json"   # میانگین/مد برای پرکردن NaN و ستون‌های ناموجود

# -------------------------------------------------

def load_model_bundle():
    """باندل مدل/اسکیلر/لیست فیچر/پنجره/آستانه‌ها را بارگذاری می‌کند."""
    if os.path.exists(MODEL_BUNDLE):
        b = joblib.load(MODEL_BUNDLE)
        return b
    # fallback
    bundle = {}
    if os.path.exists(MODEL_PKL):
        bundle["model"] = joblib.load(MODEL_PKL)
    if os.path.exists(PAYLOAD_JSON):
        with open(PAYLOAD_JSON, "r") as f:
            payload = json.load(f)
        bundle.update(payload)
    if os.path.exists(TRAIN_DIST):
        with open(TRAIN_DIST, "r") as f:
            bundle["train_distribution"] = json.load(f)
    return bundle

def align_columns(X, feats, train_dist=None):
    """ستون‌ها را دقیقا به ترتیب feats می‌چیند. اگر ستون کم باشد:
       - اول با train_dist['mean'][col] پر می‌کنیم (اگر موجود باشد)
       - وگرنه، عملیات را «NONE» می‌کنیم (با raise).
    """
    missing = [c for c in feats if c not in X.columns]
    if missing:
        if train_dist and "mean" in train_dist:
            for c in missing:
                fillv = train_dist["mean"].get(c, None)
                if fillv is None:
                    raise ValueError(f"Missing feature {c} without mean in train_distribution")
                X[c] = fillv
        else:
            raise ValueError(f"Missing features {missing} and no train_distribution to fill.")
    X = X[feats]
    return X

def dump_last_row_features(X_last, ts, tag):
    """سطر آخر را برای مقایسه ذخیره می‌کند."""
    fn = os.path.join(DUMP_DIR, f"{tag}_{str(ts).replace(':','-')}.csv")
    X_last.assign(_ts=str(ts)).to_csv(fn, index=False)

def main():
    LOG.info("=== live_like_sim_v3 starting ===")
    bundle = load_model_bundle()
    model  = bundle.get("model", None)
    scaler = bundle.get("scaler", None)
    feats  = bundle.get("selected_features") or bundle.get("feats") or []
    window = int(bundle.get("window_size", 1))
    pos_thr = float(bundle.get("pos_thr", 0.995))
    neg_thr = float(bundle.get("neg_thr", 0.005))
    train_dist = bundle.get("train_distribution", None)

    if model is None or scaler is None or not feats:
        raise RuntimeError("Model/scaler/feature list not found. Provide model_bundle.pkl or model.pkl+payload.json.")

    LOG.info(f"Model loaded | window={window} | cols={len(feats)} | neg_thr={neg_thr} | pos_thr={pos_thr}")

    # PREP مثل آموزش
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={
            "30T": TF_FILES["30T"],
            "15T": TF_FILES["15T"],
            "5T" : TF_FILES["5T"],
            "1H" : TF_FILES["1H"],
        },
        main_timeframe="30T",
        fast_mode=False,
        strict_disk_feed=False,   # تمام دیتاست آموزش را می‌خوانیم
        resample_label="right",
        resample_closed="right"
    )

    # داده خام → آماده
    raw = prep.load_data()
    LOG.info(f"[load_data] shape={raw.shape}")

    Xall, yall, tall, extra = prep.ready(
        raw,
        window=window,
        selected_features=feats,
        mode="train",            # همان مسیر آموزش (تا Y ها بسازیم)
        with_times=True,
        predict_drop_last=False
    )
    # بخش آخر برای تست
    Xseg = Xall.tail(LAST_N + 1).copy()
    yseg = yall[-(LAST_N + 1):].copy()
    tseg = tall[-(LAST_N + 1):].copy()

    # --------- چیمینی (Batch) روی همان بازه ----------
    LOG.info("[Step] Evaluate Chimney (batch) on the last-N segment")
    # ردیف آخر ورودی، تارگتش در y بعدی است؛ با predict_drop_last=False این هم‌ترازی حفظ شده
    # لذا برای ارزیابی یک قدم جلو می‌زنیم
    Xb = Xseg.iloc[:-1].copy()
    yb = yseg[1:].copy()
    tb = tseg[1:].copy()

    # اسکیل
    Xb_tr = pd.DataFrame(scaler.transform(Xb.values), columns=Xb.columns, index=Xb.index)
    prob = model.predict_proba(Xb_tr)[:, 1]
    # اعمال آستانه‌ها
    pred = np.where(prob >= pos_thr, 1,
           np.where(prob <= neg_thr, 0, -1))

    cover = np.mean(pred != -1)
    mask  = pred != -1
    acc   = np.mean((pred[mask] == yb[mask])) if cover > 0 else np.nan
    bacc  = acc  # اگر کلاس‌بندی نامتوازن داری، balanced_accuracy را جدا بساز

    correct  = int(((pred == yb) & (pred != -1)).sum())
    incorrect= int(((pred != yb) & (pred != -1)).sum())
    unpred   = int((pred == -1).sum())
    LOG.info(f"[Chimney-Batch] size={len(yb)} cover={cover:.3f} acc={acc:.3f} bAcc={bacc:.3f} "
             f"Correct={correct} Incorrect={incorrect} Unpred={unpred}")

    # Dump چند سطر آخر برای مقایسه (مرجع)
    for idx in tb.tail(20).index:
        X_last = Xb.loc[[idx]].copy()
        dump_last_row_features(X_last, tb.loc[idx], "chimney_lastrow")

    # --------- لایو قدم‌به‌قدم روی همان بازه ----------
    LOG.info("[Step] Live-like step-by-step simulation")
    wins=loses=none=0
    agree=0
    steps=0
    for i in range(len(tb)):
        steps += 1
        ts = tb.iloc[i]
        # همان ردیفِ i در Xb
        row = Xb.iloc[[i]].copy()
        try:
            # هم‌ترازی ستون‌ها (در این مسیر batch از قبل OK است؛ برای یکسانی این را می‌گذاریم)
            row2 = align_columns(row, feats, train_dist)
            row2 = pd.DataFrame(scaler.transform(row2.values), columns=row2.columns, index=row2.index)
        except Exception as e:
            LOG.warning(f"[Live step {i}] align/scale error → NONE ({e})")
            none += 1
            continue

        # اثرنگاری برای همین timestamp
        dump_last_row_features(row2, ts, "live_lastrow")

        pr = model.predict_proba(row2.values)[:,1][0]
        p  = 1 if pr >= pos_thr else (0 if pr <= neg_thr else -1)

        if p == -1:
            none += 1
        else:
            if p == yb.iloc[i]:
                wins += 1
            else:
                loses += 1

        # تطابق با خروجی batch همان اندیس
        pg = pred[i]
        if pg == p:
            agree += 1

        if (i+1) % 200 == 0:
            LOG.info(f"  live cut @ {i+1}/{len(tb)}")

    cover_live = (wins+loses)/max(1,len(tb))
    acc_live   = wins/max(1,(wins+loses))
    LOG.info(f"[Final] Chimney: acc={acc:.3f} bAcc={bacc:.3f} cover={cover:.3f} "
             f"Correct={correct} Incorrect={incorrect} Unpred={unpred}")
    LOG.info(f"[Final] Live    : acc={acc_live:.3f} cover={cover_live:.3f} "
             f"WINS={wins} LOSES={loses} NONE={none}")
    LOG.info(f"[Final] Pred agreement (chimney vs live): {agree/steps:.3f}")
    LOG.info("=== live_like_sim_v3 completed ===")

if __name__ == "__main__":
    main()
