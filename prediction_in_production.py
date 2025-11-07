#!/usr/bin/env python3
# prediction_in_production.py
from __future__ import annotations
import os, time, json, logging
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from ModelSaver import ModelSaver

# ---------- تنظیمات ----------
LIVE_SUFFIX  = "_live"
LIVE_FILES   = {
    "M5" : "XAUUSD_M5_live.csv",
    "M15": "XAUUSD_M15_live.csv",
    "M30": "XAUUSD_M30_live.csv",
    "H1" : "XAUUSD_H1_live.csv",
}
ANSWER_FILE  = "answer.txt"
LOG_FILE     = "production_predict.log"

# ---------- لاگینگ کنسول + فایل ----------
LOGGER = logging.getLogger("predictor")
LOGGER.setLevel(logging.INFO)
for h in list(LOGGER.handlers): LOGGER.removeHandler(h)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
fh  = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(fmt)
sh  = logging.StreamHandler()
sh.setFormatter(fmt)
LOGGER.addHandler(fh); LOGGER.addHandler(sh)

def _all_live_present() -> bool:
    return all(os.path.isfile(p) for p in LIVE_FILES.values())

def _no_answer_yet() -> bool:
    return not os.path.isfile(ANSWER_FILE)

def _safe_load_train_dist(path: str) -> dict[str, float] | None:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _align_columns(X: pd.DataFrame, train_cols: list[str], dist: dict[str, float] | None) -> pd.DataFrame:
    # ستون‌های مفقود را با میانه‌ی آموزش (در صورت وجود) یا 0 پر می‌کنیم
    missing = [c for c in train_cols if c not in X.columns]
    if missing:
        fill_vals = {c: (dist.get(c, 0.0) if dist else 0.0) for c in missing}
        for c, v in fill_vals.items():
            X[c] = v
    # ستون‌های اضافی را حذف کن
    X = X[train_cols]
    # NaN را هم نهایی پر کن
    if dist:
        med = pd.Series({c: dist.get(c, 0.0) for c in X.columns})
        X = X.fillna(med)
    else:
        X = X.fillna(0.0)
    return X

def _predict_one_step(prep: PREPARE_DATA_FOR_TRAIN, model_payload: dict) -> str:
    """
    برگشتی: 'BUY' / 'SELL' / 'NONE'
    """
    window      = int(model_payload.get("window_size", 1))
    train_cols  = list(model_payload.get("train_window_cols") or model_payload.get("feats") or [])
    neg_thr     = float(model_payload.get("neg_thr", 0.0))
    pos_thr     = float(model_payload.get("pos_thr", 1.0))
    pipeline    = model_payload["pipeline"]   # _CompatWrapper → predict_proba
    dist_name   = model_payload.get("train_distribution", "train_distribution.json")
    dist_dict   = _safe_load_train_dist(dist_name)

    # ۱) ادغام و فیچرینگ بدون drift/trim
    raw = prep.load_data()  # با strict_disk_feed=True چیزی trim نمی‌شود
    # حذف تعطیلات (شنبه/یکشنبه) در خود PREP انجام می‌شود

    # ۲) آماده‌سازی به شکل "predict"
    #    selected_features = train_cols تا دقیقا همان ستون‌ها ساخته شود
    X, _, _, _ = prep.ready(
        raw, window=window,
        selected_features=train_cols if train_cols else [],
        mode="predict",
        with_times=False,
        predict_drop_last=True  # آخرین ردیف ناقص را حذف کن
    )
    if X.empty:
        return "NONE"

    # ۳) آخرین ردیف برای پیش‌بینی
    X_last = X.tail(1).copy()

    # ۴) هم‌ترازسازی ستون‌ها (ترتیب و پرکردن مفقودی‌ها)
    if train_cols:
        X_last = _align_columns(X_last, train_cols, dist_dict)

    # ۵) پیش‌بینی + نگاشت آستانه‌ها
    probs = pipeline.predict_proba(X_last.values)[:, 1]
    p = float(probs[0])
    if p >= pos_thr: return "BUY"
    if p <= neg_thr: return "SELL"
    return "NONE"

def main():
    LOGGER.info("=== Predictor started ===")

    # مدل را بارگذاری کن
    saver   = ModelSaver()  # پیش‌فرض: best_model.pkl در cwd
    payload = saver.load_full()  # dict شامل: pipeline, window_size, train_window_cols, neg_thr, pos_thr, train_distribution, ...
    LOGGER.info(f"Model loaded | window={payload.get('window_size')} | cols={len(payload.get('train_window_cols') or payload.get('feats') or [])} | neg_thr={payload.get('neg_thr')} | pos_thr={payload.get('pos_thr')}")

    # PREP برای لایو: هیچ drift/trim خودکاری نداشته باشیم
    filepaths = {
        "30T": LIVE_FILES["M30"],
        "15T": LIVE_FILES["M15"],
        "5T" : LIVE_FILES["M5"],
        "1H" : LIVE_FILES["H1"],
    }
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths=filepaths,
        main_timeframe="30T",
        verbose=False,
        fast_mode=True,          # ← هیچ drift-scan/trim
        strict_disk_feed=True    # ← هیچ برش خودکار
    )

    iter_no = 0
    while True:
        # شرط آماده‌بودن: ۴ فایل حاضر و answer.txt حذف
        if _no_answer_yet() and _all_live_present():
            iter_no += 1
            try:
                ans = _predict_one_step(prep, payload)
            except Exception as e:
                LOGGER.error(f"[Iter {iter_no}] prediction failed: {e}")
                ans = "NONE"

            # نوشتن جواب
            try:
                with open(ANSWER_FILE, "w", encoding="utf-8") as f:
                    f.write(ans)
            except Exception as e:
                LOGGER.error(f"[Iter {iter_no}] cannot write answer.txt: {e}")

            # پاک‌کردن CSVهای زنده (چرخه جدید)
            for p in LIVE_FILES.values():
                try: os.remove(p)
                except Exception: pass

            LOGGER.info(f"[Iter {iter_no}] ANSWER={ans}")
            time.sleep(1)  # مکث سبک
        else:
            time.sleep(1)  # منتظر آماده‌شدن شرایط

if __name__ == "__main__":
    main()
