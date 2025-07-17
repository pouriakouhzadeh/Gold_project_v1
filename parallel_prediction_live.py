#!/usr/bin/env python3
"""
parallel_prediction_live.py  — نسخهٔ اصلاح‌شده

تغییرات اصلی:
1. حذف دوباره‌استانداردسازی (double‑scaling):
   - دو خطى که X_input را با `scaler.transform` تغییر مى‌داد، حذف شده است.
2. محاسبهٔ PSI روى همان ماتریس ویژگى (X_live) که به مدل وارد مى‌شود:
   - فراخوانى `drift_checker.compare_live` به بعد از ساخت X_live منتقل شد و
     آرگومان از `merged_df` به `X_live` تغییر کرد.

سایر بخش‌ها بدون تغییر مانده‌اند تا رفتار قبلى برنامه حفظ شود.
"""

import os
import time
import logging
import sys
import joblib
import pandas as pd
import numpy as np

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from drift_checker import DriftChecker
from dynamic_threshold_adjuster import DynamicThresholdAdjuster

BUFFER_ROWS = 3000

LOG_FILENAME = "parallel_prediction_live.log"

LIVE_M30 = "XAUUSD.F_M30_live.csv"
LIVE_M15 = "XAUUSD.F_M15_live.csv"
LIVE_M5  = "XAUUSD.F_M5_live.csv"
LIVE_H1  = "XAUUSD.F_H1_live.csv"
ANSWER_TXT = "Answer.txt"
MODEL_PATH = "best_model.pkl"


def remove_if_exists(fp):
    if os.path.exists(fp):
        os.remove(fp)


def remove_initial_files():
    """در شروع، هرچه فایل live و Answer باشد پاک می‌کنیم."""
    for f in [LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1, ANSWER_TXT]:
        remove_if_exists(f)


def setup_logger():
    handlers = []

    # فایل لاگ
    file_handler = logging.FileHandler(LOG_FILENAME, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    logging.info("--- parallel_prediction_live started ---")


def wait_for_live_files():
    """منتظر بمانیم تا هرچهار فایل CSV حاضر شوند."""
    while True:
        needed = [LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1]
        if all(os.path.exists(f) for f in needed):
            return
        logging.debug("[Live] Some CSVs missing => waiting ...")
        time.sleep(1)


def remove_live_files():
    """پس از پردازش، فایل‌های لایو را پاک می‌کنیم."""
    for fp in [LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1]:
        remove_if_exists(fp)


def main_loop():
    setup_logger()
    remove_initial_files()

    # تلاش برای لود مدل
    try:
        model_data = joblib.load(MODEL_PATH)
        pipeline = model_data['pipeline']
        neg_thr = model_data['neg_thr']
        pos_thr = model_data['pos_thr']
        scaler = model_data['scaler']  # فقط براى سازگارى؛ ديگر استفاده نمى‌شود.
        window_size = model_data['window_size']
        feats = model_data['feats']
        train_window_cols = model_data['train_window_cols']
        train_raw_window = model_data.get('train_raw_window', None)
        train_raw_window = None
        logging.info("[Live] Model loaded successfully.")
    except Exception as e:
        logging.error(f"[Live] Cannot load model => {e}")
        return

    filepaths = {
        '30T': LIVE_M30,
        '1H':  LIVE_H1,
        '15T': LIVE_M15,
        '5T':  LIVE_M5
    }
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe='30T', verbose=False)

    drift_checker = DriftChecker(quantile=False)
    try:
        drift_checker.load_train_distribution("train_distribution.json")
        logging.info("[Live] Loaded train_distribution.json for drift checking.")
    except Exception as e:
        logging.warning(f"[Live] Could not load drift distribution => {e}")

    thr_adjuster = DynamicThresholdAdjuster(
        atr_high=10.0,
        vol_low=500,
        shift=0.01
    )

    iteration_count = 0
    X_hist = pd.DataFrame(columns=train_window_cols)   # ← NEW

    while True:
        wait_for_live_files()
        iteration_count += 1
        logging.info(f"\n[Live] ****** Iteration #{iteration_count} ******")

        try:
            # 1) بارگذاری داده‌هاى خام از CSVهاى لایو
            print("Start merge DFs")
            merged_df = prep.load_data()
            logging.debug(f"[Live] merged_df shape={merged_df.shape}, tail:\n{merged_df.tail(1)}")

            ROLL_LOOKBACK = 200        # بزرگتر از 52 تا + حاشیهٔ امن

            combined = merged_df.tail(ROLL_LOOKBACK + window_size)



            # 3) آماده‌سازى X_live (ویژگى‌هاى نهایى)
            if window_size == 1:
                X_live, _, _ = prep.ready(combined, window=1, selected_features=feats, mode='predict')
            else:
                X_live, _ = prep.ready_incremental(combined, window=window_size, selected_features=feats)

            logging.debug(f"[Live] X_live shape={X_live.shape}")

            # 4) Drift Checking  ← این بار روى بافر ۵۰۰ رکوردى
            # بعد از این که X_live را به ستون‌های train_window_cols محدود کردی



            # 5) بررسى کیفیت X_live
            if X_live.empty or X_live.isna().any().any():
                logging.warning("[Live] X_live empty or has NaN => writing NAN => removing files.")
                with open(ANSWER_TXT, 'w') as f:
                    f.write("NAN,0.0\n")
                remove_live_files()
                continue

            # 6) محاسبهٔ آستانهٔ دینامیک بر اساس ATR و حجم
            last_atr = merged_df['30T_ATR_14'].iloc[-1] if '30T_ATR_14' in merged_df.columns else 1.0
            last_vol = merged_df['30T_volume'].iloc[-1] if '30T_volume' in merged_df.columns else 1000.0
            dyn_neg_thr, dyn_pos_thr = thr_adjuster.adjust(neg_thr, pos_thr, last_atr, last_vol)
            # dyn_neg_thr, dyn_pos_thr = neg_thr, pos_thr
            logging.debug(f"[Live] dyn_neg_thr={dyn_neg_thr:.3f}, dyn_pos_thr={dyn_pos_thr:.3f}")

            # 7) هم‌تراز کردن ستون‌ها با آموزش (fill_value=0 براى ستون‌هاى گم‌شده)
            X_live.columns = [str(c) for c in X_live.columns]
            missing_cols = [c for c in train_window_cols if c not in X_live.columns]
            if missing_cols:
                logging.error(f"[Live] Missing columns ⇒ {missing_cols}")
                with open(ANSWER_TXT, 'w') as f:
                    f.write("NAN,0.0\n")
                remove_live_files()
                continue

            X_live = X_live[train_window_cols].astype(float)
            X_hist = pd.concat([X_hist, X_live]).tail(BUFFER_ROWS)

            
            scaler       = pipeline.named_steps['scaler']
            X_hist_proc  = scaler.transform(X_hist)
            
            
            # امن‌ترین حالت: ستون‌ها را از X_hist فعلی می‌گیریم
            X_hist_df = pd.DataFrame(X_hist_proc, columns=X_hist.columns)


            psi_val = drift_checker.compare_live(X_hist_df, bins=10)
            print(f"PSI = {psi_val}")

            # ------------------------------------------------------------
            # آخرین ردیفی که هیچ ستونی NaN ندارد را برای پیش‌بینی می‌گیریم
            last_valid = X_live[~X_live.isna().any(axis=1)].tail(1)

            if last_valid.empty:
                logging.warning("[Live] No valid row ⇒ writing NAN and skipping iteration.")
                # همان کاری که الآن برای حالت NaN می‌کنی:
                with open(ANSWER_TXT, "w") as f:
                    f.write("NAN,0.0\n")
                remove_live_files()
                continue           # برو سرِ iteration بعدی

            # اگر ردیف سالم موجود است
            X_input = last_valid.to_numpy().reshape(1, -1)
            # ------------------------------------------------------------
            
            proba = pipeline.predict_proba(X_input)[:, 1][0]
            logging.info(f"[Live] proba={proba:.4f}")
            print(f"[Live] proba={proba:.4f}")
            print(f"[Live] dyn_neg_thr={dyn_neg_thr:.4f}")
            print(f"[Live] dyn_pos_thr={dyn_pos_thr:.4f}")
            
            # 9) تصمیم نهایى بر اساس آستانه‌هاى دینامیک
            if proba <= dyn_neg_thr:
                ai_decision = "SEL"
            elif proba >= dyn_pos_thr:
                ai_decision = "BUY"
            else:
                ai_decision = "NAN"

            with open(ANSWER_TXT, 'w') as f:
                f.write(f"{ai_decision},{proba:.4f}\n")
            logging.info(f"[Live] => {ai_decision}, proba={proba:.4f}, ATR={last_atr:.2f}, VOL={last_vol:.2f}")

        except Exception as e:
            logging.error(f"[Live] Error => {e}")
            with open(ANSWER_TXT, 'w') as f:
                f.write("NAN,0.0\n")
                print("Answer file created ...")

        # 10) پاک‌سازى فایل‌هاى لایو و آماده شدن براى تکرار بعدى
        remove_live_files()
        logging.debug("[Live] Live files removed. Ready for next iteration.")

        time.sleep(1)


if __name__ == "__main__":
    main_loop()
