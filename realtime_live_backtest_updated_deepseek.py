#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_chimney_vs_live_accuracy.py
────────────────────────────────
• تست کامل همخوانی دقت مدل در محیط دودکش (Batch) و لایو (Incremental)
• شناسایی و گزارش دقیق اختلاف‌ها
"""

from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path
from typing import List, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive

# ─── تنظیمات لاگ ────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("accuracy_test.log", encoding="utf-8")
    ]
)
LOGGER = logging.getLogger("accuracy_test")

# ─── ثابت‌های مهم ───────────────────────────────────────────────────────────
TIME_COL = "30T_time"
CLOSE_COL = "30T_close"

# ─── توابع کمکی ─────────────────────────────────────────────────────────────
def load_model(model_path: Path) -> Tuple[dict, ModelPipelineLive]:
    """بارگذاری مدل و ساخت تخمین‌گر لایو"""
    payload = joblib.load(model_path)
    model_pipe = payload["pipeline"]
    
    live_estimator = ModelPipelineLive(
        hyperparams=payload["hyperparams"],
        calibrate=payload.get("calibrate", True),
        calib_method=payload.get("calib_method", "sigmoid")
    )
    live_estimator.base_pipe = model_pipe
    
    return payload, live_estimator

def prepare_data(data_dir: Path, start_date: str = None) -> pd.DataFrame:
    """آماده‌سازی داده‌ها با در نظر گرفتن تاریخ شروع"""
    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        filepaths={
            "30T": str(data_dir / "XAUUSD_M30.csv"),
            "15T": str(data_dir / "XAUUSD_M15.csv"),
            "5T": str(data_dir / "XAUUSD_M5.csv"),
            "1H": str(data_dir / "XAUUSD_H1.csv"),
        },
        verbose=False,
    )
    
    merged = prep.load_data()
    
    if start_date:
        merged = merged[merged[TIME_COL] >= pd.Timestamp(start_date)]
    
    return merged.reset_index(drop=True)

# ─── پردازش دودکش (Batch) ───────────────────────────────────────────────────
def process_chimney(
    prep: PREPARE_DATA_FOR_TRAIN,
    data: pd.DataFrame,
    payload: dict,
    live_est: ModelPipelineLive
) -> Tuple[pd.DataFrame, np.ndarray]:
    """پردازش دسته‌ای داده‌ها (شبیه‌سازی دودکش)"""
    LOGGER.info("شروع پردازش دودکش (Batch)")
    
    # پارامترهای مدل
    window = payload["window_size"]
    feats = payload["feats"]
    all_cols = payload["train_window_cols"]
    neg_thr = payload["neg_thr"]
    pos_thr = payload["pos_thr"]
    
    # آماده‌سازی داده‌ها
    X_batch, y_batch, _, _ = prep.ready(
        data, 
        window=window,
        selected_features=feats,
        mode="train"
    )
    
    # اطمینان از وجود تمام ستون‌ها
    for col in all_cols:
        if col not in X_batch.columns:
            X_batch[col] = 0.0
    
    # پیش‌بینی
    proba = live_est.predict_proba(X_batch[all_cols])[:, 1]
    y_pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)
    
    # محاسبه دقت
    accuracy = accuracy_score(y_batch, y_pred[y_pred != -1])
    LOGGER.info("دقت دودکش: %.4f", accuracy)
    
    return X_batch, y_pred

# ─── پردازش لایو (Incremental) ──────────────────────────────────────────────
def process_live(
    prep: PREPARE_DATA_FOR_TRAIN,
    data: pd.DataFrame,
    payload: dict,
    live_est: ModelPipelineLive
) -> Tuple[pd.DataFrame, np.ndarray]:
    """پردازش افزایشی داده‌ها (شبیه‌سازی لایو)"""
    LOGGER.info("شروع پردازش لایو (Incremental)")
    
    # پارامترهای مدل
    window = payload["window_size"]
    all_cols = payload["train_window_cols"]
    neg_thr = payload["neg_thr"]
    pos_thr = payload["pos_thr"]
    
    # تنظیم state اولیه
    if not hasattr(prep, '_live_prev2'):
        prep._live_prev2 = data.iloc[:2].copy()
    
    y_pred_live = []
    scaler = live_est.base_pipe.named_steps["scaler"]
    scaler_means = scaler.mean_ if hasattr(scaler, "mean_") else None
    
    # حلقه پردازش افزایشی
    for i in tqdm(range(window, len(data)-1), desc="پردازش لایو"):
        # انتخاب زیرمجموعه داده‌ها
        sub = data.iloc[i-window:i+1].copy().reset_index(drop=True)
        
        # آماده‌سازی داده‌ها به صورت افزایشی
        X_inc, _ = prep.ready_incremental(
            sub, 
            window=window,
            selected_features=all_cols
        )
        
        # مدیریت داده‌های گم‌شده
        if X_inc.empty:
            y_pred_live.append(-1)
            continue
            
        for col in all_cols:
            if col not in X_inc.columns:
                X_inc[col] = scaler_means[all_cols.index(col)] if scaler_means else 0.0
        
        # پیش‌بینی
        proba = live_est.predict_proba(X_inc[all_cols])[:, 1]
        pred = ModelPipelineLive.apply_thresholds(proba, neg_thr, pos_thr)[0]
        y_pred_live.append(pred)
    
    # تبدیل به آرایه numpy
    y_pred_live = np.array(y_pred_live)
    
    # محاسبه دقت
    y_true = ((data[CLOSE_COL].shift(-1) - data[CLOSE_COL] > 0).astype(int))
    y_true = y_true[window:-1].reset_index(drop=True)
    valid_preds = y_pred_live != -1
    accuracy = accuracy_score(y_true[valid_preds], y_pred_live[valid_preds])
    
    LOGGER.info("دقت لایو: %.4f", accuracy)
    return y_pred_live

# ─── مقایسه نتایج ──────────────────────────────────────────────────────────
def compare_results(
    y_pred_chimney: np.ndarray, 
    y_pred_live: np.ndarray,
    data: pd.DataFrame,
    window: int
) -> Dict[str, float]:
    """مقایسه نتایج دودکش و لایو و محاسبه معیارهای اختلاف"""
    # تطابق طول آرایه‌ها
    min_len = min(len(y_pred_chimney), len(y_pred_live))
    y_pred_chimney = y_pred_chimney[:min_len]
    y_pred_live = y_pred_live[:min_len]
    
    # محاسبه اختلاف
    diff_mask = y_pred_chimney != y_pred_live
    diff_count = diff_mask.sum()
    diff_percentage = (diff_count / min_len) * 100
    
    # استخراج برچسب‌های واقعی برای بخش مشترک
    y_true = ((data[CLOSE_COL].shift(-1) - data[CLOSE_COL] > 0).astype(int)
    y_true = y_true[window:window+min_len].reset_index(drop=True)
    
    # محاسبه دقت برای بخش مشترک
    valid_chimney = y_pred_chimney != -1
    valid_live = y_pred_live != -1
    valid_both = valid_chimney & valid_live
    
    acc_chimney = accuracy_score(y_true[valid_chimney], y_pred_chimney[valid_chimney])
    acc_live = accuracy_score(y_true[valid_live], y_pred_live[valid_live])
    acc_common = accuracy_score(y_true[valid_both], y_pred_chimney[valid_both])
    
    # گزارش نتایج
    LOGGER.info("═════════════════ نتایج مقایسه ═════════════════")
    LOGGER.info("تعداد نمونه‌ها: %d", min_len)
    LOGGER.info("تعداد اختلاف: %d (%.2f%%)", diff_count, diff_percentage)
    LOGGER.info("دقت دودکش: %.4f", acc_chimney)
    LOGGER.info("دقت لایو: %.4f", acc_live)
    LOGGER.info("دقت در نمونه‌های مشترک: %.4f", acc_common)
    
    # تحلیل اختلاف‌ها
    if diff_count > 0:
        diff_indices = np.where(diff_mask)[0]
        LOGGER.warning("نمونه‌های دارای اختلاف: %s", diff_indices[:10])
        
        # ذخیره نمونه‌های مشکل‌دار برای تحلیل بیشتر
        problem_samples = []
        for idx in diff_indices[:50]:  # فقط 50 نمونه اول
            problem_samples.append({
                "index": idx,
                "chimney": y_pred_chimney[idx],
                "live": y_pred_live[idx],
                "time": data[TIME_COL].iloc[window + idx]
            })
        
        # ذخیره در فایل برای تحلیل عمیق‌تر
        pd.DataFrame(problem_samples).to_csv("problem_samples.csv", index=False)
        LOGGER.info("نمونه‌های مشکل‌دار در problem_samples.csv ذخیره شدند")
    
    return {
        "total_samples": min_len,
        "diff_count": diff_count,
        "diff_percentage": diff_percentage,
        "acc_chimney": acc_chimney,
        "acc_live": acc_live,
        "acc_common": acc_common
    }

# ─── تابع اصلی ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="تست دقت دودکش و لایو")
    parser.add_argument("--model", required=True, help="مسیر فایل مدل (pkl)")
    parser.add_argument("--data-dir", required=True, help="مسیر دایرکتوری داده‌ها")
    parser.add_argument("--start-date", help="تاریخ شروع تست (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # بارگذاری مدل
    LOGGER.info("بارگذاری مدل از %s", args.model)
    payload, live_est = load_model(Path(args.model))
    
    # آماده‌سازی داده‌ها
    LOGGER.info("بارگذاری و آماده‌سازی داده‌ها از %s", args.data_dir)
    data = prepare_data(Path(args.data_dir), args.start_date)
    
    # آماده‌سازی داده‌ها (کلاس PREPARE)
    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        filepaths={
            "30T": str(Path(args.data_dir) / "XAUUSD_M30.csv",
            "15T": str(Path(args.data_dir) / "XAUUSD_M15.csv",
            "5T": str(Path(args.data_dir) / "XAUUSD_M5.csv",
            "1H": str(Path(args.data_dir) / "XAUUSD_H1.csv",
        },
        verbose=False,
    )
    
    # پردازش دودکش
    _, y_pred_chimney = process_chimney(prep, data, payload, live_est)
    
    # پردازش لایو
    y_pred_live = process_live(prep, data, payload, live_est)
    
    # مقایسه نتایج
    results = compare_results(
        y_pred_chimney, 
        y_pred_live, 
        data,
        payload["window_size"]
    )
    
    # نتیجه نهایی
    if results["diff_percentage"] == 0:
        LOGGER.info("✅ تست موفق: هیچ اختلافی بین دودکش و لایو یافت نشد")
    else:
        LOGGER.warning("⚠️ اختلاف یافت شد: %.2f%% نمونه‌ها متفاوت هستند", 
                      results["diff_percentage"])

if __name__ == "__main__":
    main()