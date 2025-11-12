# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations

import os, sys, time, json, joblib, logging, argparse, hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "prediction_in_production_v2.log"

def setup_logging(verbosity: int = 1):
    """راه‌اندازی سیستم لاگینگ"""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity <= 1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger

def load_payload_best_model(pkl_path: str = "best_model.pkl") -> Dict:
    """بارگذاری مدل با قابلیت بازیابی خطا"""
    try:
        payload = joblib.load(pkl_path)
        
        if isinstance(payload, dict):
            # پیدا کردن مدل در بین کلیدهای ممکن
            model = None
            for key in ["pipeline", "model", "estimator", "clf", "best_estimator"]:
                if key in payload and hasattr(payload[key], "predict"):
                    model = payload[key]
                    break
            
            if model is None:
                # جستجو در بین تمام مقادیر
                for value in payload.values():
                    if hasattr(value, "predict"):
                        model = value
                        break
            
            if model is None:
                raise ValueError("مدلی در فایل یافت نشد")
            
            return {
                "pipeline": model,
                "window_size": int(payload.get("window_size", 1)),
                "train_window_cols": list(payload.get("train_window_cols") or payload.get("feats") or []),
                "neg_thr": float(payload.get("neg_thr", 0.005)),
                "pos_thr": float(payload.get("pos_thr", 0.995)),
                "scaler": payload.get("scaler")
            }
        else:
            # فرض می‌کنیم خود payload مدل است
            return {
                "pipeline": payload,
                "window_size": 1,
                "train_window_cols": [],
                "neg_thr": 0.005,
                "pos_thr": 0.995,
                "scaler": None
            }
            
    except Exception as e:
        logging.error(f"خطا در بارگذاری مدل: {e}")
        raise

def resolve_live_paths(base_dir: Path, symbol: str) -> Dict[str, str]:
    """پیدا کردن فایل‌های live با سازگاری بیشتر"""
    patterns = {
        "30T": [f"{symbol}_30T_live.csv", f"{symbol}_M30_live.csv"],
        "15T": [f"{symbol}_15T_live.csv", f"{symbol}_M15_live.csv"],
        "5T": [f"{symbol}_5T_live.csv", f"{symbol}_M5_live.csv"],
        "1H": [f"{symbol}_1H_live.csv", f"{symbol}_H1_live.csv"],
    }
    
    resolved = {}
    for tf, names in patterns.items():
        found = None
        for name in names:
            path = base_dir / name
            if path.exists():
                found = str(path)
                break
        
        if found:
            resolved[tf] = found
            logging.info(f"[live_paths] {tf} -> {found}")
    
    return resolved

def ensure_columns(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """اطمینان از وجود تمام ستون‌های مورد نیاز"""
    if not cols:
        return X
    
    X_copy = X.copy()
    missing_cols = [col for col in cols if col not in X_copy.columns]
    
    for col in missing_cols:
        X_copy[col] = 0.0
        logging.warning(f"ستون مفقوده {col} با صفر پر شد")
    
    return X_copy[cols]

class ProductionPredictor:
    """پیش‌بین تولید با قابلیت‌های پیشرفته"""
    
    def __init__(self, model_path: str, base_dir: str, symbol: str):
        self.base_dir = Path(base_dir)
        self.symbol = symbol
        self.model_payload = None
        self.prep = None
        self.main_timeframe = "30T"
        
        # بارگذاری مدل
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """بارگذاری مدل و پارامترهای مربوطه"""
        logging.info("در حال بارگذاری مدل...")
        self.model_payload = load_payload_best_model(model_path)
        
        logging.info(f"مدل با موفقیت بارگذاری شد: window_size={self.model_payload['window_size']}, "
                    f"features={len(self.model_payload['train_window_cols'])}")
    
    def initialize_data_processor(self):
        """راه‌اندازی پردازشگر داده با پارامترهای یکسان با آموزش"""
        # پیدا کردن مسیر فایل‌های live
        live_paths = resolve_live_paths(self.base_dir, self.symbol)
        
        if not live_paths:
            raise FileNotFoundError("هیچ فایل live یافت نشد")
        
        # ایجاد پردازشگر داده با پارامترهای یکسان با live_like_sim_v3
        self.prep = PREPARE_DATA_FOR_TRAIN(
            filepaths=live_paths,
            main_timeframe=self.main_timeframe,
            verbose=False,
            fast_mode=False,  # مهم: استفاده از حالت کامل پردازش
            strict_disk_feed=False
        )
        
        logging.info("پردازشگر داده با موفقیت راه‌اندازی شد")
    
    def predict_current(self) -> Tuple[str, float, Dict]:
        """انجام پیش‌بینی برای داده‌های جاری"""
        if self.prep is None:
            raise RuntimeError("پردازشگر داده راه‌اندازی نشده است")
        
        # بارگذاری و پردازش داده‌ها (همانند live_like_sim_v3)
        merged_data = self.prep.load_data()
        
        if merged_data.empty:
            logging.warning("داده‌های ادغام شده خالی هستند")
            return "NONE", 0.0, {}
        
        # آماده‌سازی داده برای پیش‌بینی
        window_size = self.model_payload["window_size"]
        feature_cols = self.model_payload["train_window_cols"]
        
        X, y_dummy, features, price_series = self.prep.ready(
            merged_data,
            window=window_size,
            selected_features=feature_cols,
            mode="predict",
            predict_drop_last=True,  # مهم: یکسان با live_like_sim_v3
            train_drop_last=False
        )
        
        if X.empty:
            logging.warning("داده‌های آماده شده برای پیش‌بینی خالی هستند")
            return "NONE", 0.0, {}
        
        # اطمینان از وجود تمام ستون‌ها
        X_ready = ensure_columns(X, feature_cols)
        
        # گرفتن آخرین نمونه برای پیش‌بینی
        X_last = X_ready.tail(1)
        
        # پیش‌بینی
        model = self.model_payload["pipeline"]
        probability = 0.0
        
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_last)
                probability = float(proba[0, 1])  # احتمال کلاس مثبت
            else:
                # برای مدل‌هایی که predict_proba ندارند
                prediction = model.predict(X_last)
                probability = float(prediction[0])
                logging.warning("مدل از predict_proba پشتیبانی نمی‌کند")
        except Exception as e:
            logging.error(f"خطا در پیش‌بینی: {e}")
            return "NONE", 0.0, {}
        
        # اعمال آستانه‌ها
        neg_thr = self.model_payload["neg_thr"]
        pos_thr = self.model_payload["pos_thr"]
        
        if probability >= pos_thr:
            decision = "BUY"
        elif probability <= neg_thr:
            decision = "SELL"
        else:
            decision = "NONE"
        
        # اطلاعات دیباگ
        debug_info = {
            "probability": probability,
            "neg_threshold": neg_thr,
            "pos_threshold": pos_thr,
            "features_used": len(feature_cols),
            "data_points": len(X),
            "window_size": window_size
        }
        
        return decision, probability, debug_info
    
    def cleanup_live_files(self):
        """پاکسازی فایل‌های موقت live"""
        patterns = [
            f"{self.symbol}_30T_live.csv", f"{self.symbol}_M30_live.csv",
            f"{self.symbol}_15T_live.csv", f"{self.symbol}_M15_live.csv", 
            f"{self.symbol}_5T_live.csv", f"{self.symbol}_M5_live.csv",
            f"{self.symbol}_1H_live.csv", f"{self.symbol}_H1_live.csv"
        ]
        
        for pattern in patterns:
            path = self.base_dir / pattern
            if path.exists():
                try:
                    path.unlink()
                    logging.debug(f"فایل {pattern} پاک شد")
                except Exception as e:
                    logging.warning(f"خطا در پاک کردن {pattern}: {e}")

def main():
    parser = argparse.ArgumentParser(description='سیستم پیش‌بینی تولید برای طلا')
    parser.add_argument('--base-dir', type=str, default='.', help='پوشه حاوی داده‌ها')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='نماد معاملاتی')
    parser.add_argument('--model-path', type=str, default='best_model.pkl', help='مسیر مدل')
    parser.add_argument('--sleep', type=float, default=0.5, help='تأخیر بین چک‌ها')
    parser.add_argument('--max-retries', type=int, default=3, help='تعداد تلاش مجاز')
    
    args = parser.parse_args()
    setup_logging(1)
    
    logging.info("=== راه‌اندازی سیستم پیش‌بینی تولید ===")
    
    # ایجاد پیش‌بین
    try:
        predictor = ProductionPredictor(args.model_path, args.base_dir, args.symbol)
    except Exception as e:
        logging.error(f"خطا در ایجاد پیش‌بین: {e}")
        return
    
    answer_path = predictor.base_dir / "answer.txt"
    retry_count = 0
    
    while True:
        try:
            # پاک کردن پاسخ قبلی اگر وجود دارد
            if answer_path.exists():
                try:
                    answer_path.unlink()
                except Exception:
                    pass
            
            # بررسی وجود فایل‌های live
            live_paths = resolve_live_paths(predictor.base_dir, predictor.symbol)
            
            if not live_paths:
                logging.debug("در انتظار فایل‌های live...")
                time.sleep(args.sleep)
                continue
            
            # راه‌اندازی پردازشگر داده (هر بار برای اطمینان از تازگی)
            predictor.initialize_data_processor()
            
            # انجام پیش‌بینی
            decision, probability, debug_info = predictor.predict_current()
            
            # نوشتن پاسخ
            try:
                answer_path.write_text(decision, encoding='utf-8')
                logging.info(f"پیش‌بینی: {decision} (احتمال: {probability:.4f}) | "
                           f"آستانه‌ها: [{debug_info['neg_threshold']:.3f}, {debug_info['pos_threshold']:.3f}]")
            except Exception as e:
                logging.error(f"خطا در نوشتن پاسخ: {e}")
            
            # پاکسازی فایل‌های live
            predictor.cleanup_live_files()
            
            retry_count = 0  # reset retry counter on success
            
        except Exception as e:
            logging.error(f"خطا در چرخه پیش‌بینی: {e}")
            retry_count += 1
            
            if retry_count >= args.max_retries:
                logging.error("تعداد خطاها از حد مجاز گذشت. خروج...")
                break
            
            # نوشتن پاسخ NONE در صورت خطا
            try:
                answer_path.write_text("NONE", encoding='utf-8')
            except Exception:
                pass
        
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()