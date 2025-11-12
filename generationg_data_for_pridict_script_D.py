# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, logging, argparse, hashlib
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "generator_v2.log"

def setup_logging(verbosity: int = 1):
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

def resolve_raw_paths(base_dir: Path, symbol: str) -> Dict[str, str]:
    """پیدا کردن مسیر فایل‌های داده با سازگاری بیشتر"""
    candidates = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"], 
        "5T": [f"{symbol}_5T.csv", f"{symbol}_M5.csv"],
        "1H": [f"{symbol}_1H.csv", f"{symbol}_H1.csv"],
    }
    
    resolved = {}
    for tf, names in candidates.items():
        found = None
        for name in names:
            path = base_dir / name
            if path.exists():
                found = str(path)
                break
        if not found:
            # جستجوی case-insensitive
            for child in base_dir.iterdir():
                if child.is_file() and any(child.name.lower() == name.lower() for name in names):
                    found = str(child)
                    break
        if found:
            logging.info(f"[paths] {tf} -> {found}")
            resolved[tf] = found
    
    if "30T" not in resolved:
        raise FileNotFoundError(f"فایل تایم‌فریم اصلی 30T یافت نشد")
    
    return resolved

class DataGenerator:
    """کلاس تولید کننده داده‌های شبیه‌سازی شده"""
    
    def __init__(self, base_dir: str, symbol: str, last_n: int = 2000):
        self.base_dir = Path(base_dir)
        self.symbol = symbol
        self.last_n = last_n
        self.prep = None
        self.raw_data = None
        self.main_timeframe = "30T"
        
    def initialize(self):
        """آماده‌سازی اولیه داده‌ها"""
        logging.info("در حال بارگذاری و پردازش داده‌ها...")
        
        # پیدا کردن مسیر فایل‌ها
        filepaths = resolve_raw_paths(self.base_dir, self.symbol)
        
        # ایجاد PREPARE_DATA_FOR_TRAIN با پارامترهای یکسان با آموزش
        self.prep = PREPARE_DATA_FOR_TRAIN(
            filepaths=filepaths,
            main_timeframe=self.main_timeframe,
            verbose=False,
            fast_mode=False,  # مهم: استفاده از حالت کامل پردازش
            strict_disk_feed=False
        )
        
        # بارگذاری و پردازش کامل داده‌ها (همانند live_like_sim_v3)
        self.raw_data = self.prep.load_data()
        
        # مرتب‌سازی بر اساس زمان
        time_col = f"{self.main_timeframe}_time"
        self.raw_data[time_col] = pd.to_datetime(self.raw_data[time_col])
        self.raw_data = self.raw_data.sort_values(time_col).reset_index(drop=True)
        
        logging.info(f"داده‌ها با موفقیت بارگذاری شدند: {len(self.raw_data)} رکورد")
        return True
    
    def generate_live_slice(self, current_index: int) -> bool:
        """تولید برش داده برای زمان فعلی"""
        if self.raw_data is None or self.prep is None:
            logging.error("داده‌ها هنوز آماده نشده‌اند")
            return False
        
        # محاسبه اندیس شروع برای حفظ تاریخچه کافی
        start_idx = max(0, current_index - self.last_n)
        data_slice = self.raw_data.iloc[start_idx:current_index + 1].copy()
        
        if len(data_slice) == 0:
            logging.warning(f"برش داده برای اندیس {current_index} خالی است")
            return False
        
        # ذخیره برش داده در فایل موقت
        temp_path = self.base_dir / "temp_slice.csv"
        data_slice.to_csv(temp_path, index=False)
        
        # تولید فایل‌های live (شبیه‌سازی خروجی MQL)
        self._create_live_files(data_slice)
        
        return True
    
    def _create_live_files(self, data_slice: pd.DataFrame):
        """ایجاد فایل‌های live برای تایم‌فریم‌های مختلف"""
        time_col = f"{self.main_timeframe}_time"
        current_time = data_slice[time_col].iloc[-1]
        
        # برای هر تایم‌فریم، داده‌های مربوطه را فیلتر و ذخیره کن
        timeframes = ["30T", "15T", "5T", "1H"]
        
        for tf in timeframes:
            # فیلتر کردن داده‌ها برای تایم‌فریم جاری
            tf_pattern = f"{tf}_"
            tf_columns = [col for col in data_slice.columns if col.startswith(tf_pattern)]
            
            if tf_columns:
                # استخراج داده‌های تایم‌فریم خاص
                base_cols = [time_col] + tf_columns
                tf_data = data_slice[base_cols].copy()
                
                # حذف ردیف‌های کاملاً خالی
                tf_data = tf_data.dropna(how='all', subset=tf_columns)
                
                # تغییر نام ستون‌ها به فرمت استاندارد
                rename_dict = {}
                for col in tf_columns:
                    standard_name = col.replace(f"{tf}_", "")
                    rename_dict[col] = standard_name
                
                tf_data = tf_data.rename(columns=rename_dict)
                
                # ذخیره فایل live
                output_path = self.base_dir / f"{self.symbol}_{tf}_live.csv"
                tf_data.to_csv(output_path, index=False)
                
                logging.debug(f"فایل {tf} با {len(tf_data)} رکورد ایجاد شد")
    
    def get_ground_truth(self, current_index: int) -> tuple:
        """دریافت برچسب واقعی برای ارزیابی"""
        if current_index >= len(self.raw_data) - 1:
            return None, None
        
        time_col = f"{self.main_timeframe}_time"
        current_time = self.raw_data.loc[current_index, time_col]
        current_close = float(self.raw_data.loc[current_index, f"{self.main_timeframe}_close"])
        next_close = float(self.raw_data.loc[current_index + 1, f"{self.main_timeframe}_close"])
        
        real_up = next_close > current_close
        return real_up, current_time

def main():
    parser = argparse.ArgumentParser(description='تولید کننده داده‌های شبیه‌سازی شده برای پیش‌بینی طلا')
    parser.add_argument('--last-n', type=int, default=2000, help='تعداد داده‌های اخیر برای استفاده')
    parser.add_argument('--base-dir', type=str, default='.', help='پوشه حاوی داده‌ها')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='نماد معاملاتی')
    parser.add_argument('--sleep', type=float, default=0.5, help='تأخیر بین مراحل (ثانیه)')
    parser.add_argument('--start-index', type=int, help='اندیس شروع (اختیاری)')
    
    args = parser.parse_args()
    setup_logging(1)
    logging.info("=== راه‌اندازی ژنراتور داده‌های پیشرفته ===")
    
    # ایجاد ژنراتور
    generator = DataGenerator(args.base_dir, args.symbol, args.last_n)
    
    if not generator.initialize():
        logging.error("خطا در راه‌اندازی ژنراتور")
        return
    
    # تعیین اندیس شروع
    total_records = len(generator.raw_data)
    if args.start_index is not None:
        start_index = args.start_index
    else:
        start_index = max(generator.last_n, total_records - args.last_n)
    
    end_index = total_records - 2  # برای داشتن برچسب واقعی
    
    logging.info(f"شبیه‌سازی از اندیس {start_index} تا {end_index} (مجموعاً {end_index - start_index + 1} مرحله)")
    
    # فایل audit
    audit_rows = []
    answer_path = generator.base_dir / "answer.txt"
    
    wins = loses = none = 0
    
    for current_index in range(start_index, end_index + 1):
        # پاک کردن پاسخ قبلی
        if answer_path.exists():
            try:
                answer_path.unlink()
            except Exception:
                pass
        
        # تولید داده‌های live
        if not generator.generate_live_slice(current_index):
            logging.warning(f"خطا در تولید داده برای اندیس {current_index}")
            continue
        
        # دریافت برچسب واقعی
        real_up, current_time = generator.get_ground_truth(current_index)
        if real_up is None:
            logging.warning(f"برچسب واقعی برای اندیس {current_index} در دسترس نیست")
            continue
        
        logging.info(f"[Step {current_index - start_index + 1}/{(end_index - start_index + 1)}] زمان: {current_time}")
        
        # انتظار برای پاسخ مدل
        wait_start = time.time()
        while not answer_path.exists():
            if time.time() - wait_start > 30:  # timeout 30 ثانیه
                logging.warning("اتمام زمان انتظار برای پاسخ")
                break
            time.sleep(args.sleep)
        
        # خواندن پاسخ
        if answer_path.exists():
            try:
                answer = answer_path.read_text(encoding='utf-8').strip().upper()
                answer_path.unlink()  # پاک کردن فایل پاسخ
            except Exception as e:
                logging.error(f"خطا در خواندن پاسخ: {e}")
                answer = "NONE"
        else:
            answer = "NONE"
        
        # محاسبه نتایج
        if answer == "NONE":
            none += 1
            result = "NO_TRADE"
        elif answer == "BUY":
            if real_up:
                wins += 1
                result = "WIN"
            else:
                loses += 1
                result = "LOSE"
        elif answer == "SELL":
            if not real_up:
                wins += 1
                result = "WIN"
            else:
                loses += 1
                result = "LOSE"
        else:
            none += 1
            result = "INVALID"
        
        # محاسبه دقت
        total_trades = wins + loses
        accuracy = wins / total_trades if total_trades > 0 else 0.0
        
        # ذخیره audit
        audit_rows.append({
            'index': current_index,
            'timestamp': current_time,
            'answer': answer,
            'real_up': real_up,
            'result': result,
            'wins': wins,
            'loses': loses,
            'none': none,
            'accuracy': accuracy,
            'total_trades': total_trades
        })
        
        logging.info(f"[Result] پاسخ: {answer} | واقعی: {'UP' if real_up else 'DOWN'} | نتیجه: {result} | "
                    f"دقت: {accuracy:.3f} | Wins: {wins} | Loses: {loses} | None: {none}")
    
    # ذخیره نتایج نهایی
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv('generator_audit_v2.csv', index=False)
    
    final_accuracy = wins / (wins + loses) if (wins + loses) > 0 else 0
    logging.info(f"=== پایان شبیه‌سازی ===")
    logging.info(f"نتایج نهایی: دقت: {final_accuracy:.3f} | Wins: {wins} | Loses: {loses} | None: {none}")

if __name__ == "__main__":
    main()