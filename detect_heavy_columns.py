#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_heavy_columns.py  –  شناسایی ستون‌های حجیم (skew‐بالا) در دیتافریم پروژه

اجرا:
    python detect_heavy_columns.py                 # حالت پیش‌فرض (threshold=10, top=30)
    python detect_heavy_columns.py --threshold 15  # آستانه سخت‌گیرانه‌تر
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ------------------ ۱) گرفتن دیتافریم با کلاس آماده‌سازی ------------------
try:
    # فرض بر این است که فایل prepare_data_for_train.py در همین دایرکتوری است
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
except ImportError:
    sys.exit("❌  کلاس PREPARE_DATA_FOR_TRAIN پیدا نشد؛ مطمئن شوید اسکریپت را در پوشه پروژه اجرا می‌کنید.")

def build_dataframe(window: int = 1, mode: str = "train") -> pd.DataFrame:
    prep = PREPARE_DATA_FOR_TRAIN()          # مسیر فایل‌ها از پیش در کلاس تعریف شده
    df, _, _ = prep.get_prepared_data(window=window, mode=mode)
    if df.empty:
        sys.exit("⚠️  دیتافریم برگشتی خالی است.")
    return df

# ------------------ ۲) تشخیص ستون‌های skew زیاد ------------------
def detect_heavy(df: pd.DataFrame, threshold: float = 10.0, top: int = 30):
    heavy = []
    num_cols = df.select_dtypes(include=["float32", "float64"]).columns
    for col in num_cols:
        q50, q99 = df[col].quantile([0.50, 0.99])
        if q50 == 0 or q99 <= 0:
            continue
        ratio = q99 / abs(q50)
        if ratio > threshold:
            heavy.append((col, q50, q99, ratio))
    heavy.sort(key=lambda x: x[3], reverse=True)
    return heavy[:top]

# ------------------ ۳) اجرای اصلی ------------------
def main():
    parser = argparse.ArgumentParser(description="Detect heavy-tailed columns in project dataframe")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="نسبت p99 / |median| جهت شناسایی ستون حجیم (پیش‌فرض: 10)")
    parser.add_argument("--top", type=int, default=30,
                        help="حداکثر ستون‌هایی که چاپ می‌شود (پیش‌فرض: 30)")
    parser.add_argument("--window", type=int, default=1,
                        help="آرگومان window برای آماده‌سازی داده (پیش‌فرض: 1)")
    args = parser.parse_args()

    df = build_dataframe(window=args.window, mode="train")
    heavy_cols = detect_heavy(df, threshold=args.threshold, top=args.top)

    if not heavy_cols:
        print(f"هیچ ستونی با نسبت بزرگ‌تر از {args.threshold} پیدا نشد.")
        return

    print(f"\nستون‌های حجیم (ratio > {args.threshold}):")
    print(f"{'column':45s}  median        p99           ratio")
    print("-" * 80)
    for col, q50, q99, r in heavy_cols:
        print(f"{col:45s}  {q50:12.4g}  {q99:12.4g}  {r:8.2f}")

if __name__ == "__main__":
    main()
