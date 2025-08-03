#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_batch_vs_live.py
~~~~~~~~~~~~~~~~~~~~~~~~
سنجش هم‌ارزی «دودکش» (batch) و «لایو» برای ۴۰۰۰ کندل آخر.
"""

from pathlib import Path
import joblib, logging, warnings, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# مسیر پروژهٔ خود را به PYTHONPATH بیفزایید یا import زیر را تنظیم کنید
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN   # ← همان فایلی که اصلاح کردید

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

N_TEST = 4000        # تعداد نمونهٔ ارزیابی
WINDOW  = 5          # همان پنجره‌ای که در آموزش استفاده کردید

def build_batch(prep: PREPARE_DATA_FOR_TRAIN) -> tuple[pd.DataFrame, pd.Series]:
    """تهیهٔ X,y دودکشی فقط از ۴۰۰۰ سطر آخر."""
    merged = prep.load_data()
    merged_tail = merged.tail(N_TEST + WINDOW)          # برای داشتن کانتکست کافی
    Xb, yb, _, _ = prep.ready(merged_tail, window=WINDOW, mode="train")
    # فقط آخرین N_TEST ردیف (بعد از پنجره‌بندی) را نگه می‌داریم
    Xb, yb = Xb.tail(N_TEST), yb.tail(N_TEST)
    return Xb.reset_index(drop=True), yb.reset_index(drop=True)

def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=300, n_jobs=-1, class_weight="balanced"))
    ])
    pipe.fit(X, y)
    return pipe

def simulate_live(prep: PREPARE_DATA_FOR_TRAIN,
                  raw: pd.DataFrame,
                  feats_keep: list[str],
                  model: Pipeline):
    """فید رکورد به رکورد روی ۴۰۰۰ کندل آخر."""
    preds, gt, diff_cols = [], [], set()
    # ← ابتدا باید دو رکورد بافر شود
    buf = raw.tail(N_TEST + WINDOW)        # همان بازه
    for i in range(len(buf)):
        row_df = buf.iloc[[i]]
        X_inc, _ = prep.ready_incremental(row_df,
                                          window=WINDOW,
                                          selected_features=feats_keep)
        if X_inc.empty:          # تا زمان پر شدن بافر (دو ردیف) چیزی نداریم
            continue
        p = model.predict(X_inc)[0]
        preds.append(p)

        # ground-truth (همان تعریف دودکش)
        if i+1 < len(buf):
            gt.append(int(buf[f"{prep.main_timeframe}_close"].iloc[i+1] -
                           buf[f"{prep.main_timeframe}_close"].iloc[i] > 0))

        # مقایسهٔ فیچرها با دودکش برای همین اندیس
        idx_batch = len(gt)-1           # چون بعد از آماده شدن پنجره، اندیس جلو افتاده
        if idx_batch < len(Xb):
            xb_row = Xb.iloc[idx_batch]
            xl_row = X_inc.iloc[0]
            diff = (np.isclose(xb_row, xl_row, atol=1e-10) == False)
            diff_cols.update(xb_row.index[diff].tolist())

    return np.array(preds), np.array(gt), diff_cols

if __name__ == "__main__":
    prep = PREPARE_DATA_FOR_TRAIN(verbose=False)

    # --------- 1) دودکش ----------
    Xb, yb = build_batch(prep)
    model  = train_model(Xb, yb)
    acc_batch = (model.predict(Xb) == yb).mean()

    # --------- 2) لایو ----------
    raw_full = prep.load_data()                         # دیتای merge-شده کامل
    preds_live, gt_live, diff_cols = simulate_live(
        prep, raw_full, Xb.columns.tolist(), model
    )
    # ممکن است یک یا دو gt کم باشد (به علت بافر)
    min_len = min(len(preds_live), len(gt_live))
    preds_live, gt_live = preds_live[:min_len], gt_live[:min_len]

    acc_live = (preds_live == gt_live).mean()
    correct  = (preds_live == gt_live).sum()
    wrong    = (preds_live != gt_live).sum()
    skipped  = N_TEST - len(preds_live)

    # --------- 3) گزارش ----------
    print("\n=== ACCURACY REPORT (last 4 000 rows) ===")
    print(f"Batch accuracy  : {acc_batch:.4f}")
    print(f"Live  accuracy  : {acc_live:.4f}")
    print(f"Total predicted : {len(preds_live)} of {N_TEST} (skipped {skipped})")
    print(f"  • Correct     : {correct}")
    print(f"  • Wrong       : {wrong}")

    # اختلاف برچسب‌هایی که وارد مدل شده‌اند (باید ۰ باشد)
    label_mismatch = (yb.tail(len(gt_live)).values != gt_live).sum()
    print(f"\nLabel mismatch between batch-y and live-y: {label_mismatch}")

    # ستون‌هایی که حداقل یک بار مقدارشان بین دودکش و لایو فرق داشته
    if diff_cols:
        print("\nFeature columns with value mismatch (live vs batch):")
        for c in sorted(diff_cols):
            print(" •", c)
    else:
        print("\nNo feature value mismatch between live and batch.")

