#!/usr/bin/env python3
# compare_batch_vs_live.py
# ----------------------------------------------------------
# مقایسۀ دقیقِ ویژگی‌ها و دقت مدل در حالت دودکش و لایو
# فقط روی 4 000 کندل آخر – پنجرهٔ 5 تایی
# ----------------------------------------------------------

from pathlib import Path
import logging, warnings
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN   # ← کلاس اصلاح‌شده

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

N_TEST  = 4000     # اندازهٔ ارزیابی
WINDOW  = 5        # طول پنجرهٔ tminus
SEED    = 2025

# ----------------------------------------------------------
def build_batch(prep: PREPARE_DATA_FOR_TRAIN) -> Tuple[pd.DataFrame, pd.Series]:
    """ساخت X و y دودکشی فقط از ۴۰۰۰ ردیف آخر."""
    merged = prep.load_data()
    merged_tail = merged.tail(N_TEST + WINDOW)      # +WINDOW برای کانتکست کافی
    Xb, yb, _, _ = prep.ready(merged_tail, window=WINDOW, mode="train")
    Xb, yb = Xb.tail(N_TEST), yb.tail(N_TEST)
    return Xb.reset_index(drop=True), yb.reset_index(drop=True)

def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, n_jobs=-1, random_state=SEED))
    ])
    pipe.fit(X, y)
    return pipe

def simulate_live(prep: PREPARE_DATA_FOR_TRAIN,
                  raw: pd.DataFrame,
                  feats_keep: List[str],
                  model: Pipeline,
                  Xb: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, set]:
    """فید رکورد-به-رکورد روی ۴۰۰۰ کندل آخر و جمع‌آوری نتایج."""
    preds, gt = [], []
    diff_cols: set = set()

    buf = raw.tail(N_TEST + WINDOW)          # همان بازهٔ ارزیابی

    for i in range(len(buf)):
        # تا وقتی حداقل WINDOW ردیف نداریم، ادامه نده
        if i + 1 < WINDOW:
            continue

        win_df = buf.iloc[i - WINDOW + 1 : i + 1]          # پنجرهٔ 5 تایی
        X_inc, _ = prep.ready_incremental(
            win_df, window=WINDOW, selected_features=feats_keep
        )
        if X_inc.empty:
            continue

        # --- NEW: تضمین ستون‌ها ---
        missing = [c for c in feats_keep if c not in X_inc.columns]
        if missing:
            for c in missing:
                X_inc[c] = 0.0
            X_inc = X_inc[feats_keep]
        # ---------------------------

        p = model.predict(X_inc)[0]
        preds.append(p)

        # ground-truth «صعود/نزول کندل بعدی»
        if i + 1 < len(buf):
            gt.append(
                int(
                    buf[f"{prep.main_timeframe}_close"].iloc[i + 1]
                    - buf[f"{prep.main_timeframe}_close"].iloc[i]
                    > 0
                )
            )

        # مقایسۀ فیچر با نسخهٔ batch برای همین اندیس
        idx_batch = len(gt) - 1
        if idx_batch < len(Xb):
            diff_mask = ~np.isclose(
                Xb.iloc[idx_batch].values,
                X_inc.iloc[0].values,
                atol=1e-10,
                equal_nan=True,
            )
            diff_cols.update(Xb.columns[diff_mask])

    return np.array(preds), np.array(gt), diff_cols

# ----------------------------------------------------------
if __name__ == "__main__":
    prep = PREPARE_DATA_FOR_TRAIN(verbose=False)

    # ---------- بخش دودکش ----------
    Xb, yb = build_batch(prep)
    model  = train_model(Xb, yb)
    acc_batch = (model.predict(Xb) == yb).mean()

    # ---------- بخش لایو ----------
    raw_full = prep.load_data()                     # دیتای ادغام-شدهٔ کامل
    preds_live, gt_live, diff_cols = simulate_live(
        prep, raw_full, Xb.columns.tolist(), model, Xb
    )

    # هم‌ترازی طول آرایه‌ها
    min_len = min(len(preds_live), len(gt_live))
    preds_live, gt_live = preds_live[:min_len], gt_live[:min_len]

    acc_live = (preds_live == gt_live).mean()
    correct  = (preds_live == gt_live).sum()
    wrong    = (preds_live != gt_live).sum()
    skipped  = N_TEST - len(preds_live)

    # ---------- گزارش ----------
    print("\n========== ACCURACY REPORT (last 4 000 rows) ==========")
    print(f"Batch accuracy : {acc_batch:.4f}")
    print(f"Live  accuracy : {acc_live:.4f}")
    print(f"Predicted rows : {len(preds_live)} / {N_TEST}   (skipped {skipped})")
    print(f"  • Correct     {correct}")
    print(f"  • Wrong       {wrong}")

    # ---------- اختلاف برچسب دودکش و لایو ----------
    tail_len = min(len(gt_live), len(yb))
    mismatch  = (yb.tail(tail_len).values != gt_live[:tail_len]).sum()
    print(f"\nLabel mismatch between batch-y and live-y : {mismatch}")


    # فیچرهای متفاوت
    if diff_cols:
        print("\nFeature columns with at least one value mismatch:")
        for c in sorted(diff_cols):
            print("  -", c)
    else:
        print("\nNo feature value mismatch between live and batch.")
