#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compare_batch_vs_live.py (v2)
# ----------------------------------------------------------
# مقایسۀ دقیق batch ↔ live با استفاده از همان pipelineِ آموزش‌دیده
# (بدون بازسازی مدل؛ کالیبراتور حفظ می‌شود)
# ----------------------------------------------------------

from __future__ import annotations
import argparse, logging, warnings
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("cmp")

# ----------------------------------------------------------
# ---------- batch features و labels (آخر N_TEST ردیف) -----
# ----------------------------------------------------------
def build_batch(
    prep: PREPARE_DATA_FOR_TRAIN,
    window: int,
    feat_list: List[str],
    n_test: int,
    raw_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    خروجی:
      Xb, yb: آخر n_test ردیف، هم‌ستون با feat_list
      tail_buf: بافر آخر (n_test + window) ردیف برای شبیه‌سازی incremental
    """
    merged = raw_data if raw_data is not None else prep.load_data()
    tail_buf = merged.tail(n_test + window).copy()
    Xb_full, yb_full, _, _ = prep.ready(
        tail_buf, window=window, selected_features=feat_list, mode="train"
    )
    # آخر n_test ردیف، ریست اندیس، حفظ ترتیب ستون‌ها
    Xb = Xb_full.tail(n_test).copy().reset_index(drop=True)
    yb = yb_full.tail(n_test).copy().reset_index(drop=True)
    # اطمینان از ترتیب ستون‌ها
    Xb = Xb[feat_list].astype("float32")
    return Xb, yb, tail_buf.reset_index(drop=True)

# ----------------------------------------------------------
# ---------- شبیه‌سازی لایو رکورد-به-رکورد -----------------
# ----------------------------------------------------------
def simulate_live(
    prep: PREPARE_DATA_FOR_TRAIN,
    tail_buf: pd.DataFrame,
    feat_cols: List[str],
    window: int,
    inference_pipe: Pipeline,
    neg_thr: float,
    pos_thr: float,
    n_test: int,
    X_batch: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[int], set]:
    """
    خروجی:
      preds_live: پیش‌بینی‌های لایو به طول n_test (در صورت امکان)
      gt_live: GT محاسبه‌شده به طول برابر
      decided_idx: ایندکس‌های تصمیم‌دار (≠ -1) در بازۀ 0..n_test-1
      diff_cols: مجموعه‌ی نام ستون‌هایی که در هر نقطه اختلاف عددی داشتند
    """
    preds, gt = [], []
    diff_cols: set = set()

    # ما می‌خواهیم خروجی دقیقاً برای n_test ردیف «آخر» (همان batch) باشد.
    # پس، تا وقتی که به اندازه‌ی n_test «نمونهٔ محاسبه‌شده» نداریم، جلو می‌رویم.
    produced = 0
    # از ابتدای tail_buf حرکت می‌کنیم تا warm-up خودش پر شود
    for i in range(len(tail_buf)):
        row = tail_buf.iloc[[i]]
        X_inc, _ = prep.ready_incremental(
            row, window=window, selected_features=feat_cols
        )
        if X_inc.empty:
            continue  # هنوز warm-up تکمیل نشده

        # وقتی به نقطه‌ای رسیدیم که آماده است، این رکورد متناظر با
        # X_batch.iloc[produced] باید باشد (تا سقف n_test)
        if produced < len(X_batch):
            # اجباری کردن ترتیب و dtype
            X_inc = X_inc[feat_cols].astype("float32")
            # کنترل اختلافِ فیچر
            v_live = X_inc.iloc[0].values
            v_batch = X_batch.iloc[produced].values
            diff_mask = ~np.isclose(v_live, v_batch, atol=1e-10, equal_nan=True)
            if diff_mask.any():
                # جمع آوری نام ستون‌های متفاوت
                for c in X_batch.columns[diff_mask]:
                    diff_cols.add(c)

            # پیش‌بینی
            proba = inference_pipe.predict_proba(X_inc)[:, 1]
            lab = -1
            if proba[0] <= neg_thr:
                lab = 0
            elif proba[0] >= pos_thr:
                lab = 1
            preds.append(lab)

            # GT: مانند batch، برچسب واقعی کندل بعدی (diff قیمت)
            # برای آخرین ردیف tail_buf ممکن است GT نداشته باشیم
            if i + 1 < len(tail_buf):
                nxt = float(tail_buf[f"{prep.main_timeframe}_close"].iloc[i + 1])
                cur = float(tail_buf[f"{prep.main_timeframe}_close"].iloc[i])
                gt.append(int(nxt - cur > 0))

            produced += 1

        if produced >= n_test:
            break

    # هم‌طول‌سازی
    L = min(len(preds), len(gt), n_test)
    preds = np.asarray(preds[:L], dtype=int)
    gt    = np.asarray(gt[:L], dtype=int)

    decided_idx = np.where(preds != -1)[0].tolist()
    return preds, gt, decided_idx, diff_cols

# ----------------------------------------------------------
# ---------------------   CLI   ----------------------------
# ----------------------------------------------------------
def cli():
    p = argparse.ArgumentParser("Compare batch vs live outputs (v2)")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--n-test", type=int, default=4_000, help="Rows to compare (tail)")
    p.add_argument("--data-dir", default=".", help="Folder of CSV price files")
    return p.parse_args()

# ----------------------------------------------------------
# -----------------------  MAIN  ---------------------------
# ----------------------------------------------------------
if __name__ == "__main__":
    args     = cli()
    mdl_path = Path(args.model).expanduser().resolve()
    if not mdl_path.is_file():
        raise FileNotFoundError(f"❌  {mdl_path} not found")

    payload   = joblib.load(mdl_path)
    pipe_fit  : Pipeline = payload["pipeline"]               # ← همان پایپلاین آموزش‌دیده
    window    : int      = int(payload["window_size"])
    neg_thr   : float    = float(payload["neg_thr"])
    pos_thr   : float    = float(payload["pos_thr"])
    feat_cols : List[str]= list(payload["train_window_cols"])  # ترتیب ستون‌ها

    # ---------- آماده‌سازی داده ----------
    csv_dir = Path(args.data_dir).resolve()
    prep = PREPARE_DATA_FOR_TRAIN(
        main_timeframe="30T",
        filepaths={
            "30T": str(csv_dir / "XAUUSD_M30.csv"),
            "15T": str(csv_dir / "XAUUSD_M15.csv"),
            "5T" : str(csv_dir / "XAUUSD_M5.csv"),
            "1H" : str(csv_dir / "XAUUSD_H1.csv"),
        },
        verbose=False,
    )

    # برای جلوگیری از دوباره‌کاری و اختلاف ایندکس‌ها، یک بار raw را می‌خوانیم
    raw_full = prep.load_data()

    # ---------- دودکش ----------
    Xb, yb, tail_buf = build_batch(prep, window, feat_cols, args.n_test, raw_data=raw_full)
    prob_b = pipe_fit.predict_proba(Xb)[:, 1]
    yb_hat = np.full(len(yb), -1, dtype=int)
    yb_hat[prob_b <= neg_thr] = 0
    yb_hat[prob_b >= pos_thr] = 1
    decided_batch = (yb_hat != -1)
    acc_batch = (yb_hat[decided_batch] == yb[decided_batch].values).mean() if decided_batch.any() else float("nan")

    # ---------- لایو (با همان pipe_fit) ----------
    preds_live, gt_live, decided_idx, diff_cols = simulate_live(
        prep, tail_buf, feat_cols, window,
        inference_pipe=pipe_fit,   # ← هیچ بازسازی‌ای در کار نیست
        neg_thr=neg_thr, pos_thr=pos_thr,
        n_test=args.n_test,
        X_batch=Xb,
    )

    # هم‌ترازسازی
    L = min(len(preds_live), len(gt_live), len(yb_hat))
    preds_live = preds_live[:L]
    gt_live    = gt_live[:L]
    yb_hat     = yb_hat[:L]  # برای مقایسه‌ی label mismatch

    decided_live_mask = (preds_live != -1)
    acc_live = (preds_live[decided_live_mask] == gt_live[decided_live_mask]).mean() if decided_live_mask.any() else float("nan")

    # ---------- گزارش ----------
    print("\n========== ACCURACY REPORT (last {:,} rows) ==========".format(args.n_test))
    print(f"Batch accuracy (decided rows) : {acc_batch:.4f}")
    print(f"Live  accuracy (decided rows) : {acc_live:.4f}")
    print(f"Predicted rows (live)         : {decided_live_mask.sum()} / {L}")
    print(f"  • Correct                   : {(preds_live[decided_live_mask]==gt_live[decided_live_mask]).sum()}")
    print(f"  • Wrong                     : {(preds_live[decided_live_mask]!=gt_live[decided_live_mask]).sum()}")
    print(f"  • Unpredicted (-1)          : {L - decided_live_mask.sum()}")

    # ---------- اختلاف برچسب batch ↔ live ----------
    label_mismatch = int((yb_hat[:L] != preds_live[:L]).sum())
    print(f"\nLabel mismatch between batch-pred و live-pred : {label_mismatch}")

    # ---------- اختلاف فیچر ----------
    if diff_cols:
        print(f"\n⚠️ Feature columns with any mismatch ({len(diff_cols)}):")
        for c in sorted(diff_cols):
            print("  -", c)
    else:
        print("\n✅  هیچ اختلاف مقداری بین فیچرهای batch و live یافت نشد.")

    # نکته: اگر باز هم Predicted rows = 0 شد، یعنی احتمال‌های لایو بین neg_thr و pos_thr مانده‌اند.
    # با توجه به استفاده‌ی همین pipe_fit، این فقط زمانی رخ می‌دهد که
    #  ➜ آستانه‌ها با این مدل/این بازه همخوان نباشند، یا
    #  ➜ y/GT بر اساس تعریف متفاوتی ساخته شده باشد.
