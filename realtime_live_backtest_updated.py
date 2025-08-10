#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compare_batch_vs_live.py (v2 — stepwise recompute, robust)
# ----------------------------------------------------------
# مقایسۀ دقیق batch ↔ live با استفاده از همان pipelineِ آموزش‌دیده
# لایو به‌صورت گام‌به‌گام و با بازتولید فیچرها از tail_buf ساخته می‌شود
# و به‌صورت امن طول‌ها را به واقعیتِ داده sync می‌کند.
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
      Xb, yb: آخر n_test ردیف، هم‌ستون با feat_list (ممکن است کمتر از n_test شود)
      tail_buf: بافر آخر (n_test + window) ردیف برای شبیه‌سازی stepwise
    """
    merged = raw_data if raw_data is not None else prep.load_data()
    tail_buf = merged.tail(n_test + window).copy()

    Xb_full, yb_full, _, _ = prep.ready(
        tail_buf, window=window, selected_features=feat_list, mode="train"
    )

    # ممکن است به‌دلیل lookback داخلی اندیکاتورها، خروجی کامل‌تر نشود
    take = min(n_test, len(Xb_full), len(yb_full))
    if take <= 0:
        raise RuntimeError("خروجی batch خالی است؛ داده یا پارامترها را بررسی کن.")

    Xb = Xb_full.tail(take).copy().reset_index(drop=True)
    yb = yb_full.tail(take).copy().reset_index(drop=True)

    # ترتیب ستون‌ها و dtype
    Xb = Xb[feat_list].astype("float32")
    return Xb, yb, tail_buf.reset_index(drop=True)

# ----------------------------------------------------------
# ---------- شبیه‌سازی لایو با بازتولید گام‌به‌گام ----------
# ----------------------------------------------------------
def simulate_live_stepwise(
    prep: PREPARE_DATA_FOR_TRAIN,
    tail_buf: pd.DataFrame,
    feat_cols: List[str],
    window: int,
    inference_pipe: Pipeline,
    neg_thr: float,
    pos_thr: float,
    X_batch: pd.DataFrame,
) -> Tuple[np.ndarray, set]:
    """
    به‌جای ready_incremental، هر بار تا اندیس i از tail_buf را به ready می‌دهیم
    و آخرین ردیف فیچر را به‌عنوان «رکورد لایو» برمی‌داریم.
    خروجی:
      preds_live: برچسب‌های لایو به طول len(X_batch)
      diff_cols  : نام ستون‌هایی که حتی یک اختلاف کوچک با batch داشتند
    """
    total = len(X_batch)
    if total == 0:
        return np.asarray([], dtype=int), set()

    # tail_buf طولش = window + n_test بود. اگر X_batch کوتاه‌تر شد، base را مطابق آن تنظیم می‌کنیم
    # base = تعداد ردیف‌هایی که قبل از اولین نمونهٔ batch در tail_buf می‌آید (عملاً warm-up)
    # حداقل warm-up باید window باشد؛ اگر tail_buf کوتاه‌تر است، total را پایین می‌آوریم.
    if len(tail_buf) < window + total:
        # حداکثر تعداد قابل‌پیش‌بینی با این tail_buf
        possible = max(0, len(tail_buf) - window)
        if possible <= 0:
            return np.asarray([], dtype=int), set()
        total = min(total, possible)
        X_batch = X_batch.tail(total).reset_index(drop=True)

    base = len(tail_buf) - total
    base = max(base, window)  # تضمین حداقل warm-up

    preds_live: List[int] = []
    diff_cols: set = set()

    for k in range(total):
        end = base + k + 1  # از ابتدای tail_buf تا این اندیس را داریم
        cur_slice = tail_buf.iloc[:end]

        X_cur, _, _, _ = prep.ready(
            cur_slice, window=window, selected_features=feat_cols, mode="train"
        )
        if X_cur.empty:
            # warm-up کامل نشده؛ منطقی‌ترین رفتار: عدم تصمیم
            preds_live.append(-1)
            continue

        # آخرین ردیف فیچر، معادل رکورد «لایو» در این لحظه
        x_inc = X_cur.iloc[[-1]][feat_cols].astype("float32")  # 1×d

        # کنترل اختلافِ فیچر با batch (همان ردیف k)
        v_live  = x_inc.iloc[0].values
        v_batch = X_batch.iloc[k].values
        diff_mask = ~np.isclose(v_live, v_batch, atol=1e-10, equal_nan=True)
        if diff_mask.any():
            for c in X_batch.columns[diff_mask]:
                diff_cols.add(c)

        # پیش‌بینی با همان pipeline ذخیره‌شده (کالیبراسیون حفظ می‌شود)
        proba = float(inference_pipe.predict_proba(x_inc)[:, 1][0])
        if proba <= neg_thr:
            preds_live.append(0)
        elif proba >= pos_thr:
            preds_live.append(1)
        else:
            preds_live.append(-1)

    return np.asarray(preds_live, dtype=int), diff_cols

# ----------------------------------------------------------
# ---------------------   CLI   ----------------------------
# ----------------------------------------------------------
def cli():
    p = argparse.ArgumentParser("Compare batch vs live outputs (v2 stepwise robust)")
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

    raw_full = prep.load_data()

    # ---------- دودکش ----------
    Xb, yb, tail_buf = build_batch(prep, window, feat_cols, args.n_test, raw_data=raw_full)
    prob_b = pipe_fit.predict_proba(Xb)[:, 1]
    yb_hat = np.full(len(yb), -1, dtype=int)
    yb_hat[prob_b <= neg_thr] = 0
    yb_hat[prob_b >= pos_thr] = 1
    decided_batch = (yb_hat != -1)
    acc_batch = (yb_hat[decided_batch] == yb[decided_batch].values).mean() if decided_batch.any() else float("nan")

    # ---------- لایو (گام‌به‌گامِ امن) ----------
    preds_live, diff_cols = simulate_live_stepwise(
        prep, tail_buf, feat_cols, window,
        inference_pipe=pipe_fit,   # ← همون pipeline
        neg_thr=neg_thr, pos_thr=pos_thr,
        X_batch=Xb,
    )

    # هم‌طول‌سازی کامل با batch
    L = min(len(preds_live), len(yb_hat))
    preds_live = preds_live[:L]
    yb_hat = yb_hat[:L]
    yb_cut = yb.values[:L]

    decided_live_mask = (preds_live != -1)
    acc_live = (preds_live[decided_live_mask] == yb_cut[decided_live_mask]).mean() if decided_live_mask.any() else float("nan")

    # ---------- گزارش ----------
    if L < args.n_test:
        LOG.info("ℹ️ تعداد ردیف‌های قابل‌مقایسه کمتر از n_test شد: L=%d (n_test=%d). احتمالاً به‌خاطر warm-up طولانی برخی اندیکاتورها.", L, args.n_test)

    print("\n========== ACCURACY REPORT (last {:,} rows requested; compared {:,}) ==========".format(args.n_test, L))
    print(f"Batch accuracy (decided rows) : {acc_batch:.4f}")
    print(f"Live  accuracy (decided rows) : {acc_live:.4f}")
    print(f"Predicted rows (live)         : {decided_live_mask.sum()} / {L}")
    print(f"  • Correct                   : {(preds_live[decided_live_mask]==yb_cut[decided_live_mask]).sum()}")
    print(f"  • Wrong                     : {(preds_live[decided_live_mask]!=yb_cut[decided_live_mask]).sum()}")
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
