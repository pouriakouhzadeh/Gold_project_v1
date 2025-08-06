#!/usr/bin/env python3
# compare_batch_vs_live.py
# ----------------------------------------------------------
# مقایسۀ دقیق ویژگی‌ها و خروجی مدل در حالت دودکش (batch) و لایو
# با تکیه بر artefact های ذخیره‌شده در best_model.pkl
# ----------------------------------------------------------

from __future__ import annotations
import argparse, logging, warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from model_pipeline_live import ModelPipelineLive          # نسخهٔ بدون SMOTE

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

# ----------------------------------------------------------
# --------- helper: بساز نسخهٔ لایوِ بدون SMOTE ------------
# ----------------------------------------------------------
def build_live_estimator(fitted_pipe: Pipeline,
                         keep_calibrator: bool = True) -> ModelPipelineLive:
    """
    از Pipe آموزش-دیده (scaler + smote + clf یا CalibratedCLF) یک
    ModelPipelineLive می‌سازد که *فقط* «scaler + clf» را نگه می‌دارد.
    """
    scaler      = fitted_pipe.named_steps["scaler"]
    trained_clf = fitted_pipe.named_steps["classifier"]

    # اگر در GA بخش calibrate فعال بوده، خودِ CalibratedCLF را نگه می‌داریم
    final_clf   = trained_clf
    hp_for_live = getattr(trained_clf, "estimator_", trained_clf).get_params()

    live = ModelPipelineLive(hyperparams=hp_for_live, calibrate=False)
    live.base_pipe = Pipeline([("scaler", scaler), ("clf", final_clf)])

    if keep_calibrator and isinstance(trained_clf, CalibratedClassifierCV):
        live._calibrator = trained_clf            # pylint: disable=protected-access
    return live

# ----------------------------------------------------------
# ---------- batch features و labels (آخر N_TEST ردیف) -----
# ----------------------------------------------------------
def build_batch(prep: PREPARE_DATA_FOR_TRAIN,
                window: int,
                feat_list: List[str],
                n_test: int) -> Tuple[pd.DataFrame, pd.Series]:
    merged      = prep.load_data()
    tail_slice  = merged.tail(n_test + window)              # کانتکستِ کافی
    Xb, yb, _, _ = prep.ready(
        tail_slice, window=window,
        selected_features=feat_list, mode="train"
    )
    Xb = Xb.tail(n_test).reset_index(drop=True)
    yb = yb.tail(n_test).reset_index(drop=True)
    return Xb, yb

# ----------------------------------------------------------
# ---------- شبیه‌سازی لایو رکورد-به-رکورد -----------------
# ----------------------------------------------------------
def simulate_live(prep: PREPARE_DATA_FOR_TRAIN,
                  raw: pd.DataFrame,
                  feat_cols: List[str],
                  window: int,
                  live_est,
                  neg_thr: float,
                  pos_thr: float,
                  n_test: int,
                  X_batch: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, set]:
    preds, gt = [], []
    diff_cols: set = set()

    buf = raw.tail(n_test + window).reset_index(drop=True)

    for i in range(len(buf)):
        new_row = buf.iloc[[i]]

        X_inc, _ = prep.ready_incremental(
            new_row, window=window, selected_features=feat_cols
        )
        if X_inc.empty:
            continue

        X_inc = X_inc[feat_cols].astype("float32")

        prob  = live_est.predict_proba(X_inc)[:, 1]
        label = np.full(1, -1, dtype=int)
        label[prob <= neg_thr] = 0
        label[prob >= pos_thr] = 1
        preds.append(label[0])

        if i + 1 < len(buf):          # برچسب آینده
            gt.append(int(buf[f"{prep.main_timeframe}_close"].iloc[i + 1]
                           - buf[f"{prep.main_timeframe}_close"].iloc[i] > 0))

        # اختلاف ویژگی با batch
        idx_b = len(preds) - 1
        if idx_b < len(X_batch):
            diff_mask = ~np.isclose(X_batch.iloc[idx_b].values,
                                    X_inc.iloc[0].values,
                                    atol=1e-10, equal_nan=True)
            if diff_mask.any():
                diff_cols.update(X_batch.columns[diff_mask])

    return np.asarray(preds), np.asarray(gt), diff_cols

# ----------------------------------------------------------
# ---------------------   CLI   ----------------------------
# ----------------------------------------------------------
def cli():
    p = argparse.ArgumentParser("Compare batch vs live outputs")
    p.add_argument("--model", default="best_model.pkl", help="Path to best_model.pkl")
    p.add_argument("--n-test", type=int, default=4_000, help="Rows to compare")
    p.add_argument("--data-dir", default=".", help="Folder of CSV price files")
    return p.parse_args()

# ----------------------------------------------------------
# -----------------------  MAIN  ---------------------------
# ----------------------------------------------------------
if __name__ == "__main__":
    args      = cli()
    mdl_path  = Path(args.model).expanduser().resolve()
    if not mdl_path.is_file():
        raise FileNotFoundError(f"❌  {mdl_path} not found")

    payload   = joblib.load(mdl_path)
    pipe_fit  : Pipeline = payload["pipeline"]
    window    : int      = int(payload["window_size"])
    neg_thr   : float    = float(payload["neg_thr"])
    pos_thr   : float    = float(payload["pos_thr"])
    feat_cols : List[str]= payload["train_window_cols"]   # ترتیبِ نهاییِ ستون‌ها

    # ---------- نسخهٔ لایوِ بدون SMOTE ----------
    live_est = build_live_estimator(pipe_fit)

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

    # ---------- دودکش ----------
    Xb, yb = build_batch(prep, window, feat_cols, args.n_test)
    prob_b = pipe_fit.predict_proba(Xb)[:, 1]
    yb_hat = np.full_like(yb.values, -1, dtype=int)
    yb_hat[prob_b <= neg_thr] = 0
    yb_hat[prob_b >= pos_thr] = 1
    acc_batch = (yb_hat[yb_hat != -1] == yb[yb_hat != -1]).mean()

    # ---------- لایو ----------
    raw_full = prep.load_data()
    preds_live, gt_live, diff_cols = simulate_live(
        prep, raw_full, feat_cols, window,
        live_est, neg_thr, pos_thr,
        args.n_test, Xb
    )

    # هم‌ترازسازی
    L = min(len(preds_live), len(gt_live))
    preds_live, gt_live = preds_live[:L], gt_live[:L]
    decided_mask        = preds_live != -1

    acc_live = (preds_live[decided_mask] == gt_live[decided_mask]).mean() \
               if decided_mask.any() else float("nan")

    # ---------- گزارش ----------
    print("\n========== ACCURACY REPORT (last {:,} rows) ==========".format(args.n_test))
    print(f"Batch accuracy (decided rows) : {acc_batch:.4f}")
    print(f"Live  accuracy (decided rows) : {acc_live:.4f}")
    print(f"Predicted rows (live)         : {decided_mask.sum()} / {args.n_test}")
    print(f"  • Correct                   : {(preds_live[decided_mask]==gt_live[decided_mask]).sum()}")
    print(f"  • Wrong                     : {(preds_live[decided_mask]!=gt_live[decided_mask]).sum()}")
    print(f"  • Unpredicted (-1)          : {args.n_test - decided_mask.sum()}")

    # ---------- اختلاف برچسب batch ↔ live ----------
    label_mismatch = int((yb_hat[:L] != preds_live[:L]).sum())
    print(f"\nLabel mismatch between batch-pred و live-pred : {label_mismatch}")

    # ---------- اختلاف فیچر ----------
    if diff_cols:
        print(f"\nFeature columns with any mismatch ({len(diff_cols)}):")
        for c in sorted(diff_cols):
            print("  -", c)
    else:
        print("\n✅  هیچ اختلاف مقداری بین فیچرهای batch و live یافت نشد.")
