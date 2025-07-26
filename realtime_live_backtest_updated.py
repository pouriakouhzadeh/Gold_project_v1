#!/usr/bin/env python3
"""
realtime_live_backtest_updated.py
─────────────────────────────────────────────────────────────────────────────
Consistency test (CHIMNEY batch vs. LIVE incremental) برای مدل GA-trained.

اصلاحات:
• حلقهٔ LIVE اکنون از  prep.ready_incremental  استفاده می‌کند (کاملاً همسان با منطق
  آنلاین) و در مرحلهٔ «گرم‌کردن» فقط بافر را پُر می‌کند؛ هیچ پیش‌بینی‌ای
  ثبت نمی‌شود.
• نگهبان ستون زمان، PAD پویا، پر کردن NaN با medianِ دودکش و …
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ─────────────────────────────  ثابت‌ها  ────────────────────────────────
MODEL_PATH = Path("best_model.pkl")
RAW_FILES = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T": "XAUUSD_M5.csv",
    "1H": "XAUUSD_H1.csv",
}
MAIN_TIMEFRAME = "30T"
TIME_COL = f"{MAIN_TIMEFRAME}_time"
SNAPSHOT_CSV = "chimney_snapshot.csv"
LIVE_CSV = "live_snapshot.csv"
COMPARE_CSV = "comparison_differences.csv"

STRICT = False          # در صورت True بلافاصله روی خطای suspect متوقف می‌شود
WARM_UP_EXTRA = 2       # چند برابر PAD برای بافرِ گرم‌سازی (۰ ⇒ بدون گرم‌سازی)

# ────────────────────────────  لاگینگ  ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("tester")
logging.getLogger("prepare_data_for_train").setLevel(logging.ERROR)

# ───────────────────────────  توابع کمکی  ──────────────────────────────
def align_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[ordered_cols]

def has_bad_tail(row: pd.Series, binary_mask: dict[str, bool]) -> List[str]:
    bad = []
    for col, val in row.items():
        if binary_mask.get(col, False):
            continue
        if pd.isna(val) or np.isclose(val, 0.0, atol=1e-12):
            bad.append(col)
    return bad

def ensure_time_col(df: pd.DataFrame, tag: str) -> None:
    global TIME_COL
    if TIME_COL in df.columns:
        return
    alts = [c for c in df.columns if c.endswith("_time")]
    if not alts:
        raise KeyError(f"[{tag}] DataFrame has no <*_time> column!")
    LOG.warning("⚠️  %s: '%s' not found – using '%s' instead", tag, TIME_COL, alts[0])
    df.rename(columns={alts[0]: TIME_COL}, inplace=True)

# ─────────────────────────────  MAIN  ──────────────────────────────────
def main(max_rows: int | None = None) -> None:
    # 1) بارگذاری مدل
    if not MODEL_PATH.exists():
        LOG.error("Model file %s not found!", MODEL_PATH)
        sys.exit(1)
    mdl = joblib.load(MODEL_PATH)
    pipe       = mdl["pipeline"]
    window     = int(mdl["window_size"])
    feats      = list(mdl["feats"])
    final_cols = list(mdl.get("train_window_cols", feats))
    neg_thr    = float(mdl["neg_thr"])
    pos_thr    = float(mdl["pos_thr"])

    # بزرگ‌ترین look-back
    lookbacks = [int(m.group(1)) for m in map(re.search, [r"_(\d{1,3})(?:$|_)"]*len(final_cols), final_cols) if m]
    max_lb = max(lookbacks) if lookbacks else 1
    PAD    = window + max_lb + 2
    LOG.info("📏 max-lookback=%d  |  PAD=%d", max_lb, PAD)

    # 2) بارگذاری CSVها
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={tf: str(Path(p)) for tf, p in RAW_FILES.items()},
        main_timeframe=MAIN_TIMEFRAME,
        verbose=False,
    )
    merged = prep.load_data()
    ensure_time_col(merged, "merged")

    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    merged.sort_values(TIME_COL, inplace=True, ignore_index=True)

    if not max_rows or max_rows <= 0:
        max_rows = max(4000, 3 * 20 + window + 20)
    required = max_rows + PAD * (1 + WARM_UP_EXTRA)
    if len(merged) < required:
        LOG.error("❌ Need %d rows, have %d", required, len(merged))
        sys.exit(1)

    merged = merged.tail(required).reset_index(drop=True)
    LOG.info("📂 merged df  rows=%d  cols=%d", *merged.shape)

    # 3) دودکش (batch)
    X_all, y_all, _, _ = prep.ready(
        merged, window=window, selected_features=feats, mode="train"
    )
    if X_all.empty:
        LOG.error("❌ prep.ready() returned empty feature matrix.")
        sys.exit(1)

    X_eval = align_columns(X_all.copy(), final_cols).astype("float32")
    X_eval.replace([np.inf, -np.inf], np.nan, inplace=True)
    TRAIN_MEDIANS = X_eval.median()
    X_eval.fillna(TRAIN_MEDIANS, inplace=True)

    times = merged[TIME_COL].iloc[window : window + len(X_eval)].reset_index(drop=True)

    bad_tail = has_bad_tail(X_eval.tail(1).iloc[0], (X_eval.nunique() <= 2).to_dict())
    if bad_tail:
        LOG.warning("⚠️  CHIMNEY last row has %d suspect features", len(bad_tail))
        if STRICT:
            sys.exit(1)

    proba_ch  = pipe.predict_proba(X_eval)[:, 1]
    label_ch  = np.full_like(proba_ch, -1, dtype=int)
    label_ch[proba_ch <= neg_thr] = 0
    label_ch[proba_ch >= pos_thr] = 1

    pd.concat(
        [
            X_eval,
            pd.Series(times, name=TIME_COL),
            pd.Series(proba_ch, name="proba"),
            pd.Series(label_ch, name="label"),
        ],
        axis=1,
    ).to_csv(SNAPSHOT_CSV, index=False)

    mask_ch = label_ch != -1
    acc_ch  = accuracy_score(y_all[mask_ch], label_ch[mask_ch]) if mask_ch.any() else 0.0
    f1_ch   = f1_score(y_all[mask_ch], label_ch[mask_ch]) if mask_ch.any() else 0.0
    LOG.info("🏭 CHIMNEY metrics | Acc=%.4f | F1=%.4f | decided=%d/%d",
             acc_ch, f1_ch, int(mask_ch.sum()), len(label_ch))

    # 4) حلقهٔ LIVE (incremental واقعی)
    y_true_lst, y_pred_lst, live_rows = [], [], []

    for idx in range(PAD, len(merged) - 1):
        # گرم‌سازی: فقط بافر را پُر می‌کنیم
        if idx < PAD * WARM_UP_EXTRA:
            _ = prep.ready_incremental(
                merged.iloc[idx - PAD : idx + 1],
                window=window,
                selected_features=feats,
            )
            continue

        win_df = merged.iloc[idx - PAD : idx + 1]

        X_live, _ = prep.ready_incremental(
            win_df,
            window=window,
            selected_features=feats,
        )
        if X_live.empty:
            continue

        X_live = align_columns(X_live, final_cols).astype("float32")
        X_live.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_live.fillna(TRAIN_MEDIANS, inplace=True)

        proba = float(pipe.predict_proba(X_live)[0, 1])
        label = -1
        if proba <= neg_thr:
            label = 0
        elif proba >= pos_thr:
            label = 1

        cur_close  = float(merged.at[idx, f"{MAIN_TIMEFRAME}_close"])
        next_close = float(merged.at[idx + 1, f"{MAIN_TIMEFRAME}_close"])
        y_true = int(next_close > cur_close)

        if label != -1:
            y_true_lst.append(y_true)
            y_pred_lst.append(label)

        live_rows.append({
            **X_live.iloc[0].to_dict(),
            TIME_COL: merged.at[idx, TIME_COL].strftime("%Y-%m-%d %H:%M:%S"),
            "proba": proba,
            "label": label,
            "y_true": y_true,
        })

    df_live = pd.DataFrame(live_rows)
    if df_live.empty:
        LOG.error("❌ LIVE loop produced zero valid rows.")
        sys.exit(1)
    df_live.to_csv(LIVE_CSV, index=False)

    if y_pred_lst:
        acc_live = accuracy_score(y_true_lst, y_pred_lst)
        f1_live  = f1_score(y_true_lst, y_pred_lst)
        LOG.info("🎯 LIVE metrics | Acc=%.4f | F1=%.4f | decided=%d/%d",
                 acc_live, f1_live, len(y_pred_lst), len(df_live))
    else:
        LOG.warning("⚠️  No confident LIVE predictions.")

    # 5) مقایسه
    chimney = pd.read_csv(SNAPSHOT_CSV)
    live    = pd.read_csv(LIVE_CSV)
    ensure_time_col(chimney, "chimney")
    ensure_time_col(live, "live")

    chimney[TIME_COL] = pd.to_datetime(chimney[TIME_COL])
    live[TIME_COL]    = pd.to_datetime(live[TIME_COL])

    chimney = chimney[chimney[TIME_COL].isin(live[TIME_COL])].reset_index(drop=True)
    live    = live[live[TIME_COL].isin(chimney[TIME_COL])].reset_index(drop=True)

    min_len = min(len(chimney), len(live))
    chimney = chimney.tail(min_len).reset_index(drop=True)
    live    = live.tail(min_len).reset_index(drop=True)

    diff_cols = [c for c in final_cols if c in chimney.columns]
    diff_rows, lbl_diff = [], 0
    for i in range(min_len):
        feat_diffs = [
            c for c in diff_cols
            if not ((pd.isna(chimney.at[i, c]) and pd.isna(live.at[i, c]))
                    or np.isclose(chimney.at[i, c], live.at[i, c], atol=1e-6, rtol=1e-3))
        ]
        if chimney.at[i, "label"] != live.at[i, "label"]:
            lbl_diff += 1
        if feat_diffs or chimney.at[i, "label"] != live.at[i, "label"]:
            diff_rows.append({
                "row": i,
                "time": chimney.at[i, TIME_COL],
                "num_feat_diffs": len(feat_diffs),
                "feat_diffs": feat_diffs[:10],
                "lbl_chimney": int(chimney.at[i, "label"]),
                "lbl_live": int(live.at[i, "label"]),
            })

    pd.DataFrame(diff_rows).to_csv(COMPARE_CSV, index=False)

    from collections import Counter
    top_feats = Counter(sum((r["feat_diffs"] for r in diff_rows), [])).most_common(15)
    if top_feats:
        LOG.info("🧐 Top mismatch features (count) → %s", top_feats)
    LOG.info("📊 Comparison done | label mismatches=%d | differences → %s",
             lbl_diff, COMPARE_CSV)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser("Batch vs Live consistency check")
    cli.add_argument("--max-rows", type=int, default=-1,
                     help="Tail N rows from raw data (auto if <=0)")
    main(cli.parse_args().max_rows)
