#!/usr/bin/env python3
"""
realtime_live_backtest_updated.py
End-to-end consistency test for the GA-trained model.
The LIVE loop now feeds **window+2** candles into `ready_incremental` 
so that, after the internal `shift(1).diff()` operation, at least
`window` valid rows remain and the annoying "Not enough rows" warnings
are eliminated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("tester")

# Silence verbose warnings coming from PREPARE_DATA_FOR_TRAIN
logging.getLogger("prepare_data_for_train").setLevel(logging.ERROR)

STRICT = False  # abort on first bad‑row if True

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def align_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[ordered_cols]


def has_bad_tail(row: pd.Series, binary_mask: dict[str, bool]) -> List[str]:
    bad: List[str] = []
    for col, val in row.items():
        if binary_mask.get(col, False):
            continue
        if pd.isna(val) or np.isclose(val, 0.0, atol=1e-12):
            bad.append(col)
    return bad

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(max_rows: int | None = None):
    # 1️⃣  Load model artefacts -------------------------------------------------
    if not MODEL_PATH.exists():
        LOG.error("Model file %s not found!", MODEL_PATH)
        sys.exit(1)

    LOG.info("🔍 Loading artefacts from %s …", MODEL_PATH)
    mdl = joblib.load(MODEL_PATH)
    pipe = mdl["pipeline"]
    window = int(mdl["window_size"])
    feats = list(mdl["feats"])
    final_cols = list(mdl.get("train_window_cols", feats))
    neg_thr = float(mdl["neg_thr"])
    pos_thr = float(mdl["pos_thr"])

    # ──🔍 بزرگ‌ترین look-back اندیکاتورها را به‌طور پویا پیدا کن ──
    import re
    ROLLING_REGEX = re.compile(r'_(\d{1,3})(?:$|_)')   # می‌گیرد  _14 , _52_ , _200
    lookbacks = [int(m.group(1)) for m in map(ROLLING_REGEX.search, final_cols) if m]
    ROLLING_LOOKBACK = max(lookbacks) if lookbacks else 1
    LOG.info("📏 Max rolling-lookback inferred = %d", ROLLING_LOOKBACK)

    # 2️⃣  Load & merge raw CSVs ----------------------------------------------
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

    filepaths = {tf: str(Path(fn)) for tf, fn in RAW_FILES.items()}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=MAIN_TIMEFRAME, verbose=False)
    merged = prep.load_data()
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    merged.sort_values(TIME_COL, inplace=True, ignore_index=True)
    
    pad = window + ROLLING_LOOKBACK + 1      # ← حالا مقدار درست

    # ─── moved below pad so it’s defined ───
    if not max_rows or max_rows <= 0:
        max_rows = max(4000, 3 * 20 + window + 20)

    required = max_rows + pad

    if len(merged) < required:
        LOG.error("❌ Need %d rows but only %d available after preprocessing.", required, len(merged))
        sys.exit(1)

    merged = merged.tail(required).reset_index(drop=True)
    LOG.info("📂 Merged dataframe → rows=%d, cols=%d", *merged.shape)

    # 3️⃣  Chimney snapshot ----------------------------------------------------
    X_all, y_all, _, _ = prep.ready(merged, window=window, selected_features=feats, mode="train")
    if X_all.empty:
        LOG.error("❌ Feature matrix empty after prep.ready – aborting.")
        sys.exit(1)

    X_eval = align_columns(X_all.copy(), final_cols).astype("float32")
    times = merged[TIME_COL].iloc[window : window + len(X_eval)].reset_index(drop=True)

    binary_mask = (X_eval.nunique() <= 2).to_dict()
    if bad_tail := has_bad_tail(X_eval.tail(1).iloc[0], binary_mask):
        LOG.warning("⚠️  Chimney last row contains %d suspect features", len(bad_tail))
        if STRICT:
            sys.exit(1)

    proba_ch = pipe.predict_proba(X_eval)[:, 1]
    label_ch = np.full_like(proba_ch, -1, dtype=int)
    label_ch[proba_ch <= neg_thr] = 0
    label_ch[proba_ch >= pos_thr] = 1

    df_ch = X_eval.assign(**{TIME_COL: times, "proba": proba_ch, "label": label_ch})
    df_ch.to_csv(SNAPSHOT_CSV, index=False)
    LOG.info("✅ Chimney snapshot → %s (rows=%d)", SNAPSHOT_CSV, len(df_ch))

    # 👉 Chimney metrics
    mask_ch = label_ch != -1
    if mask_ch.any():
        acc_ch = accuracy_score(y_all[mask_ch], label_ch[mask_ch])
        f1_ch = f1_score(y_all[mask_ch], label_ch[mask_ch])
        correct_ch = int(((label_ch == y_all.values) & mask_ch).sum())
        incorrect_ch = int(mask_ch.sum() - correct_ch)
        undecided_ch = int(len(label_ch) - mask_ch.sum())
        LOG.info("🏭 CHIMNEY metrics | Acc=%.4f | F1=%.4f | correct=%d | incorrect=%d | undecided=%d",
                 acc_ch, f1_ch, correct_ch, incorrect_ch, undecided_ch)
    else:
        LOG.warning("🏭 CHIMNEY produced no confident predictions.")

    # 4️⃣  LIVE simulation -----------------------------------------------------
    LOG.info("🎮 LIVE simulation …")
    live_rows: List[dict] = []
    y_true_all, y_pred_all = [], []

    # Feed window + ROLLING_LOOKBACK + 1 rows → همهٔ اندیکاتورها مقدار معتبر دارند
    pad = window + ROLLING_LOOKBACK + 1

    for idx in range(pad, len(merged) - 1):
        win_df = merged.iloc[idx - pad : idx + 1]
        X_live, _ = prep.ready_incremental(win_df, window=window, selected_features=feats)
        if X_live.empty:
            continue  # still skip but no warning flood
        X_live = align_columns(X_live, final_cols).astype("float32")

        proba = float(pipe.predict_proba(X_live)[0, 1])
        label = -1
        if proba <= neg_thr:
            label = 0
        elif proba >= pos_thr:
            label = 1

        cur_close = float(merged.iloc[idx][f"{MAIN_TIMEFRAME}_close"])
        next_close = float(merged.iloc[idx + 1][f"{MAIN_TIMEFRAME}_close"])
        y_true = int(next_close > cur_close)

        if label != -1:
            y_true_all.append(y_true)
            y_pred_all.append(label)

        live_rows.append({
            **X_live.iloc[0].to_dict(),
            TIME_COL: merged.at[idx, TIME_COL].strftime("%Y-%m-%d %H:%M:%S"),
            "proba": proba,
            "label": label,
            "y_true": y_true,
        })

    df_live = pd.DataFrame(live_rows)
    if df_live.empty:
        LOG.error("❌ Still no valid LIVE rows even after padding – investigate data/feature prep.")
        sys.exit(1)

    df_live.to_csv(LIVE_CSV, index=False)
    LOG.info("✅ LIVE snapshot → %s (rows=%d)", LIVE_CSV, len(df_live))

    if y_pred_all:
        acc = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all)
        correct = int(sum(yp == yt for yp, yt in zip(y_pred_all, y_true_all)))
        incorrect = int(len(y_pred_all) - correct)
        undecided = int(len(df_live) - len(y_pred_all))
        LOG.info("🎯 LIVE metrics | Acc=%.4f | F1=%.4f | correct=%d | incorrect=%d | undecided=%d",
                 acc, f1, correct, incorrect, undecided)
    else:
        LOG.warning("⚠️  No confident LIVE predictions.")
        LOG.warning("⚠️  No confident LIVE predictions.")

    # 5️⃣  Comparison ----------------------------------------------------------
    chimney = pd.read_csv(SNAPSHOT_CSV)
    live = pd.read_csv(LIVE_CSV)

    chimney[TIME_COL] = pd.to_datetime(chimney[TIME_COL])
    live[TIME_COL] = pd.to_datetime(live[TIME_COL])

    common = chimney[TIME_COL].isin(live[TIME_COL])
    chimney = chimney[common].reset_index(drop=True)
    live = live[live[TIME_COL].isin(chimney[TIME_COL])].reset_index(drop=True)

    min_len = min(len(chimney), len(live))
    chimney = chimney.tail(min_len).reset_index(drop=True)
    live = live.tail(min_len).reset_index(drop=True)

    diff_cols = [c for c in final_cols if c in chimney]
    diff_rows, lbl_diff = [], 0

    for i in range(min_len):
        col_diffs = [c for c in diff_cols if not (
            (pd.isna(chimney.at[i, c]) and pd.isna(live.at[i, c])) or
            np.isclose(chimney.at[i, c], live.at[i, c], atol=1e-6, rtol=1e-3)
        )]
        l1, l2 = int(chimney.at[i, "label"]), int(live.at[i, "label"])
        if l1 != l2:
            lbl_diff += 1
        if col_diffs or l1 != l2:
            diff_rows.append({
                "idx": i,
                "time": chimney.at[i, TIME_COL],
                "num_feat_diffs": len(col_diffs),
                "feat_diffs": col_diffs[:10],
                "lbl_chimney": l1,
                "lbl_live": l2,
            })

    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(COMPARE_CSV, index=False)

    # 🔝 most frequent differing features
    from collections import Counter
    feat_counter = Counter()
    for row in diff_rows:
        feat_counter.update(row["feat_diffs"])
    top_feats = feat_counter.most_common(15)
    if top_feats:
        LOG.info("🧐 Top mismatch features (count): %s", top_feats)

    LOG.info("📊 Comparison → %s | rows=%d | label mismatches=%d", COMPARE_CSV, min_len, lbl_diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch vs Live consistency check")
    parser.add_argument("--max-rows", type=int, default=-1, help="Tail N rows from raw data (auto if <=0)")
    args = parser.parse_args()

    main(max_rows=args.max_rows)
