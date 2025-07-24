#!/usr/bin/env python3
"""
realtime_live_backtest_updated.py  ─────────────────────────────────────────────
Consistency test (batch‑vs‑live) for the GA‑trained model.

Main fixes in this version (2025‑07‑22)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Robust time‑column guard** – auto‑detects/renames the *_time column when it
  is missing (e.g. after CSV round‑trip) both right after `load_data()` *and*
  at the beginning of the comparison block.
* **Unified PAD calculation** based on `max_lookback` + `window`.
* Keeps the median‑imputation + ∞ → NaN cleaning that prevents the previous
  ValueError.
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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
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

STRICT = False   # if True ⇒ abort as soon as a suspect row/column is detected
WARM_UP_EXTRA = 2
# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("tester")
logging.getLogger("prepare_data_for_train").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def align_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    """Add missing columns (filled with NaN) & reorder."""
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[ordered_cols]


def has_bad_tail(row: pd.Series, binary_mask: dict[str, bool]) -> List[str]:
    """Return non‑binary columns whose last value is NaN or ~0."""
    bad = []
    for col, val in row.items():
        if binary_mask.get(col, False):
            continue
        if pd.isna(val) or np.isclose(val, 0.0, atol=1e-12):
            bad.append(col)
    return bad


def ensure_time_col(df: pd.DataFrame, tag: str) -> None:
    """Guarantee that TIME_COL is present; if not, rename the first *_time col."""
    global TIME_COL
    if TIME_COL in df.columns:
        return
    # seek alternative
    alts = [c for c in df.columns if c.endswith("_time")]
    if not alts:
        raise KeyError(f"[{tag}] DataFrame has no <*_time> column!")
    LOG.warning("⚠️  %s: '%s' not found – using '%s' instead", tag, TIME_COL, alts[0])
    df.rename(columns={alts[0]: TIME_COL}, inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# Main routine
# ══════════════════════════════════════════════════════════════════════════════

def main(max_rows: int | None = None) -> None:
    # 1️⃣  ── Load model ───────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        LOG.error("Model file %s not found!", MODEL_PATH)
        sys.exit(1)

    mdl = joblib.load(MODEL_PATH)
    pipe      = mdl["pipeline"]
    window    = int(mdl["window_size"])
    feats     = list(mdl["feats"])
    final_cols = list(mdl.get("train_window_cols", feats))
    neg_thr   = float(mdl["neg_thr"])
    pos_thr   = float(mdl["pos_thr"])

    # Detect the largest rolling look‑back from column names
    import re
    rk = re.compile(r"_(\d{1,3})(?:$|_)")
    lookbacks = [int(m.group(1)) for m in map(rk.search, final_cols) if m]
    max_lb    = max(lookbacks) if lookbacks else 1
    PAD       = window + max_lb + 2
    LOG.info("📏 max‑lookback=%d  |  PAD=%d", max_lb, PAD)

    # 2️⃣  ── Load & merge raw CSVs ────────────────────────────────────────────
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # heavy import

    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={tf: str(Path(fp)) for tf, fp in RAW_FILES.items()},
        main_timeframe=MAIN_TIMEFRAME,
        verbose=False,
    )
    merged = prep.load_data()

    # ‑‑ Guard #1 : ensure TIME_COL exists in merged
    ensure_time_col(merged, "merged")

    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    merged.sort_values(TIME_COL, inplace=True, ignore_index=True)

    if not max_rows or max_rows <= 0:
        max_rows = max(4000, 3 * 20 + window + 20)

    required = max_rows + PAD
    if len(merged) < required:
        LOG.error("❌ Need %d rows but only %d after preprocessing", required, len(merged))
        sys.exit(1)

    merged = merged.tail(required).reset_index(drop=True)
    LOG.info("📂 merged df  rows=%d  cols=%d", *merged.shape)

    # 3️⃣  ── Chimney (batch) snapshot ────────────────────────────────────────
    X_all, y_all, _, _ = prep.ready(merged, window=window, selected_features=feats, mode="train")
    if X_all.empty:
        LOG.error("❌ prep.ready() returned empty feature matrix – aborting.")
        sys.exit(1)

    X_eval = align_columns(X_all.copy(), final_cols).astype("float32")
    X_eval.replace([np.inf, -np.inf], np.nan, inplace=True)   # مثل live
    TRAIN_MEDIANS = X_eval.median()                           # محاسبه‌ی مدین واحد
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

    pd.concat([
        X_eval,
        pd.Series(times, name=TIME_COL),
        pd.Series(proba_ch, name="proba"),
        pd.Series(label_ch, name="label"),
    ], axis=1).to_csv(SNAPSHOT_CSV, index=False)

    mask_ch = label_ch != -1
    acc_ch  = accuracy_score(y_all[mask_ch], label_ch[mask_ch]) if mask_ch.any() else 0.0
    f1_ch   = f1_score(y_all[mask_ch], label_ch[mask_ch])       if mask_ch.any() else 0.0
    LOG.info("🏭 CHIMNEY metrics | Acc=%.4f | F1=%.4f | decided=%d/%d",
             acc_ch, f1_ch, int(mask_ch.sum()), len(label_ch))

    # 4️⃣  ── LIVE loop ───────────────────────────────────────────────────────
    y_true_lst: list[int] = []
    y_pred_lst: list[int] = []
    live_rows:  list[dict] = []

    for idx in range(PAD, len(merged) - 1):
            # مرحلهٔ گرم‌کردن؛ هنوز پیش‌بینی نمی‌گیریم
        if idx < PAD * WARM_UP_EXTRA:
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
        LOG.error("❌ LIVE loop produced zero valid rows – investigate!")
        sys.exit(1)

    df_live.to_csv(LIVE_CSV, index=False)

    if y_pred_lst:
        acc_live = accuracy_score(y_true_lst, y_pred_lst)
        f1_live  = f1_score(y_true_lst, y_pred_lst)
        LOG.info("🎯 LIVE metrics | Acc=%.4f | F1=%.4f | decided=%d/%d",
                 acc_live, f1_live, len(y_pred_lst), len(df_live))
    else:
        LOG.warning("⚠️  No confident LIVE predictions.")

    # 5️⃣  ── Comparison ──────────────────────────────────────────────────────
    chimney = pd.read_csv(SNAPSHOT_CSV)
    live    = pd.read_csv(LIVE_CSV)

    # Guard #2 – ensure TIME_COL exists in snapshot dataframes
    ensure_time_col(chimney, "chimney")
    ensure_time_col(live,    "live")

    chimney[TIME_COL] = pd.to_datetime(chimney[TIME_COL])
    live[TIME_COL]    = pd.to_datetime(live[TIME_COL])

    # Align on common timestamps
    common_mask = chimney[TIME_COL].isin(live[TIME_COL])
    chimney = chimney[common_mask].reset_index(drop=True)
    live    = live[live[TIME_COL].isin(chimney[TIME_COL])].reset_index(drop=True)

    # Compare last N rows (use the shorter length just in case)
    min_len   = min(len(chimney), len(live))
    chimney   = chimney.tail(min_len).reset_index(drop=True)
    live      = live.tail(min_len).reset_index(drop=True)

    diff_cols = [c for c in final_cols if c in chimney.columns]
    lbl_diff  = 0
    diff_rows = []
    for i in range(min_len):
        feat_diffs = [c for c in diff_cols if not (
            (pd.isna(chimney.at[i, c]) and pd.isna(live.at[i, c])) or
            np.isclose(chimney.at[i, c], live.at[i, c], atol=1e-6, rtol=1e-3)
        )]
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

    # Show top differing features
    from collections import Counter
    top_feats = Counter(sum((r["feat_diffs"] for r in diff_rows), [])).most_common(15)
    if top_feats:
        LOG.info("🧐 Top mismatch features (count) → %s", top_feats)
    LOG.info("📊 Comparison done | label mismatches=%d | top feat diffs written to %s",
             lbl_diff, COMPARE_CSV)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    argp = argparse.ArgumentParser("Batch vs. live consistency check")
    argp.add_argument("--max-rows", type=int, default=-1,
                      help="Tail N rows of raw data (auto if <=0)")
    main(argp.parse_args().max_rows)
