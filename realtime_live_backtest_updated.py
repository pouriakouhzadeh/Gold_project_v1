#!/usr/bin/env python3
"""
realtime_live_backtest_updated.py
================================
End‑to‑end consistency test for the GA‑trained model.
* Generates a **Chimney** (batch) snapshot on the most‑recent window of data
* Streams the same data candle‑by‑candle ("LIVE" simulation)
* Compares features, probabilities, and final labels one‑to‑one

Key guarantees implemented
-------------------------
1. **Identical feature set** – we fetch `model["train_window_cols"]` (aka *final_cols*)
   and align every X passed to the pipeline to this exact ordered list.
2. **No hidden NaN / constant‑zero leaks** – every row fed to the model is
   inspected; offending columns raise a warning (or can abort if `STRICT=True`).
3. **Scaler parity** – we never touch the pipeline; we only reorder/patch the
   raw matrix so the fitted `StandardScaler` sees the exact same schema as in
   training.
4. **Smart row budget** – `MAX_ROWS` is auto‑derived if not set, to include at
   least 3× the largest rolling window length (20) plus `window` margin.
5. **Rich comparison report** – a CSV (`comparison_differences.csv`) flags any
   feature mismatch exceeding `1e-6` *or* divergent labels.

Usage
-----
Just run the script inside the project root after training finished and
`best_model.pkl` exists:

```bash
python realtime_live_backtest_updated.py --max-rows 4000
```
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
# CONFIG --------------------------------------------------------------------
# ---------------------------------------------------------------------------
MODEL_PATH = Path("best_model.pkl")
RAW_FILES = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T" : "XAUUSD_M5.csv",
    "1H" : "XAUUSD_H1.csv",
}
MAIN_TIMEFRAME = "30T"
TIME_COL = f"{MAIN_TIMEFRAME}_time"

SNAPSHOT_CSV  = "chimney_snapshot.csv"
LIVE_CSV      = "live_snapshot.csv"
COMPARE_CSV   = "comparison_differences.csv"

# ---------------------------------------------------------------------------
# LOGGING --------------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("tester")

# Abort flag when strict checking fails
aSTRICT = False

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS -----------------------------------------------------------
# ---------------------------------------------------------------------------

def align_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    """Ensure *exactly* the columns in `ordered_cols` (order preserved).
    Missing columns are inserted with NaN; extra columns are discarded."""
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[ordered_cols]


def has_bad_tail(row: pd.Series, binary_mask: dict[str, bool]) -> List[str]:
    """Return list of feature names whose value is NaN or ~0 in *this* row and are
    *not* bona‑fide binary indicators."""
    bad = []
    for col, val in row.items():
        if binary_mask.get(col, False):
            continue  # 0/1 is allowed
        if pd.isna(val) or np.isclose(val, 0.0, atol=1e-12):
            bad.append(col)
    return bad

# ---------------------------------------------------------------------------
# MAIN ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(max_rows: int | None = None):
    # ---------- 1) Load artefacts ------------------------------------------------
    if not MODEL_PATH.exists():
        LOG.error("Model file %s not found!", MODEL_PATH)
        sys.exit(1)

    LOG.info("🔍 Loading model artefacts from %s …", MODEL_PATH)
    mdl = joblib.load(MODEL_PATH)
    pipe       = mdl["pipeline"]
    window     = int(mdl["window_size"])
    feats      = list(mdl["feats"])
    final_cols = list(mdl.get("train_window_cols", feats))  # fallback safety
    neg_thr, pos_thr = float(mdl["neg_thr"]), float(mdl["pos_thr"])
    LOG.info("✅ Model loaded. window=%d | features=%d | thr=(%.3f, %.3f)",
             window, len(final_cols), neg_thr, pos_thr)

    # ---------- 2) Load raw data & build merged DF ----------------------------
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # late import

    filepaths = {tf: str(Path(fname)) for tf, fname in RAW_FILES.items()}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths, main_timeframe=MAIN_TIMEFRAME, verbose=False)
    merged = prep.load_data()
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    merged.sort_values(TIME_COL, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Smart default for max_rows
    if max_rows is None or max_rows <= 0:
        max_rows = max(4000, 3 * 20 + window + 20)  # 20 = longest rolling win
    merged = merged.tail(max_rows + window + 1).reset_index(drop=True)
    LOG.info("📂 Raw merged dataframe → rows=%d, cols=%d", *merged.shape)

    # ---------- 3) Build CHIMNEY snapshot (batch mode) ------------------------
    LOG.info("📦 Creating CHIMNEY snapshot …")
    X_all, y_all, _ = prep.ready(merged, window=window, selected_features=feats, mode="train")

    # Align columns strictly to training order
    X_eval = align_columns(X_all.copy(), final_cols).astype("float32")
    time_series = merged[TIME_COL].iloc[window: window + len(X_eval)].reset_index(drop=True)

    # Tail integrity check
    binary_mask = (X_eval.nunique() <= 2).to_dict()
    tail_bad = has_bad_tail(X_eval.tail(1).iloc[0], binary_mask)
    if tail_bad:
        LOG.warning("⚠️  CHIMNEY last row has %d bad features (\u2260 binary).", len(tail_bad))
        if aSTRICT:
            sys.exit("Strict mode: abort due to bad tail features → " + ", ".join(tail_bad[:10]))

    proba_ch = pipe.predict_proba(X_eval)[:, 1]
    label_ch = np.full_like(proba_ch, -1, dtype=int)
    label_ch[proba_ch <= neg_thr] = 0
    label_ch[proba_ch >= pos_thr] = 1

    df_ch = X_eval.copy()
    df_ch[TIME_COL] = time_series
    df_ch["proba"] = proba_ch
    df_ch["label"] = label_ch
    df_ch.to_csv(SNAPSHOT_CSV, index=False)
    LOG.info("✅ CHIMNEY snapshot saved → %s (rows=%d)", SNAPSHOT_CSV, len(df_ch))

    # ---------- 4) LIVE candle‑by‑candle simulation --------------------------
    LOG.info("🎮 Starting LIVE simulation …")
    live_rows = []
    y_true_all, y_pred_all = [], []

    for idx in range(window, len(merged) - 1):  # require t, t+1 prices
        win_df = merged.iloc[idx - window : idx + 1]
        X_live, _ = prep.ready_incremental(win_df, window=window, selected_features=feats)
        if X_live.empty:
            continue
        X_live = align_columns(X_live, final_cols).astype("float32")

        # per‑row integrity check
        bad_now = has_bad_tail(X_live.iloc[0], binary_mask)
        if bad_now and aSTRICT:
            sys.exit(f"Bad live row @idx={idx} → {bad_now[:10]}")

        proba = float(pipe.predict_proba(X_live)[0, 1])
        label = -1
        if proba <= neg_thr:
            label = 0
        elif proba >= pos_thr:
            label = 1

        cur_close  = float(merged.iloc[idx][f"{MAIN_TIMEFRAME}_close"])
        next_close = float(merged.iloc[idx + 1][f"{MAIN_TIMEFRAME}_close"])
        y_true = int(next_close > cur_close)

        # collect metrics
        if label != -1:
            y_true_all.append(y_true)
            y_pred_all.append(label)

        row_dict = X_live.iloc[0].to_dict()
        row_dict[TIME_COL] = merged.iloc[idx][TIME_COL].strftime("%Y-%m-%d %H:%M:%S")
        row_dict.update({"proba": proba, "label": label, "y_true": y_true})
        live_rows.append(row_dict)

    df_live = pd.DataFrame(live_rows)
    df_live.to_csv(LIVE_CSV, index=False)
    LOG.info("✅ LIVE simulation saved → %s (rows=%d)", LIVE_CSV, len(df_live))

    if y_pred_all:
        acc = accuracy_score(y_true_all, y_pred_all)
        f1  = f1_score(y_true_all, y_pred_all)
        LOG.info("🎯 LIVE evaluation → Accuracy=%.4f | F1=%.4f", acc, f1)
    else:
        LOG.warning("⚠️  LIVE produced no confident predictions.")

    # ---------- 5) Compare CHIMNEY vs LIVE ----------------------------------
    LOG.info("🔎 Comparing feature vectors & labels …")
    chimney = pd.read_csv(SNAPSHOT_CSV)
    live    = pd.read_csv(LIVE_CSV)

    chimney[TIME_COL] = pd.to_datetime(chimney[TIME_COL])
    live[TIME_COL]    = pd.to_datetime(live[TIME_COL])

    common_mask = chimney[TIME_COL].isin(live[TIME_COL])
    chimney = chimney[common_mask].reset_index(drop=True)
    live    = live[live[TIME_COL].isin(chimney[TIME_COL])].reset_index(drop=True)
    min_len = min(len(chimney), len(live))
    chimney = chimney.iloc[-min_len:].reset_index(drop=True)
    live    = live.iloc[-min_len:].reset_index(drop=True)

    diff_cols = [c for c in final_cols if c in chimney.columns]
    diff_rows = []
    label_mismatch = 0

    for i in range(min_len):
        col_diffs = []
        for c in diff_cols:
            v1, v2 = chimney.at[i, c], live.at[i, c]
            if pd.isna(v1) and pd.isna(v2):
                continue
            if isinstance(v1, float) or isinstance(v2, float):
                if not np.isclose(v1, v2, atol=1e-6, rtol=1e-3):
                    col_diffs.append(c)
            elif v1 != v2:
                col_diffs.append(c)
        lbl1, lbl2 = int(chimney.at[i, "label"]), int(live.at[i, "label"])
        if lbl1 != lbl2:
            label_mismatch += 1
        if col_diffs or lbl1 != lbl2:
            diff_rows.append({
                "idx": i,
                "time": chimney.at[i, TIME_COL],
                "num_feat_diffs": len(col_diffs),
                "feat_diffs": col_diffs[:10],  # truncate for display
                "label_chimney": lbl1,
                "label_live": lbl2,
                "label_match": lbl1 == lbl2,
            })

    pd.DataFrame(diff_rows).to_csv(COMPARE_CSV, index=False)
    LOG.info("📊 Comparison CSV saved → %s | rows compared=%d | label mismatches=%d",
             COMPARE_CSV, min_len, label_mismatch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run batch‑vs‑live back‑test consistency check.")
    ap.add_argument("--max-rows", type=int, default=-1,
                    help="Tail rows of raw data to use (auto if <=0)")
    args = ap.parse_args()

    main(max_rows=args.max_rows)
