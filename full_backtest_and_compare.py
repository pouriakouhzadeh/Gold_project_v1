#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""full_backtest_and_compare_plus.py – snapshot, live, diff & smart diagnostics.

This script generates a chimney snapshot, simulates a live sliding‑window run,
compares the two, and prints / stores heuristic diagnostics so that the user
can quickly spot why inputs or accuracies diverge (data leakage, last‑row
NaNs/zeros, one‑row shift, scale mismatch, borderline thresholds …).

Main additions over the classic version
--------------------------------------
• `--stage-dump`   – dump every major dataframe stage as Parquet for offline diff
• `--focus-time`   – print column‑wise diff for a specific timestamp
• `--print-cols`   – regex to limit printed columns in focus diff
• `--html-report`  – generate colour‑coded HTML diff table
• `diagnostics_report.txt` – textual summary of root‑cause candidates

Exit codes: 0 (OK / compare skipped), 1 (IO / arg error), 2 (mismatches exist).
"""
from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_cli = argparse.ArgumentParser("full_backtest_and_compare_plus", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_cli.add_argument("--model", default="best_model.pkl")
_cli.add_argument("--data-dir", default=".")
_cli.add_argument("--main-tf", default="30T")
_cli.add_argument("--start")
_cli.add_argument("--rows", type=int, default=0)
_cli.add_argument("--snapshot", default="chimney_snapshot.csv")
_cli.add_argument("--live", default="live_snapshot.csv")
_cli.add_argument("--compare-with")
_cli.add_argument("--diff-csv", default="comparison_differences.csv")
_cli.add_argument("--abs-tol", type=float, default=1e-6)
_cli.add_argument("--rel-tol", type=float, default=1e-3)
_cli.add_argument("--neg-thr", type=float, help="Override negative threshold")
_cli.add_argument("--pos-thr", type=float, help="Override positive threshold")
_cli.add_argument("--dyn-thr", action="store_true")
_cli.add_argument("--whole-pass", action="store_true")
_cli.add_argument("--stage-dump", action="store_true")
_cli.add_argument("--focus-time")
_cli.add_argument("--print-cols")
_cli.add_argument("--html-report")
_cli.add_argument("--verbose", action="store_true")
ARGS = _cli.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if ARGS.verbose else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("full_backtest_and_compare_plus.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("full")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def smart_nan(val):
    """Return True if value is NaN/None/0.0 (likely meaningless)."""
    if val is None:
        return True
    if isinstance(val, float):
        return np.isnan(val) or val == 0.0
    return False


def ratio_stats(a: np.ndarray, b: np.ndarray):
    """Mean & std of a/b where both valid."""
    eps = 1e-12
    mask = (~np.isnan(a)) & (~np.isnan(b)) & (np.abs(b) > eps)
    if mask.sum() == 0:
        return np.nan, np.nan
    r = a[mask] / b[mask]
    return float(np.nanmean(r)), float(np.nanstd(r))

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
model_path = Path(ARGS.model).expanduser().resolve()
if not model_path.is_file():
    log.error("Model not found: %s", model_path)
    sys.exit(1)
mdl = joblib.load(model_path)
pipe = mdl["pipeline"]
neg_thr, pos_thr = float(mdl["neg_thr"]), float(mdl["pos_thr"])
if ARGS.neg_thr is not None:
    neg_thr = ARGS.neg_thr
if ARGS.pos_thr is not None:
    pos_thr = ARGS.pos_thr
log.info("Effective thresholds → neg=%.3f | pos=%.3f", neg_thr, pos_thr)
window = int(mdl["window_size"])
feats: List[str] = mdl["feats"]
all_cols: List[str] = mdl["train_window_cols"]
log.info("Model loaded | window=%d | thr=(%.3f, %.3f)", window, neg_thr, pos_thr)

# ---------------------------------------------------------------------------
# 2. Load & prepare data via PREPARE_DATA_FOR_TRAIN
# ---------------------------------------------------------------------------
RAW = {
    "30T": "XAUUSD_M30.csv",
    "15T": "XAUUSD_M15.csv",
    "5T": "XAUUSD_M5.csv",
    "1H": "XAUUSD_H1.csv",
}
filepaths = {k: str(Path(ARGS.data_dir) / v) for k, v in RAW.items()}
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # heavy import
prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=ARGS.main_tf, verbose=False)

time_col = f"{ARGS.main_tf}_time"
merged = prep.load_data()
merged[time_col] = pd.to_datetime(merged[time_col])
merged.sort_values(time_col, inplace=True)
merged.reset_index(drop=True, inplace=True)
if ARGS.start:
    merged = merged[merged[time_col] >= pd.Timestamp(ARGS.start)].reset_index(drop=True)
if ARGS.rows > 0 and len(merged) > ARGS.rows + window + 1:
    merged = merged.tail(ARGS.rows + window + 1).reset_index(drop=True)
log.info("Merged rows=%d", len(merged))

dump_dir = Path("stage_dump")
if ARGS.stage_dump:
    dump_dir.mkdir(exist_ok=True)
    merged.to_parquet(dump_dir / "stage0_clean.parquet")

# ---------------------------------------------------------------------------
# 3. Chimney snapshot
# ---------------------------------------------------------------------------
r = prep.ready(merged, window=window, selected_features=feats, mode="train")
X_all, y_all, feats_sel = r[:3]          # <-- فقط سه عضؤ اول را می‌گیریم

for c in all_cols:
    if c not in X_all.columns:
        X_all[c] = np.nan
X_eval = X_all[all_cols].astype("float32")
chimney_times = merged[time_col].iloc[window : window + len(X_eval)].dt.strftime("%Y-%m-%d %H:%M:%S")
chimney_df = X_eval.assign(**{time_col: chimney_times})
chimney_path = Path(ARGS.snapshot).expanduser().resolve()
chimney_df.to_csv(chimney_path, index=False)
log.info("Chimney snapshot → %s (%d rows)", chimney_path.name, len(chimney_df))
if ARGS.stage_dump:
    chimney_df.to_parquet(dump_dir / "stage1_window.parquet")
if ARGS.whole_pass:
    sys.exit(0)

# ---------------------------------------------------------------------------
# 4. Live simulation
# ---------------------------------------------------------------------------
try:
    from dynamic_threshold_adjuster import DynamicThresholdAdjuster
except ImportError:
    DynamicThresholdAdjuster = None
    if ARGS.dyn_thr:
        log.error("--dyn-thr requested but module not found")
        sys.exit(1)
_dyn = DynamicThresholdAdjuster() if (ARGS.dyn_thr and DynamicThresholdAdjuster) else None

rows_live: List[Dict] = []
y_true_all: List[int] = []
y_pred_all: List[int] = []
for idx in range(window, len(merged) - 1):
    win_df = merged.iloc[idx - window : idx + 1]
    X_live, _ = prep.ready_incremental(win_df, window=window, selected_features=feats)
    if X_live.empty:
        continue
    for c in all_cols:
        if c not in X_live.columns:
            X_live[c] = np.nan
    X_live = X_live[all_cols].astype("float32")

    n_thr, p_thr = neg_thr, pos_thr
    if _dyn is not None:
        last_atr = win_df.get(f"{ARGS.main_tf}_atr_14", pd.Series([np.nan])).iloc[-1]
        last_vol = win_df.get(f"{ARGS.main_tf}_volume", pd.Series([np.nan])).iloc[-1]
        n_thr, p_thr = _dyn.adjust(neg_thr, pos_thr, last_atr=last_atr, last_volume=last_vol)

    proba = float(pipe.predict_proba(X_live)[0, 1])
    label = -1
    if proba <= n_thr:
        label = 0
    elif proba >= p_thr:
        label = 1

    cur_close = merged.iloc[idx][f"{ARGS.main_tf}_close"]
    next_close = merged.iloc[idx + 1][f"{ARGS.main_tf}_close"]
    y_true = int(next_close > cur_close)

    if label != -1:
        y_true_all.append(y_true)
        y_pred_all.append(label)

    rows_live.append({
        **X_live.iloc[0].to_dict(),
        time_col: merged.iloc[idx][time_col].strftime("%Y-%m-%d %H:%M:%S"),
        "proba": proba,
        "label": label,
        "y_true": y_true,
    })


live_path = Path(ARGS.live).expanduser().resolve()
if rows_live:
    live_df = pd.DataFrame(rows_live)
else:
    log.warning("No confident predictions – writing empty CSV with headers only")
    live_df = pd.DataFrame(columns=all_cols + [time_col, "proba", "label", "y_true"])
live_df.to_csv(live_path, index=False)

log.info("Live snapshot → %s (%d rows)", live_path.name, len(live_df))
if ARGS.stage_dump:
    live_df.to_parquet(dump_dir / "stage_live_window.parquet")
if y_pred_all:
    log.info("LIVE metrics | N=%d | Acc=%.4f | F1=%.4f", len(y_pred_all), accuracy_score(y_true_all, y_pred_all), f1_score(y_true_all, y_pred_all))
else:
    log.warning("No confident predictions – metrics skipped")

# ---------------------------------------------------------------------------
# 5. Comparison & diagnostics
# ---------------------------------------------------------------------------
cmp_path = Path(ARGS.compare_with).expanduser().resolve() if ARGS.compare_with else chimney_path
if not cmp_path.is_file():
    log.info("Compare file not found – skip diff")
    sys.exit(0)

chimney_df = pd.read_csv(cmp_path)
live_df = pd.read_csv(live_path)
if live_df.empty or chimney_df.empty:
    log.warning("One of the snapshots is empty – skipping diff and exiting with code 2")
    sys.exit(2)


chimney_df[time_col] = pd.to_datetime(chimney_df[time_col])
live_df[time_col] = pd.to_datetime(live_df[time_col])

# Align rows by time
merge_times = chimney_df[time_col].isin(live_df[time_col])
chimney_df = chimney_df[merge_times].reset_index(drop=True)
live_df = live_df[live_df[time_col].isin(chimney_df[time_col])].reset_index(drop=True)
min_len = min(len(chimney_df), len(live_df))
chimney_df = chimney_df.iloc[-min_len:].reset_index(drop=True)
live_df = live_df.iloc[-min_len:].reset_index(drop=True)

feature_cols = [c for c in chimney_df.columns if c in live_df.columns and c not in {time_col, "proba", "label", "y_true"}]

diff_records: List[Dict] = []
label_mismatch = 0
col_counters: Dict[str, Counter] = defaultdict(Counter)

for i in range(min_len):
    feat_diffs = []
    for col in feature_cols:
        v1, v2 = chimney_df.at[i, col], live_df.at[i, col]
        if isinstance(v1, float) or isinstance(v2, float):
            if not np.isclose(v1, v2, equal_nan=True, atol=ARGS.abs_tol, rtol=ARGS.rel_tol):
                feat_diffs.append(col)
                col_counters[col]["_numeric_diff"] += 1
                if smart_nan(v2) and not smart_nan(v1):
                    col_counters[col]["insufficient_history"] += 1
        elif v1 != v2:
            feat_diffs.append(col)
            col_counters[col]["categorical_diff"] += 1
    if chimney_df.at[i, "label"] != live_df.at[i, "label"]:
        label_mismatch += 1

    if feat_diffs or chimney_df.at[i, "label"] != live_df.at[i, "label"]:
        diff_records.append({
            "idx": i,
            "time": chimney_df.at[i, time_col],
            "num_feat_diffs": len(feat_diffs),
            "feat_list": ";".join(feat_diffs),
            "label_ch": chimney_df.at[i, "label"],
            "label_live": live_df.at[i, "label"],
            "y_true": live_df.at[i, "y_true"] if "y_true" in live_df.columns else np.nan,
        })

# Save diff CSV
Path(ARGS.diff_csv).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(diff_records).to_csv(ARGS.diff_csv, index=False)
log.info("Diff CSV saved → %s | rows compared=%d | label mismatches=%d", ARGS.diff_csv, min_len, label_mismatch)

# ---------------------------------------------------------------------------
# Diagnostics report
# ---------------------------------------------------------------------------
report_lines = [f"Rows compared: {min_len}", f"Label mismatches: {label_mismatch}", "", "Feature diagnostics:"]
for col, ctr in sorted(col_counters.items(), key=lambda x: -sum(x[1].values())):
    total = sum(ctr.values())
    insuff = ctr.get("insufficient_history", 0)
    if insuff / total > 0.6:
        reason = "insufficient_history"
    else:
        mean_ratio, std_ratio = ratio_stats(live_df[col].values, chimney_df[col].values)
        if not np.isnan(mean_ratio) and abs(mean_ratio - 1) > 0.2 and std_ratio < 0.2:
            reason = "scale_mismatch"
        else:
            reason = "other"
    report_lines.append(f"- {col}: {total} diffs | top_reason={reason}")

with open("diagnostics_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log.info("Diagnostics report → diagnostics_report.txt")

# ---------------------------------------------------------------------------
# Focused diff print
# ---------------------------------------------------------------------------
if ARGS.focus_time:
    ts = pd.Timestamp(ARGS.focus_time)
    row_l = live_df[live_df[time_col] == ts]
    row_c = chimney_df[chimney_df[time_col] == ts]
    if not row_l.empty and not row_c.empty:
        regex = re.compile(ARGS.print_cols) if ARGS.print_cols else None
        cols_show = [c for c in feature_cols if (regex.search(c) if regex else True)] + ["label", "proba"]
        diff_focus = row_l[cols_show].transpose().merge(row_c[cols_show].transpose(), left_index=True, right_index=True, suffixes=("_live", "_chimney"))
        print("\n=== Focus diff @", ts, "===")
        print(diff_focus.to_string())
    else:
        log.warning("focus-time not found in aligned data")

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
if ARGS.html_report:
    df_html = pd.DataFrame(diff_records)
    def colour_row(row):
        colour = "" if row["num_feat_diffs"] == 0 and row["label_ch"] == row["label_live"] else "background-color: #ffdce0"  # light red
        return [colour] * len(row)
    styled = df_html.style.apply(colour_row, axis=1)
    html_out = Path(ARGS.html_report).expanduser().resolve()
    html_out.write_text(styled.to_html())
    log.info("HTML diff → %s", html_out.name)

# ---------------------------------------------------------------------------
# Exit code
# ---------------------------------------------------------------------------
exit_code = 2 if diff_records or label_mismatch else 0
sys.exit(exit_code)
