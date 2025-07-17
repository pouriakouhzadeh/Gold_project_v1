#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_live_vs_chimney.py – Full comparison between LIVE and CHIMNEY predictions and inputs
"""
import pandas as pd
import numpy as np
import argparse
import logging
import sys
from pathlib import Path

cli = argparse.ArgumentParser(description="Compare chimney vs live predictions and inputs")
cli.add_argument("--chimney", default="chimney_snapshot.csv", help="Offline snapshot file")
cli.add_argument("--live", default="live_snapshot.csv", help="Live simulated snapshot file")
cli.add_argument("--time-col", default="30T_time", help="Time column name")
cli.add_argument("--output", default="comparison_differences.csv")
cli.add_argument("--verbose", action="store_true")
args = cli.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("compare")

# Load files
chimney = pd.read_csv(args.chimney)
live = pd.read_csv(args.live)

chimney[args.time_col] = pd.to_datetime(chimney[args.time_col])
live[args.time_col] = pd.to_datetime(live[args.time_col])

# Align on time
common_times = chimney[args.time_col].isin(live[args.time_col])
chimney = chimney[common_times].reset_index(drop=True)
live = live[live[args.time_col].isin(chimney[args.time_col])].reset_index(drop=True)

min_len = min(len(chimney), len(live))
chimney = chimney.iloc[-min_len:].reset_index(drop=True)
live = live.iloc[-min_len:].reset_index(drop=True)

log.info("Common rows aligned: %d", min_len)

# Identify comparison columns
exclude_cols = {"label", "proba", "y_true", args.time_col}
common_cols = [c for c in chimney.columns if c in live.columns and c not in exclude_cols]

# Compare rows
diffs = []
total_diff_rows = 0
label_mismatch = 0
ytrue_mismatch = 0

for i in range(min_len):
    row_diffs = []
    for col in common_cols:
        v1, v2 = chimney.at[i, col], live.at[i, col]
        if pd.isnull(v1) and pd.isnull(v2):
            continue
        if isinstance(v1, float) or isinstance(v2, float):
            if not np.isclose(v1, v2, atol=1e-6, rtol=1e-3):
                row_diffs.append(col)
        elif v1 != v2:
            row_diffs.append(col)

    label1 = chimney.at[i, "label"] if "label" in chimney.columns else None
    label2 = live.at[i, "label"] if "label" in live.columns else None
    ytrue = live.at[i, "y_true"] if "y_true" in live.columns else None

    row_summary = {
        "index": i,
        "time": chimney.at[i, args.time_col],
        "diff_cols": row_diffs,
        "num_feature_diffs": len(row_diffs),
        "label_chimney": label1,
        "label_live": label2,
        "label_match": label1 == label2,
        "y_true": ytrue
    }

    if len(row_diffs) > 0 or (label1 != label2):
        total_diff_rows += 1
        if label1 != label2:
            label_mismatch += 1
        diffs.append(row_summary)

# Save results
df_out = pd.DataFrame(diffs)
df_out.to_csv(args.output, index=False)

log.info("Finished comparison.")
print(f"\n🔍 Total compared rows      : {min_len}")
print(f"🔁 Rows with any difference : {total_diff_rows}")
print(f"❌ Label mismatches         : {label_mismatch}")
print(f"📄 Report saved to          : {args.output}")
