#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_tail_rows.py
────────────────────────────────────────────────────────
هدف
────
مقایسهٔ «آخرین ردیف» Features بین دو مسیر آماده‌سازی داده در پروژهٔ Gold:

  1. **مسیر آفلاین**  ــ فراخوانی  `PREP.ready()`  روی کل تاریخچهٔ داده.
  2. **مسیر لایو**    ــ فراخوانی  `PREP.ready_incremental()`  روی همان تاریخچه
     به همراه آخرین کندل (آنچه واقعاً در  realtime_like_backtest  استفاده
     می‌شود).

اسکریپت ستون‌هایی را گزارش می‌کند که:
  * مقدارشان در یکی از دو مسیر  `NaN` یا `Inf` باشد؛ یا
  * اختلاف مطلقشان از آستانهٔ `--abs-thr` بیشتر باشد؛ یا
  * اختلاف نسبی‌شان از آستانهٔ `--rel-thr` بیشتر باشد.

هر نتیجه در یک فایل CSV ذخیره می‌شود و اگر ‌`--verbose`  بدهید خلاصهٔ ستون‌های
مشکل‌دار روی کنسول چاپ می‌شود.

مثال اجرا:
──────────
```bash
python compare_tail_rows.py \
       --model best_model.pkl \
       --abs-thr 0.05 --rel-thr 0.1 \
       --out tail_diff_report.csv --verbose
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Union

import math
import joblib
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ────────────────────────── CLI ──────────────────────────
parser = argparse.ArgumentParser("Compare last feature row of offline vs. live pipeline")
parser.add_argument("-m", "--model", default="best_model.pkl", help="مسیر مدل ذخیره‌شده (pickle)")
parser.add_argument("-o", "--out", default="tail_diff_report.csv", help="نام فایل خروجی گزارش")
parser.add_argument("--abs-thr", type=float, default=1e-4, help="آستانهٔ اختلاف مطلق")
parser.add_argument("--rel-thr", type=float, default=1e-2, help="آستانهٔ اختلاف نسبی (٪)")
parser.add_argument("--verbose", action="store_true", help="چاپ ستون‌های مشکل‌دار در کنسول")
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

ABS_THR = args.abs_thr
REL_THR = args.rel_thr

# ───────────────────── بارگذاری مدل و داده ─────────────────────
logging.info("Loading model: %s", args.model)
mdl = joblib.load(args.model)
WINDOW: int = mdl.get("window_size", 1)
FEATS: list[str] = mdl.get("feats", [])

prep = PREPARE_DATA_FOR_TRAIN(main_timeframe="30T", verbose=False)
logging.info("Loading merged dataset …")
MERGED = prep.load_data()
logging.info("Merged shape: %s", MERGED.shape)

# ───────────────────── آماده‌سازی آفلاین ─────────────────────
logging.info("Preparing features (offline)…")
try:
    offline_ready = prep.ready(MERGED.copy(), window=WINDOW, selected_features=FEATS, mode="predict")
except TypeError:
    # برخی نسخه‌ها ممکن است پارامتر mode نپذیرند
    offline_ready = prep.ready(MERGED.copy(), window=WINDOW, selected_features=FEATS)

# تابع ممکن است فقط X یا (X, y) برگرداند
if isinstance(offline_ready, tuple):
    X_offline: pd.DataFrame = offline_ready[0]
else:
    X_offline = offline_ready

if X_offline.empty:
    raise RuntimeError("PREP.ready() returned an empty dataframe — cannot compare.")

row_off = X_offline.tail(1).copy()
row_off.index = ["offline"]

# ───────────────────── آماده‌سازی لایو ─────────────────────
logging.info("Preparing features (live/incremental)…")
try:
    live_ready = prep.ready_incremental(MERGED.copy(), window=WINDOW, selected_features=FEATS)
except TypeError:
    live_ready = prep.ready_incremental(MERGED.copy(), window=WINDOW)

if isinstance(live_ready, tuple):
    X_live: pd.DataFrame = live_ready[0]
else:
    X_live = live_ready

if X_live.empty:
    raise RuntimeError("PREP.ready_incremental() returned an empty dataframe — cannot compare.")

row_live = X_live.tail(1).copy()
row_live.index = ["live"]

# ───────────────────── ادغام و مقایسه ─────────────────────
# اطمینان از یکسان بودن ترتیب ستون‌ها
row_live = row_live[row_off.columns]

combined = pd.concat([row_off, row_live])

records: list[dict[str, Union[str, float, bool]]] = []
for col in combined.columns:
    v_off = combined.loc["offline", col]
    v_live = combined.loc["live", col]

    # Handle NaN/Inf flags
    is_off_nan = pd.isna(v_off) or math.isinf(v_off)
    is_live_nan = pd.isna(v_live) or math.isinf(v_live)

    abs_diff = np.nan
    rel_diff = np.nan
    flag = False

    if not (is_off_nan or is_live_nan):
        abs_diff = float(abs(v_off - v_live))
        rel_diff = float(abs_diff / (abs(v_off) + 1e-12))  # avoid zero‑div
        flag = (abs_diff > ABS_THR) or (rel_diff > REL_THR)
    else:
        flag = True  # هرگونه NaN/Inf باید گزارش شود

    records.append({
        "column": col,
        "offline": v_off,
        "live": v_live,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "offline_nan_inf": is_off_nan,
        "live_nan_inf": is_live_nan,
        "flagged": flag,
    })

report_df = pd.DataFrame(records)

# ───────────────────── ذخیرهٔ گزارش ─────────────────────
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
report_df.to_csv(out_path, index=False)
logging.info("Report written → %s (flagged columns = %d)", out_path.resolve(), report_df["flagged"].sum())

# چاپ روی کنسول در حالت verbose
if args.verbose:
    flagged = report_df[report_df["flagged"]]
    if flagged.empty:
        print("\n✅  No significant differences detected. (abs_thr=%.4g, rel_thr=%.4g)" % (ABS_THR, REL_THR))
    else:
        print("\n⚠️  Columns with issues (|Δ|>%.4g or rel>%.4g or NaN/Inf):" % (ABS_THR, REL_THR))
        print(flagged[["column", "offline", "live", "abs_diff", "rel_diff", "offline_nan_inf", "live_nan_inf"]].to_string(index=False))
