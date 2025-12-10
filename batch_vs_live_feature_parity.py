#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_vs_live_feature_parity.py

مقایسه‌ی فیچرهای حالت *batch* (آموزش) و حالت *live-simulated*
با منطق نزدیک به generating_data_for_predict_script.py و
prediction_in_production_next_candle.py.

ایده‌ی اصلی:

1) از روی CSVهای خام:
       XAUUSD_M30.csv / XAUUSD_M15.csv / XAUUSD_M5.csv / XAUUSD_H1.csv
   یک بار دیتای merge-شده را با PREPARE_DATA_FOR_TRAIN.load_data() می‌سازیم.

2) روی همین merged:
   - حالت batch (train):  PREP.ready(mode="train") روی کل دیتاست
     → X_batch, y_batch, t_batch
   - برای N کندل آخر M30 (idx_range مثل ژنراتور):
       * ts_now = زمان کندل M30
       * sub = merged[time_col <= ts_now]
       * X_all = PREP.ready(sub, mode="predict", predict_drop_last=True)
       * X_last, t_feat = آخرین سطر پایدار فیچرها
       * X_live[t_feat] = X_last

3) در انتها، برای زمان‌هایی که هم در batch و هم در live داریم
   (اشتراک t_batch_tail و index(X_live))، مقدار هر فیچر را مقایسه می‌کنیم
   و گزارش کامل + خلاصه می‌نویسیم:

   - features_parity_full.csv
   - features_parity_summary.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ---------------------------------------------------------
#  Helperهای مربوط به مسیرهای RAW و اسم *_live.csv
# ---------------------------------------------------------

try:
    from generating_data_for_predict_script import resolve_raw_paths, live_name
except Exception:
    # نسخه‌ی ساده‌تر در صورت در دسترس نبودن ماژول اصلی
    def resolve_raw_paths(base: Path, symbol: str) -> Dict[str, Path]:
        p_1h = base / f"{symbol}_M1H.csv"
        if not p_1h.is_file():
            p_1h = base / f"{symbol}_H1.csv"
        return {
            "30T": base / f"{symbol}_M30.csv",
            "15T": base / f"{symbol}_M15.csv",
            "5T": base / f"{symbol}_M5.csv",
            "1H": p_1h,
        }

    def live_name(path: Path) -> Path:
        return path.with_name(path.stem + "_live" + path.suffix)


# ---------------------------------------------------------
#  1) بارگذاری CSVهای خام
# ---------------------------------------------------------

def load_raw_csvs(base: Path, symbol: str) -> Tuple[Dict[str, Path], Dict[str, pd.DataFrame]]:
    """بارگذاری CSVهای خام برای هر TF، با تمیزکردن ستون time مثل ژنراتور."""
    raw_paths = resolve_raw_paths(base, symbol)
    raw: Dict[str, pd.DataFrame] = {}

    for tf, p in raw_paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Raw CSV for {tf} not found: {p}")
        df = pd.read_csv(p)

        if "time" not in df.columns:
            raise ValueError(f"'time' column missing in {p}")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        raw[tf] = df

    return raw_paths, raw


# ---------------------------------------------------------
#  2) Batch features (همان حالت TRAIN)
# ---------------------------------------------------------

def build_batch_features(
    prep: PREPARE_DATA_FOR_TRAIN,
    merged: pd.DataFrame,
    window: int,
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """فیچرهای batch برای کل دیتاست (همان حالتی که در TRAIN استفاده می‌شود)."""
    X_b, y_b, feats_b, close_b, t_idx_b = prep.ready(
        merged,
        window=window,
        selected_features=selected_features,
        mode="train",
        with_times=True,
        predict_drop_last=False,
        train_drop_last=False,
    )

    t_batch = pd.to_datetime(t_idx_b)
    X_b = pd.DataFrame(X_b).copy()
    X_b.index = t_batch

    if isinstance(y_b, pd.Series):
        y_s = y_b.copy()
        y_s.index = t_batch
    else:
        y_s = pd.Series(y_b, index=t_batch)

    price_s = pd.Series(close_b, index=t_batch)

    return X_b, y_s, price_s


# ---------------------------------------------------------
#  3) شبیه‌سازی مسیر LIVE
# ---------------------------------------------------------

def simulate_live_features(
    prep: PREPARE_DATA_FOR_TRAIN,
    merged: pd.DataFrame,
    raw_paths: Dict[str, Path],
    raw: Dict[str, pd.DataFrame],
    window: int,
    selected_features: Optional[List[str]],
    last_n_live: int,
    write_live_csv: bool = True,
) -> pd.DataFrame:
    """
    شبیه‌سازی مسیر live:
    - روی آخرین last_n_live کندل M30 حلقه می‌زند (مثل ژنراتور).
    - در هر ts_now:
        * در صورت نیاز *_live.csv می‌سازد (مثل ژنراتور).
        * sub = merged[time_col <= ts_now]
        * X_all = PREP.ready(sub, mode="predict", predict_drop_last=True)
        * X_last روی زمان t_feat را نگه می‌دارد.
    - در انتها، یک DataFrame با index=ts_feat (زمان فیچر) و ستون‌های فیچر برمی‌گرداند.
    """

    if "30T" not in raw:
        raise KeyError("raw['30T'] missing (need M30 data)")

    df30 = raw["30T"]
    if len(df30) < 2:
        raise ValueError("Not enough M30 data")

    total = len(df30)
    # مثل ژنراتور؛ برای y(t) = 1{close_{t+1} > close_t}
    max_idx_for_label = total - 2
    if max_idx_for_label <= 0:
        raise ValueError("Not enough rows in M30 for labels (need at least 2)")

    n_steps = min(last_n_live, max_idx_for_label)
    start_idx = max(0, max_idx_for_label - n_steps + 1)
    idx_range = range(start_idx, start_idx + n_steps)

    # مانند ژنراتور: طول برش‌ها در *_live
    SL = {"30T": 1000, "15T": 2000, "5T": 5000, "1H": 1000}

    # ستون زمان در merged
    tcol = f"{prep.main_timeframe}_time"
    if tcol not in merged.columns:
        if "time" in merged.columns:
            tcol = "time"
        else:
            raise ValueError(f"Time column not found in merged (expected {tcol} or 'time')")

    merged_local = merged.copy()
    merged_local[tcol] = pd.to_datetime(merged_local[tcol], errors="coerce")
    merged_local.dropna(subset=[tcol], inplace=True)
    merged_local.sort_values(tcol, inplace=True)
    merged_local.reset_index(drop=True, inplace=True)

    live_rows: List[pd.Series] = []

    for step, idx in enumerate(idx_range, start=1):
        ts_now = df30.loc[idx, "time"]

        # --- 1) ساخت *_live.csv مثل ژنراتور ---
        if write_live_csv:
            for tf, df_tf in raw.items():
                cut = df_tf[df_tf["time"] <= ts_now]
                if cut.empty:
                    continue
                tail_n = SL.get(tf, 500)
                cut = cut.tail(tail_n).copy()
                out_path = live_name(raw_paths[tf])
                cut.to_csv(out_path, index=False)

        # --- 2) مسیر دپلوی: sub تا ts_now از merged ---
        sub = merged_local[merged_local[tcol] <= ts_now].copy()
        if sub.empty:
            continue

        X_all, _, _, _, t_idx = prep.ready(
            sub,
            window=window,
            selected_features=selected_features,
            mode="predict",
            with_times=True,
            predict_drop_last=True,   # سطر آخر (کندل ناپایدار فعلی) حذف شود
            train_drop_last=False,
        )

        if len(X_all) == 0:
            continue

        t_idx = pd.to_datetime(t_idx)
        t_feat = t_idx.iloc[-1]  # زمان آخرین سطر پایدار فیچر

        X_all_df = pd.DataFrame(X_all)
        last_row = X_all_df.iloc[-1]

        s = pd.concat(
            [
                pd.Series({"ts_now": ts_now, "ts_feat": t_feat}),
                last_row.astype(float),
            ]
        )
        live_rows.append(s)

    if not live_rows:
        return pd.DataFrame()

    live_df = pd.DataFrame(live_rows)
    live_df["ts_feat"] = pd.to_datetime(live_df["ts_feat"])
    live_df.sort_values("ts_feat", inplace=True)

    # فقط آخرین مشاهده برای هر ts_feat را نگه داریم
    live_df = live_df.drop_duplicates(subset=["ts_feat"], keep="last")

    feature_cols = [c for c in live_df.columns if c not in ("ts_now", "ts_feat")]
    X_live_by_time = live_df.set_index("ts_feat")[feature_cols].copy()

    return X_live_by_time


# ---------------------------------------------------------
#  4) مقایسه‌ی batch vs live و ذخیره گزارش
# ---------------------------------------------------------

def compare_batch_vs_live(
    base_dir: Path,
    symbol: str = "XAUUSD",
    main_tf: str = "30T",
    window: int = 16,
    last_n_live: int = 2000,
    last_n_batch: int = 200,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> None:
    """اجرای کامل مقایسه batch vs live و نوشتن دو CSV خروجی."""
    base_dir = base_dir.resolve()
    print(f"Base dir = {base_dir} | symbol = {symbol}")

    # ---------- 1) بارگذاری CSVهای خام ----------
    raw_paths, raw = load_raw_csvs(base_dir, symbol)
    print("Raw paths:")
    for tf, p in raw_paths.items():
        print(f"  {tf}: {p}")

    # ---------- 2) ساخت PREPARE و merged ----------
    filepaths = {tf: str(p) for tf, p in raw_paths.items()}
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths=filepaths,
        main_timeframe=main_tf,
        verbose=True,
        fast_mode=False,          # مثل TRAIN
        strict_disk_feed=False,
    )
    merged = prep.load_data()

    # ستون زمان اصلی در merged
    tcol = f"{main_tf}_time"
    if tcol not in merged.columns and "time" in merged.columns:
        tcol = "time"

    merged[tcol] = pd.to_datetime(merged[tcol], errors="coerce")
    merged.dropna(subset=[tcol], inplace=True)
    merged.sort_values(tcol, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # ---------- 3) فیچرهای batch (حالت TRAIN) ----------
    print("=== Step 1: batch features on full dataset ===")
    X_batch, y_batch, price_batch = build_batch_features(
        prep=prep,
        merged=merged,
        window=window,
        selected_features=None,   # همه‌ی فیچرهای پایدار بعد از ready
    )

    if X_batch.empty:
        print("No rows from ready() in batch mode; aborting.")
        return

    # فقط last_n_batch ردیف آخر را برای مقایسه انتخاب می‌کنیم
    X_batch_tail = X_batch.tail(last_n_batch).copy()
    times_batch_tail = X_batch_tail.index.to_list()
    print(
        f"Batch tail: {len(X_batch_tail)} rows from {times_batch_tail[0]} to {times_batch_tail[-1]}"
    )

    # ---------- 4) فیچرهای live-simulated ----------
    print("=== Step 2: live-simulated features (generator + deploy style) ===")
    X_live = simulate_live_features(
        prep=prep,
        merged=merged,
        raw_paths=raw_paths,
        raw=raw,
        window=window,
        selected_features=list(X_batch.columns),  # همان ستون‌هایی که در batch داریم
        last_n_live=last_n_live,
        write_live_csv=True,
    )

    if X_live.empty:
        print("No live features produced; aborting.")
        return

    times_live = X_live.index.to_list()
    print(
        f"Live features: {len(X_live)} unique ts_feat from {times_live[0]} to {times_live[-1]}"
    )

    # ---------- 5) هم‌ترازسازی زمان‌ها ----------
    common_times = X_batch_tail.index.intersection(X_live.index)
    if len(common_times) == 0:
        print("No common timestamps between batch tail and live features; aborting.")
        return

    Xb = X_batch_tail.loc[common_times].copy()
    Xl = X_live.loc[common_times].copy()

    # هم‌ترازسازی ستون‌ها
    common_cols = [c for c in Xb.columns if c in Xl.columns]
    Xb = Xb[common_cols]
    Xl = Xl[common_cols]

    print(
        f"Comparing {len(common_times)} timestamps × {len(common_cols)} features (batch tail vs live)"
    )

    # ---------- 6) محاسبه‌ی اختلاف‌ها ----------
    full_records = []
    col_stats = defaultdict(
        lambda: {"count": 0, "n_diff": 0, "max_diff": 0.0, "sum_diff": 0.0}
    )

    for t in common_times:
        row_batch = Xb.loc[t]
        row_live = Xl.loc[t]

        diff = (row_batch - row_live).astype(float)
        abs_diff = diff.abs()

        for c in common_cols:
            db = float(row_batch[c])
            dl = float(row_live[c])
            d = float(abs_diff[c])
            is_diff = not np.isclose(db, dl, rtol=rtol, atol=atol)

            full_records.append(
                {
                    "time": t,
                    "feature": c,
                    "batch_value": db,
                    "live_value": dl,
                    "abs_diff": d,
                    "is_diff": int(is_diff),
                }
            )

            s = col_stats[c]
            s["count"] += 1
            s["max_diff"] = max(s["max_diff"], d)
            s["sum_diff"] += d
            if is_diff:
                s["n_diff"] += 1

    full_path = base_dir / "features_parity_full.csv"
    full_df = pd.DataFrame(full_records)
    full_df.to_csv(full_path, index=False)
    print(f"Full parity log written to: {full_path}")

    # ---------- 7) خلاصه به تفکیک هر فیچر ----------
    summary_records = []
    for c, s in col_stats.items():
        if s["count"] == 0:
            continue
        summary_records.append(
            {
                "feature": c,
                "tests": s["count"],
                "n_diff": s["n_diff"],
                "ratio_diff": s["n_diff"] / s["count"],
                "max_diff": s["max_diff"],
                "mean_abs_diff": s["sum_diff"] / s["count"],
            }
        )

    summ_df = pd.DataFrame(summary_records).sort_values(
        "ratio_diff", ascending=False
    )
    summ_path = base_dir / "features_parity_summary.csv"
    summ_df.to_csv(summ_path, index=False)
    print(f"Summary written to: {summ_path}")


# ---------------------------------------------------------
#  5) entry point
# ---------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--window", default=16, type=int)
    ap.add_argument("--last-n-live", default=2000, type=int,
                    help="تعداد استپ‌های لایو (M30) که شبیه‌سازی می‌شود.")
    ap.add_argument("--last-n-batch", default=200, type=int,
                    help="تعداد ردیف‌های batch_tail برای مقایسه.")
    ap.add_argument("--atol", default=1e-9, type=float)
    ap.add_argument("--rtol", default=1e-6, type=float)
    args = ap.parse_args()

    compare_batch_vs_live(
        base_dir=Path(args.base_dir),
        symbol=args.symbol,
        main_tf="30T",
        window=int(args.window),
        last_n_live=int(args.last_n_live),
        last_n_batch=int(args.last_n_batch),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )


if __name__ == "__main__":
    main()
