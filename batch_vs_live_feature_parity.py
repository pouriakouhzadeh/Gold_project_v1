#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_vs_live_feature_parity.py

مقایسه‌ی فیچرهای حالت batch و حالت live-simulated
برای N کندل آخر دیتاست.

خروجی:
- features_parity_full.csv
- features_parity_summary.csv
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN


def build_batch_features(
    filepaths: dict[str, str],
    main_timeframe: str,
    window: int,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """فیچرهای batch برای کل دیتاست."""
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths=filepaths,
        main_timeframe=main_timeframe,
        verbose=True,
    )
    merged = prep.load_data()

    X_batch, y_batch, feats_batch, price_batch, t_idx_batch = prep.ready(
        merged,
        window=window,
        selected_features=selected_features,
        mode="train",           # چون y(t+1) می‌سازی
        with_times=True,
        predict_drop_last=False,
        train_drop_last=False,
    )

    # اندیس X را برابر زمان قرار بده
    t_batch = pd.to_datetime(t_idx_batch)
    X_batch.index = t_batch

    return X_batch, pd.Series(y_batch, index=t_batch), pd.Series(price_batch, index=t_batch)


def build_live_row_for_time(
    prep: PREPARE_DATA_FOR_TRAIN,
    merged: pd.DataFrame,
    t_now: pd.Timestamp,
    window: int,
    selected_features: list[str],
) -> pd.Series | None:
    """
    شبیه‌سازی حالت لایو برای یک timestamp مشخص:
    - فقط دیتا تا t_now را می‌گیرد
    - ready(mode="predict") را صدا می‌زند
    - آخرین ردیف X مربوط به همان t_now را برمی‌گرداند
    """
    sub = merged[merged["time"] <= t_now].copy()
    if sub.empty:
        return None

    X_live, _, _, _, t_idx_live = prep.ready(
        sub,
        window=window,
        selected_features=selected_features,
        mode="predict",
        with_times=True,
        predict_drop_last=False,   # در مسیر PREDICT این پارامتر معنی دارد
        train_drop_last=False,
    )

    if len(X_live) == 0:
        return None

    t_live = pd.to_datetime(t_idx_live)
    X_live.index = t_live

    if t_now not in X_live.index:
        # ممکن است به دلیل window یا drop ها، t_now موجود نباشد
        return None

    return X_live.loc[t_now]


def compare_batch_vs_live(
    base_dir: Path,
    symbol: str = "XAUUSD",
    main_tf: str = "30T",
    window: int = 16,
    last_n: int = 2000,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> None:
    """مقایسه‌ی N سطر آخر batch vs live."""
    filepaths = {
        "30T": str(base_dir / f"{symbol}_M30.csv"),
        "15T": str(base_dir / f"{symbol}_M15.csv"),
        "5T": str(base_dir / f"{symbol}_M5.csv"),
        "1H": str(base_dir / f"{symbol}_H1.csv"),
    }

    print("=== Step 1: batch features on full dataset ===")
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths=filepaths,
        main_timeframe=main_tf,
        verbose=True,
    )
    merged = prep.load_data()
    X_batch, y_batch, price_batch = build_batch_features(
        filepaths=filepaths,
        main_timeframe=main_tf,
        window=window,
        selected_features=None,
    )

    if len(X_batch) == 0:
        print("No rows from ready() in batch mode; aborting.")
        return

    # آخرین last_n ردیف را برای مقایسه انتخاب کن
    n = min(last_n, len(X_batch))
    X_tail = X_batch.tail(n).copy()
    times_tail = X_tail.index.to_list()
    cols = list(X_tail.columns)

    print(f"Comparing last {n} rows (from {times_tail[0]} to {times_tail[-1]})")

    # جمع‌آوری اختلاف‌ها
    full_records = []
    col_stats = defaultdict(lambda: {"count": 0, "n_diff": 0, "max_diff": 0.0, "sum_diff": 0.0})

    for i, t_now in enumerate(times_tail, start=1):
        if i % 50 == 0 or i == n:
            print(f"[{i}/{n}] t_now={t_now}")

        row_batch = X_tail.loc[t_now]

        row_live = build_live_row_for_time(
            prep=prep,
            merged=merged,
            t_now=t_now,
            window=window,
            selected_features=cols,
        )
        if row_live is None:
            continue

        # هم‌ستون‌سازی
        common_cols = [c for c in cols if c in row_live.index]
        row_batch = row_batch[common_cols]
        row_live = row_live[common_cols]

        diff = (row_batch - row_live).astype(float)
        abs_diff = diff.abs()

        # ردیف کامل برای لاگ
        for c in common_cols:
            db = float(row_batch[c])
            dl = float(row_live[c])
            d = float(abs_diff[c])
            is_diff = not np.isclose(db, dl, rtol=rtol, atol=atol)
            full_records.append(
                {
                    "time": t_now,
                    "feature": c,
                    "batch_value": db,
                    "live_value": dl,
                    "abs_diff": d,
                    "is_diff": int(is_diff),
                }
            )

            # به‌روزرسانی آمار ستونی
            s = col_stats[c]
            s["count"] += 1
            s["max_diff"] = max(s["max_diff"], d)
            s["sum_diff"] += d
            if is_diff:
                s["n_diff"] += 1

    full_df = pd.DataFrame(full_records)
    full_path = base_dir / "features_parity_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Full parity log written to: {full_path}")

    # خلاصه به تفکیک هر فیچر
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

    summ_df = pd.DataFrame(summary_records).sort_values("ratio_diff", ascending=False)
    summ_path = base_dir / "features_parity_summary.csv"
    summ_df.to_csv(summ_path, index=False)
    print(f"Summary written to: {summ_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--window", default=16, type=int)
    ap.add_argument("--last-n", default=2000, type=int)
    ap.add_argument("--atol", default=1e-9, type=float)
    ap.add_argument("--rtol", default=1e-6, type=float)
    args = ap.parse_args()

    compare_batch_vs_live(
        base_dir=Path(args.base_dir).resolve(),
        symbol=args.symbol,
        main_tf="30T",
        window=int(args.window),
        last_n=int(args.last_n),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )


if __name__ == "__main__":
    main()
