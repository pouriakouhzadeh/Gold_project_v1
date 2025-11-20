#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generating_data_for_predict_script.py  —  نسخه‌ی جدید ژنراتور

نقش:
- تولید تست آفلاین برای اسکریپت دپلوی با N استپ آخر
- استفاده از همان window و همان train_window_cols که مدل با آن آموزش دیده
- بر پایه‌ی PREPARE_DATA_FOR_TRAIN.ready(..., with_times=True)
- هماهنگ با prediction_in_production_parity.py از طریق:
    • ساخت XAUUSD_*_live.csv تا timestamp مشخص
    • انتظار برای ردیف متناظر در deploy_predictions.csv
    • نوشتن generator_predictions.csv با متریک‌های نهایی
"""

from __future__ import annotations
import os, argparse, time, logging, json, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# اطمینان از اینکه مسیر پروژه در sys.path هست
for cand in (Path(__file__).resolve().parent, Path.cwd()):
    p = str(cand)
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

LOGFMT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT, datefmt="%Y-%m-%d %H:%M:%S")
L = logging.getLogger("generator")


# ---------------- Meta ----------------
def load_meta(model_dir: Path) -> Tuple[Dict[str, Any], int, List[str]]:
    meta_path = model_dir / "best_model.meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"best_model.meta.json not found in {model_dir}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    window = int(meta.get("window_size") or meta.get("window") or 1)
    train_cols = meta.get("train_window_cols") \
                 or meta.get("feats") \
                 or meta.get("feature_names") \
                 or []
    train_cols = list(train_cols)
    if not train_cols:
        raise RuntimeError("No train_window_cols / feats in best_model.meta.json")

    return meta, window, train_cols


# ---------------- Live writer ----------------
def write_live_csvs(base_dir: Path, symbol: str, ts: pd.Timestamp,
                    raw_paths: Dict[str, str]) -> None:
    """تا timestamp مشخص، برای هر TF یک *_live.csv می‌سازد."""
    for tf, fp in raw_paths.items():
        df = pd.read_csv(fp)
        if "time" not in df.columns:
            raise RuntimeError(f"{fp} has no 'time' column")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[df["time"] <= ts].copy()

        if tf == "30T":
            suffix = "M30"
        elif tf == "15T":
            suffix = "M15"
        elif tf == "5T":
            suffix = "M5"
        elif tf == "1H":
            suffix = "H1"
        else:
            suffix = tf

        out = base_dir / f"{symbol}_{suffix}_live.csv"
        df.to_csv(out, index=False)


# ---------------- Wait for deploy ----------------
def wait_for_deploy_prediction(base_dir: Path,
                               ts: pd.Timestamp,
                               poll_sec: float = 0.5,
                               timeout: float = 600.0) -> Dict[str, Any]:
    """
    منتظر می‌ماند تا prediction_in_production_parity.py
    یک ردیف با timestamp == ts در deploy_predictions.csv بنویسد.
    """
    pred_path = base_dir / "deploy_predictions.csv"
    t0 = time.time()

    while True:
        if pred_path.is_file():
            try:
                df = pd.read_csv(pred_path, parse_dates=["timestamp"])
                hit = df[df["timestamp"] == ts]
                if not hit.empty:
                    return hit.iloc[-1].to_dict()
            except Exception:
                pass

        if timeout is not None and (time.time() - t0) > timeout:
            raise TimeoutError(f"deploy did not write a row for {ts} within {timeout} seconds")

        time.sleep(poll_sec)


# ---------------- MAIN ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--symbol",   default="XAUUSD")
    ap.add_argument("--last-n",   type=int, default=200)
    ap.add_argument("--poll-sec", type=float, default=0.5)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    if args.verbosity <= 0:
        L.setLevel(logging.WARNING)

    base_dir = Path(args.base-dir if hasattr(args, "base-dir") else args.base_dir).resolve()
    symbol   = args.symbol

    L.info("=== Generator (NEW) started ===")
    L.info("Base dir: %s | Symbol: %s | last-n=%d", base_dir, symbol, args.last_n)

    # --- Meta ---
    meta, window, train_cols = load_meta(base_dir)
    L.info("Meta loaded | window=%d | train_cols=%d", window, len(train_cols))

    # --- Raw CSV paths ---
    filepaths = {
        "30T": base_dir / f"{symbol}_M30.csv",
        "15T": base_dir / f"{symbol}_M15.csv",
        "5T":  base_dir / f"{symbol}_M5.csv",
        "1H":  base_dir / f"{symbol}_H1.csv",
    }
    for tf, fp in filepaths.items():
        L.info("[raw] %s -> %s", tf, fp)

    # --- PREPARE_DATA_FOR_TRAIN برای گرفتن timestamp و y_true ---
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={k: str(v) for k, v in filepaths.items()},
        main_timeframe="30T",
        verbose=True,
        fast_mode=False,
        strict_disk_feed=False
    )
    merged = prep.load_data()

    X_all, y_all, _, price_ser, t_idx = prep.ready(
        merged,
        window=window,
        selected_features=train_cols,
        mode="train",
        with_times=True,
        predict_drop_last=False,
        train_drop_last=True
    )
    times = pd.to_datetime(pd.Series(t_idx))
    y_all = pd.Series(y_all).astype(int)

    idx_df = pd.DataFrame({"timestamp": times, "y_true": y_all})
    if idx_df.empty:
        raise RuntimeError("No windows produced by PREPARE_DATA_FOR_TRAIN; nothing to test.")

    if len(idx_df) < args.last_n:
        L.warning("Only %d windows available; reducing last-n to that.", len(idx_df))
        last_n = len(idx_df)
    else:
        last_n = args.last_n

    steps_df = idx_df.tail(last_n).reset_index(drop=True)
    L.info("Using last %d windows for generator test.", last_n)

    raw_paths_simple = {
        "30T": str(filepaths["30T"]),
        "15T": str(filepaths["15T"]),
        "5T":  str(filepaths["5T"]),
        "1H":  str(filepaths["1H"]),
    }

    ans_path        = base_dir / "answer.txt"
    gen_pred_path   = base_dir / "generator_predictions.csv"

    wins = loses = none = 0

    for i, row in steps_df.iterrows():
        ts     = row["timestamp"]
        y_true = int(row["y_true"])
        step   = i + 1

        # پاک کردن پاسخ قبلی (اگر هست)
        try:
            if ans_path.is_file():
                ans_path.unlink()
        except Exception:
            pass

        # نوشتن live CSVها تا ts
        write_live_csvs(base_dir, symbol, ts, raw_paths_simple)
        L.info("[Step %d/%d] live CSVs written for %s; waiting for deploy...", step, last_n, ts)

        # انتظار برای خروجی دپلوی
        pred = wait_for_deploy_prediction(base_dir, ts, poll_sec=args.poll_sec)
        act   = str(pred.get("action", "NONE")).upper()
        y_prob = float(pred.get("y_prob", np.nan))
        cover_dep = float(pred.get("cover_cum", np.nan))

        # به‌روزرسانی آمار
        if act == "NONE":
            none += 1
        else:
            if act == "BUY":
                if y_true == 1:
                    wins += 1
                else:
                    loses += 1
            elif act == "SELL":
                if y_true == 0:
                    wins += 1
                else:
                    loses += 1

        total_steps = step
        traded = wins + loses
        cover_gen = traded / max(1, total_steps)
        acc_gen   = wins / max(1, traded) if traded > 0 else 0.0

        L.info(
            "[Step %d] ts=%s act=%s y_true=%d | cover_gen=%.3f acc_gen=%.3f (wins=%d loses=%d none=%d) | cover_dep=%.3f",
            step, ts, act, y_true, cover_gen, acc_gen, wins, loses, none, cover_dep
        )

        # نوشتن generator_predictions.csv
        row_gen = {
            "timestamp": ts,
            "y_true": y_true,
            "action": act,
            "y_prob": y_prob,
            "cover_cum_deploy": cover_dep,
            "cover_cum_gen": cover_gen,
            "acc_cum_gen": acc_gen,
        }
        hdr = not gen_pred_path.is_file()
        pd.DataFrame([row_gen]).to_csv(
            gen_pred_path, mode="a", header=hdr, index=False
        )

    final_cover = (wins + loses) / max(1, len(steps_df))
    final_acc   = wins / max(1, wins + loses) if (wins + loses) > 0 else 0.0
    L.info(
        "[Final] acc_gen=%.3f cover_gen=%.3f wins=%d loses=%d none=%d",
        final_acc, final_cover, wins, loses, none
    )


if __name__ == "__main__":
    main()
