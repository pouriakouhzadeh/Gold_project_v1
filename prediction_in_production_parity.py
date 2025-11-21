#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_in_production_parity.py  —  اسکریپت دپلوی هماهنگ با ژنراتور (MT4-like)

نقش کلی
---------
- منتظر 4 فایل *_live.csv (M5 / M15 / M30 / H1) می‌ماند.
- روی merged تا ts_now دقیقاً مثل TRAIN فیچر می‌سازد (PREPARE_DATA_FOR_TRAIN.ready با همان پارامترها).
- مدل را اجرا می‌کند و نتایج را در این فایل‌ها می‌نویسد:
    • deploy_X_feed_log.csv      ← تمام فیچر + متادیتا برای هر استپ
    • deploy_X_feed_tail200.csv  ← آخرین ۲۰۰ ردیف فیچر
    • deploy_predictions.csv     ← خلاصه‌ی پیش‌بینی‌ها (timestamp, action, y_prob, y_true)
    • answer.txt                 ← فقط اکشن برای ژنراتور / MT4

نکته‌ی مهم:
- y_true در این اسکریپت مستقیماً از PREPARE_DATA_FOR_TRAIN می‌آید و با live_like_sim_v3 یکسان است.
- ژنراتور جدید برای محاسبه‌ی دقت، به جای محاسبه‌ی دستی تارگت، همین y_true را استفاده می‌کند.
"""

from __future__ import annotations

import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# مطمئن شویم مسیر پروژه در sys.path هست
for cand in (Path(__file__).resolve().parent, Path.cwd()):
    p = str(cand)
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

LOG = logging.getLogger("deploy_parity")


# ---------- Logging ----------
def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity > 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------- Artifacts ----------
def load_artifacts(model_dir: Path) -> Tuple[object, dict, List[str], int, float, float]:
    """
    مدل، متا و اطلاعات موردنیاز (ستون‌ها، window، آستانه‌ها) را برمی‌گرداند.
    """
    meta_path = model_dir / "best_model.meta.json"
    pkl_path = model_dir / "best_model.pkl"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Model file not found: {pkl_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    obj = joblib.load(pkl_path)
    # اگر مدل داخل دیکشنری ذخیره شده باشد
    if isinstance(obj, dict) and "pipeline" in obj:
        model = obj["pipeline"]
    else:
        model = obj

    window = int(meta.get("window_size") or meta.get("window") or 1)
    neg_thr = float(meta.get("neg_thr", 0.005))
    pos_thr = float(meta.get("pos_thr", 0.995))

    train_cols: List[str] = (
        meta.get("train_window_cols")
        or meta.get("feats")
        or meta.get("feature_names")
        or []
    )
    train_cols = list(train_cols)
    if not train_cols:
        raise RuntimeError("No train_window_cols / feats / feature_names in meta")

    return model, meta, train_cols, window, neg_thr, pos_thr


# ---------- Live files ----------
def live_files_ready(base_dir: Path, symbol: str) -> Tuple[bool, Dict[str, Path]]:
    files = {
        "30T": base_dir / f"{symbol}_M30_live.csv",
        "15T": base_dir / f"{symbol}_M15_live.csv",
        "5T": base_dir / f"{symbol}_M5_live.csv",
        "1H": base_dir / f"{symbol}_H1_live.csv",
    }
    ok = all(p.is_file() for p in files.values())
    return ok, files


def read_last_timestamp(m30_live: Path) -> Optional[pd.Timestamp]:
    """
    آخرین time موجود در فایل M30_live را برمی‌گرداند.
    """
    try:
        df = pd.read_csv(m30_live)
        if "time" not in df.columns or df.empty:
            return None
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        if df.empty:
            return None
        return pd.Timestamp(df["time"].iloc[-1])
    except Exception:
        return None


def remove_live_files(files: Dict[str, Path]) -> None:
    for p in files.values():
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# ---------- Feature alignment ----------
def align_columns(
    X: pd.DataFrame,
    train_cols: List[str],
    train_distribution_path: Path,
) -> pd.DataFrame:
    """
    X را طوری تنظیم می‌کند که دقیقاً ستون‌ها و ترتیب train_cols را داشته باشد.
    اگر ستونی نبود، از train_distribution.json میانه‌اش را می‌گذارد؛ اگر نبود → 0.0
    """
    X = X.copy()

    medians: Dict[str, float] = {}
    try:
        if train_distribution_path.is_file():
            td = json.loads(train_distribution_path.read_text(encoding="utf-8"))
            medians = td.get("medians", {}) or {}
    except Exception:
        medians = {}

    # ستون‌های مفقود را بساز
    for c in train_cols:
        if c not in X.columns:
            X[c] = float(medians.get(c, 0.0))

    # فقط همین ستون‌ها و همین ترتیب
    X = X[[c for c in train_cols if c in X.columns]]

    # نوع عددی و پر کردن NaN
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(0.0).astype("float64", copy=False)

    return X


# ---------- Decision ----------
def proba_to_action(p: float, neg_thr: float, pos_thr: float) -> str:
    if p <= neg_thr:
        return "SELL"
    if p >= pos_thr:
        return "BUY"
    return "NONE"


# ---------- MAIN ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--poll-sec", default=1.0, type=float)
    ap.add_argument(
        "--max-steps",
        default=0,
        type=int,
        help="برای تست آفلاین (مثلاً ۲۰۰ استپ) اینجا تعداد استپ را وارد کن؛ ۰ یعنی بی‌نهایت",
    )
    ap.add_argument("--verbosity", default=1, type=int)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    base_dir = Path(args.base_dir).resolve()
    symbol = args.symbol

    LOG.info("=== prediction_in_production_parity (DEPLOY) started ===")
    LOG.info("Base dir=%s | Symbol=%s", base_dir, symbol)

    # --- Load model & meta ---
    model, meta, train_cols, window, neg_thr, pos_thr = load_artifacts(base_dir)
    LOG.info(
        "Artifacts loaded | window=%d | thr=(%.3f, %.3f) | train_cols=%d",
        window,
        neg_thr,
        pos_thr,
        len(train_cols),
    )

    # --- Prepare PREPARE_DATA_FOR_TRAIN once (merged ثابت) ---
    filepaths = {
        "30T": base_dir / f"{symbol}_M30.csv",
        "15T": base_dir / f"{symbol}_M15.csv",
        "5T": base_dir / f"{symbol}_M5.csv",
        "1H": base_dir / f"{symbol}_H1.csv",
    }
    for tf, fp in filepaths.items():
        if not fp.is_file():
            LOG.error("Raw CSV for %s not found: %s", tf, fp)
            return
        LOG.info("[paths] %s -> %s", tf, fp)

    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={k: str(v) for k, v in filepaths.items()},
        main_timeframe="30T",
        verbose=True,
        fast_mode=False,
        strict_disk_feed=False,
    )
    merged = prep.load_data()
    tcol = "30T_time" if "30T_time" in merged.columns else "time"
    LOG.info("Merged data loaded: shape=%s (time column=%s)", merged.shape, tcol)

    # --- Output paths ---
    ans_path = base_dir / "answer.txt"
    feed_log_path = base_dir / "deploy_X_feed_log.csv"
    feed_tail_path = base_dir / "deploy_X_feed_tail200.csv"
    deploy_pred_path = base_dir / "deploy_predictions.csv"
    train_dist_path = base_dir / "train_distribution.json"

    # پاک کردن answer.txt قبلی اگر هست
    ans_path.unlink(missing_ok=True)

    # اگر می‌خواهی تست جدید تمیز باشد، این سه تا را پاک می‌کنیم
    feed_log_path.unlink(missing_ok=True)
    feed_tail_path.unlink(missing_ok=True)
    deploy_pred_path.unlink(missing_ok=True)

    steps_done = 0
    predicted_cnt = 0
    last_ts_seen: Optional[pd.Timestamp] = None

    LOG.info("Waiting for *_live.csv files from generator / MT4 ...")

    while True:
        if args.max_steps > 0 and steps_done >= args.max_steps:
            LOG.info("Max steps (%d) reached, stopping deploy.", args.max_steps)
            break

        # اگر ژنراتور هنوز answer.txt قبلی را نخورده، دست نزن
        if ans_path.exists():
            time.sleep(args.poll_sec)
            continue

        ok, live_files = live_files_ready(base_dir, symbol)
        if not ok:
            time.sleep(args.poll_sec)
            continue

        ts_now = read_last_timestamp(live_files["30T"])
        if ts_now is None:
            time.sleep(args.poll_sec)
            continue

        if last_ts_seen is not None and ts_now <= last_ts_seen:
            # استپ تازه‌ای نیست؛ اجازه بده ژنراتور لایو بعدی را بسازد
            time.sleep(args.poll_sec)
            continue

        # برش merged تا ts_now
        sub = merged[merged[tcol] <= ts_now].copy()
        if sub.empty:
            LOG.warning("No merged rows <= %s", ts_now)
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        # ساخت فیچرها مثل TRAIN (mode='train' تا y_true هم ساخته شود)
        try:
            X_all, y_all, _, price_ser, t_idx = prep.ready(
                sub,
                window=window,
                selected_features=train_cols,
                mode="train",
                with_times=True,
                predict_drop_last=False,
                train_drop_last=True,
            )
        except Exception as e:
            LOG.error("prep.ready failed at %s: %s", ts_now, e)
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        if X_all.empty or len(t_idx) == 0:
            LOG.warning("Empty features for ts_now=%s", ts_now)
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        # آخرین نمونه
        X_last = X_all.tail(1).reset_index(drop=True)
        X_last = align_columns(X_last, train_cols, train_dist_path)

        ts_feat = pd.to_datetime(pd.Series(t_idx).iloc[-1])
        y_true = None
        if hasattr(y_all, "__len__") and len(y_all):
            try:
                y_true = int(pd.Series(y_all).iloc[-1])
            except Exception:
                y_true = None

        # پیش‌بینی
        try:
            prob = float(model.predict_proba(X_last)[:, 1][0])
        except Exception as e:
            LOG.error("model.predict_proba failed at %s: %s", ts_now, e)
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        action = proba_to_action(prob, neg_thr, pos_thr)

        steps_done += 1
        if action != "NONE":
            predicted_cnt += 1
        cover_cum = predicted_cnt / max(1, steps_done)

        # --- 1) لاگ کامل فیچرها + متادیتا (feed_log) ---
        feat_row = X_last.copy()
        feat_row.insert(0, "timestamp", ts_feat)
        feat_row.insert(1, "timestamp_trigger", ts_now)
        feat_row["y_true"] = y_true
        feat_row["y_prob"] = prob
        feat_row["action"] = action
        feat_row["cover_cum"] = cover_cum
        feat_row["neg_thr"] = neg_thr
        feat_row["pos_thr"] = pos_thr

        header_log = not feed_log_path.is_file()
        feat_row.to_csv(
            feed_log_path,
            mode="a",
            header=header_log,
            index=False,
        )

        # tail 200 برای مقایسه با sim_X_feed_tail200
        try:
            (
                pd.read_csv(
                    feed_log_path,
                    parse_dates=["timestamp", "timestamp_trigger"],
                )
                .sort_values("timestamp")
                .tail(2000)
                .to_csv(feed_tail_path, index=False)
            )
        except Exception:
            pass

        # --- 2) خلاصه‌ی پیش‌بینی‌ها (deploy_predictions.csv) ---
        pred_row = pd.DataFrame(
            [
                {
                    "timestamp": ts_feat,
                    "timestamp_trigger": ts_now,
                    "action": action,
                    "y_true": y_true,
                    "y_prob": prob,
                    "cover_cum": cover_cum,
                }
            ]
        )
        header_pred = not deploy_pred_path.is_file()
        pred_row.to_csv(
            deploy_pred_path,
            mode="a",
            header=header_pred,
            index=False,
        )

        # --- 3) نوشتن answer.txt (برای ژنراتور / MT4) ---
        try:
            with ans_path.open("w", encoding="utf-8") as f:
                f.write(action)
        except Exception as e:
            LOG.error("Failed to write answer.txt: %s", e)

        msg = (
            "[Predict] step=%d ts_feat=%s ts_now=%s | p=%.6f → %s | y_true=%s | cover_cum=%.3f"
            % (steps_done, ts_feat, ts_now, prob, action, str(y_true), cover_cum)
        )
        print(msg)
        LOG.info(msg)

        # --- 4) پاک‌سازی فایل‌های live برای استپ بعدی ---
        remove_live_files(live_files)
        last_ts_seen = ts_now

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
