#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_in_production_parity.py  —  نسخه‌ی جدید دپلوی با پاریتی کامل

ویژگی‌ها:
- بارگذاری آرتیفکت‌ها با joblib و استخراج pipeline از داخل دیکت ذخیره‌شده
- استفاده از PREPARE_DATA_FOR_TRAIN با همان window و train_window_cols مثل ترین
- ساخت X,y,times در حالت mode="train", train_drop_last=True, predict_drop_last=False
- برای هر استپ:
    • پیدا کردن ts_now از XAUUSD_M30_live.csv
    • ساخت زیرمجموعه‌ی merged تا ts_now (بدون دیتا از آینده)
    • آماده‌سازی فیچر آخر (X_last) و y_true متناسب
    • پیش‌بینی با model.predict_proba
    • تبدیل به BUY/SELL/NONE با آستانه‌ها
    • محاسبه‌ی cover_cum (کاور تجمعی)
    • نوشتن:
        - answer.txt (فقط اکشن)
        - deploy_predictions.csv (timestamp, y_true, y_prob, action, thr, cover_cum)
        - deploy_X_feed_log.csv و deploy_X_feed_tail200.csv (فیچرها + متادیتا)
    • حذف تمام *_live.csv برای استپ بعدی
- حلقه با پارامتر --max-steps برای تست آفلاین روی N استپ
"""

from __future__ import annotations
import os, sys, time, json, argparse, logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# اطمینان از اینکه مسیر پروژه در sys.path هست
for cand in (Path(__file__).resolve().parent, Path.cwd()):
    p = str(cand)
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

LOGFMT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT, datefmt="%Y-%m-%d %H:%M:%S")
L = logging.getLogger("deploy")


# ---------------- Artifacts ----------------
def load_artifacts(model_dir: Path) -> Tuple[Any, Dict[str, Any], List[str], int, float, float]:
    """
    best_model.pkl       → معمولا یک dict است که داخلش pipeline نگه‌داری شده
    best_model.meta.json → شامل window_size, neg_thr, pos_thr, train_window_cols
    """
    pkl_path  = model_dir / "best_model.pkl"
    meta_path = model_dir / "best_model.meta.json"

    if not pkl_path.is_file():
        raise FileNotFoundError(f"model pickle not found: {pkl_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta json not found: {meta_path}")

    # بارگذاری با joblib
    obj = joblib.load(pkl_path)

    # ⚠️ نکته مهم: در نسخه‌ی جدید ModelSaver، این obj یک dict است
    # که داخلش `pipeline` و بقیه اطلاعات ذخیره شده.
    # همین باعث شده بود قبلاً روی خود dict → predict_proba صدا بزنیم و خطا بگیریم.
    if isinstance(obj, dict) and "pipeline" in obj:
        model = obj["pipeline"]
    else:
        # اگر به هر دلیل مستقیماً خود مدل ذخیره شده بود
        model = obj

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    window = int(meta.get("window_size") or meta.get("window") or 1)
    neg_thr = float(meta.get("neg_thr", 0.005))
    pos_thr = float(meta.get("pos_thr", 0.995))

    train_cols = meta.get("train_window_cols") \
                 or meta.get("feats") \
                 or meta.get("feature_names") \
                 or []
    train_cols = list(train_cols)

    if not train_cols:
        raise RuntimeError("No train_window_cols / feats found in best_model.meta.json")

    return model, meta, train_cols, window, neg_thr, pos_thr


# ---------------- File helpers ----------------
def live_files_ready(base_dir: Path, symbol: str) -> Tuple[bool, Dict[str, Path]]:
    files = {
        "30T": base_dir / f"{symbol}_M30_live.csv",
        "15T": base_dir / f"{symbol}_M15_live.csv",
        "5T":  base_dir / f"{symbol}_M5_live.csv",
        "1H":  base_dir / f"{symbol}_H1_live.csv",
    }
    ok = all(p.is_file() for p in files.values())
    return ok, files


def read_last_timestamp(m30_live: Path) -> Optional[pd.Timestamp]:
    try:
        d = pd.read_csv(m30_live)
        if "time" not in d.columns or d.empty:
            return None
        d["time"] = pd.to_datetime(d["time"], errors="coerce")
        d = d.dropna(subset=["time"]).sort_values("time")
        if d.empty:
            return None
        return pd.Timestamp(d["time"].iloc[-1])
    except Exception:
        return None


def remove_live_files(files: Dict[str, Path]) -> None:
    for p in files.values():
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------- Feature alignment ----------------
def align_columns(X: pd.DataFrame, train_cols: List[str]) -> pd.DataFrame:
    """
    تضمین می‌کند X دقیقاً ستون‌ها و ترتیبِ train_cols را دارد.
    هر ستون مفقود با 0.0 پر می‌شود؛ ستون‌های اضافه حذف می‌شوند.
    """
    X = X.copy()
    for c in train_cols:
        if c not in X.columns:
            X[c] = 0.0

    # فقط همین ستون‌ها و به همین ترتیب
    X = X[train_cols]

    # عددی کردن و پر کردن NaN با 0.0
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(0.0).astype("float64", copy=False)

    return X


# ---------------- Decision ----------------
def proba_to_decision(p: float, neg_thr: float, pos_thr: float) -> str:
    if p <= neg_thr:
        return "SELL"
    if p >= pos_thr:
        return "BUY"
    return "NONE"


# ---------------- MAIN ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol",   default="XAUUSD", type=str)
    ap.add_argument("--poll-sec", default=1.0, type=float)
    ap.add_argument("--max-steps", default=0, type=int,
                   help="برای تست آفلاین روی N استپ، این را روی N تنظیم کن (۰ یعنی بی‌نهایت)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    symbol   = args.symbol

    L.info("=== prediction_in_production_parity (NEW) started ===")
    L.info("Base dir: %s | Symbol: %s", base_dir, symbol)

    # --- Load artifacts ---
    model, meta, train_cols, window, neg_thr, pos_thr = load_artifacts(base_dir)
    L.info("Artifacts loaded | window=%d | thr=(%.3f, %.3f) | cols=%d",
           window, neg_thr, pos_thr, len(train_cols))

    # --- Prepare PREPARE_DATA_FOR_TRAIN once ---
    filepaths = {
        "30T": base_dir / f"{symbol}_M30.csv",
        "15T": base_dir / f"{symbol}_M15.csv",
        "5T":  base_dir / f"{symbol}_M5.csv",
        "1H":  base_dir / f"{symbol}_H1.csv",
    }
    for tf, fp in filepaths.items():
        L.info("[raw] %s -> %s", tf, fp)

    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={k: str(v) for k, v in filepaths.items()},
        main_timeframe="30T",
        verbose=True,
        fast_mode=False,
        strict_disk_feed=False
    )
    merged = prep.load_data()
    tcol = "30T_time" if "30T_time" in merged.columns else "time"
    L.info("[load_data] merged shape=%s", merged.shape)

    # --- Paths for IO ---
    ans_path          = base_dir / "answer.txt"
    deploy_pred_path  = base_dir / "deploy_predictions.csv"
    deploy_feat_log   = base_dir / "deploy_X_feed_log.csv"
    deploy_feat_tail  = base_dir / "deploy_X_feed_tail200.csv"

    steps_done    = 0
    predicted_cnt = 0
    last_ts_processed: Optional[pd.Timestamp] = None

    while True:
        # شرط اتمام در تست آفلاین
        if args.max_steps > 0 and steps_done >= args.max_steps:
            L.info("Max-steps (%d) reached → exiting.", args.max_steps)
            break

        ok, live_files = live_files_ready(base_dir, symbol)
        if not ok:
            time.sleep(args.poll_sec)
            continue

        ts_now = read_last_timestamp(live_files["30T"])
        if ts_now is None:
            L.warning("Could not read timestamp from *_live.csv, skipping this batch.")
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        if last_ts_processed is not None and ts_now <= last_ts_processed:
            # اگر به هر دلیل فایل‌های قدیمی یا تکراری دیدیم، پاک می‌کنیم و صبر
            L.warning("Seen non-increasing timestamp (%s <= %s) → removing live files and waiting.",
                      ts_now, last_ts_processed)
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        # --- زیرمجموعه‌ی merged تا ts_now (بدون دیتا از آینده) ---
        sub = merged[merged[tcol] <= ts_now].copy()
        if sub.empty:
            L.warning("No merged data <= %s, skipping.", ts_now)
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        # --- ساخت X,y,times مشابه ترین ---
        X_all, y_all, _, price_ser, t_idx = prep.ready(
            sub,
            window=window,
            selected_features=train_cols,
            mode="train",
            with_times=True,
            predict_drop_last=False,
            train_drop_last=True
        )
        if X_all.empty or len(t_idx) == 0:
            L.warning("Empty X_all or times for ts_now=%s, skipping.", ts_now)
            remove_live_files(live_files)
            time.sleep(args.poll_sec)
            continue

        # آخرین نمونه
        X_last = X_all.tail(1).reset_index(drop=True)
        X_last = align_columns(X_last, train_cols)

        y_true = None
        if hasattr(y_all, "__len__") and len(y_all) > 0:
            y_true = int(pd.Series(y_all).iloc[-1])
        ts_feat = pd.to_datetime(t_idx.iloc[-1])

        # --- Predict ---
        prob = float(model.predict_proba(X_last)[:, 1][0])
        action = proba_to_decision(prob, neg_thr, pos_thr)

        steps_done += 1
        is_pred = 1 if action != "NONE" else 0
        predicted_cnt += is_pred
        cover_cum = predicted_cnt / max(1, steps_done)

        # --- answer.txt (برای EA) ---
        try:
            with ans_path.open("w", encoding="utf-8") as f:
                f.write(action)
        except Exception as e:
            L.error("Failed to write answer.txt: %s", e)

        # --- deploy_predictions.csv (برای مقایسه با live_like_sim_v3) ---
        row_pred = {
            "timestamp": ts_feat,
            "timestamp_trigger": ts_now,
            "y_true": y_true,
            "y_prob": prob,
            "action": action,
            "neg_thr": neg_thr,
            "pos_thr": pos_thr,
            "cover_cum": cover_cum,
        }
        hdr = not deploy_pred_path.is_file()
        pd.DataFrame([row_pred]).to_csv(
            deploy_pred_path, mode="a", header=hdr, index=False
        )

        # --- deploy_X_feed_log.csv + tail200 (برای مقایسه‌ی فیچرها) ---
        feat_row = X_last.copy()
        feat_row.insert(0, "timestamp", ts_feat)
        feat_row["timestamp_trigger"] = ts_now
        feat_row["y_true"] = y_true
        feat_row["y_prob"] = prob
        feat_row["action"] = action
        feat_row["cover_cum"] = cover_cum

        hdr_feat = not deploy_feat_log.is_file()
        feat_row.to_csv(deploy_feat_log, mode="a", header=hdr_feat, index=False)

        try:
            pd.read_csv(deploy_feat_log).tail(200).to_csv(deploy_feat_tail, index=False)
        except Exception:
            pass

        # --- پاک کردن *_live.csv بعد از مصرف ---
        remove_live_files(live_files)

        L.info(
            "[Predict] step=%d ts_feat=%s ts_now=%s | prob=%.6f → %s | y_true=%s | cover_cum=%.3f",
            steps_done, ts_feat, ts_now, prob, action, str(y_true), cover_cum,
        )

        last_ts_processed = ts_now
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
