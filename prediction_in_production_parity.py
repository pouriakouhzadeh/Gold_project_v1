#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_in_production_parity.py  —  نسخهٔ هماهنگ با ژنراتور

- استفاده از joblib برای بارگذاری مدل
- ساخت فیچرها دقیقاً مثل Train (mode="train", train_drop_last=True)
- چیدن ستون‌ها بر اساس train_window_cols
- نوشتن:
    * deploy_X_feed_log.csv      (لاگ کامل هر استپ)
    * deploy_X_feed_tail200.csv  (۲۰۰ سطر آخر فیچرها)
    * deploy_predictions.csv     (برای ژنراتور)
    * answer.txt                 (اکشن نهایی، برای تست دستی)
"""

from __future__ import annotations
import sys, time, json, argparse, logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

# مطمئن شویم که پوشهٔ پروژه در sys.path هست
for cand in (Path(__file__).resolve().parent, Path.cwd()):
    p = str(cand)
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

LOG = logging.getLogger("deploy_parity")


# ---------------- Logging ----------------
def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity > 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------- Artifacts ----------------
def load_artifacts(model_dir: Path):
    meta_path = model_dir / "best_model.meta.json"
    pkl_path  = model_dir / "best_model.pkl"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Model file not found: {pkl_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # مدل با joblib (فایل zlib-compressed)
    model = joblib.load(pkl_path)

    window = int(meta.get("window_size", 1))
    neg_thr = float(meta.get("neg_thr", 0.005))
    pos_thr = float(meta.get("pos_thr", 0.995))

    # ترتیب دقیق ستون‌های پنجره
    train_cols: List[str] = (
        meta.get("train_window_cols")
        or meta.get("feats")
        or meta.get("feature_names")
        or []
    )

    return model, meta, train_cols, window, neg_thr, pos_thr


# ---------------- File IO ----------------
def live_files_ready(base_dir: Path, symbol: str):
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
def align_columns(
    X: pd.DataFrame,
    train_cols: List[str],
    train_distribution_path: Path,
) -> pd.DataFrame:
    """
    X را طوری تنظیم می‌کند که دقیقاً همان ستون‌ها و ترتیب train_cols را داشته باشد.
    اگر ستونی در X نبود، اول سعی می‌کند از train_distribution.json میانه را بردارد،
    اگر نبود، 0.0 می‌گذارد.
    """
    X = X.copy()

    med: Dict[str, float] = {}
    try:
        if train_distribution_path.is_file():
            td = json.loads(train_distribution_path.read_text(encoding="utf-8"))
            med = td.get("medians", {}) or {}
    except Exception:
        med = {}

    for c in train_cols:
        if c not in X.columns:
            X[c] = float(med.get(c, 0.0))

    X = X.loc[:, [c for c in train_cols if c in X.columns]]

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
    ap.add_argument(
        "--max-steps",
        default=0,
        type=int,
        help="برای تست آفلاین (مثلاً ۲۰۰ استپ)، اینجا تعداد استپ را وارد کن؛ صفر یعنی بی‌نهایت",
    )
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    base_dir = Path(args.base_dir).resolve()
    model_dir = base_dir

    LOG.info("=== prediction_in_production_parity started ===")

    # --- Load artifacts ---
    model, meta, train_cols, window, neg_thr, pos_thr = load_artifacts(model_dir)
    LOG.info(
        "Artifacts loaded | window=%d | thr=(%.3f, %.3f) | cols=%d",
        window,
        neg_thr,
        pos_thr,
        len(train_cols),
    )

    # --- Prepare data loader (merge all TFs once) ---
    filepaths = {
        "30T": base_dir / f"{args.symbol}_M30.csv",
        "15T": base_dir / f"{args.symbol}_M15.csv",
        "5T":  base_dir / f"{args.symbol}_M5.csv",
        "1H":  base_dir / f"{args.symbol}_H1.csv",
    }
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={k: str(v) for k, v in filepaths.items()},
        main_timeframe="30T",
        verbose=True,
    )
    merged = prep.load_data()
    tcol = "30T_time" if "30T_time" in merged.columns else "time"

    # --- Paths for I/O ---
    ans_path    = base_dir / "answer.txt"
    log_path    = base_dir / "deploy_X_feed_log.csv"
    feat_full   = base_dir / "deploy_X_feed_full.csv"
    feat_tail   = base_dir / "deploy_X_feed_tail200.csv"
    pred_path   = base_dir / "deploy_predictions.csv"
    td_path     = base_dir / "train_distribution.json"

    steps_done = 0
    predicted_cnt = 0  # برای cover تجمعی
    last_ts_seen: Optional[pd.Timestamp] = None

    LOG.info("Waiting for *_live.csv files from generator...")

    while True:
        ok, files = live_files_ready(base_dir, args.symbol)
        if not ok:
            time.sleep(args.poll_sec)
            continue

        ts_now = read_last_timestamp(files["30T"])
        if ts_now is None:
            time.sleep(args.poll_sec)
            continue

        if last_ts_seen is not None and ts_now <= last_ts_seen:
            # استپ جدیدی نیامده
            time.sleep(args.poll_sec)
            continue

        # --- برش دیتای ادغام‌شده تا لحظهٔ ts_now ---
        sub = merged[merged[tcol] <= ts_now].copy()
        if sub.empty:
            time.sleep(args.poll_sec)
            continue

        # --- ساخت X, y, time همانند train ---
        X_all, y_all, _, price_ser, t_idx = prep.ready(
            sub,
            window=window,
            selected_features=train_cols,
            mode="train",
            with_times=True,
            predict_drop_last=False,
            train_drop_last=True,
        )
        if X_all.empty or len(t_idx) == 0:
            time.sleep(args.poll_sec)
            continue

        # آخرین نمونه
        X_last = X_all.tail(1).reset_index(drop=True)
        y_true = None
        if hasattr(y_all, "__len__") and len(y_all):
            y_true = int(pd.Series(y_all).iloc[-1])
        ts_feat = pd.to_datetime(pd.Series(t_idx).iloc[-1])

        # هم‌ترازی ستون‌ها با ستون‌های train
        X_last = align_columns(X_last, train_cols, td_path)

        # --- Predict ---
        p = float(model.predict_proba(X_last)[:, 1][0])
        dec = proba_to_decision(p, neg_thr, pos_thr)

        # --- cover تجمعی ---
        steps_done += 1
        is_pred = 1 if dec != "NONE" else 0
        predicted_cnt += is_pred
        cover_cum = predicted_cnt / max(1, steps_done)

        # --- write answer.txt (برای مشاهده، هندشیک اصلی روی deploy_predictions.csv است) ---
        with ans_path.open("w", encoding="utf-8") as f:
            f.write(dec)

        # --- Log row (summary) ---
        row_log = {
            "timestamp_feature": ts_feat,
            "timestamp_trigger": ts_now,
            "prob": p,
            "decision": dec,
            "y_true": y_true,
            "cover_cum": cover_cum,
            "neg_thr": neg_thr,
            "pos_thr": pos_thr,
        }
        df_log = pd.DataFrame([row_log])
        df_log.to_csv(
            log_path,
            mode="a",
            header=not log_path.is_file(),
            index=False,
        )

        # --- ذخیرهٔ فیچر همین استپ برای مقایسه با live_like_sim ---
        feat_row = X_last.copy()
        feat_row.insert(0, "timestamp_feature", ts_feat)
        feat_row.insert(1, "timestamp_trigger", ts_now)
        feat_row.to_csv(
            feat_full,
            mode="a",
            header=not feat_full.is_file(),
            index=False,
        )
        try:
            pd.read_csv(
                feat_full,
                parse_dates=["timestamp_feature", "timestamp_trigger"],
            ).sort_values("timestamp_feature").tail(200).to_csv(
                feat_tail, index=False
            )
        except Exception:
            pass

        # --- فایل مخصوص ژنراتور: deploy_predictions.csv ---
        row_pred = {
            "timestamp": ts_now,
            "action": dec,
            "y_prob": p,
            "cover_cum": cover_cum,
            "y_true": y_true,
        }
        df_pred = pd.DataFrame([row_pred])
        df_pred.to_csv(
            pred_path,
            mode="a",
            header=not pred_path.is_file(),
            index=False,
        )

        # --- پاک کردن *_live.csv برای استپ بعدی ---
        remove_live_files(files)

        LOG.info(
            "[Predict] ts_feat=%s ts_now=%s | prob=%.6f → %s | cover_cum=%.3f",
            ts_feat,
            ts_now,
            p,
            dec,
            cover_cum,
        )

        last_ts_seen = ts_now

        if args.max_steps > 0 and steps_done >= args.max_steps:
            LOG.info(
                "Reached max_steps=%d, stopping deploy parity loop.",
                args.max_steps,
            )
            break

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
