# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prediction_in_production_parity.py  —  PARITY-SAFE DEPLOY

ویژگی‌ها:
- بارگذاری امن آرتیفکت‌ها با joblib (نه pickle)
- آماده‌سازی فیچرها دقیقاً هم‌تراز Train (mode="train", train_drop_last=True)
- هم‌ترازی ستون‌ها با train_window_cols (از best_model.meta.json)
- ثبت لاگ کامل: timestamp_feature, timestamp_trigger, score, decision, y_true(در آفلاین), cover تجمعی
- حذف خودکار *_live.csv پس از نوشتن answer.txt
"""

from __future__ import annotations
import os, sys, time, json, argparse, logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

# === وارد کردن کلاس‌های پروژه ===
# اطمینان از اینکه پوشهٔ اسکریپت و cwd در sys.path باشند
for cand in (Path(__file__).resolve().parent, Path.cwd()):
    p = str(cand)
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

# ---------------- Logging ----------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# ---------------- Artifacts ----------------
def load_artifacts(model_dir: Path) -> Tuple[Any, Dict[str, Any], List[str], int, float, float]:
    """
    best_model.pkl       → pipeline/model wrapper (joblib)
    best_model.meta.json → meta شامل window_size, neg_thr, pos_thr, train_window_cols (یا feats)
    """
    pkl_path  = model_dir / "best_model.pkl"
    meta_path = model_dir / "best_model.meta.json"

    if not pkl_path.is_file():
        raise FileNotFoundError(f"model pickle not found: {pkl_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta json not found: {meta_path}")

    # بارگذاری امن با joblib (فایل فشرده zlib)
    model = joblib.load(pkl_path)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    window = int(meta.get("window_size", 1))
    neg_thr = float(meta.get("neg_thr", 0.005))
    pos_thr = float(meta.get("pos_thr", 0.995))

    # ترتیب دقیق ستون‌ها بعد از window (اولویت اول)
    train_cols = meta.get("train_window_cols") or []
    if not train_cols:
        # اگر train_window_cols موجود نبود، از لیست «feats» استفاده می‌کنیم
        # توجه: اگر feats پایه است و window>1 باشد، کلاس PREPARE_DATA_FOR_TRAIN با selected_features=feats
        # خودش ستون‌های _tminus را می‌سازد و در خروجی X همان ترتیب را نگه می‌دارد.
        train_cols = meta.get("feats") or []

    return model, meta, list(train_cols), window, neg_thr, pos_thr

# ---------------- File IO ----------------
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
def align_columns(X: pd.DataFrame, train_cols: List[str], train_distribution_path: Path) -> pd.DataFrame:
    """
    تضمین می‌کند X دقیقاً ستون‌ها و ترتیبِ train_cols را دارد.
    اگر ستونی نبود، با میانهٔ آموزش (train_distribution.json) یا صفر پر می‌شود.
    """
    X = X.copy()
    # خواندن میانه‌ها از train_distribution.json (اختیاری)
    med = {}
    try:
        td = json.loads(train_distribution_path.read_text(encoding="utf-8"))
        # انتظار می‌رود کلیدهایی شبیه {'medians': {'colA': 0.1, ...}} یا مشابه داشته باشد.
        # در پروژهٔ شما ساختار متفاوت است؛ پس ایمن پر می‌کنیم.
        # اگر 'medians' نبود، از مقدار 0.0 استفاده می‌کنیم.
        med = td.get("medians", {})
    except Exception:
        med = {}

    # افزودن ستون‌های مفقود
    for c in train_cols:
        if c not in X.columns:
            fillv = med.get(c, 0.0)
            X[c] = float(fillv)

    # حذف ستون‌های اضافه و مرتب‌سازی
    X = X.loc[:, [c for c in train_cols if c in X.columns]]

    # تضمین نوع float64
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        X[c] = X[c].astype("float64", copy=False)
    return X

# ---------------- Decision ----------------
def proba_to_decision(p: float, neg_thr: float, pos_thr: float) -> str:
    if p <= neg_thr:
        return "SELL"
    if p >= pos_thr:
        return "BUY"
    return "NONE"

# ---------------- MAIN ----------------
def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol",   default="XAUUSD", type=str)
    ap.add_argument("--poll-sec", default=1.0, type=float)
    ap.add_argument("--max-steps", default=0, type=int, help="برای تست آفلاین روی 200 استپ، 200 بگذار؛ صفر یعنی بی‌نهایت")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    model_dir = base_dir

    logging.info("=== prediction_in_production_parity started ===")

    # --- Load artifacts ---
    model, meta, train_cols, window, neg_thr, pos_thr = load_artifacts(model_dir)
    logging.info("Artifacts loaded | window=%d | thr=(%.3f, %.3f) | cols=%d",
                 window, neg_thr, pos_thr, len(train_cols))

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
        fast_mode=False,           # برای پاریتی با شبیه‌ساز
        strict_disk_feed=False
    )
    merged = prep.load_data()
    tcol = "30T_time" if "30T_time" in merged.columns else "time"

    # --- Paths for I/O ---
    ans_path   = base_dir / "answer.txt"
    log_path   = base_dir / "deploy_X_feed_log.csv"
    tail_path  = base_dir / "deploy_X_feed_tail200.csv"
    td_path    = base_dir / "train_distribution.json"  # برای میانه‌ها (در align_columns)

    wrote_header = not log_path.is_file()
    steps_done = 0
    predicted_cnt = 0  # برای cover تجمعی

    # --- Loop ---
    last_ts_seen: Optional[pd.Timestamp] = None
    while True:
        # در تستِ ژنراتور، اگر answer.txt هنوز مصرف نشده باشد، صبر کن
        if ans_path.exists():
            time.sleep(args.poll_sec); continue

        ok, files = live_files_ready(base_dir, args.symbol)
        if not ok:
            time.sleep(args.poll_sec); continue

        ts_now = read_last_timestamp(files["30T"])
        if ts_now is None:
            time.sleep(args.poll_sec); continue
        if last_ts_seen is not None and ts_now <= last_ts_seen:
            # هنوز استپ جدیدی نیامده
            time.sleep(args.poll_sec); continue

        # --- برش دیتای ادغام‌شده تا لحظهٔ ts_now (هیچ دیتای آینده‌ای مصرف نمی‌شود) ---
        sub = merged[merged[tcol] <= ts_now].copy()
        if sub.empty:
            time.sleep(args.poll_sec); continue

        # --- ساخت X,y,times همانند Train (drop-last در Train) ---
        # selected_features = train_cols → PREPARED کلاس خودش tminusها را در صورت نیاز می‌سازد
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
            time.sleep(args.poll_sec); continue

        # آخرین نمونهٔ قابل‌برچسب (نه ردیف آینده)
        X_last = X_all.tail(1).reset_index(drop=True)
        y_true = int(pd.Series(y_all).iloc[-1]) if hasattr(y_all, "__len__") and len(y_all) else None
        ts_feat = pd.to_datetime(t_idx.iloc[-1])

        # هم‌ترازی دقیق ستون‌ها با train_window_cols (یا feats)
        X_last = align_columns(X_last, train_cols, td_path)

        # --- Predict ---
        # مدل آموزش‌دیده در pipeline خودش StandardScaler دارد؛ کافی‌ست predict_proba
        p = float(model.predict_proba(X_last)[:, 1][0])
        dec = proba_to_decision(p, neg_thr, pos_thr)

        # --- cover تجمعی ---
        is_pred = 1 if dec != "NONE" else 0
        predicted_cnt += is_pred
        steps_done += 1
        cover_cum = predicted_cnt / max(steps_done, 1)

        # --- write answer.txt ---
        with ans_path.open("w", encoding="utf-8") as f:
            f.write(dec)

        # --- log row ---
        row = {
            "timestamp_feature": ts_feat,
            "timestamp_trigger": ts_now,
            "prob": p,
            "decision": dec,
            "y_true": y_true,                # در تست آفلاین موجود است؛ در لایو واقعی None خواهد بود
            "cover_cum": cover_cum,
            "neg_thr": neg_thr,
            "pos_thr": pos_thr
        }
        # append to CSV
        df_row = pd.DataFrame([row])
        if wrote_header and log_path.is_file():
            # اگر از قبل فایل وجود داشت، یعنی هدر نوشته شده
            wrote_header = False
        df_row.to_csv(log_path, mode="a", header=not log_path.is_file(), index=False)
        try:
            # tail 200 برای مشاهدهٔ سریع
            pd.read_csv(log_path, parse_dates=["timestamp_feature", "timestamp_trigger"])\
              .sort_values("timestamp_feature")\
              .tail(200).to_csv(tail_path, index=False)
        except Exception:
            pass

        # --- تمیزکاری: مصرف شد، پاک کن تا با استپ بعدی اشتباه نشود ---
        remove_live_files(files)

        logging.info("[Predict] ts_feat=%s ts_now=%s | p=%.6f → %s | cover=%.3f | wrote=%s",
                     ts_feat, ts_now, p, dec, cover_cum, str(ans_path.resolve()))

        last_ts_seen = ts_now

        # --- termination for offline test ---
        if args.max_steps > 0 and steps_done >= args.max_steps:
            break

        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
