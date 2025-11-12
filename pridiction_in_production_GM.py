# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, joblib, logging, argparse, hashlib
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN # فرض بر این است که این فایل در دسترس است

LOG_FILE = "prediction_in_production.log"

# ---------- Logging ----------
def setup_logging(verbosity:int=1):
    """لاگ‌گیری را برای فایل و کنسول تنظیم می‌کند."""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    
    # هندلر فایل
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    except Exception as e:
        print(f"Warning: Could not open log file {LOG_FILE}: {e}")
        
    # هندلر کنسول
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

# ---------- Utils ----------
def md5_list(xs:list[str]) -> str:
    """هش MD5 از لیستی از رشته‌ها ایجاد می‌کند."""
    h = hashlib.md5()
    for s in xs:
        h.update(str(s).encode("utf-8"))
    return h.hexdigest()

def load_payload_best_model(pkl_path: str="best_model.pkl") -> dict:
    """مدل و تنظیمات ذخیره شده را از فایل best_model.pkl بارگیری می‌کند."""
    if not Path(pkl_path).exists():
        logging.error(f"Fatal: Model file not found at {pkl_path}")
        raise FileNotFoundError(f"Model file not found at {pkl_path}")
        
    payload = joblib.load(pkl_path)
    if isinstance(payload, dict):
        model = payload.get("pipeline") or payload.get("model") or payload.get("estimator") or payload.get("clf") or payload.get("best_estimator")
        if model is None:
            raise ValueError("Could not find model inside best_model.pkl dictionary.")
        out = {
            "pipeline": model,
            "window_size": int(payload.get("window_size", 1)),
            "train_window_cols": list(payload.get("train_window_cols") or payload.get("feats") or []),
            "neg_thr": float(payload.get("neg_thr", 0.005)),
            "pos_thr": float(payload.get("pos_thr", 0.995)),
        }
        if "scaler" in payload: out["scaler"] = payload["scaler"]
        return out
    # payload was a raw estimator
    return {"pipeline": payload, "window_size": 1, "train_window_cols": [], "neg_thr": 0.005, "pos_thr": 0.995}

def resolve_live_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    """مسیر فایل‌های _live.csv را پیدا می‌کند (برای استفاده به عنوان تریگر)."""
    patterns = {
        "30T": [f"{symbol}_30T_live.csv", f"{symbol}_M30_live.csv"],
        "15T": [f"{symbol}_15T_live.csv", f"{symbol}_M15_live.csv"],
        "5T" : [f"{symbol}_5T_live.csv",  f"{symbol}_M5_live.csv" ],
        "1H" : [f"{symbol}_1H_live.csv",  f"{symbol}_H1_live.csv" ],
    }
    resolved = {}
    for tf, names in patterns.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found = str(p); break
        if not found:
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            resolved[tf] = found
    return resolved

def resolve_raw_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    """مسیر فایل‌های CSV خام *اصلی* را پیدا می‌کند (برای بارگیری تاریخچه کامل)."""
    cands = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    res = {}
    for tf, names in cands.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found = str(p); break
        if not found:
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            res[tf] = found
    if "30T" not in res:
        logging.error(f"Fatal: Main timeframe 30T file not found in {base_dir}")
        raise FileNotFoundError(f"Main timeframe 30T file not found in {base_dir}")
    return res

def ensure_columns(X:pd.DataFrame, cols:list[str]) -> pd.DataFrame:
    """اطمینان حاصل می‌کند که X دقیقاً ستون‌های مورد نیاز مدل را دارد."""
    if not cols: return X
    X2 = X.copy()
    miss = [c for c in cols if c not in X2.columns]
    for c in miss: X2[c] = 0.0
    # ستون‌های اضافی را حذف می‌کند و ترتیب را تضمین می‌کند
    return X2[cols]

# ---------- Adaptive Thresholds ----------
class AdaptiveThresholds:
    """کلاس مدیریت آستانه‌های پویا (اختیاری، از کد اصلی شما)."""
    def __init__(self, base_neg: float, base_pos: float, buf_len:int=1000, q_low=0.10, q_high=0.90, warm_min:int=500, fallback_soft=(0.30, 0.70)):
        self.base_neg = float(base_neg)
        self.base_pos = float(base_pos)
        self.buf = deque(maxlen=buf_len)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.warm_min = int(warm_min)
        self.fallback_soft = (float(fallback_soft[0]), float(fallback_soft[1]))

    def update_and_get(self, p: float) -> tuple[float,float,str]:
        self.buf.append(float(p))
        if len(self.buf) >= self.warm_min:
            arr = np.asarray(self.buf, dtype=float)
            ql = float(np.nanquantile(arr, self.q_low))
            qh = float(np.nanquantile(arr, self.q_high))
            return ql, qh, "adaptive"
        if (self.base_neg <= 0.01 and self.base_pos >= 0.99):
            return self.fallback_soft[0], self.fallback_soft[1], "soft-fallback"
        return self.base_neg, self.base_pos, "model"

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()
    
    setup_logging(args.verbosity)
    logging.info("=== prediction_in_production (FIXED) started ===")

    try:
        # 1. بارگیری مدل
        payload = load_payload_best_model("best_model.pkl")
        model = payload["pipeline"]
        window = int(payload["window_size"])
        cols = payload.get("train_window_cols") or []
        neg0, pos0 = float(payload["neg_thr"]), float(payload["pos_thr"])
        
        if not hasattr(model, "predict_proba"):
            raise TypeError("Loaded model does not support predict_proba().")
            
        logging.info(f"Model loaded | window={window} | feats={len(cols)} | neg_thr0={neg0:.3f} | pos_thr0={pos0:.3f}")

        base = Path(args.base_dir).resolve()
        ans_path = base / "answer.txt"

        # 2. بارگیری *کامل* تاریخچه خام (فقط یک بار)
        #    این بخش کلیدی راه‌حل است
        original_paths = resolve_raw_paths(base, args.symbol)
        logging.info(f"Loading FULL original history from: {list(original_paths.keys())}")
        
        # fast_mode=True از اسکن دریفت صرفنظر می‌کند اما همچنان کل فایل‌ها را می‌خواند
        prep_full = PREPARE_DATA_FOR_TRAIN(filepaths=original_paths, main_timeframe="30T",
                                           verbose=True, fast_mode=True, strict_disk_feed=False)
        
        # این تابع، تمام کار سنگین (اندیکاتور، ریسمپل، ادغام) را *یک بار* انجام می‌دهد
        FULL_MERGED_HISTORY = prep_full.load_data()
        TCOL = f"{prep_full.main_timeframe}_time"
        FULL_MERGED_HISTORY[TCOL] = pd.to_datetime(FULL_MERGED_HISTORY[TCOL], errors='coerce')
        FULL_MERGED_HISTORY.sort_values(TCOL, inplace=True)
        FULL_MERGED_HISTORY.reset_index(drop=True, inplace=True)
        
        logging.info(f"Full history loaded. Shape={FULL_MERGED_HISTORY.shape}. Last timestamp: {FULL_MERGED_HISTORY[TCOL].iloc[-1]}")

        # 3. آماده‌سازی برای حلقه اصلی
        live_paths_map = resolve_live_paths(base, args.symbol)
        if "30T" not in live_paths_map:
            logging.error(f"Could not find 30T_live.csv trigger file pattern. Exiting.")
            return
        trigger_file_path = Path(live_paths_map["30T"]) # فقط 30T_live به عنوان تریگر لازم است

        ath = AdaptiveThresholds(neg0, pos0)
        step = 0
        
        logging.info("Initialization complete. Entering prediction loop...")

        # 4. حلقه اصلی پیش‌بینی
        while True:
            if ans_path.exists():
                time.sleep(args.sleep)
                continue
                
            # منتظر سیگنال (فایل تریگر) از ژنراتور می‌مانیم
            if not trigger_file_path.exists():
                time.sleep(args.sleep)
                continue

            try:
                # سیگنال دریافت شد. فایل تریگر را می‌خوانیم تا زمان فعلی را بفهمیم
                df_30T_live = pd.read_csv(trigger_file_path)
                if "time" not in df_30T_live.columns:
                    logging.warning("Trigger file 30T_live.csv has no 'time' column. Skipping.")
                    time.sleep(args.sleep)
                    continue
                    
                df_30T_live["time"] = pd.to_datetime(df_30T_live["time"], errors="coerce")
                ts_now = df_30T_live["time"].dropna().iloc[-1] # آخرین زمان از فایل ژنراتور

                # 5. برش (Slice) تاریخچه کامل
                #    تاریخچه کامل و پردازش‌شده را تا این زمان برش می‌زنیم
                df_cut = FULL_MERGED_HISTORY[FULL_MERGED_HISTORY[TCOL] <= ts_now].copy()
                
                if df_cut.empty:
                    logging.warning(f"Timestamp {ts_now} not found in full history. Skipping.")
                    ans_path.write_text("NONE", encoding="utf-8")
                else:
                    # 6. آماده‌سازی (Ready)
                    #    df_cut را به prep.ready() می‌دهیم (مانند live_like_sim_v3)
                    X, _, _, _ = prep_full.ready(
                        df_cut,
                        window=window,
                        selected_features=cols,
                        mode="predict",
                        with_times=False,
                        predict_drop_last=True # برای همخوانی با شبیه‌ساز
                    )
                    
                    if X.empty:
                        ans_path.write_text("NONE", encoding="utf-8")
                        logging.info(f"[{ts_now}] X empty after prep.ready() → answer=NONE")
                    else:
                        # 7. پیش‌بینی
                        X = ensure_columns(X, cols)
                        x1 = X.tail(1)
                        prob = float(model.predict_proba(x1)[:, 1][0])
                        
                        neg_thr, pos_thr, tmode = ath.update_and_get(prob)
                        
                        if   prob >= pos_thr: ans="BUY"
                        elif prob <= neg_thr: ans="SELL"
                        else:                 ans="NONE"
                        
                        ans_path.write_text(ans, encoding="utf-8")
                        step += 1
                        if step % 50 == 1: # هر 50 پیش‌بینی لاگ می‌اندازیم
                             logging.info(f"[{ts_now}] prob={prob:.6f} → {ans} | thr=({neg_thr:.4f},{pos_thr:.4f})[{tmode}]")

                # 8. پاک‌سازی
                #    تمام فایل‌های _live را پاک می‌کنیم تا ژنراتور برای مرحله بعد آماده شود
                all_live_paths = resolve_live_paths(base, args.symbol) # دوباره چک می‌کنیم
                for p_str in all_live_paths.values():
                    try: Path(p_str).unlink(missing_ok=True)
                    except Exception: pass

            except Exception as e:
                logging.exception(f"[ERROR] Unhandled error in prediction loop: {e}")
                try: ans_path.write_text("NONE", encoding="utf-8")
                except Exception: pass
                
                # پاک‌سازی در صورت خطا
                all_live_paths = resolve_live_paths(base, args.symbol)
                for p_str in all_live_paths.values():
                    try: Path(p_str).unlink(missing_ok=True)
                    except Exception: pass
            
            time.sleep(args.sleep)

    except Exception as e:
        logging.exception(f"FATAL error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()