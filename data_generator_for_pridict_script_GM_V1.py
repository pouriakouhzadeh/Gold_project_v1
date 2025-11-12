# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, logging, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

LOG_FILE = "prediction.log"

def setup_logging(verbosity:int=1):
    """لاگ‌گیری را برای فایل و کنسول تنظیم می‌کند."""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    
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

def resolve_raw_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    """مسیر فایل‌های CSV خام *اصلی* را پیدا می‌کند."""
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
            # جستجوی غیر حساس به بزرگی و کوچکی حروف
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            logging.info(f"[raw] {tf} -> {found}")
            res[tf] = found
    
    if "30T" not in res:
        logging.error(f"Fatal: Main timeframe 30T file not found in {base_dir}")
        raise FileNotFoundError(f"Main timeframe 30T file not found in {base_dir}")
    return res

def pick_slice_sizes() -> dict[str,int]:
    """تعداد ردیف‌های لازم برای هر تایم‌فریم در فایل‌های _live.csv"""
    # این مقادیر باید به اندازه‌ای بزرگ باشند که اندیکاتورها دچار خطا نشوند
    # اما در منطق جدید (prediction_in_production_fixed.py) این مقادیر دیگر مهم نیستند
    # چون اسکریپت پروداکشن به تاریخچه کامل دسترسی دارد.
    # ما فقط از 30T_live.csv به عنوان تریگر استفاده می‌کنیم.
    return {"30T": 500, "15T": 1000, "5T": 3000, "1H": 300}

def live_name(path: str) -> str:
    """نام فایل _live.csv را بر اساس نام فایل اصلی می‌سازد."""
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000, help="تعداد آخرین کندل‌ها برای شبیه‌سازی")
    ap.add_argument("--base-dir", type=str, default=".", help="پوشه‌ی حاوی فایل‌های CSV خام")
    ap.add_argument("--symbol", type=str, default="XAUUSD", help="نماد (Symbol)")
    ap.add_argument("--sleep", type=float, default=0.5, help="تأخیر بین هر مرحله (ثانیه)")
    args = ap.parse_args()
    
    setup_logging(1)
    logging.info("=== Generator started ===")
    
    base = Path(args.base_dir).resolve()
    paths = resolve_raw_paths(base, args.symbol)
    
    # بارگیری تمام داده‌های خام
    raw = {}
    for tf in paths:
        try:
            raw[tf] = pd.read_csv(paths[tf])
        except Exception as e:
            logging.error(f"Could not read {tf} file at {paths[tf]}: {e}")
            return
            
    # پیش‌پردازش تمام داده‌های خام
    for tf, df in raw.items():
        if "time" not in df.columns:
            logging.error(f"[{tf}] 'time' column missing.")
            return
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    df30 = raw["30T"]
    total = len(df30)
    if total < args.last_n + 2:
        logging.warning(f"Dataset size ({total}) is small vs last_n ({args.last_n}); proceeding anyway.")
        
    SL = pick_slice_sizes()
    wins = loses = none = 0
    audit_rows = []
    
    start_idx = max(1, total - args.last_n) # از 1 شروع می‌کنیم تا y_true قابل محاسبه باشد
    ans_path = base / "answer.txt"

    logging.info(f"Starting simulation loop from index {start_idx} to {total-1}...")
    
    for i, idx in enumerate(range(start_idx, total), start=1):
        ts_now = df30.loc[idx, "time"]
        
        # 1. نوشتن فایل‌های _live.csv (به عنوان تریگر)
        for tf in raw.keys():
            k = SL.get(tf, 500)
            df = raw[tf]
            # برشی از دیتا تا این زمان فعلی
            df_cut = df[df["time"] <= ts_now].tail(k).copy()
            out_path = live_name(paths[tf])
            try:
                df_cut.to_csv(out_path, index=False)
            except Exception as e:
                logging.warning(f"Failed to write {out_path}: {e}")

        logging.info(f"[Step {i}/{args.last_n}] Live CSVs written at {ts_now} — waiting for answer.txt ...")

        # 2. منتظر پاسخ ماندن
        start_wait = time.time()
        while not ans_path.exists():
            time.sleep(args.sleep)
            if time.time() - start_wait > 30: # 30 ثانیه مهلت
                logging.warning("Timeout waiting for answer.txt. Predictor might be stuck.")
                break
        
        if not ans_path.exists():
            continue # برو به مرحله بعد

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"

        # 3. محاسبه واقعیت (Ground-truth)
        # y_true برای ts_now (کندل idx) بر اساس کندل بعدی (idx+1) است
        real_up = None
        if idx < len(df30) - 1: # اطمینان از وجود کندل بعدی
            c0 = float(df30.loc[idx, "close"])
            c1 = float(df30.loc[idx + 1, "close"])
            real_up = (c1 > c0)

        # 4. به‌روزرسانی آمار
        if ans == "NONE" or real_up is None:
            none += 1
        elif ans == "BUY":
            if real_up: wins += 1;
            else: loses += 1
        elif ans == "SELL":
            if not real_up: wins += 1;
            else: loses += 1
        else:
            none += 1

        acc = wins / max(1, wins + loses)
        logging.info(f"[Result {i}] ans={ans} | real_up={real_up} | WINS={wins} LOSES={loses} NONE={none} ACC={acc:.3f}")
        
        audit_rows.append({
            "i": i, "time": ts_now, "answer": ans, "real_up": real_up,
            "wins": wins, "loses": loses, "none": none, "acc": acc
        })

        # 5. پاک کردن فایل پاسخ
        try: ans_path.unlink(missing_ok=True)
        except Exception: pass

        if i >= args.last_n:
            break
            
    # ذخیره گزارش نهایی
    pd.DataFrame(audit_rows).to_csv("generator_audit.csv", index=False)
    final_acc = wins / max(1, wins + loses)
    logging.info(f"=== Generator finished === | Final ACC=%.3f (on {wins+loses} predicted samples)" , final_acc)

if __name__ == "__main__":
    main()