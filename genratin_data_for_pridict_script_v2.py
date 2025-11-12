# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, logging, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "prediction_generator_v2.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

def resolve_raw_paths(base_dir: Path, symbol: str) -> dict[str,str]:
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
            logging.info(f"[raw] {tf} -> {found}")
            res[tf] = found
    if "30T" not in res:
        raise FileNotFoundError("30T file not found.")
    return res

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def md5_of_columns(df: pd.DataFrame) -> str:
    h = hashlib.md5()
    for c in df.columns:
        h.update(c.encode("utf-8"))
    return h.hexdigest()

def compute_lookbacks() -> dict[str,int]:
    """
    حاشیه‌ی امن برای هر TF تا اثر لبه‌ی اندیکاتورها/ایچیموکو/رولینگ از بین برود.
    می‌توانی در صورت نیاز بزرگ‌تر کنی.
    """
    return {"5T": 3000, "15T": 1200, "30T": 500, "1H": 600}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.2, help="polling sleep for answer.txt")
    ap.add_argument("--use-full-prefix", action="store_true",
                    help="اگر بگذاری، کل تاریخ تا هر تایم‌استمپ را می‌نویسد (فایل‌ها بزرگ می‌شوند اما ۱:۱ ترین حالت است).")
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== Generator v2 started ===")

    base = Path(args.base_dir).resolve()
    raw_paths = resolve_raw_paths(base, args.symbol)

    # 1) خواندن کامل CSV خام هر TF (برای برش‌های تکرارشونده سریع)
    raw = {}
    for tf, fp in raw_paths.items():
        df = pd.read_csv(fp)
        if "time" not in df.columns:
            raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        raw[tf] = df

    # 2) ساخت تایم‌لاین 30T “پس از ادغام/ری‌سمپل” با همان PREP (کلید شباهت با live_like_sim_v3)
    prep_for_timeline = PREPARE_DATA_FOR_TRAIN(
        filepaths=raw_paths, main_timeframe="30T",
        verbose=False, fast_mode=False, strict_disk_feed=False
    )
    merged = prep_for_timeline.load_data()
    tcol = f"{prep_for_timeline.main_timeframe}_time"
    if tcol not in merged.columns:
        raise RuntimeError(f"Timeline column '{tcol}' not found in merged data.")
    merged[tcol] = pd.to_datetime(merged[tcol])
    merged = merged.sort_values(tcol).reset_index(drop=True)

    total_merged = len(merged)
    if total_merged < args.last_n + 2:
        logging.warning("Merged dataset is small vs last_n; proceeding anyway.")

    start_idx = max(1, total_merged - args.last_n)
    ans_path = base / "answer.txt"

    LB = compute_lookbacks()

    audit_rows = []
    wins = loses = none = 0
    acc = 0.0

    # 3) شبیه‌سازی: برای هر timestampِ ادغام‌شده، فایل‌های live هر TF را تا همان زمان (به‌صورت تجمعی) می‌نویسیم
    for i, ridx in enumerate(range(start_idx, total_merged), start=1):
        ts_now = merged.loc[ridx, tcol]

        for tf, df in raw.items():
            # محدوده‌ی امن lookback
            if args.use_full_prefix:
                df_cut = df[df["time"] <= ts_now].copy()
            else:
                lb = LB.get(tf, 500)
                # زمان شروع = ts_now - lb بار از همان TF (به‌صورت تقریبی)
                # به‌جای تبدیل تقویمی پیچیده، ساده: آخرین lb ردیف قبل از ts_now
                df_cut = df[df["time"] <= ts_now]
                if len(df_cut) > lb:
                    df_cut = df_cut.tail(lb).copy()
                else:
                    df_cut = df_cut.copy()

            out_path = live_name(raw_paths[tf])
            df_cut.to_csv(out_path, index=False)

        logging.info(f"[Step {i}/{args.last_n}] Live CSVs written at {ts_now} — waiting for answer.txt …")

        # 4) انتظار پاسخ predictor
        while not ans_path.exists():
            time.sleep(args.sleep)

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"

        # 5) برچسب واقعی بر اساس 30T خامِ ادغام‌شده (بعد از PREP: همان رفتار تست)
        #    y(t) = (close_{t+1} > close_t)
        j = ridx
        real_up = None
        if j < total_merged - 1:
            c0 = float(merged.loc[j, f"30T_close"])
            c1 = float(merged.loc[j + 1, f"30T_close"])
            real_up = (c1 > c0)

        # 6) به‌روزرسانی آمار
        if ans == "NONE" or real_up is None:
            none += 1
        elif ans == "BUY":
            wins += 1 if real_up else 0
            loses += 0 if real_up else 1
        elif ans == "SELL":
            wins += 1 if (not real_up) else 0
            loses += 0 if (not real_up) else 1
        else:
            none += 1

        acc = wins / max(1, wins + loses)
        logging.info(f"[Result {i}] ans={ans} | real_up={real_up} | WINS={wins} LOSES={loses} NONE={none} ACC={acc:.3f}")

        audit_rows.append({
            "i": i, "time": ts_now, "answer": ans, "real_up": real_up,
            "wins": wins, "loses": loses, "none": none, "acc": acc
        })

        try:
            ans_path.unlink(missing_ok=True)
        except Exception:
            pass

        if i >= args.last_n:
            break

    pd.DataFrame(audit_rows).to_csv("generator_audit_v2.csv", index=False)
    logging.info("=== Generator v2 finished === | Final ACC=%.3f (on predicted samples only)", acc)

if __name__ == "__main__":
    main()
