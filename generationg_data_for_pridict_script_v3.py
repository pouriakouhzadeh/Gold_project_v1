# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
generatin_data_for_pridict_script.py
------------------------------------
سازندهٔ فایل‌های *_live.csv برای شبیه‌سازی لایو. برای هر timestamp روی تایم‌لاین 30T:
- از CSV خام، تا همان لحظه کات می‌نویسد (با lookback امن برای TFهای دیگر)،
- منتظر answer.txt از اسکریپت پیش‌بینی می‌ماند،
- نتیجه را با real_up می‌سنجد و آمار می‌نویسد.
"""

from __future__ import annotations
import os, sys, time, logging, argparse
from pathlib import Path
import pandas as pd
from prepare_data_for_train_production import PREPARE_DATA_FOR_TRAIN_PRODUCTION

LOG_FILE = "generatin_data_for_pridict_script.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

def resolve_raw_paths(base_dir: Path, symbol: str) -> dict:
    cands = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    res = {}
    for tf, names in cands.items():
        for nm in names:
            p = base_dir / nm
            if p.exists():
                res[tf] = str(p); break
    if "30T" not in res:
        raise FileNotFoundError("30T file not found.")
    return res

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def compute_lookbacks() -> dict:
    return {"5T": 3000, "15T": 1200, "30T": 500, "1H": 600}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.2, help="polling sleep for answer.txt")
    ap.add_argument("--use-full-prefix", action="store_true", help="اگر ست شود، کل تاریخ تا ts_now ذخیره می‌شود.")
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== generatin_data_for_pridict_script started ===")

    base = Path(args.base_dir).resolve()
    raw_paths = resolve_raw_paths(base, args.symbol)

    # خواندن خام برای نوشتن live prefix
    raw = {}
    for tf, fp in raw_paths.items():
        df = pd.read_csv(fp)
        if "time" not in df.columns:
            raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        raw[tf] = df

    # مرجع تایم‌لاین: ادغام‌شده‌ی کامل از کلاس (همانی که در Batch استفاده می‌کنی)
    prep = PREPARE_DATA_FOR_TRAIN_PRODUCTION(filepaths=raw_paths, main_timeframe="30T", verbose=False)
    prep.load_all()
    merged = prep.load_data_up_to(pd.Timestamp.max)
    tcol = f"{prep.main_timeframe}_time"
    if tcol not in merged.columns or merged.empty:
        logging.error("Merged timeline is empty or time column missing.")
        return
    merged[tcol] = pd.to_datetime(merged[tcol])
    merged = merged.sort_values(tcol).reset_index(drop=True)

    total_merged = len(merged)
    start_idx = max(1, total_merged - args.last_n)
    ans_path = base / "answer.txt"
    LB = compute_lookbacks()

    audit_rows = []
    wins = loses = none = 0
    acc = 0.0

    for i, ridx in enumerate(range(start_idx, total_merged), start=1):
        ts_now = merged.loc[ridx, tcol]

        # نوشتن live prefix برای هر TF
        for tf, df in raw.items():
            if args.use_full_prefix:
                df_cut = df[df["time"] <= ts_now].copy()
            else:
                lb = LB.get(tf, 500)
                df_cut = df[df["time"] <= ts_now]
                df_cut = df_cut.tail(lb).copy() if len(df_cut) > lb else df_cut.copy()
            out_path = base / Path(live_name(raw_paths[tf])).name
            df_cut.to_csv(out_path, index=False)

        logging.info(f"[Step {i}/{args.last_n}] Live CSVs written @ {ts_now} — waiting for answer.txt …")

        # انتظار predictor
        t0 = time.time()
        while not ans_path.exists():
            time.sleep(args.sleep)
            # fail-safe: اگر فایل ثبت نشد، از حلقه بیرون نرو؛ همینطور منتظر بمان

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"

        # y واقعی روی merged (next candle 30T)
        j = ridx
        real_up = None
        if j < total_merged - 1:
            c0 = float(merged.loc[j, f"30T_close"])
            c1 = float(merged.loc[j + 1, f"30T_close"])
            real_up = (c1 > c0)

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

    pd.DataFrame(audit_rows).to_csv("generator_audit_production.csv", index=False)
    logging.info("=== generatin_data_for_pridict_script finished | Final ACC=%.3f ===", acc)

if __name__ == "__main__":
    main()
