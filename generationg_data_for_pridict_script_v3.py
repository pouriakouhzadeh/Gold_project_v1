# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, logging, argparse
from pathlib import Path
import numpy as np
import pandas as pd

LOG_FILE = "generator_v3.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
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
            if p.exists():
                found = str(p); break
        if not found:
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            logging.info(f"[paths] {tf} -> {found}")
            res[tf] = found

    if "30T" not in res:
        raise FileNotFoundError("Main timeframe 30T not found.")
    return res

# حداقل سابقه برای جلوگیری از لبه‌های ناقص در ریسمپل و پنجره‌های بزرگ اندیکاتورها
# (ایچیموکو تا 52، KST تا ~45، EMA تا 50، پنجره مدل هم w؛ برای اطمینان حاشیه اضافه)
WARMUP_BARS = {
    "5T":  3000,   # ~10 روز کاری 5 دقیقه‌ای
    "15T": 1200,   # ~6-7 روز کاری 15 دقیقه‌ای
    "30T": 800,    # ~4-5 روز کاری 30 دقیقه‌ای
    "1H":  400,    # ~3-4 هفته 1 ساعته
}

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000, help="تعداد گام‌های شبیه‌سازی روی 30T")
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== Generator v3 started ===")

    base = Path(args.base_dir).resolve()
    paths = resolve_raw_paths(base, args.symbol)
    raw = {}

    # 1) بارگذاری خامِ هر TF و مرتب‌سازی زمانی
    for tf, fp in paths.items():
        df = pd.read_csv(fp)
        if "time" not in df.columns:
            raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        raw[tf] = df

    df30 = raw["30T"]
    total = len(df30)
    if total < args.last_n + 2:
        logging.warning("Dataset length < last_n; proceeding anyway.")

    ans_path = base / "answer.txt"
    audit_rows = []

    # 2) حلقه‌ی شبیه‌سازی روی 30T: در هر گام، تا timestamp جاری برش می‌زنیم
    start_idx = max(1, total - args.last_n)
    wins = loses = none = 0

    for i, idx in enumerate(range(start_idx, total), start=1):
        ts_now = df30.loc[idx, "time"]

        # 2-الف) برای هر TF برش با WARMUP کافی تا ts_now
        for tf, df in raw.items():
            warm = WARMUP_BARS.get(tf, 800)
            # subset تا زمان فعلی
            df_upto = df[df["time"] <= ts_now]
            if df_upto.empty:
                # احتمالاً هنوز آن TF تا این زمان داده‌ای ندارد
                out = df.head(1).copy()
            else:
                # آخرین ایندکس ردیف ts_now (یا قبلش)
                end = df_upto.index[-1]
                beg = max(0, end - warm + 1)
                out = df.iloc[beg:end+1].copy()

            out_path = live_name(paths[tf])
            out.to_csv(out_path, index=False)

        logging.info(f"[Step {i}/{args.last_n}] Live CSVs written at {ts_now} — waiting for answer.txt …")

        # 2-ب) منتظر پاسخ پرِدیکتور
        t0 = time.time()
        while not ans_path.exists():
            time.sleep(args.sleep)
            # اگر خیلی طول کشید، بار دیگر لاگ
            if time.time() - t0 > 60:
                logging.info("still waiting for answer.txt ...")
                t0 = time.time()

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"

        # 2-ج) برچسب واقعی
        real_up = None
        if idx < len(df30) - 1:
            c0 = float(df30.loc[idx, "close"])
            c1 = float(df30.loc[idx+1, "close"])
            real_up = (c1 > c0)

        if ans == "NONE" or real_up is None:
            none += 1
        elif ans == "BUY":
            if real_up: wins += 1
            else:       loses += 1
        elif ans == "SELL":
            if not real_up: wins += 1
            else:           loses += 1
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

    pd.DataFrame(audit_rows).to_csv("generator_audit_v3.csv", index=False)
    logging.info("=== Generator v3 finished === | Final ACC=%.3f (on predicted samples only)", acc)

if __name__ == "__main__":
    main()
