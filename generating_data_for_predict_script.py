# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, logging, argparse
from pathlib import Path
import pandas as pd

LOG_FILE = "prediction.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
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
            if p.exists(): found = str(p); break
        if not found:
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            logging.info(f"[raw] {tf} -> {found}")
            res[tf] = found
    # ⛔ بدون 30T کار نکن
    if "30T" not in res:
        raise FileNotFoundError("30T file not found. The generator must have 30T to align targets/resamples.")
    return res

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def parse_slices(s: str|None):
    # دیفالتِ عین v3
    d = {"30T":500, "15T":1000, "5T":3000, "1H":300}
    if not s: return d
    for part in s.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            k = k.strip().upper()
            try:
                d[k]=int(v)
            except Exception:
                pass
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--slices", type=str, default=None,
                    help='e.g. "30T:500,15T:1000,5T:3000,1H:300"')
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== Generator started ===")

    base = Path(args.base_dir).resolve()
    paths = resolve_raw_paths(base, args.symbol)

    raw = {tf: pd.read_csv(paths[tf]) for tf in paths}
    for tf, df in raw.items():
        if "time" not in df.columns:
            raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    df30 = raw["30T"]
    total = len(df30)
    SL = parse_slices(args.slices)

    wins = loses = none = 0
    start_idx = max(1, total - args.last_n)

    ans_path = base / "answer.txt"

    for step_idx, idx in enumerate(range(start_idx, total), start=1):
        ts_now = df30.loc[idx, "time"]

        # پاک‌سازی احتمالی فایل‌های قبلی _live
        for tf, p in paths.items():
            try: Path(live_name(p)).unlink(missing_ok=True)
            except Exception: pass

        # نوشتن فایل‌های زنده بر مبنای ts_now (همه TFها تا همین لبه)
        for tf in raw.keys():
            k = SL.get(tf, 500)
            df = raw[tf]
            df_cut = df[df["time"] <= ts_now].tail(k).copy()
            out_path = live_name(paths[tf])
            df_cut.to_csv(out_path, index=False)
        logging.info(f"[Step {step_idx}/{args.last_n}] Live CSVs written at {ts_now} — waiting for answer.txt …")

        # انتظار پاسخ
        while not ans_path.exists():
            time.sleep(args.sleep)

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"

        # تارگت «واقعی» از 30T
        j = df30.index[df30["time"] == ts_now]
        real_up = None
        if len(j) > 0 and j[0] < len(df30) - 1:
            j = int(j[0])
            c0 = float(df30.loc[j, "close"])
            c1 = float(df30.loc[j+1, "close"])
            real_up = (c1 > c0)

        if ans == "NONE" or real_up is None:
            none += 1
        elif ans == "BUY":
            wins += 1 if real_up else 0; loses += 0 if real_up else 1
        elif ans == "SELL":
            wins += 1 if (not real_up) else 0; loses += 0 if (not real_up) else 1
        else:
            none += 1

        acc = wins / max(1, wins + loses)
        logging.info(f"[Result {step_idx}] ans={ans} | real_up={real_up} | WINS={wins} LOSES={loses} NONE={none} ACC={acc:.3f}")

        # پاک‌سازی پاسخ
        try: ans_path.unlink(missing_ok=True)
        except Exception: pass

        if step_idx >= args.last_n:
            break

if __name__ == "__main__":
    main()
