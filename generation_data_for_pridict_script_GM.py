# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, logging, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

LOG_FILE = "prediction.log"

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

def pick_slice_sizes() -> dict[str,int]:
    # Increased slice sizes significantly to cover full history for accurate resampling and indicator calculations.
    # In simulation, this ensures the slices are large enough to mimic full dataset behavior.
    # In production, adjust based on available data; MQL should provide sufficient history if possible.
    return {"30T": 20000, "15T": 40000, "5T": 120000, "1H": 10000}  # Large enough for 2000+ rows in 30T

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== Generator started ===")

    base = Path(args.base_dir).resolve()
    paths = resolve_raw_paths(base, args.symbol)
    raw = {tf: pd.read_csv(paths[tf]) for tf in paths}
    for tf, df in raw.items():
        if "time" not in df.columns: raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce"); df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True); df.reset_index(drop=True, inplace=True)

    df30 = raw["30T"]
    total = len(df30)
    if total < args.last_n + 2:
        logging.warning("Dataset is small vs last_n; proceeding anyway.")

    SL = pick_slice_sizes()
    wins = loses = none = 0

    # Output file audit
    audit_rows = []
    start_idx = max(1, total - args.last_n)
    ans_path = base / "answer.txt"

    for i, idx in enumerate(range(start_idx, total), start=1):
        ts_now = df30.loc[idx, "time"]
        # Write live CSV slices: use large tail to ensure full history for resampling
        for tf in raw.keys():
            k = SL.get(tf, 20000)  # Use large k
            df = raw[tf]
            df_cut = df[df["time"] <= ts_now].tail(k).copy()  # Large tail to include sufficient history
            out_path = live_name(paths[tf])
            df_cut.to_csv(out_path, index=False)

        logging.info(f"[Step {i}/{args.last_n}] Live CSVs written at {ts_now} — waiting for answer.txt …")
        # Wait for prediction
        while not ans_path.exists():
            time.sleep(args.sleep)

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"

        # Compute ground-truth from 30T
        j = df30.index[df30["time"] == ts_now]
        real_up = None
        if len(j) > 0 and j[0] < len(df30) - 1:
            j = int(j[0])
            c0 = float(df30.loc[j, "close"]); c1 = float(df30.loc[j+1, "close"])
            real_up = (c1 > c0)

        # Update stats
        if ans == "NONE" or real_up is None:
            none += 1
        elif ans == "BUY":
            wins += 1 if real_up else 0; loses += 0 if real_up else 1
        elif ans == "SELL":
            wins += 1 if (not real_up) else 0; loses += 0 if (not real_up) else 1
        else:
            none += 1

        acc = wins / max(1, wins + loses)
        logging.info(f"[Result {i}] ans={ans} | real_up={real_up} | WINS={wins} LOSES={loses} NONE={none} ACC={acc:.3f}")

        audit_rows.append({
            "i": i, "time": ts_now, "answer": ans, "real_up": real_up,
            "wins": wins, "loses": loses, "none": none, "acc": acc
        })

        try: ans_path.unlink(missing_ok=True)
        except Exception: pass

        if i >= args.last_n:
            break

    # Save audit
    pd.DataFrame(audit_rows).to_csv("generator_audit.csv", index=False)
    logging.info("=== Generator finished === | Final ACC=%.3f (on predicted samples only)", acc)

if __name__ == "__main__":
    main()