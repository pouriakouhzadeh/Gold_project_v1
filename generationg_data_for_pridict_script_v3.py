# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
generating_data_for_predict_script_v5.py
- Simulates live by writing *_live.csv (cut BEFORE resample window) in the SAME folder
- Waits for answer.txt each step
- Uses your training merge (via PREPARE_DATA_FOR_TRAIN) to build 30T timeline
"""

from __future__ import annotations
import os, sys, time, json, logging, argparse
from pathlib import Path
import pandas as pd
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "generator_v5.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

def live_name(raw_path: str) -> str:
    p = Path(raw_path)
    if p.stem.endswith("_live"):
        return str(p)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def resolve_raw_paths(symbol: str) -> dict[str,str]:
    here = Path(".").resolve()
    candidates = {
        "30T": [f"{symbol}_M30.csv", f"{symbol}_30T.csv"],
        "15T": [f"{symbol}_M15.csv", f"{symbol}_15T.csv"],
        "5T" : [f"{symbol}_M5.csv",  f"{symbol}_5T.csv" ],
        "1H" : [f"{symbol}_H1.csv",  f"{symbol}_1H.csv" ],
    }
    res = {}
    for tf, names in candidates.items():
        for nm in names:
            p = here / nm
            if p.exists():
                res[tf] = str(p); break
    if "30T" not in res:
        raise FileNotFoundError("30T raw file not found in current folder.")
    return res

def compute_lookbacks() -> dict[str,int]:
    # Safe lookbacks to avoid indicator edge-effects
    return {"5T": 3000, "15T": 1200, "30T": 500, "1H": 600}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.2, help="polling for answer.txt")
    ap.add_argument("--use-full-prefix", action="store_true")
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== generator_v5 started ===")

    raw_paths = resolve_raw_paths(args.symbol)

    # Read full raw per TF
    raw = {}
    for tf, fp in raw_paths.items():
        df = pd.read_csv(fp)
        if "time" not in df.columns:
            raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        raw[tf] = df

    # Build merged 30T timeline using training pipeline (for timestamps only)
    prep_for_timeline = PREPARE_DATA_FOR_TRAIN(
        filepaths=raw_paths, main_timeframe="30T",
        verbose=False, fast_mode=False, strict_disk_feed=False
    )
    merged = prep_for_timeline.load_data()
    tcol = f"{prep_for_timeline.main_timeframe}_time"
    if tcol not in merged.columns:
        raise RuntimeError(f"Timeline column '{tcol}' not found.")
    merged[tcol] = pd.to_datetime(merged[tcol])
    merged = merged.sort_values(tcol).reset_index(drop=True)

    total = len(merged)
    start_idx = max(1, total - args.last_n)
    LB = compute_lookbacks()
    ans_path = Path("answer.txt")

    wins = loses = none = 0
    audit = []

    for i, ridx in enumerate(range(start_idx, total), start=1):
        ts_now = merged.loc[ridx, tcol]

        # write live per TF
        for tf, df in raw.items():
            if args.use_full_prefix:
                df_cut = df[df["time"] <= ts_now].copy()
            else:
                lb = LB.get(tf, 500)
                df_cut = df[df["time"] <= ts_now]
                df_cut = df_cut.tail(lb).copy() if len(df_cut) > lb else df_cut.copy()
            Path(live_name(raw_paths[tf])).write_text(df_cut.to_csv(index=False))

        logging.info("[Step %d/%d] live CSVs written @ %s — waiting for answer.txt ...", i, args.last_n, ts_now)

        # wait for answer.txt
        t0 = time.time()
        while not ans_path.exists():
            time.sleep(args.sleep)
            if time.time() - t0 > 60:  # safety
                logging.warning("answer.txt timeout @ %s; counting as NONE", ts_now)
                break

        # read answer if present
        if ans_path.exists():
            try:
                ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
            except Exception:
                ans = "NONE"
            try:
                ans_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            ans = "NONE"

        # compute real_up by 30T merged (y(t) = close_{t+1} > close_t)
        real_up = None
        if ridx < total - 1:
            c0 = float(merged.loc[ridx, "30T_close"])
            c1 = float(merged.loc[ridx + 1, "30T_close"])
            real_up = (c1 > c0)

        if ans == "NONE" or real_up is None:
            none += 1
        elif ans == "BUY":
            if real_up: wins += 1
            else: loses += 1
        elif ans == "SELL":
            if not real_up: wins += 1
            else: loses += 1
        else:
            none += 1

        acc = wins / max(1, wins + loses)
        logging.info("[Result %d] ans=%s | real_up=%s | WINS=%d LOSES=%d NONE=%d ACC=%.3f",
                     i, ans, real_up, wins, loses, none, acc)

        audit.append({"i": i, "time": ts_now, "answer": ans, "real_up": real_up,
                      "wins": wins, "loses": loses, "none": none, "acc": acc})

        if i >= args.last_n:
            break

    pd.DataFrame(audit).to_csv("generator_audit_v5.csv", index=False)
    logging.info("=== generator_v5 finished ===")

if __name__ == "__main__":
    main()
