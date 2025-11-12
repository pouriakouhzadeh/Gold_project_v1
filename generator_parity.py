# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
generator_parity.py
-------------------
Writes *_live.csv trigger files for the last-N timestamps taken from the SAME
merged timeline as the simulator (PREPARE_DATA_FOR_TRAIN.load_data()).
Prediction script reads only XAUUSD_M30_live.csv (ts_now trigger) and ignores
others for feature-building (features are built from original raw CSVs).

Usage:
  python generator_parity.py --last-n 200 --base-dir . --symbol XAUUSD
"""

from __future__ import annotations
import os, sys, time, logging, argparse
from pathlib import Path
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOG_FILE = "generator_parity.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

def resolve_timeframe_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    candidates = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    resolved: dict[str,str] = {}
    for tf, names in candidates.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): 
                found = str(p); break
        if found:
            logging.info(f"[raw] {tf} -> {found}")
            resolved[tf] = found
    if "30T" not in resolved:
        raise FileNotFoundError(f"Main timeframe '30T' not found under {base_dir}")
    return resolved

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=200)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== generator_parity started ===")

    base = Path(args.base_dir).resolve()
    raw_paths = resolve_timeframe_paths(base, args.symbol)

    # Build the EXACT same merged timeline as simulator
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=raw_paths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    merged[tcol] = pd.to_datetime(merged[tcol])
    merged.sort_values(tcol, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    total = len(merged)
    start_idx = max(1, total - args.last_n)
    ans_path = Path("answer.txt")

    for i, ridx in enumerate(range(start_idx, total), start=1):
        ts_now = merged.loc[ridx, tcol]

        # Only M30_live is actually used by predictor. Others are optional triggers.
        m30_live = pd.DataFrame({"time":[ts_now], "open":[None], "high":[None], "low":[None], "close":[None], "volume":[None]})
        (Path(live_name(raw_paths["30T"]))).write_text(m30_live.to_csv(index=False))

        # (اختیاری) بنویسیم تا اگر خواستی در آینده استفاده کنی:
        for tf in ("15T","5T","1H"):
            if tf in raw_paths:
                dummy = pd.DataFrame({"time":[ts_now]})
                Path(live_name(raw_paths[tf])).write_text(dummy.to_csv(index=False))

        logging.info("[Step %d/%d] wrote *_live.csv @ %s — waiting for answer.txt ...", i, args.last_n, ts_now)

        t0 = time.time()
        while not ans_path.exists():
            time.sleep(args.sleep)
            if time.time() - t0 > 30:
                logging.warning("answer.txt timeout @ %s; counting as NONE", ts_now)
                break

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

        logging.info("[Result %d] ts=%s answer=%s", i, ts_now, ans)

        if i >= args.last_n:
            break

    logging.info("=== generator_parity finished ===")

if __name__ == "__main__":
    main()
