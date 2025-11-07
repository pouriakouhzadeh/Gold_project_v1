#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, logging
import pandas as pd

LOG_FILE = "prediction_generator.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")]
)

CSV_BASE = {
    "5T":  "XAUUSD_M5.csv",
    "15T": "XAUUSD_M15.csv",
    "30T": "XAUUSD_M30.csv",
    "1H":  "XAUUSD_H1.csv",
}
CSV_LIVE = {tf: fn.replace(".csv", "_live.csv") for tf, fn in CSV_BASE.items()}
ANSWER_FILE = "answer.txt"

# اندازه‌ی برشِ عقب‌گرد برای هر TF (بزرگ‌تر از بزرگ‌ترین رولینگ)
WINDOW_BACK = {"5T": 3000, "15T": 1000, "30T": 500, "1H": 300}

def load_all() -> dict:
    bases = {}
    for tf, f in CSV_BASE.items():
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        bases[tf] = df
    return bases

def write_live_slices(anchor_ts: pd.Timestamp, bases: dict):
    for tf, df in bases.items():
        if anchor_ts not in df.index:
            idx = df.index.searchsorted(anchor_ts, side="right") - 1
            if idx < 0: idx = 0
            anchor = df.index[idx]
        else:
            anchor = anchor_ts
        n_back = WINDOW_BACK[tf]
        pos = df.index.get_loc(anchor)
        s = max(0, pos - n_back + 1)
        cut = df.iloc[s:pos+1].copy().reset_index()
        cut.to_csv(CSV_LIVE[tf], index=False)

def main():
    logging.info("=== Generator started ===")
    bases = load_all()
    main30 = bases["30T"]

    steps = 2000
    if len(main30) < steps + 1:
        steps = max(1, len(main30) - 1)
    start_i = len(main30) - steps - 1

    wins = loses = none = 0
    for k in range(steps):
        anchor_ts = main30.index[start_i + k]
        write_live_slices(anchor_ts, bases)
        logging.info(f"[Step {k+1}/{steps}] Live CSVs written at {anchor_ts} — waiting for {ANSWER_FILE} …")

        # منتظر پاسخ
        while not os.path.exists(ANSWER_FILE):
            time.sleep(1.0)

        # خواندن پاسخ
        with open(ANSWER_FILE, "r") as f:
            ans = f.read().strip().upper()

        # نمره‌دهی: جهت کندل بعدی در 30T
        i = main30.index.get_loc(anchor_ts)
        real_up = None
        if i < len(main30) - 1:
            c0 = float(main30.iloc[i]["close"])
            c1 = float(main30.iloc[i+1]["close"])
            real_up = (c1 > c0)

        if ans == "NONE" or real_up is None:
            none += 1
        else:
            pred_up = (ans == "BUY")
            if pred_up == real_up:
                wins += 1
            else:
                loses += 1

        try:
            os.remove(ANSWER_FILE)
        except FileNotFoundError:
            pass

        total_scored = wins + loses
        acc = wins / total_scored if total_scored else 0.0
        logging.info(f"[Result {k+1}] ans={ans} | real_up={real_up} | "
                     f"WINS={wins} LOSES={loses} NONE={none} ACC={acc:.3f}")

    logging.info("=== Generator completed ===")

if __name__ == "__main__":
    main()
