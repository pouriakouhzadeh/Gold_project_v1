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

MAIN_TF = "30T"
CSV_MAIN = "XAUUSD_M30.csv"
CSV_MAP = {
    "5T":  "XAUUSD_M5.csv",
    "15T": "XAUUSD_M15.csv",
    "30T": "XAUUSD_M30.csv",
    "1H":  "XAUUSD_H1.csv",
}
LIVE_SUFFIX = "_live.csv"

# برش‌های بزرگ‌تر از ماکس رولینگ‌های رایج (در صورت نیاز تغییر دهید)
WINDOW_HINTS = {
    "5T": 3000,
    "15T": 1000,
    "30T": 500,
    "1H": 300,
}

ANSWER_FILE = "answer.txt"

def write_live_csv_slices(anchor_ts: pd.Timestamp, bases: dict):
    """از زمان anchor_ts در هر تایم‌فریم برش مناسب را می‌سازد و با پسوند _live.csv می‌نویسد."""
    for tf, fname in CSV_MAP.items():
        df = bases[tf]
        if anchor_ts not in df.index:
            # نزدیک‌ترین ایندکس <= anchor
            anchor = df.index[df.index.searchsorted(anchor_ts, side="right") - 1]
        else:
            anchor = anchor_ts
        n_back = WINDOW_HINTS[tf]
        pos = df.index.get_loc(anchor)
        s = max(0, pos - n_back + 1)
        cut = df.iloc[s:pos+1].copy()
        live_name = fname.replace(".csv", LIVE_SUFFIX)
        cut.to_csv(live_name, index=True)
    return

def read_answer_and_score(ans_path: str, main_df: pd.DataFrame, anchor_ts: pd.Timestamp):
    """خواندن پاسخ BUY/SELL/NONE و مقایسه با جهت کندل بعدی در TF اصلی (30T)"""
    with open(ans_path, "r") as f:
        ans = f.read().strip().upper()
    # جهت واقعی
    if anchor_ts not in main_df.index:
        anchor = main_df.index[main_df.index.searchsorted(anchor_ts, side="right") - 1]
    else:
        anchor = anchor_ts
    i = main_df.index.get_loc(anchor)
    if i >= len(main_df) - 1:
        # کندل بعدی نداریم → امتیازدهی نکن
        real_up = None
    else:
        c0 = float(main_df.iloc[i]["close"])
        c1 = float(main_df.iloc[i+1]["close"])
        real_up = (c1 > c0)

    return ans, real_up

def main():
    logging.info("=== Generator started ===")

    # لود دیتای اصلی
    bases = {}
    for tf, fname in CSV_MAP.items():
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing file: {fname}")
        df = pd.read_csv(fname)
        # انتظار: ستون‌های time, open, high, low, close, volume
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        bases[tf] = df

    # محدوده‌ی حرکت (2000 قدم آخر در تایم‌فریم اصلی)
    main = bases["30T"]
    steps = 2000
    if len(main) < steps + 1:
        steps = max(1, len(main) - 1)
    # از steps کندل قبل از آخرین ردیف شروع می‌کنیم
    start_i = len(main) - steps - 1

    wins = loses = none = 0
    for k in range(steps):
        idx = start_i + k
        anchor_ts = main.index[idx]  # زمان مرجع در 30T

        # 1) تولید ۴ CSV لایو
        write_live_csv_slices(anchor_ts, bases)
        logging.info(f"[Step {k+1}/{steps}] Live CSVs written at {anchor_ts} — waiting for {ANSWER_FILE} …")

        # 2) انتظار برای answer.txt
        while not os.path.exists(ANSWER_FILE):
            time.sleep(1.0)

        # 3) خواندن پاسخ و امتیازدهی
        ans, real_up = read_answer_and_score(ANSWER_FILE, main, anchor_ts)
        if ans == "NONE":
            none += 1
        elif real_up is None:
            # کندل بعدی نداریم → امتیازدهی نکن، ولی none تلویحاً
            none += 1
        else:
            # BUY یعنی انتظار صعود؛ SELL یعنی انتظار نزول
            pred_up = (ans == "BUY")
            if pred_up == real_up:
                wins += 1
            else:
                loses += 1

        # 4) پاک کردن پاسخ برای قدم بعد
        try:
            os.remove(ANSWER_FILE)
        except FileNotFoundError:
            pass

        # 5) گزارش لحظه‌ای
        total_scored = wins + loses
        acc = (wins / total_scored) if total_scored > 0 else 0.0
        logging.info(f"[Result {k+1}] ans={ans} | real_up={real_up} | "
                     f"WINS={wins} LOSES={loses} NONE={none} ACC={acc:.3f}")

    logging.info("=== Generator completed ===")

if __name__ == "__main__":
    main()
