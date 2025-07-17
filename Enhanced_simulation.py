#!/usr/bin/env python3
"""
enhanced_simulation.py

شبیه‌ساز (Producer) با هماهنگی همگام + پاک‌کردن فایل‌های لایو در شرایط NAN
+ حذف فایل‌های باقی‌مانده در ابتدای کار
+ لاگ در فایل و کنسول
"""

import pandas as pd
import os
import time
import logging
import json
import multiprocessing as mp
import sys

# --- تنظیمات ---
MAX_ROWS = 4999
SLEEP_TIME = 1
TIMEOUT_ANSWER = 300
LOG_FILENAME = "simulation_synchronous.log"
REPORT_CSV = "simulation_report.csv"

# فایل‌های منبع
SOURCE_M30 = "XAUUSD_M30.csv"
SOURCE_M15 = "XAUUSD_M15.csv"
SOURCE_M5  = "XAUUSD_M5.csv"
SOURCE_H1  = "XAUUSD_H1.csv"

# فایل‌های لایو
LIVE_M30 = "XAUUSD.F_M30_live.csv"
LIVE_M15 = "XAUUSD.F_M15_live.csv"
LIVE_M5  = "XAUUSD.F_M5_live.csv"
LIVE_H1  = "XAUUSD.F_H1_live.csv"
ANSWER_TXT = "Answer.txt"

def remove_if_exists(fp):
    if os.path.exists(fp):
        os.remove(fp)

def remove_initial_files():
    """
    در ابتدای برنامه، اگر فایلی از اجرای قبلی مانده باشد، پاک می‌کنیم 
    تا گیر نکنیم.
    """
    for f in [LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1, ANSWER_TXT]:
        remove_if_exists(f)

def setup_logger():
    handlers = []
    # فایل لاگ
    file_handler = logging.FileHandler(LOG_FILENAME, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers
    )
    logging.info("--- Enhanced Simulation Started ---")

def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df.dropna(subset=["time"], inplace=True)

    # اطمینان از عددی بودن ستون‌های OHLCV
    num_cols = ["open", "high", "low", "close", "volume"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=num_cols, inplace=True)

    df.sort_values("time", inplace=True, ignore_index=True)
    logging.debug(f"[load_and_prepare] {path} => shape={df.shape}")
    return df


def find_index_by_time(df, target_time, freq='M5'):
    if freq == 'H1':
        target_time = target_time.replace(minute=0, second=0, microsecond=0)
    elif freq == 'M15':
        mod_ = target_time.minute % 15
        target_time = target_time.replace(minute=target_time.minute - mod_, second=0, microsecond=0)
    elif freq == 'M5':
        mod_ = target_time.minute % 5
        target_time = target_time.replace(minute=target_time.minute - mod_, second=0, microsecond=0)

    idx = df['time'].searchsorted(target_time, side='left')
    if idx == len(df):
        return -1
    if df.iloc[idx]['time'] == target_time:
        return idx
    return -1

def make_csv_for_time(df, target_time, freq, output_csv):
    idx = find_index_by_time(df, target_time, freq=freq)
    if idx == -1:
        logging.warning(f"[make_csv_for_time] {freq} not found => {target_time}")
        return False
    df_part = df.iloc[:idx+1].tail(MAX_ROWS)
    df_part.to_csv(output_csv, index=False)
    return True

def create_csvs(time_, df_m5, df_m15, df_m30, df_h1):
    tasks = [
        (df_m5, time_, 'M5',  LIVE_M5),
        (df_m15, time_, 'M15', LIVE_M15),
        (df_m30, time_, 'M30', LIVE_M30),
        (df_h1, time_, 'H1',  LIVE_H1)
    ]
    success = True
    with mp.Pool(processes=len(tasks)) as pool:
        results = [pool.apply_async(make_csv_for_time, t) for t in tasks]
        pool.close()
        pool.join()
        for r in results:
            if not r.get():  # اگر یکی از تایم‌فریم‌ها پیدا نشود
                success = False
    return success

def wait_until_files_removed():
    while any(os.path.exists(fp) for fp in [LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1]):
        logging.debug("[wait_until_files_removed] Live files exist => wait ...")
        time.sleep(SLEEP_TIME)

def wait_for_answer():
    waited = 0
    while not os.path.isfile(ANSWER_TXT):
        time.sleep(SLEEP_TIME)
        waited += SLEEP_TIME
        if waited > TIMEOUT_ANSWER:
            logging.warning("[wait_for_answer] timed out => (NAN,0.0)")
            return ("NAN", 0.0)
    with open(ANSWER_TXT, 'r') as f:
        txt = f.read().split(',')
    remove_if_exists(ANSWER_TXT)
    if len(txt) == 2:
        return (txt[0].strip().upper(), float(txt[1]))
    elif len(txt) == 1:
        return (txt[0].strip().upper(), 0.0)
    return ("NAN", 0.0)

def remove_live_files():
    """
    اگر قرار باشد iteration را رد کنیم (NAN) 
    بهتر است هر فایلی که ساخته شده حذف کنیم.
    """
    for fp in [LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1]:
        remove_if_exists(fp)

def main_simulation():
    setup_logger()
    # ابتدا فایل‌های قدیمی را پاک می‌کنیم
    remove_initial_files()

    df_m5  = load_and_prepare(SOURCE_M5)
    df_m15 = load_and_prepare(SOURCE_M15)
    df_m30 = load_and_prepare(SOURCE_M30)
    df_h1  = load_and_prepare(SOURCE_H1)
# --- NEW: earliest common timestamp across all TFs -------------------------
    earliest_common = max(
        df_m30["time"].min(),
        df_m15["time"].min(),
        df_m5 ["time"].min(),
        df_h1 ["time"].min(),
        pd.Timestamp("2024-01-01")   # حداقل تاریخی که می‌خواهی
    )

    df_m30 = df_m30[df_m30["time"] >= earliest_common].copy()
    df_m15 = df_m15[df_m15["time"] >= earliest_common].copy()
    df_m5  = df_m5 [df_m5 ["time"] >= earliest_common].copy()
    df_h1  = df_h1 [df_h1 ["time"] >= earliest_common].copy()
# ---------------------------------------------------------------------------




    wins, loses, nan_count = 0, 0, 0
    profit_pips = 0.0
    history = []

    total_len = len(df_m30)
    logging.info(f"[main_simulation] M30 length={total_len}")
    skip = 1
    for idx in range(skip,total_len - 1):
        row_cur  = df_m30.iloc[idx]
        row_next = df_m30.iloc[idx + 1]
        wait_until_files_removed()
        c0 = row_cur["close"]
        c1 = row_next["close"]

        delta = c1 - c0                 # ↑ آینده منهای حال
        true_pos = "BUY" if delta > 0 else "SEL"


        time_ = row_cur["time"] if "time" in row_cur else row_cur.name
        if time_.weekday() >= 5:          # 5=Saturday, 6=Sunday در تقویم پایتون
            continue
        if not all(find_index_by_time(df, time_, freq=f) != -1 
            for df, f in [(df_m5,'M5'), (df_m15,'M15'), (df_m30,'M30'), (df_h1,'H1')]):
            continue           # رد کن، چون دادهٔ ناقص است

        logging.info(f"\n[Sim Iter={idx+1}] time={time_}, True={true_pos}, Δ={delta:.2f}")
        success = create_csvs(time_, df_m5, df_m15, df_m30, df_h1)

        if not success:
            nan_count += 1
            logging.warning("[Sim] Missing time in TF => treat as NAN iteration => removing CSVs.")
            # اگر یکی از تایم‌فریم‌ها تاریخ را نداشت => این iteration را رد می‌کنیم
            remove_live_files()  # فایل‌های تولیدشده را حذف کنیم تا آنالیزگر منتظرشان نماند
            continue

        # حالا منتظر پاسخ
        ai_decision, proba_val = wait_for_answer()

        correct = False
        if ai_decision == true_pos:
            correct = True
            wins += 1
            profit_pips += abs(delta)
        elif ai_decision in ["BUY", "SEL"]:
            loses += 1
            profit_pips -= abs(delta)
        else:                     # decision == "NAN"
            nan_count += 1
            # continue              # این نمونه در Acc/F1 حساب نشود

        # فقط BUY یا SEL به این نقطه مى‌رسند
        total_moves = wins + loses
        acc = wins / total_moves if total_moves else 0.0


        rec = {
            "time": str(time_),
            "ai_decision": ai_decision,
            "proba": proba_val,
            "true_direction": true_pos,
            "pip_change": float(delta),
            "correct": correct,
            "cumulative_wins": wins,
            "cumulative_loses": loses,
            "cumulative_profit": profit_pips,
            "cumulative_acc": acc,
            "nan_count": nan_count
        }
        history.append(rec)

        logging.info(
            f"[Sim Iter={idx+1}] Decision={ai_decision}, Proba={proba_val:.4f}, "
            f"Correct={correct}, Acc={acc:.4f}, Wins={wins}, Loses={loses}, NAN={nan_count}"
        )

    # در انتها گزارش را ذخیره کنیم
    df_report = pd.DataFrame(history)
    df_report.to_csv(REPORT_CSV, index=False)
    logging.info(f"[Sim] DONE => {REPORT_CSV}")

if __name__ == "__main__":
    main_simulation()
