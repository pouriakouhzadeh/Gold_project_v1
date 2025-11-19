#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, time, logging
import pandas as pd
import numpy as np
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

LOGFMT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT, datefmt="%Y-%m-%d %H:%M:%S")
L = logging.getLogger("generator")

def write_live_csvs(base_dir, symbol, ts, raw_paths):
    """تا زمان ts شامل، برای هر TF یک *_live.csv می‌سازد."""
    for tf, fp in raw_paths.items():
        df = pd.read_csv(fp)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[df["time"] <= ts].copy()
        out = os.path.join(base_dir, f"{symbol}_{'M30' if tf=='30T' else tf}_live.csv")
        df.to_csv(out, index=False)

def wait_for_deploy_row(base_dir, ts):
    """منتظر می‌ماند تا deploy_predictions.csv شامل یک ردیف با timestamp==ts شود."""
    pred_path = os.path.join(base_dir, "deploy_predictions.csv")
    ans_path  = os.path.join(base_dir, "answer.txt")
    while True:
        # اول تلاش با deploy_predictions.csv
        if os.path.exists(pred_path):
            try:
                df = pd.read_csv(pred_path)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    hit = df.loc[df["timestamp"] == ts]
                    if len(hit) > 0:
                        # خروجی
                        act = str(hit["action"].iloc[-1])
                        y_prob = float(hit["y_prob"].iloc[-1])
                        cover_cum = float(hit.get("cover_cum", pd.Series([np.nan])).iloc[-1])
                        return act, y_prob, cover_cum
            except Exception:
                pass
        # اگر نبود، به answer.txt بسنده می‌کنیم
        if os.path.exists(ans_path):
            try:
                with open(ans_path, "r", encoding="utf-8") as f:
                    act = f.read().strip().upper()
                if act in ("BUY","SELL","NONE"):
                    return act, np.nan, np.nan
            except Exception:
                pass
        time.sleep(0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--last-n", type=int, default=200)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()
    if args.verbosity <= 0: L.setLevel(logging.WARNING)

    L.info("=== Generator started ===")
    base = args.base_dir
    raw_paths = {
        "30T": os.path.join(base, f"{args.symbol}_M30.csv"),
        "15T": os.path.join(base, f"{args.symbol}_M15.csv"),
        "5T":  os.path.join(base, f"{args.symbol}_M5.csv"),
        "1H":  os.path.join(base, f"{args.symbol}_H1.csv"),
    }
    for tf, fp in raw_paths.items():
        L.info("[raw] %s -> %s", tf, fp)

    # برای محاسبه y_true
    main_df = pd.read_csv(raw_paths["30T"])
    main_df["time"]  = pd.to_datetime(main_df["time"], errors="coerce")
    main_df.sort_values("time", inplace=True)
    main_df.reset_index(drop=True, inplace=True)

    # آخرین N timestamp (به‌جز آخرین ردیف که y_true ندارد)
    if len(main_df) < args.last_n + 1:
        args.last_n = max(1, len(main_df)-1)
    ts_list = main_df["time"].tail(args.last_n+1).reset_index(drop=True)
    ts_list = ts_list.iloc[:-1]  # آخرین را کنار بگذاریم چون y_true ندارد

    wins=loses=none=0
    for i, ts in enumerate(ts_list, start=1):
        # 1) نوشتن فایل‌های live
        write_live_csvs(base, args.symbol, ts, raw_paths)
        L.info("[Step %d/%d] Live CSVs written at %s — waiting for deploy …",
               i, len(ts_list), ts)

        # 2) انتظار برای پاسخ deploy
        act, y_prob, cover_cum = wait_for_deploy_row(base, ts)

        # 3) محاسبه y_true از CSV اصلی
        idx = main_df.index[main_df["time"] == ts]
        if len(idx) == 0 or idx[0] >= len(main_df)-1:
            y_true = np.nan
        else:
            i0 = int(idx[0])
            y_true = int(main_df["close"].iloc[i0+1] > main_df["close"].iloc[i0])

        # 4) به‌روزرسانی آمار
        if act == "NONE":
            none += 1
        elif pd.notna(y_true):
            if (act == "BUY" and y_true==1) or (act == "SELL" and y_true==0):
                wins += 1
            else:
                loses += 1

        cover = (wins+loses)/max(1, i)
        acc   = wins/max(1, wins+loses) if (wins+loses)>0 else 0.0
        L.info("[Step %d] cover=%.3f acc=%.3f wins=%d loses=%d none=%d",
               i, cover, acc, wins, loses, none)

        # 5) ذخیره‌ی رکورد این استپ
        row = {
            "timestamp": ts,
            "action": act,
            "y_true": y_true,
            "y_prob": y_prob,
            "cover_cum": cover_cum
        }
        hdr = not os.path.exists(os.path.join(base, "generator_predictions.csv"))
        pd.DataFrame([row]).to_csv(os.path.join(base, "generator_predictions.csv"),
                                   mode="a", header=hdr, index=False)

    L.info("[Final] acc=%.3f cover=%.3f wins=%d loses=%d none=%d",
           (wins/max(1, wins+loses)), ((wins+loses)/max(1,len(ts_list))), wins, loses, none)

if __name__ == "__main__":
    main()
