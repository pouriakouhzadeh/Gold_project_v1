#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generating_data_for_predict_script.py
- برای هر استپ، ۴ CSV زنده می‌سازد (M5/M15/M30/H1)
- برش‌ها «کافی» هستند تا هیچ NaN نهایی در فیچر آخر نماند
- تارگت را از PREP آموزش استخراج می‌کند (y_map)، نه از close خام
- منتظر answer.txt می‌ماند و نتیجه را می‌سنجد و گزارش می‌دهد
"""

import os, time, json, logging, joblib, warnings
import pandas as pd
warnings.filterwarnings("ignore")

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO: %(message)s"
)
LOGGER = logging.getLogger("gen")

# ------------------ تنظیمات ------------------
SRC = {
    "M5" : "XAUUSD_M5.csv",
    "M15": "XAUUSD_M15.csv",
    "M30": "XAUUSD_M30.csv",
    "H1" : "XAUUSD_H1.csv",
}
LIVE = {
    "M5" : "XAUUSD_M5_live.csv",
    "M15": "XAUUSD_M15_live.csv",
    "M30": "XAUUSD_M30_live.csv",
    "H1" : "XAUUSD_H1_live.csv",
}
MAIN_TF = "M30"

# تاریخچهٔ کافی برای حذف NaN ناشی از اندیکاتورهای بلند:
LOOKBACK = {
    "M5" : 12000,
    "M15": 4000,
    "M30": 2000,
    "H1" : 800
}

ANSWER_FILE = "answer.txt"
LAST_N_STEPS = 2000       # چند استپ پایانی روی M30
SLEEP_SEC = 1.0

# ---------------------------------------------

def build_target_map(window, feats):
    """تمام آموزش را با PREP می‌سازد و y_map: timestamp → y می‌سازد."""
    prep_full = PREPARE_DATA_FOR_TRAIN(
        filepaths={
            "30T": SRC["M30"],
            "15T": SRC["M15"],
            "5T" : SRC["M5"],
            "1H" : SRC["H1"],
        },
        main_timeframe="30T",
        fast_mode=False,
        strict_disk_feed=False,
        resample_label="right",
        resample_closed="right"
    )
    raw = prep_full.load_data()
    Xall, yall, tall, _ = prep_full.ready(
        raw,
        window=window,
        selected_features=feats,
        mode="train",
        with_times=True,
        predict_drop_last=False
    )
    y_map = dict(zip(tall.astype(str).tolist(), yall.tolist()))
    return y_map

def read_payload():
    bpath = "model_bundle.pkl"
    if os.path.exists(bpath):
        b = joblib.load(bpath)
        return {
            "window": int(b.get("window_size", 1)),
            "feats":  b.get("selected_features") or b.get("feats") or []
        }
    # fallback
    with open("payload.json","r") as f:
        j = json.load(f)
    return {
        "window": int(j.get("window_size", 1)),
        "feats":  j.get("selected_features") or j.get("feats") or []
    }

def main():
    LOGGER.info("=== Generator started ===")
    payload = read_payload()
    window  = payload["window"]
    feats   = payload["feats"]
    if not feats:
        raise RuntimeError("No feature list found in bundle/payload.")

    # y_map از آموزش
    y_map = build_target_map(window, feats)

    # داده‌ی اصلی 30m برای استخراج فهرست تایم‌ها
    df30 = pd.read_csv(SRC["M30"])
    # فرض: ستون زمان = 'time' یا 'datetime' → پیدا کن:
    tcol = "time" if "time" in df30.columns else ("datetime" if "datetime" in df30.columns else df30.columns[0])
    # آخرین LAST_N_STEPS تایم‌استمپ
    times = pd.to_datetime(df30[tcol])
    times = times.tail(LAST_N_STEPS + 1).reset_index(drop=True)  # +1 چون label یک گام جلوتر است
    # استپ‌ها را از 1 تا LAST_N_STEPS می‌گیریم؛ ts_i = times[i] (با شروع از 1)

    wins = loses = none = skipped = 0

    for i in range(1, LAST_N_STEPS+1):
        ts = times[i]  # همان تایم کندل هدف
        ts_str = str(ts)

        # اگر این timestamp در آموزش حذف شده (تعطیلات/گپ)، از ارزیابی رد می‌شویم:
        y_true = y_map.get(ts_str, None)
        if y_true is None:
            skipped += 1
            LOGGER.info(f"[Step {i}/{LAST_N_STEPS}] ts={ts_str} skipped (filtered in training)")
            continue

        # برش هر تایم‌فریم با تاریخچه کافی تا ts (شامل خودش)
        for tf, src in SRC.items():
            d = pd.read_csv(src)
            tcol_tf = "time" if "time" in d.columns else ("datetime" if "datetime" in d.columns else d.columns[0])
            d[tcol_tf] = pd.to_datetime(d[tcol_tf])
            # همه‌ی ردیف‌هایی که <= ts باشند
            d = d[d[tcol_tf] <= ts].tail(LOOKBACK[tf])
            d.to_csv(LIVE[tf], index=False)

        LOGGER.info(f"[Step {i}/{LAST_N_STEPS}] Live CSVs written at {ts} — waiting for answer.txt …")

        # صبر برای پاسخ پردیکتور
        while not os.path.exists(ANSWER_FILE):
            time.sleep(SLEEP_SEC)

        with open(ANSWER_FILE, "r") as f:
            ans = f.read().strip().upper()

        # پاک‌سازی answer تا دور بعدی
        os.remove(ANSWER_FILE)

        # NONE → فقط شمارش
        if ans == "NONE":
            none += 1
            LOGGER.info(f"[Result {i}] ans=NONE | ACC={(wins/(wins+loses)) if (wins+loses)>0 else 0:.3f} "
                        f"WINS={wins} LOSES={loses} NONE={none} SKIPPED={skipped}")
            continue

        real_up = bool(int(y_true))  # 1=UP, 0=DOWN (مثل آموزش)
        if (ans == "BUY" and real_up) or (ans == "SELL" and (not real_up)):
            wins += 1
            ok = True
        else:
            loses += 1
            ok = False

        LOGGER.info(f"[Result {i}] ans={ans} | real_up={real_up} | "
                    f"WINS={wins} LOSES={loses} NONE={none} ACC={wins/max(1,(wins+loses)):.3f}")

    LOGGER.info(f"=== Finished | ACC={wins/max(1,(wins+loses)):.3f} "
                f"WINS={wins} LOSES={loses} NONE={none} SKIPPED={skipped} ===")

if __name__ == "__main__":
    main()
