#!/usr/bin/env python3
"""
Realtime‑like back‑test
--------------------------------------------------------------
* سازگار با نسخهٔ قبلی ولی با امکانات جدید:
  • ثبت تغییرات قیمت (close و delta)
  • دقت تجمعی و میانگین غلتان ۵۰ تایی
  • لاگ کامل در فایل و کنسول
  • CSV خروجی شامل ستون‌های جدید: price, delta_price, rolling_acc50

Usage (مثل قبل):
    python realtime_like_backtest.py --start 2024-01-02 [-m best.pkl] [--dyn-thr] [--verbose]
"""
import os, logging, argparse, joblib
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from dynamic_threshold_adjuster import DynamicThresholdAdjuster

RAW_FILES = {
    '30T': "XAUUSD_M30.csv",
    '15T': "XAUUSD_M15.csv",
    '5T' : "XAUUSD_M5.csv",
    '1H' : "XAUUSD_H1.csv"
}
LIVE_CSV = {
    '30T': "XAUUSD.F_M30_live.csv",
    '15T': "XAUUSD.F_M15_live.csv",
    '5T' : "XAUUSD.F_M5_live.csv",
    '1H' : "XAUUSD.F_H1_live.csv"
}
MAX_ROWS, LOGFILE = 4_999, "realtime_like_backtest.log"

# ───────── CLI ─────────
cli = argparse.ArgumentParser()
cli.add_argument('-m', '--model', default='best_model.pkl')
cli.add_argument('-o', '--out',   default='realtime_like_report.csv')
cli.add_argument('--dyn-thr',     action='store_true')
cli.add_argument('--start',       default='2024-01-02')
cli.add_argument('--verbose',     action='store_true')
args = cli.parse_args()

# ───────── Logger ─────────
file_hd = logging.FileHandler(LOGFILE, mode='w')
file_hd.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                       '%Y-%m-%d %H:%M:%S'))
file_hd.setLevel(logging.DEBUG)

console_hd = logging.StreamHandler()
console_hd.setFormatter(logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S'))
console_hd.setLevel(logging.DEBUG if args.verbose else logging.INFO)

logging.basicConfig(level=logging.DEBUG, handlers=[file_hd, console_hd], force=True)
log = logging.getLogger(__name__)

# ───────── helpers ─────────

def load_src(path: str) -> pd.DataFrame:
    """read CSV & sort by time"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File '{path}' not found.")
    return pd.read_csv(path, parse_dates=['time']).sort_values('time', ignore_index=True)

def dump_until(df: pd.DataFrame, t: pd.Timestamp, tf: str) -> bool:
    """Dump slice up to timestamp **t** into LIVE_CSV[tf] (max MAX_ROWS rows)."""
    idx = df['time'].searchsorted(t, side='right') - 1
    if idx < 0:
        return False
    df.iloc[:idx+1].tail(MAX_ROWS).to_csv(LIVE_CSV[tf], index=False)
    return True

def clean_live():
    for fp in LIVE_CSV.values():
        if os.path.exists(fp):
            os.remove(fp)

# ───────── load raw sources ─────────
raw_src = {tf: load_src(fp) for tf, fp in RAW_FILES.items()}
log.info("✅ Source CSVs loaded.")

# ───────── load model ─────────
mdl     = joblib.load(args.model)
pipe    = mdl['pipeline']
neg_thr = mdl['neg_thr']
pos_thr = mdl['pos_thr']
win     = mdl['window_size']
feats   = mdl['feats']
cols    = mdl['train_window_cols']
log.info("✅ Model loaded | window=%d  features=%d", win, len(cols))

thr_adj = DynamicThresholdAdjuster(atr_high=10.0, vol_low=500, shift=0.01)

# ───────── initialisations ─────────
prep   = PREPARE_DATA_FOR_TRAIN(filepaths=LIVE_CSV, main_timeframe='30T', verbose=False)
acc_hist: list[int] = []   # تاریخچهٔ صحت برای rolling‑acc
records, y_true, y_pred = [], [], []

m30         = raw_src['30T']
start_idx   = m30['time'].searchsorted(pd.Timestamp(args.start), side='left')
iterations  = len(m30) - start_idx - 1
log.info("🚀 Start %s  | iterations=%d", args.start, iterations)

# ───────── main loop ─────────
for k, i in enumerate(range(start_idx, len(m30) - 1), 1):
    now = m30.at[i, 'time']
    nxt, cur = m30.at[i + 1, 'close'], m30.at[i, 'close']
    delta    = nxt - cur
    lbl      = int(nxt > cur)            # برچسب حقیقت
    y_true.append(lbl)

    # نمایش سادهٔ پیشرفت در کنسول (فلاشر)
    print(f"[{k}/{iterations}] preparing {now}", flush=True)

    # ساخت CSV‑های live
    clean_live()
    if not all(dump_until(raw_src[tf], now, tf) for tf in LIVE_CSV):
        log.warning("❌ missing slice up to %s – skipped", now)
        y_pred.append(-1)
        acc_hist.append(0)
        continue

    merged = prep.load_data()
    X_live, _ = prep.ready_incremental(merged, window=win, selected_features=feats)
    if X_live.empty or X_live.isna().any().any():
        log.warning("⚠️ NaN/empty at %s", now)
        y_pred.append(-1)
        acc_hist.append(0)
        continue

    X_live = X_live[cols].astype(float)
    if win == 1:                 # window_size = 1  →  many rows returned
        X_live = X_live.tail(1)
    proba  = pipe.predict_proba(X_live)[:, 1][0]

    # آستانه‌های پویا / ثابت
    last_atr = merged['30T_ATR_14'].shift(1).iloc[-1] if '30T_ATR_14' in merged.columns else 1.0
    last_vol = merged['30T_volume'].shift(1).iloc[-1] if '30T_volume' in merged.columns else 1_000.0
    n_thr, p_thr = (
        thr_adj.adjust(neg_thr, pos_thr, last_atr, last_vol) if args.dyn_thr else (neg_thr, pos_thr)
    )

    # تصمیم
    dec, txt = -1, "NAN"
    if proba <= n_thr:
        dec, txt = 0, "SEL"
    elif proba >= p_thr:
        dec, txt = 1, "BUY"
    y_pred.append(dec)

    # صحت جاری و تجمعی
    correct = (dec != -1) and (dec == lbl)
    if dec != -1:
        acc_hist.append(int(correct))
    else:
        acc_hist.append(0)
    cum_acc = np.mean([a for a, p in zip(acc_hist, y_pred) if p != -1]) if any(p != -1 for p in y_pred) else 0.0
    roll_acc = np.mean(acc_hist[-50:]) if len(acc_hist) >= 50 else np.nan

    # لاگ
    log.info(
        "[%d/%d] %s p=%.4f thr=(%.3f,%.3f) → %-3s true=%s price=%.2f Δ=%.2f acc=%.3f roll50=%.3f",
        k, iterations, now, proba, n_thr, p_thr, txt,
        "BUY" if lbl else "SEL", cur, delta, cum_acc, roll_acc,
    )

    # ذخیرهٔ رکورد
    records.append({
        'iteration': k,
        'time': str(now),
        'price': cur,
        'delta_price': delta,
        'proba': proba,
        'neg_thr': n_thr,
        'pos_thr': p_thr,
        'decision': txt,
        'true_dir': "BUY" if lbl else "SEL",
        'correct': correct,
        'cumulative_acc': cum_acc,
        'rolling_acc50': roll_acc,
    })

# ───────── metrics & export ─────────
y_true, y_pred = np.array(y_true), np.array(y_pred)
mask  = y_pred != -1
conf  = mask.mean()
f1    = f1_score(y_true[mask], y_pred[mask]) if mask.any() else 0.0
acc   = accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else 0.0

out_path = Path(args.out)
pd.DataFrame(records).to_csv(out_path, index=False)
clean_live()

print(f"\nConf‑ratio: {conf:.3f}   F1: {f1:.4f}   Acc: {acc:.4f}")
print(f"CSV  → {out_path.resolve()}")
print(f"LOG  → {Path(LOGFILE).resolve()}")
