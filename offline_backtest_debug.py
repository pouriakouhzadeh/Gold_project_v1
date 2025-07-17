#!/usr/bin/env python3
# offline_backtest_debug_fixed.py
# --------------------------------------------------------------
#  گام‌به‌گام، هم‌زمان روی کنسول + لاگ‌فایل، بدون افت دقت
# --------------------------------------------------------------
import argparse, logging, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from dynamic_threshold_adjuster import DynamicThresholdAdjuster

# ─────────────── CLI ───────────────
cli = argparse.ArgumentParser()
cli.add_argument('-m', '--model', default='best_model.pkl')
cli.add_argument('-o', '--out', default='offline_backtest_report.csv')
cli.add_argument('--dyn-thr', action='store_true',
                 help='enable DynamicThresholdAdjuster')
cli.add_argument('--verbose', action='store_true',
                 help='console DEBUG printing')
args = cli.parse_args()

# ───────────── logging ─────────────
LOGFILE = 'offline_backtest.log'
file_handler = logging.FileHandler(LOGFILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                      datefmt='%H:%M:%S'))

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
log = logging.getLogger(__name__)

# ─────────── load model ───────────
mdl = joblib.load(args.model)
pipe, neg_thr, pos_thr = mdl['pipeline'], mdl['neg_thr'], mdl['pos_thr']
win, feats, cols = mdl['window_size'], mdl['feats'], mdl['train_window_cols']
log.info("✅ Model loaded | window=%d  features=%d", win, len(cols))

# ───────── build features ─────────
prep = PREPARE_DATA_FOR_TRAIN(main_timeframe='30T', verbose=False)
raw  = prep.load_data()
X_all, y_all, _ = prep.ready(raw, window=win,
                             selected_features=feats, mode='train')
X_all = X_all[cols].astype(float)
log.info("🔄 Feature matrix: %s", X_all.shape)

time_col = f"{prep.main_timeframe}_time"
time_ser = raw[time_col].iloc[-len(X_all):].reset_index(drop=True)

atr_ser = (raw['30T_ATR_14'].shift(1).fillna(0)
           .iloc[-len(X_all):].reset_index(drop=True)
           if '30T_ATR_14' in raw.columns else pd.Series(1., index=X_all.index))
vol_ser = (raw['30T_volume'].shift(1).fillna(0)
           .iloc[-len(X_all):].reset_index(drop=True)
           if '30T_volume' in raw.columns else pd.Series(1_000., index=X_all.index))
thr_adj = DynamicThresholdAdjuster(atr_high=10., vol_low=500, shift=0.01)

# ────────── back-test loop ─────────
y_pred = np.full_like(y_all, -1)
records = []
for i in range(len(X_all)):
    proba = pipe.predict_proba(X_all.iloc[i:i+1])[:, 1][0]
    n_thr, p_thr = (thr_adj.adjust(neg_thr, pos_thr, atr_ser.iat[i], vol_ser.iat[i])
                    if args.dyn_thr else (neg_thr, pos_thr))

    decision = "NAN"
    if proba <= n_thr:  decision, y_pred[i] = "SEL", 0
    elif proba >= p_thr: decision, y_pred[i] = "BUY", 1

    true_dir = "BUY" if y_all.iat[i] else "SEL"
    correct  = decision != "NAN" and ((decision == "BUY") == bool(y_all.iat[i]))

    msg = (f"[{i:05d}] {time_ser.iat[i]}  "
           f"proba={proba:.4f}  thr=({n_thr:.3f},{p_thr:.3f})  "
           f"→ {decision}  true={true_dir}  {'✔' if correct else '✘'}")
    log.info(msg)             # هم کنسول، هم فایل
    print(msg, flush=True)    # چاپ فوری کنسول

    records.append({
        'idx': i, 'time': str(time_ser.iat[i]), 'proba': proba,
        'neg_thr': n_thr, 'pos_thr': p_thr,
        'decision': decision, 'true': true_dir, 'correct': correct
    })

# ─────────── metrics ───────────
mask = y_pred != -1
conf_ratio = float(mask.mean())
f1  = f1_score(y_all[mask], y_pred[mask]) if mask.any() else 0.0
acc = accuracy_score(y_all[mask], y_pred[mask]) if mask.any() else 0.0
log.info("🏁 FINISHED | Conf-ratio=%.3f  F1=%.4f  Acc=%.4f",
         conf_ratio, f1, acc)
print(f"\nConf-ratio: {conf_ratio:.3f}   F1: {f1:.4f}   Acc: {acc:.4f}\n", flush=True)

# ───────── save CSV ─────────
pd.DataFrame(records).to_csv(args.out, index=False)
print(f"CSV report  → {Path(args.out).resolve()}", flush=True)
print(f"Full log    → {Path(LOGFILE).resolve()}", flush=True)
