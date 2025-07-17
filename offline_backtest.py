#!/usr/bin/env python3
# offline_backtest.py
"""
Fast, alignment-proof back-tester for the CatBoost/LogReg model.

• Feeds the model one row at a time in chronological order.
• Uses PREPARE_DATA_FOR_TRAIN → ready(...) with *mode='train'* so the
  feature/target alignment is **identical** to training/quick_eval.
• Optional DynamicThresholdAdjuster is kept, but can be disabled with
  --static-thr for a pure static-threshold run.
• Generates `offline_backtest_report.csv` with per-candle results and
  prints final Conf-ratio, F1 and Acc.

Run:
    python offline_backtest.py               # dynamic thresholds
    python offline_backtest.py --static-thr  # static thresholds (quick_eval style)
"""

import argparse, logging, joblib, pandas as pd, numpy as np
from pathlib import Path
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN     # :contentReference[oaicite:0]{index=0}
from dynamic_threshold_adjuster import DynamicThresholdAdjuster  # :contentReference[oaicite:1]{index=1}

# ─────────────────────────────── CLI ───────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('-m','--model', default='best_model.pkl', help='trained model pkl')
ap.add_argument('-o','--out'  , default='offline_backtest_report.csv', help='output CSV')
ap.add_argument('--static-thr', action='store_true',
                help='disable DynamicThresholdAdjuster (use fixed neg_thr / pos_thr)')
args = ap.parse_args()

# ─────────────────────────── logging ───────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

# ───────────────────────── load model ──────────────────────────────
saved   = joblib.load(args.model)
pipe    = saved['pipeline']
neg_thr = saved['neg_thr']
pos_thr = saved['pos_thr']
w_size  = saved['window_size']
feats   = saved['feats']
cols    = saved['train_window_cols']
logging.info("Model loaded (window=%d, |cols|=%d)", w_size, len(cols))

# ──────────────────── build full feature set ───────────────────────
prep  = PREPARE_DATA_FOR_TRAIN(main_timeframe='30T', verbose=False)
raw   = prep.load_data()                           # full merged DF (all TFs)
X_all, y_all, _ = prep.ready(                      # identical to training
    raw, window=w_size, selected_features=feats, mode='train')
X_all = X_all[cols].astype(float)                  # keep training col order
logging.info("Feature matrix ready: %s", X_all.shape)

# ATR / volume streams for optional dynamic thresholds
atr_ser = raw['30T_ATR_14'].shift(1).fillna(0).iloc[w_size:].reset_index(drop=True) \
            if '30T_ATR_14' in raw.columns else pd.Series(1.0, index=X_all.index)
vol_ser = raw['30T_volume' ].shift(1).fillna(0).iloc[w_size:].reset_index(drop=True) \
            if '30T_volume' in raw.columns else pd.Series(1000.0, index=X_all.index)
thr_adj = DynamicThresholdAdjuster(atr_high=10.0, vol_low=500, shift=0.01)

# ───────────────────── back-test loop ──────────────────────────────
wins = loses = nan = 0
rows = []
for i in range(len(X_all)):
    x = X_all.iloc[i:i+1]
    proba = pipe.predict_proba(x)[:,1][0]

    if args.static_thr:
        n_thr, p_thr = neg_thr, pos_thr
    else:
        n_thr, p_thr = thr_adj.adjust(neg_thr, pos_thr,
                                      atr_ser.iat[i], vol_ser.iat[i])

    decision = "NAN"
    if proba <= n_thr:
        decision = "SEL"
    elif proba >= p_thr:
        decision = "BUY"

    true_dir = "BUY" if y_all.iat[i] == 1 else "SEL"
    correct  = decision == true_dir and decision != "NAN"

    if decision == "NAN":
        nan += 1
    elif correct:
        wins += 1
    else:
        loses += 1

    acc_so_far = wins / (wins + loses) if (wins + loses) else 0.0
    rows.append({
        "index": i,
        "decision": decision,
        "proba": proba,
        "true": true_dir,
        "correct": correct,
        "cumulative_acc": acc_so_far,
    })

# ─────────────────────── results & save ────────────────────────────
df_rep = pd.DataFrame(rows)
df_rep.to_csv(args.out, index=False)
mask = df_rep.decision != "NAN"
conf_ratio = mask.mean()
f1 = (2 * (df_rep.correct & mask).sum()) / ((df_rep.decision == "BUY").sum()
       + (df_rep.decision == "SEL").sum() + (df_rep.true == "BUY").sum()
       + (df_rep.true == "SEL").sum()) if mask.any() else 0
acc = (df_rep.correct & mask).mean() if mask.any() else 0
logging.info("Done → %s", Path(args.out).name)
print(f"Conf-ratio: {conf_ratio:.3f}   F1: {f1:.4f}   Acc: {acc:.4f}")
