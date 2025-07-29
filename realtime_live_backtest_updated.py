#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_live_backtest_retrain.py
─────────────────────────────────
✓ مقایسهٔ snapshot دودکش با لایو  
✓ پایش PSI و ری‌کالیبرِ سریع  
✓ خروجی ستون‌های پُراختلاف
"""

from __future__ import annotations
import argparse, logging, sys, json
from collections import Counter, deque
from pathlib import Path
from typing import List

import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from drift_checker import DriftChecker
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ═══════════════ پارامترهای ثابت ═══════════════
MODEL_PATH            = Path("best_model.pkl")
RAW_FILES             = {"30T":"XAUUSD_M30.csv","15T":"XAUUSD_M15.csv",
                         "5T":"XAUUSD_M5.csv","1H":"XAUUSD_H1.csv"}
MAIN_TF               = "30T"
TIME_COL              = f"{MAIN_TF}_time"
SNAP_BATCH_CSV        = "chimney_snapshot.csv"
SNAP_LIVE_CSV         = "live_snapshot.csv"
DIFF_CSV              = "comparison_differences.csv"
TOP_UNSTABLE_FEATURES = "top_unstable_features.txt"

PSI_ALERT        = 0.50      # هشدار
RECENT_CALIB_WIN = 350       # طول بافر داده برای ری-کالیبر
MIN_CALIB_SAMPLES= 250       # حداقل رکورد جهت ری-کالیبر
MAX_ROWS_RAW     = 8000

# ═══════════════ لاگینگ ═══════════════
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("live-test")

# ═══════════════ توابع کمکی ═══════════════
def align_cols(df: pd.DataFrame, ordered: List[str]) -> pd.DataFrame:
    for c in ordered:
        if c not in df.columns:
            df[c] = np.nan
    return df[ordered]

def ensure_time(df: pd.DataFrame, tag:str)->None:
    global TIME_COL
    if TIME_COL in df: return
    cand=[c for c in df if c.endswith("_time")]
    if cand:
        LOG.warning("%s: '%s' نبود؛ '%s' جایگزین شد", tag, TIME_COL, cand[0])
        df.rename(columns={cand[0]:TIME_COL},inplace=True)
    else:
        raise KeyError(f"{tag}: ستون زمان یافت نشد")

# ═══════════════ اسکریپت اصلی ═══════════════
def main():
    # ── مدل ─────────────────────────────
    mdl   = joblib.load(MODEL_PATH)
    pipe  = mdl["pipeline"]
    window= int(mdl["window_size"])
    neg_thr, pos_thr = float(mdl["neg_thr"]), float(mdl["pos_thr"])
    final_cols       = list(mdl.get("train_window_cols", mdl["feats"]))

    # ── داده خام ────────────────────────
    prep = PREPARE_DATA_FOR_TRAIN({k:str(Path(v)) for k,v in RAW_FILES.items()},
                                  main_timeframe=MAIN_TF, verbose=False)
    raw  = prep.load_data()
    ensure_time(raw,"raw")
    raw[TIME_COL] = pd.to_datetime(raw[TIME_COL])
    raw.sort_values(TIME_COL, inplace=True, ignore_index=True)
    raw = raw.tail(MAX_ROWS_RAW).reset_index(drop=True)

    LOG.info("✔ raw rows=%d  cols=%d", *raw.shape)

    # ── دودکش (Batch) snapshot ───────────
    X_b, y_b, *_ = prep.ready(raw, window=window,
                              selected_features=mdl["feats"], mode="train")
    X_b = align_cols(X_b.copy(), final_cols).astype("float32")
    med = X_b.median(); X_b.fillna(med, inplace=True)

    proba_b = pipe.predict_proba(X_b)[:,1]
    lbl_b   = np.full_like(proba_b, -1,dtype=int)
    lbl_b[proba_b<=neg_thr]=0; lbl_b[proba_b>=pos_thr]=1
    times_b = raw[TIME_COL].iloc[window:window+len(X_b)].reset_index(drop=True)
    pd.concat([X_b, times_b.rename(TIME_COL),
               pd.Series(proba_b,name="proba"),
               pd.Series(lbl_b,name="label")],axis=1
             ).to_csv(SNAP_BATCH_CSV,index=False)

    # ── پایش PSI ─────────────────────────
    psi_chk = DriftChecker(verbose=False)
    psi_chk.load_train_distribution("train_distribution.json")

    # ── بافر ری-کالیبر ───────────────────
    recent_X = deque(maxlen=RECENT_CALIB_WIN)
    recent_y = deque(maxlen=RECENT_CALIB_WIN)

    # ── متغیرهای آمار ─────────────────────
    y_true_all, y_pred_all = [], []
    live_rows, diff_feat_counter = [], Counter()

    # ── اجرای لایو روی هر رکورد ───────────
    for idx in range(window, len(raw)-1):
        window_df = raw.iloc[idx-window:idx+1].copy()
        X_l,_ = prep.ready_incremental(window_df, window=window,
                                       selected_features=mdl["feats"])
        if X_l.empty: continue
        X_l = align_cols(X_l, final_cols).astype("float32")
        X_l.fillna(med,inplace=True)

        psi_val = psi_chk.compare_live(X_l)
        if psi_val>PSI_ALERT:
            LOG.warning("⚠️ PSI %.3f > %.2f @ %s",
                        psi_val, PSI_ALERT,
                        raw.at[idx,TIME_COL])

        # ---------- پیش‌بینی ----------
        p = float(pipe.predict_proba(X_l)[0,1])
        lbl=-1
        if p<=neg_thr: lbl=0
        elif p>=pos_thr: lbl=1

        # ---------- هدف واقعی ----------
        cur_close  = float(raw.at[idx,   f"{MAIN_TF}_close"])
        nxt_close  = float(raw.at[idx+1, f"{MAIN_TF}_close"])
        y_true     = int(nxt_close>cur_close)

        # ---------- آمار ----------
        if lbl!=-1:
            y_true_all.append(y_true)
            y_pred_all.append(lbl)

        recent_X.append(X_l.iloc[0].values)
        recent_y.append(y_true)

        live_rows.append({**X_l.iloc[0].to_dict(),
                          TIME_COL:raw.at[idx,TIME_COL].strftime("%Y-%m-%d %H:%M:%S"),
                          "proba":p,"label":lbl,"y_true":y_true})

        # ---------- ری-کالیبر سریع ----------
        if psi_val>PSI_ALERT and len(recent_y)>=MIN_CALIB_SAMPLES:
            LOG.info("🔄 Re-calibrating on last %d samples …", len(recent_y))
            X_cal = pd.DataFrame(list(recent_X), columns=final_cols)
            y_cal = np.array(recent_y, dtype=int)

            # اسکیلر + Logistic + کالیبره (کل پایپ لاین) با warm_start دوباره fit
            # برای سرعت، max_iter لایهٔ LR را کم می‌کنیم
            clf = pipe.named_steps["classifier"]
            lr  = clf.estimator
            old_iter = lr.max_iter
            lr.max_iter = 50
            pipe.fit(X_cal, y_cal)
            lr.max_iter = old_iter           # بازگرداندن مقدار قبلی
            LOG.info("✅ Re-calibration done.")

            # پس از ری-کالیبر احتمالاً توزیع proba عوض می‌شود → می‌توان
            # آستانه‌ها را با ThresholdFinder هم دوباره محاسبه کرد (اختیاری).

            # پاک‌سازی بافر تا نوبت بعد
            recent_X.clear(); recent_y.clear()

    # ── ذخیره لایو snapshot ───────────────
    pd.DataFrame(live_rows).to_csv(SNAP_LIVE_CSV,index=False)

    # ── دقت لایو ─────────────────────────
    if y_pred_all:
        LOG.info("🎯 LIVE Acc=%.4f  F1=%.4f (decided=%d)",
                 accuracy_score(y_true_all,y_pred_all),
                 f1_score      (y_true_all,y_pred_all),
                 len(y_pred_all))

    # ── مقایسه دودکش ↔ لایو ───────────────
    ch = pd.read_csv(SNAP_BATCH_CSV); lv = pd.read_csv(SNAP_LIVE_CSV)
    ensure_time(ch,"batch"); ensure_time(lv,"live")
    ch[TIME_COL]=pd.to_datetime(ch[TIME_COL]); lv[TIME_COL]=pd.to_datetime(lv[TIME_COL])
    lv = lv[lv[TIME_COL].isin(ch[TIME_COL])]
    ch = ch[ch[TIME_COL].isin(lv[TIME_COL])]
    ch,lv = ch.reset_index(drop=True), lv.reset_index(drop=True)
    n=min(len(ch),len(lv)); ch,ch=ch.tail(n),lv,lv # noqa

    diff_rows=[]
    for i in range(n):
        bad=[c for c in final_cols
             if not((pd.isna(ch.at[i,c]) and pd.isna(lv.at[i,c])) or
                    np.isclose(ch.at[i,c],lv.at[i,c],atol=1e-6,rtol=1e-3))]
        diff_feat_counter.update(bad)
        if bad or ch.at[i,"label"]!=lv.at[i,"label"]:
            diff_rows.append({"row":i,"time":ch.at[i,TIME_COL],
                              "mismatch":len(bad),"cols":bad[:8],
                              "lbl_ch":int(ch.at[i,'label']),
                              "lbl_lv":int(lv.at[i,'label'])})
    pd.DataFrame(diff_rows).to_csv(DIFF_CSV,index=False)

    # ── 10 ستون پُراختلاف ────────────────
    top10=[c for c,_ in diff_feat_counter.most_common(10)]
    Path(TOP_UNSTABLE_FEATURES).write_text("\n".join(top10),encoding="utf-8")
    LOG.info("📝 top mismatched features saved to %s", TOP_UNSTABLE_FEATURES)

# ═══════════════ Entry ═══════════════
if __name__=="__main__":
    argparse.ArgumentParser(description="Live vs Batch checker").parse_args([])
    main()
