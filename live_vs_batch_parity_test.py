# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import joblib, json, logging
from pathlib import Path
import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

def load_payload_best_model(path="best_model.pkl"):
    p = joblib.load(path)
    if isinstance(p, dict):
        model = p.get("pipeline") or p.get("model") or p.get("estimator") or p.get("clf") or p.get("best_estimator")
        if model is None:
            raise ValueError("No model in pkl")
        return dict(
            pipeline=model,
            window_size=int(p.get("window_size",1)),
            train_window_cols=list(p.get("train_window_cols") or p.get("feats") or []),
            neg_thr=float(p.get("neg_thr",0.005)),
            pos_thr=float(p.get("pos_thr",0.995)),
        )
    return {"pipeline": p, "window_size":1, "train_window_cols":[], "neg_thr":0.005, "pos_thr":0.995}

def main():
    base = Path(".").resolve()
    symbol = "XAUUSD"
    # map to your real raw CSV names
    filepaths = {
        "30T": base / f"{symbol}_M30.csv",
        "15T": base / f"{symbol}_M15.csv",
        "5T" : base / f"{symbol}_M5.csv",
        "1H" : base / f"{symbol}_H1.csv",
    }
    # keep only existing
    filepaths = {k: str(v) for k,v in filepaths.items() if Path(v).exists()}
    assert "30T" in filepaths, "need 30T"

    payload = load_payload_best_model("best_model.pkl")
    model   = payload["pipeline"]
    window  = int(payload["window_size"])
    cols    = payload["train_window_cols"]
    neg_thr = payload["neg_thr"]; pos_thr = payload["pos_thr"]

    # ===== CHIMNEY (batch) =====
    prepB = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                   verbose=False, fast_mode=True, strict_disk_feed=False)
    mergedB = prepB.load_data()
    last_n = 2000
    cutB = mergedB.tail(last_n+1)  # +1 برای target
    Xb, yb, _, _ = prepB.ready(cutB, window=window, selected_features=cols, mode="predict",
                               with_times=False, predict_drop_last=True)
    if Xb.empty:
        print("Batch X empty"); return
    Xb = Xb.reindex(columns=cols, fill_value=0.0)
    probB = model.predict_proba(Xb)[:,1]
    decB = np.where(probB >= pos_thr, 1, np.where(probB <= neg_thr, 0, -1))

    # ===== LIVE-LIKE (in-memory) =====
    prepL = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                   verbose=False, fast_mode=True, strict_disk_feed=False)
    mergedL = prepL.load_data()
    # همان برش زمانی را قدم به قدم simulate می‌کنیم:
    tcol = "30T_time" if "30T_time" in mergedL.columns else "time"
    tail_index = mergedL.tail(last_n+1).index
    probs_live = []
    for k in range(tail_index.min(), tail_index.max()+1):
        sub = mergedL.loc[:k].tail(500)  # مثل ژنراتور؛ صرفاً کافی‌ست چندصد سطر آخر را نگه داریم
        Xk, _, _, _ = prepL.ready(sub, window=window, selected_features=cols, mode="predict",
                                  with_times=False, predict_drop_last=True)
        if Xk.empty:
            probs_live.append(np.nan); continue
        Xk = Xk.reindex(columns=cols, fill_value=0.0).tail(1)
        pk = float(model.predict_proba(Xk)[:,1][0])
        probs_live.append(pk)
    # هم طول کن
    probs_live = np.asarray([p for p in probs_live if not np.isnan(p)], dtype=float)[-len(probB):]
    decL = np.where(probs_live >= pos_thr, 1, np.where(probs_live <= neg_thr, 0, -1))

    # ===== Compare =====
    m = min(len(probB), len(probs_live))
    prob_diff = np.abs(probB[-m:] - probs_live[-m:])
    dec_equal = (decB[-m:] == decL[-m:]).mean()
    print(f"Parity — probs max|diff|={prob_diff.max():.6g} | mean|diff|={prob_diff.mean():.6g} | decisions match={dec_equal:.3f}")

    out = pd.DataFrame({
        "prob_batch": probB[-m:],
        "prob_live": probs_live[-m:],
        "absdiff": prob_diff,
        "dec_batch": decB[-m:],
        "dec_live": decL[-m:]
    })
    out.to_csv("parity_check.csv", index=False)
    print("Saved: parity_check.csv")

if __name__ == "__main__":
    main()
