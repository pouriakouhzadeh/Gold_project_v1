\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_replay_deploy.py
---------------------
Deploy-ready live replay simulator.
"""
from __future__ import annotations

import os, sys, argparse, tempfile, shutil, logging, json
from typing import Dict
import numpy as np
import pandas as pd
import joblib

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

TF_MAP = {"30T":"M30","15T":"M15","5T":"M5","1H":"H1"}

def setup_logger(path: str, verbose: bool)->logging.Logger:
    lg = logging.getLogger("live-replay")
    lg.setLevel(logging.DEBUG if verbose else logging.INFO)
    lg.propagate=False
    for h in list(lg.handlers): lg.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.DEBUG if verbose else logging.INFO); sh.setFormatter(fmt)
    lg.addHandler(sh)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg

def expect_time_col(df: pd.DataFrame)->pd.DataFrame:
    if "time" not in df.columns:
        for c in ("Time","timestamp","datetime","Date"):
            if c in df.columns: df = df.rename(columns={c:"time"}); break
    if "time" not in df.columns: raise KeyError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def load_raw_csvs(data_dir: str, symbol: str)->Dict[str,pd.DataFrame]:
    paths = {tf: os.path.join(data_dir, f"{symbol}_{TF_MAP[tf]}.csv") for tf in TF_MAP}
    raw = {}
    for tf, p in paths.items():
        if not os.path.isfile(p):
            if tf == "30T": raise FileNotFoundError(f"Main TF (30T) missing: {p}")
            else: continue
        raw[tf] = expect_time_col(pd.read_csv(p))
    if "30T" not in raw: raise FileNotFoundError("30T dataframe missing; abort.")
    return raw

def write_tail_csvs(raw_df: Dict[str,pd.DataFrame], symbol: str, cutoff: pd.Timestamp,
                    out_dir: str, hist_rows: Dict[str,int])->Dict[str,str]:
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for tf, df in raw_df.items():
        sub = df.loc[df["time"] <= cutoff]
        if sub.empty: continue
        need = int(hist_rows.get(tf, 200))
        if need > 0 and len(sub) > need: sub = sub.tail(need)
        mapped = TF_MAP.get(tf, tf)
        out_path = os.path.join(out_dir, f"{symbol}_{mapped}.csv")
        cols = ["time"] + [c for c in sub.columns if c != "time"]
        sub[cols].to_csv(out_path, index=False)
        out[tf] = out_path
    return out

def decide(prob: float, neg_thr: float, pos_thr: float)->int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

def build_baseline(args, log):
    filepaths = {tf: f"{args.data_dir}/{args.symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=True, strict_disk_feed=True)
    merged = prep.load_data()
    payload = joblib.load(os.path.join(args.model_dir,"best_model.pkl"))
    pipe    = payload["pipeline"]; final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    window  = int(payload.get("window_size", 1))
    X, _, _, _, t_idx = prep.ready(merged, window=window, selected_features=final_cols,
                                   mode="predict", with_times=True, predict_drop_last=True)
    if not X.empty and all(c in X.columns for c in final_cols):
        X = X[final_cols]
    probs = pipe.predict_proba(X)[:,1] if len(X) else np.array([], dtype=float)
    neg_thr = float(payload.get("neg_thr",0.5)); pos_thr = float(payload.get("pos_thr",0.5))
    preds = np.full(len(probs), -1, dtype=int); preds[probs<=neg_thr]=0; preds[probs>=pos_thr]=1

    t_idx = pd.to_datetime(t_idx).reset_index(drop=True) if t_idx is not None else pd.Series([], dtype="datetime64[ns]")
    tcol = "30T_time" if "30T_time" in merged.columns else "time"
    mtimes = pd.to_datetime(merged[tcol]).reset_index(drop=True)
    mclose = merged["30T_close" if "30T_close" in merged.columns else "close"].reset_index(drop=True)

    time_to_idx = {tt:i for i,tt in enumerate(mtimes)}
    y_true_list = []
    for tt in t_idx:
        mi = time_to_idx.get(tt, None)
        if mi is None or (mi+1) >= len(mclose): y_true_list.append(np.nan)
        else: y_true_list.append(1 if float(mclose.iloc[mi+1]) - float(mclose.iloc[mi]) > 0 else 0)

    valid = ~pd.isna(y_true_list)
    base_df = pd.DataFrame({
        "time": t_idx[valid].values,
        "prob": probs[valid],
        "pred": preds[valid],
        "true": pd.Series(y_true_list, dtype="float")[valid].astype(int).values
    }).reset_index(drop=True)
    log.info("[baseline] built (N=%d)", len(base_df))
    return base_df, X.reset_index(drop=True).loc[valid,:], payload, window, final_cols

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tail-iters", type=int, default=50)
    p.add_argument("--log-file", default="live_replay.log")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hist-30t", type=int, default=6000)
    p.add_argument("--hist-1h",  type=int, default=3200)
    p.add_argument("--hist-15t", type=int, default=12000)
    p.add_argument("--hist-5t",  type=int, default=36000)
    args = p.parse_args()

    log = setup_logger(os.path.join(args.model_dir, args.log_file), args.verbose)
    log.info("=== live_replay starting ===")
    raw_df = load_raw_csvs(args.data_dir, args.symbol)
    main_df = raw_df["30T"].copy()
    base_df, base_X, payload, window, final_cols = build_baseline(args, log)
    base_times = pd.to_datetime(base_df["time"])
    base_time_to_idx = pd.Series(np.arange(len(base_times), dtype=int), index=base_times)

    neg_thr = float(payload.get("neg_thr",0.5)); pos_thr = float(payload.get("pos_thr",0.5))
    hist_rows = {"30T": int(args.hist_30t), "1H": int(args.hist_1h),
                 "15T": int(args.hist_15t), "5T": int(args.hist_5t)}
    log.info("[live] tails: %s", hist_rows)

    N = len(main_df); end_idx = N - 2; start_idx = max(0, end_idx - int(args.tail_iters) + 1)
    total = end_idx - start_idx + 1
    log.info("[live] iters: start_idx=%d end_idx=%d total=%d", start_idx, end_idx, total)

    rows = []; tmp_root = tempfile.mkdtemp(prefix="live_replay_")
    try:
        for k in range(total):
            i = start_idx + k
            cutoff = pd.to_datetime(main_df.loc[i, "time"])
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            fps = write_tail_csvs(raw_df, args.symbol, cutoff, iter_dir, hist_rows)
            if "30T" not in fps:
                shutil.rmtree(iter_dir, ignore_errors=True); continue

            prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                          verbose=False, fast_mode=True, strict_disk_feed=True)
            merged = prep.load_data()
            X, _, _, _, t_idx = prep.ready(merged, window=window, selected_features=final_cols,
                                           mode="predict", with_times=True, predict_drop_last=True)
            if X.empty:
                shutil.rmtree(iter_dir, ignore_errors=True); continue
            want = list(final_cols)
            if not all(c in X.columns for c in want):
                # try fill by train medians if present
                med = None
                try:
                    path = payload.get("train_distribution") or "train_distribution.json"
                    if not os.path.isabs(path): path = os.path.join(args.model_dir, path)
                    with open(path,"r",encoding="utf-8") as f: j=json.load(f)
                    src = j.get("columns", j); med = {c:(d.get("median", d.get("q50"))) for c,d in src.items() if isinstance(d, dict)}
                except Exception:
                    med = None
                X = X.reindex(columns=want, fill_value=np.nan)
                if med is not None:
                    for c in want:
                        if X[c].isna().any():
                            mv = med.get(c, None)
                            if mv is not None: X[c] = X[c].fillna(float(mv))
                X = X.fillna(0.0)
            else:
                X = X[want]

            t_feat = pd.to_datetime(t_idx.iloc[-1]) if (t_idx is not None and len(t_idx)>0) else None
            prob_live = float(payload["pipeline"].predict_proba(X.tail(1))[:,1])
            pred_live = decide(prob_live, neg_thr, pos_thr)

            c0 = float(main_df.loc[i,"close"]); c1 = float(main_df.loc[i+1,"close"])
            y_true = 1 if (c1 - c0) > 0 else 0

            base_prob = base_pred = base_true = np.nan
            l2 = maxabs = np.nan
            if t_feat is not None:
                bj = base_time_to_idx.get(t_feat, None)
                if bj is not None and 0 <= bj < len(base_df):
                    base_prob = float(base_df.loc[bj,"prob"]); base_pred = int(base_df.loc[bj,"pred"]); base_true = int(base_df.loc[bj,"true"])
                    row_live = X.tail(1).iloc[0].values.astype(float)
                    row_base = base_X.iloc[bj][want].values.astype(float)
                    diff = row_live - row_base
                    l2 = float(np.linalg.norm(diff)); maxabs = float(np.max(np.abs(diff)))

            rows.append({
                "iter": k+1, "cutoff": cutoff, "feat_time": t_feat,
                "prob_live": prob_live, "pred_live": int(pred_live), "true": int(y_true),
                "base_prob": float(base_prob) if not np.isnan(base_prob) else np.nan,
                "base_pred": int(base_pred) if not np.isnan(base_prob) else -9,
                "base_true": int(base_true) if not np.isnan(base_prob) else -9,
                "l2_diff": float(l2) if not np.isnan(l2) else np.nan,
                "maxabs_diff": float(maxabs) if not np.isnan(maxabs) else np.nan,
            })

            log.info("[live %5d/%5d] cutoff=%s feat_time=%s prob=%.4f -> pred=%s true=%d | L2=%.3g MaxAbs=%.3g",
                     k+1, total, str(cutoff), str(t_feat), prob_live,
                     {1:"BUY ",0:"SELL",-1:"NONE"}[pred_live], y_true,
                     (l2 if not np.isnan(l2) else -1.0), (maxabs if not np.isnan(maxabs) else -1.0))

            shutil.rmtree(iter_dir, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_csv = os.path.join(args.model_dir, "deploy_live.csv")
    live_df = pd.DataFrame(rows); live_df.to_csv(out_csv, index=False)
    log.info("[live] saved -> %s (N=%d)", out_csv, len(live_df))

    common = live_df.dropna(subset=["base_prob"]).copy()
    pred_agree = (common["base_pred"].values == common["pred_live"].values).mean() * 100.0 if len(common)>0 else 0.0
    feat_ok    = ((common["l2_diff"].fillna(0) <= 1e-6) & (common["maxabs_diff"].fillna(0) <= 1e-6)).mean() * 100.0 if len(common)>0 else 0.0
    log.info("[parity] pred_agree=%.3f%% feat_ok=%.3f%% on %d rows", pred_agree, feat_ok, len(common))
    log.info("=== live_replay finished ===")

if __name__ == "__main__":
    main()



# python3 live_replay_deploy.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 600 \
#   --hist-30t 6000 \
#   --hist-1h 3200 \
#   --hist-15t 12000 \
#   --hist-5t 36000 \
#   --log-file live_replay.log \
#   --verbose
