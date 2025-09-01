#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_incremental_vs_train.py

هدف:
- سنجش دقیق و هم‌ترازِ "TRAIN" با منطق PREDICT/INCREMENTAL
- کشف آف-بای-وان، اختلاف آستانه‌ها، و اختلاف ستون/ویژگی
- تولید گزارش‌های baseline/incremental/sweep برای ریشه‌یابی اختلاف Acc

خروجی‌ها (در --model-dir):
- baseline_predict.csv
- incremental_log.csv
- sweep_report.csv
و لاگ مفصل: align_incremental_vs_train.log
"""

from __future__ import annotations
import os, sys, json, argparse, logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

# ======================================================================================
# Logging
# ======================================================================================

def setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("align-inc-train")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# ======================================================================================
# Project imports
# ======================================================================================

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}

try:
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore
except Exception as e:
    print("FATAL: cannot import PREPARE_DATA_FOR_TRAIN:", e, file=sys.stderr)
    sys.exit(1)

# ======================================================================================
# Args
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align incremental vs train with thorough diagnostics.")
    p.add_argument("--data-dir", required=True, help="Folder with CSVs")
    p.add_argument("--symbol", required=True, help="Symbol, e.g., XAUUSD")
    p.add_argument("--model-dir", required=True, help="Folder containing best_model.pkl")
    p.add_argument("--tail-iters", type=int, default=4000, help="How many last feature rows to iterate in incremental view")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", default="align_incremental_vs_train.log")
    # override thresholds if needed (optional)
    p.add_argument("--pos-thr", type=float, default=None)
    p.add_argument("--neg-thr", type=float, default=None)
    return p.parse_args()

# ======================================================================================
# Utils
# ======================================================================================

def load_model(model_dir: str):
    payload = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    pipeline   = payload["pipeline"]
    window     = int(payload.get("window_size", payload.get("window", 1)))
    neg_thr    = float(payload.get("neg_thr", 0.01))
    pos_thr    = float(payload.get("pos_thr", 0.985))
    final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    if not isinstance(final_cols, list):
        final_cols = list(final_cols)
    return payload, pipeline, window, neg_thr, pos_thr, final_cols

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

def build_filepaths(data_dir: str, symbol: str) -> Dict[str, str]:
    return {tf: os.path.join(data_dir, f"{symbol}_{TF_MAP[tf]}.csv") for tf in TF_MAP}

# ======================================================================================
# Core
# ======================================================================================

def build_predict_baseline(pipeline, filepaths: Dict[str, str], final_cols: List[str],
                           window: int, *, predict_drop_last: bool, log: logging.Logger):
    """Prepare PREDICT features on full data and compute y_true from next-bar delta."""
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()
    # prepare features
    Xp, _, _, _, t_idx = prep.ready(
        merged,
        window=window,
        selected_features=final_cols,
        mode="predict",
        with_times=True,
        predict_drop_last=predict_drop_last
    )
    if Xp.empty:
        raise RuntimeError("Empty X for predict baseline; check data/feature config.")
    # enforce column order
    missing = [c for c in final_cols if c not in Xp.columns]
    if missing:
        log.error("Missing %d train columns in predict baseline, e.g.: %s", len(missing), missing[:10])
        raise RuntimeError("Predict features miss train columns.")
    Xp = Xp[final_cols]

    # model scores
    probs = pipeline.predict_proba(Xp)[:, 1].astype(float)

    # true labels aligned by time: y_true(t) = 1{close[t+1] - close[t] > 0}
    t_idx = pd.to_datetime(t_idx).reset_index(drop=True)
    tcol = "30T_time" if "30T_time" in merged.columns else "time"
    mtimes = pd.to_datetime(merged[tcol]).reset_index(drop=True)
    ccol = "30T_close" if "30T_close" in merged.columns else ("close" if "close" in merged.columns else None)
    if ccol is None:
        raise RuntimeError("Cannot find close column in merged.")
    mclose = merged[ccol].reset_index(drop=True).astype(float)

    # map last occurrence of time->index
    last_map = {}
    for i, tt in enumerate(mtimes):
        last_map[tt] = i

    y_true = []
    for tt in t_idx:
        mi = last_map.get(tt, None)
        if mi is None or mi + 1 >= len(mclose):
            y_true.append(np.nan)
        else:
            y_true.append(1 if (mclose.iloc[mi+1] - mclose.iloc[mi]) > 0 else 0)
    y_true = pd.Series(y_true, dtype="float")

    # keep valid rows only
    valid = y_true.notna().values
    df = pd.DataFrame({
        "time": t_idx[valid].values,
        "prob": probs[valid],
        "true": y_true[valid].astype(int).values
    })
    df.reset_index(drop=True, inplace=True)
    return df, Xp.reset_index(drop=True).loc[valid, :]

def evaluate_with_thresholds(df: pd.DataFrame, pos_thr: float, neg_thr: float) -> Tuple[float, float, dict]:
    probs = df["prob"].values
    y = df["true"].values
    preds = np.full_like(y, -1, dtype=int)
    preds[probs <= neg_thr] = 0
    preds[probs >= pos_thr] = 1
    mask = preds != -1
    predicted = mask.sum()
    unpred = len(y) - predicted
    wins = int(((preds == y) & mask).sum())
    losses = predicted - wins
    acc = (wins / predicted * 100.0) if predicted > 0 else 0.0
    cover = (predicted / len(y) * 100.0) if len(y) else 0.0
    stats = dict(P=int(predicted), W=int(wins), L=int(losses), U=int(unpred))
    return acc, cover, stats

def do_shift_sweep(df: pd.DataFrame, Xp: pd.DataFrame,
                   base_pos: float, base_neg: float, window: int,
                   model_dir: str, log: logging.Logger) -> pd.DataFrame:
    """Try label shifts and threshold sweeps to detect off-by-one & over-tight thresholds."""
    shifts = [-2, -1, 0, +1, +2]
    pos_grid = [0.95, 0.985, 0.99, 0.995, 0.999]
    neg_grid = [0.05, 0.02, 0.01, 0.005, 0.001]
    rows = []
    y0 = df["true"].values.copy()
    probs = df["prob"].values.copy()
    for sh in shifts:
        if sh == 0:
            y = y0.copy()
        elif sh > 0:
            y = np.concatenate([y0[sh:], np.full(sh, -9)])
        else:
            k = -sh
            y = np.concatenate([np.full(k, -9), y0[:-k]])
        mask = y >= 0
        df_sh = pd.DataFrame({"prob": probs[mask], "true": y[mask]})
        for pos in pos_grid:
            for neg in neg_grid:
                acc, cov, st = evaluate_with_thresholds(df_sh, pos, neg)
                rows.append({
                    "shift": sh,
                    "pos_thr": pos,
                    "neg_thr": neg,
                    "acc_pct": acc,
                    "cover_pct": cov,
                    "P": st["P"], "W": st["W"], "L": st["L"], "U": st["U"]
                })
    rep = pd.DataFrame(rows).sort_values(["shift", "acc_pct", "cover_pct"], ascending=[True, False, False]).reset_index(drop=True)
    out = os.path.join(model_dir, "sweep_report.csv")
    rep.to_csv(out, index=False)
    log.info("[sweep] saved → %s", out)
    # log top lines per shift
    for sh in shifts:
        best = rep[rep["shift"]==sh].head(3)
        for _, r in best.iterrows():
            log.info("[sweep] shift=%+d pos=%.3f neg=%.3f  Acc=%.2f%% Cover=%.2f%% P=%d W=%d L=%d U=%d",
                     sh, r["pos_thr"], r["neg_thr"], r["acc_pct"], r["cover_pct"], r["P"], r["W"], r["L"], r["U"])
    return rep

def incremental_like_log(df: pd.DataFrame, pos_thr: float, neg_thr: float, tail_iters: int,
                         model_dir: str, log: logging.Logger):
    """Produce step-by-step incremental-style log on top of the baseline predict df."""
    n = len(df)
    end = n - 1
    start = max(0, end - tail_iters + 1)
    total = end - start + 1

    P = W = L = U = 0
    buy_n = sell_n = none_n = 0

    rows = []
    for k, i in enumerate(range(start, end + 1), start=1):
        prob = float(df.loc[i, "prob"])
        true = int(df.loc[i, "true"])
        pred = decide(prob, neg_thr, pos_thr)
        if pred == -1:
            U += 1; none_n += 1; verdict = "UNPRED"
        else:
            P += 1
            if pred == 1: buy_n += 1
            else: sell_n += 1
            if pred == true:
                W += 1; verdict = "WIN"
            else:
                L += 1; verdict = "LOSS"
        acc = (W / P * 100.0) if P > 0 else 0.0
        cov = (P / (P + U) * 100.0) if (P + U) > 0 else 0.0
        log.info("[ %5d/%5d] feat_time=%s  prob=%.3f → pred=%s  true=%d → %s  | P=%d W=%d L=%d U=%d  Acc=%.2f%% Cover=%.2f%%  | BUY=%d SELL=%d NONE=%d",
                 k, total, str(df.loc[i, "time"]), prob,
                 {1:"BUY ", 0:"SELL", -1:"NONE"}[pred], true, verdict,
                 P, W, L, U, acc, cov, buy_n, sell_n, none_n)
        rows.append({
            "iter": k,
            "time": df.loc[i, "time"],
            "prob": prob,
            "pred": int(pred),
            "true": true,
            "verdict": verdict
        })
    out = os.path.join(model_dir, "incremental_log.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Incremental-like log saved → %s", out)

# ======================================================================================
# Main
# ======================================================================================

def main():
    args = parse_args()
    log = setup_logger(args.log_file, args.verbose)
    log.info("=== align_incremental_vs_train starting ===")
    log.info("data-dir=%s symbol=%s model-dir=%s", args.data_dir, args.symbol, args.model_dir)

    # Load model & thresholds
    payload, pipeline, window, neg_thr_m, pos_thr_m, final_cols = load_model(args.model_dir)
    pos_thr = float(args.pos_thr) if args.pos_thr is not None else pos_thr_m
    neg_thr = float(args.neg_thr) if args.neg_thr is not None else neg_thr_m
    log.info("Model window=%d  thr(neg=%.3f,pos=%.3f)  final_cols=%d", window, neg_thr, pos_thr, len(final_cols))

    filepaths = build_filepaths(args.data_dir, args.symbol)

    # -------- Baseline: predict_drop_last=True (واقع‌گرایانه‌ترین حالت)
    df_pred_drop, Xp_drop = build_predict_baseline(
        pipeline, filepaths, final_cols, window, predict_drop_last=True, log=log
    )
    out_base = os.path.join(args.model_dir, "baseline_predict.csv")
    df_pred_drop.to_csv(out_base, index=False)
    acc, cov, st = evaluate_with_thresholds(df_pred_drop, pos_thr, neg_thr)
    log.info("[baseline/predict_drop_last=True] N=%d  predicted=%d unpred=%d  Acc=%.4f%% Cover=%.4f%%",
             len(df_pred_drop), st["P"], st["U"], acc, cov)

    # -------- Optional: predict_drop_last=False (برای مقایسه)
    df_pred_keep, _ = build_predict_baseline(
        pipeline, filepaths, final_cols, window, predict_drop_last=False, log=log
    )
    acc2, cov2, st2 = evaluate_with_thresholds(df_pred_keep, pos_thr, neg_thr)
    log.info("[baseline/predict_drop_last=False] N=%d  predicted=%d unpred=%d  Acc=%.4f%% Cover=%.4f%%",
             len(df_pred_keep), st2["P"], st2["U"], acc2, cov2)

    # -------- Sweep برای کشف آف-بای-وان و آستانه‌ی بهینه (با گزارش CSV)
    _ = do_shift_sweep(df_pred_drop, Xp_drop, pos_thr, neg_thr, window, args.model_dir, log)

    # -------- Incremental-like log روی baseline (برای مقایسه سرراست با لاگ شما)
    incremental_like_log(df_pred_drop, pos_thr, neg_thr, args.tail_iters, args.model_dir, log)

    log.info("=== align_incremental_vs_train finished ===")

if __name__ == "__main__":
    main()



# python3 -u align_incremental_vs_train.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 4000 \
#   --verbose
