#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_replay_validator.py

شبیه‌ساز لایوِ ۴۰۰۰ قدمی با Auto-tail مبتنی بر «تطابق عددی فیچرها».
هدف: اگر دقت این شبیه‌ساز با گزارش TRAIN هم‌راستا بود → دپلوی.

خروجی‌ها (در --model-dir):
- smart_replay_predict.csv   ← time, prob, pred, true, verdict
- smart_live_tail.csv        ← جزئیات هر قدم + اختلاف با baseline
- smart_incremental.csv      ← خلاصهٔ قدم‌به‌قدم
- smart_live_replay.log      ← لاگ کامل

نکته مهم: مسیر محاسبهٔ baseline و LIVE هر دو:
    fast_mode=True, strict_disk_feed=True, predict_drop_last=True
"""

from __future__ import annotations
import os, sys, math, json, shutil, tempfile, argparse, logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import joblib

# --- پروژه شما ---
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}
ALL_TFS = ("30T", "1H", "15T", "5T")

# ---------------- Logging ----------------
def setup_logger(path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("smart-live-replay")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# ---------------- IO helpers ----------------
def expect_time_col(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        for c in ("Time","timestamp","datetime","Date"):
            if c in df.columns:
                df = df.rename(columns={c:"time"}); break
    if "time" not in df.columns:
        raise KeyError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def load_raw_csvs(args) -> Dict[str, pd.DataFrame]:
    paths = {tf: os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP[tf]}.csv") for tf in TF_MAP}
    out = {}
    for tf, p in paths.items():
        if not os.path.isfile(p):
            if tf == "30T":
                raise FileNotFoundError(f"Main TF (30T) missing: {p}")
            else:
                continue
        out[tf] = expect_time_col(pd.read_csv(p))
    if "30T" not in out:
        raise FileNotFoundError("30T dataframe missing; abort.")
    return out

def write_tail_csvs(raw_df: Dict[str, pd.DataFrame], symbol: str, cutoff: pd.Timestamp,
                    out_dir: str, hist_rows: Dict[str,int]) -> Dict[str,str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for tf, df in raw_df.items():
        sub = df.loc[df["time"] <= cutoff]
        if sub.empty:
            continue
        need = int(hist_rows.get(tf, 200))
        if need > 0 and len(sub) > need:
            sub = sub.tail(need)
        mapped = TF_MAP.get(tf, tf)
        cols = ["time"] + [c for c in sub.columns if c != "time"]
        out_path = os.path.join(out_dir, f"{symbol}_{mapped}.csv")
        sub[cols].to_csv(out_path, index=False)
        paths[tf] = out_path
    return paths

# ---------------- Model ----------------
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

def load_train_medians(model_dir: str, payload: dict) -> Optional[dict]:
    p = payload.get("train_distribution") or "train_distribution.json"
    if not os.path.isabs(p): p = os.path.join(model_dir, p)
    if not os.path.isfile(p): return None
    try:
        j = json.load(open(p, "r", encoding="utf-8"))
        src = j.get("columns", j)
        med = {}
        for c, st in src.items():
            if isinstance(st, dict):
                if "median" in st: med[c] = st["median"]
                elif "q50" in st:  med[c] = st["q50"]
        return med or None
    except Exception:
        return None

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

# ---------------- Baseline on FULL data (strict path = LIVE path) ----------------
def build_baseline_predict_strict(args, log, pipeline, window, final_cols):
    filepaths = {tf: os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP[tf]}.csv") for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=True, strict_disk_feed=True)
    merged = prep.load_data()
    X, _, _, _, t_idx = prep.ready(
        merged,
        window=window,
        selected_features=final_cols,
        mode="predict",
        with_times=True,
        predict_drop_last=True,
    )
    if X.empty:
        raise RuntimeError("Empty features for baseline/predict_strict.")
    if not all(c in X.columns for c in final_cols):
        missing = [c for c in final_cols if c not in X.columns]
        raise RuntimeError(f"Baseline X missing {len(missing)} train cols, e.g. {missing[:10]}")
    X = X[final_cols].reset_index(drop=True)
    t_idx = pd.to_datetime(t_idx).reset_index(drop=True)
    # map time → last index
    t2i = pd.Series(np.arange(len(t_idx), dtype=int), index=t_idx)
    return {"X": X, "times": t_idx, "time_to_idx": t2i, "merged": merged}

# ---------------- Value-matching Auto-tail (ارتقاء‌یافته) ----------------
def autotail_value_match(args, log, raw_df, symbol, baseline_X, baseline_times,
                         window, final_cols, *, samples=3,
                         base_lo=64, base_hi=6000,
                         mult_1h=0.8, mult_15t=3.0, mult_5t=9.0,
                         tol_l2=1e-6, tol_maxabs=1e-6) -> Dict[str,int]:
    """
    به‌جای «فقط وجود ستون»، این تابع کمینه‌ی tail را می‌یابد که فیچرهای آخرین ردیف
    در چند cutoff نمونه با baseline «هم‌ارز عددی» باشند.
    """
    main = raw_df["30T"]
    N = len(main)
    step = max(1, N // (samples + 1))
    idxs = [N - 1 - i * step for i in range(samples)]
    idxs = [i for i in idxs if 0 <= i < N - 1]
    cutoffs = [pd.to_datetime(main.loc[i, "time"]) for i in sorted(set(idxs))]

    def tails_from_base(b30: int) -> Dict[str,int]:
        return {
            "30T": int(b30),
            "1H":  int(max(64, math.ceil(b30 * mult_1h))),
            "15T": int(max(128, math.ceil(b30 * mult_15t))),
            "5T":  int(max(256, math.ceil(b30 * mult_5t))),
        }

    best_needed_30 = 0
    tmp_root = tempfile.mkdtemp(prefix="autotail_valmatch_")
    try:
        for ci, cutoff in enumerate(cutoffs, 1):
            lo, hi = base_lo, min(base_hi, N - 2)
            ok = None
            while lo <= hi:
                mid = (lo + hi) // 2
                tails = tails_from_base(mid)
                it_dir = os.path.join(tmp_root, f"c{ci}_b{mid}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, it_dir, tails)
                if "30T" not in fps:
                    shutil.rmtree(it_dir, ignore_errors=True)
                    lo = mid + 1
                    continue
                prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                              verbose=False, fast_mode=True, strict_disk_feed=True)
                merged = prep.load_data()
                X, _, _, _, t_idx = prep.ready(
                    merged, window=window, selected_features=final_cols,
                    mode="predict", with_times=True, predict_drop_last=True
                )
                shutil.rmtree(it_dir, ignore_errors=True)
                if X.empty or not all(c in X.columns for c in final_cols):
                    lo = mid + 1
                    continue
                X = X[final_cols]
                t_feat = pd.to_datetime(t_idx.iloc[-1]) if (t_idx is not None and len(t_idx)) else None
                if t_feat is None or t_feat not in baseline_times.values:
                    lo = mid + 1
                    continue
                # مقایسه با baseline در همان time
                bj = int(np.where(baseline_times.values == t_feat)[0][-1])
                v_live = X.tail(1).iloc[0].values.astype(float)
                v_base = baseline_X.iloc[bj].values.astype(float)
                diff = v_live - v_base
                l2 = float(np.linalg.norm(diff))
                mx = float(np.max(np.abs(diff)))
                if (l2 <= tol_l2) and (mx <= tol_maxabs):
                    ok = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            if ok is None:
                log.warning("[auto-tail] cutoff=%s not satisfied up to %d; will use max.", str(cutoff), base_hi)
                best_needed_30 = max(best_needed_30, base_hi)
            else:
                log.info("[auto-tail] cutoff=%s → min30T=%d (value-matched)", str(cutoff), ok)
                best_needed_30 = max(best_needed_30, ok)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    tails = tails_from_base(best_needed_30)
    log.info("[auto-tail] chosen tails: 30T=%d 1H=%d 15T=%d 5T=%d (value-matching)", tails["30T"], tails["1H"], tails["15T"], tails["5T"])
    return tails

# ---------------- Live replay over last K steps ----------------
def run_live_replay(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols,
                    baseline, *, K: int, hist_rows: Dict[str,int], allow_missing_cols=True):
    raw_df = load_raw_csvs(args)
    main_df = raw_df["30T"]
    N = len(main_df)
    end_idx = N - 2
    start_idx = max(0, end_idx - K + 1)
    total = end_idx - start_idx + 1
    log.info("[live] iteration range: start_idx=%d end_idx=%d total=%d", start_idx, end_idx, total)

    base_df_times = baseline["times"]
    base_X = baseline["X"]
    # true از سری قیمت اصلی
    mclose = main_df["close"].astype(float).reset_index(drop=True)

    tmp_root = tempfile.mkdtemp(prefix="smart_live_replay_")
    train_medians = load_train_medians(args.model_dir, payload)
    rows_pred, rows_tail, rows_incr = [], [], []
    P = W = L = U = 0
    buy_n = sell_n = none_n = 0

    try:
        for k in range(total):
            i = start_idx + k
            cutoff = pd.to_datetime(main_df.loc[i, "time"])
            it_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            fps = write_tail_csvs(raw_df, args.symbol, cutoff, it_dir, hist_rows)
            if "30T" not in fps:
                shutil.rmtree(it_dir, ignore_errors=True)
                continue
            prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                          verbose=False, fast_mode=True, strict_disk_feed=True)
            merged = prep.load_data()
            X, _, _, _, t_idx = prep.ready(
                merged, window=window, selected_features=final_cols,
                mode="predict", with_times=True, predict_drop_last=True
            )
            if X.empty:
                shutil.rmtree(it_dir, ignore_errors=True)
                log.debug("[live/skip] empty X @ cutoff=%s", str(cutoff))
                continue

            want = list(final_cols)
            missing = [c for c in want if c not in X.columns]
            filled_median = False
            if missing:
                if not allow_missing_cols:
                    shutil.rmtree(it_dir, ignore_errors=True)
                    log.error("[live/fatal] missing %d train cols @%s e.g. %s", len(missing), str(cutoff), missing[:10])
                    break
                if train_medians is None:
                    shutil.rmtree(it_dir, ignore_errors=True)
                    log.error("[live/fatal] --allow-missing-cols but train medians not found")
                    break
                X = X.reindex(columns=want, fill_value=np.nan)
                for c in missing:
                    mv = train_medians.get(c, None)
                    if mv is None or not np.isfinite(mv):
                        shutil.rmtree(it_dir, ignore_errors=True)
                        log.error("[live/fatal] no median for column %s", c)
                        break
                    X[c] = float(mv)
                X = X.fillna(0.0)
                filled_median = True
            else:
                X = X[want]

            t_feat = pd.to_datetime(t_idx.iloc[-1]) if (t_idx is not None and len(t_idx)) else None
            prob = float(pipeline.predict_proba(X.tail(1))[:, 1])
            pred = decide(prob, neg_thr, pos_thr)

            # true = حرکت کندل بعدی
            c0 = float(mclose.iloc[i]); c1 = float(mclose.iloc[i+1])
            y_true = 1 if (c1 - c0) > 0 else 0

            verdict = "UNPRED"
            if pred == -1:
                U += 1; none_n += 1
            else:
                P += 1
                if pred == 1: buy_n += 1
                else:         sell_n += 1
                if pred == y_true:
                    W += 1; verdict = "WIN"
                else:
                    L += 1; verdict = "LOSS"

            acc = (W / P * 100.0) if P > 0 else 0.0
            cov = (P / (P + U) * 100.0) if (P + U) > 0 else 0.0

            # مقایسه با baseline (اختیاری، برای دیباگ)
            base_prob = base_pred = np.nan
            l2 = mx = np.nan
            if t_feat is not None:
                idxs = np.where(base_df_times.values == t_feat)[0]
                if len(idxs):
                    bj = int(idxs[-1])
                    base_prob = float(np.clip(pipeline.predict_proba(base_X.iloc[[bj]])[:,1], 0, 1))
                    base_pred = decide(base_prob, neg_thr, pos_thr)
                    try:
                        v_live = X.tail(1).iloc[0].values.astype(float)
                        v_base = base_X.iloc[bj][want].values.astype(float)
                        diff = v_live - v_base
                        l2 = float(np.linalg.norm(diff))
                        mx = float(np.max(np.abs(diff)))
                    except Exception:
                        pass

            log.info("[LIVE %5d/%5d] cutoff=%s feat_time=%s prob=%.3f → pred=%s true=%d → %s | base_prob=%s base_pred=%s | L2=%.3g MaxAbs=%.3g miss_cols=%d filled_median=%s | Acc=%.2f%% Cov=%.2f%%",
                     k+1, total, str(cutoff), str(t_feat), prob,
                     {1:"BUY ",0:"SELL",-1:"NONE"}[pred], y_true, verdict,
                     (f"{base_prob:.3f}" if not np.isnan(base_prob) else "NA"),
                     ({1:"BUY ",0:"SELL",-1:"NONE"}[int(base_pred)] if not np.isnan(base_pred) else "NA"),
                     (l2 if not np.isnan(l2) else 0.0), (mx if not np.isnan(mx) else 0.0),
                     len(missing), filled_median, acc, cov)

            rows_pred.append({
                "iter": k+1, "time": t_feat, "prob": prob,
                "pred": int(pred), "true": int(y_true), "verdict": verdict
            })
            rows_tail.append({
                "iter": k+1, "cutoff": cutoff, "feat_time": t_feat,
                "prob": prob, "pred": int(pred), "true": int(y_true),
                "base_prob": float(base_prob) if not np.isnan(base_prob) else np.nan,
                "base_pred": int(base_pred) if not np.isnan(base_prob) else -9,
                "l2_diff": float(l2) if not np.isnan(l2) else np.nan,
                "maxabs_diff": float(mx) if not np.isnan(mx) else np.nan,
                "miss_cols": len(missing), "filled_median": filled_median
            })
            rows_incr.append({
                "k": k+1, "acc_pct": acc, "cover_pct": cov,
                "P": P, "W": W, "L": L, "U": U, "BUY": buy_n, "SELL": sell_n, "NONE": none_n
            })

            shutil.rmtree(it_dir, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ذخیره خروجی‌ها
    out_pred = os.path.join(args.model_dir, "smart_replay_predict.csv")
    out_tail = os.path.join(args.model_dir, "smart_live_tail.csv")
    out_incr = os.path.join(args.model_dir, "smart_incremental.csv")
    pd.DataFrame(rows_pred).to_csv(out_pred, index=False)
    pd.DataFrame(rows_tail).to_csv(out_tail, index=False)
    pd.DataFrame(rows_incr).to_csv(out_incr, index=False)
    log.info("[live] saved → %s | %s | %s", out_pred, out_tail, out_incr)

    final_acc = (W / P * 100.0) if P > 0 else 0.0
    final_cov = (P / (P + U) * 100.0) if (P + U) > 0 else 0.0
    log.info("[summary] steps=%d  predicted=%d unpred=%d  Acc=%.4f%%  Cover=%.4f%%  BUY=%d SELL=%d NONE=%d",
             len(rows_pred), P, U, final_acc, final_cov, buy_n, sell_n, none_n)
    return dict(acc=final_acc, cover=final_cov, P=P, U=U, W=W, L=L)

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart LIVE replay (4000 steps) with value-matching auto-tail.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tail-iters", type=int, default=4000)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", default="smart_live_replay.log")

    # tolerances برای value-matching
    p.add_argument("--tol-l2", type=float, default=1e-6)
    p.add_argument("--tol-maxabs", type=float, default=1e-6)

    # auto-tail نمونه‌ها/سقف‌ها
    p.add_argument("--auto-tail-samples", type=int, default=3)
    p.add_argument("--auto-tail-max30t", type=int, default=8000)
    p.add_argument("--mult-1h", type=float, default=0.8)
    p.add_argument("--mult-15t", type=float, default=3.0)
    p.add_argument("--mult-5t", type=float, default=9.0)

    return p.parse_args()

# ---------------- MAIN ----------------
def main():
    args = parse_args()
    log = setup_logger(os.path.join(args.model_dir, args.log_file), args.verbose)
    log.info("=== smart_live_replay starting ===")
    log.info("data-dir=%s  symbol=%s  model-dir=%s  tail-iters=%d", args.data_dir, args.symbol, args.model_dir, args.tail_iters)

    # 1) Load model & strict baseline (full history)
    payload, pipeline, window, neg_thr, pos_thr, final_cols = load_model(args.model_dir)
    log.info("Model: window=%d thr(neg=%.3f,pos=%.3f) final_cols=%d", window, neg_thr, pos_thr, len(final_cols))
    baseline = build_baseline_predict_strict(args, log, pipeline, window, final_cols)

    # 2) Value-matching auto-tail
    raw_df = load_raw_csvs(args)
    hist_rows = autotail_value_match(
        args, log, raw_df, args.symbol,
        baseline["X"], baseline["times"],
        window, final_cols,
        samples=max(1, int(args.auto_tail_samples)),
        base_hi=int(args.auto_tail_max30t),
        mult_1h=float(args.mult_1h), mult_15t=float(args.mult_15t), mult_5t=float(args.mult_5t),
        tol_l2=float(args.tol_l2), tol_maxabs=float(args.tol_maxabs)
    )

    # 3) LIVE replay over last K
    stats = run_live_replay(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols,
                            baseline, K=int(args.tail_iters), hist_rows=hist_rows, allow_missing_cols=True)

    log.info("=== smart_live_replay finished ===  Acc=%.4f%% Cover=%.4f%%", stats["acc"], stats["cover"])
    sys.exit(0)

if __name__ == "__main__":
    main()



# python3 -u live_replay_validator.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 4000 \
#   --verbose
