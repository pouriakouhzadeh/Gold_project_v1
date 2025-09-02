\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_ready_guard_fixed.py
---------------------------
Fixed version of deploy_ready_guard with two key fixes:
1) Robust auto-tail sampling even when --auto-tail-samples=1 (no empty cutoffs).
2) Safer minimum 30T history based on window size to satisfy long-memory indicators.

Outputs (in --model-dir):
- deploy_baseline.csv
- deploy_live.csv
- deploy_guard.log
Exit code: 0 (READY) or 2 (NOT READY)
"""
from __future__ import annotations

import os, sys, math, json, shutil, tempfile, argparse, logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

# ==== Project imports ====
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}
ALL_TFS = ("30T", "1H", "15T", "5T")

# ==== Logging ====
def setup_logger(path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("deploy-guard")
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

# ==== CLI ====
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy Guard (fixed): parity LIVE vs baseline with numeric convergence.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tail-iters", type=int, default=4000)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", default="deploy_guard.log")

    # live/baseline common settings
    p.add_argument("--predict-drop-last", action="store_true", default=True)

    # parity tolerances
    p.add_argument("--tol-l2", type=float, default=1e-8, help="max L2 between LIVE and baseline feature row")
    p.add_argument("--tol-maxabs", type=float, default=1e-8, help="max |delta| per-feature")
    p.add_argument("--pred-parity-min", type=float, default=99.9, help="% rows where pred_live == base_pred")
    p.add_argument("--feat-parity-min", type=float, default=99.9, help="% rows where L2/MaxAbs under tol")

    # train-vs-baseline (optional)
    p.add_argument("--check-train-acc", action="store_true", help="also compare TRAIN replay acc with baseline acc")
    p.add_argument("--acc-diff-max", type=float, default=0.5, help="max abs difference (percentage points)")

    # auto-tail (with numeric convergence)
    p.add_argument("--auto-tail", action="store_true", default=True)
    p.add_argument("--auto-tail-samples", type=int, default=3)
    p.add_argument("--auto-tail-max30t", type=int, default=12000)
    p.add_argument("--mult-1h", type=float, default=0.5)
    p.add_argument("--mult-15t", type=float, default=2.0)
    p.add_argument("--mult-5t", type=float, default=6.0)

    # manual tails (used when --auto-tail is false)
    p.add_argument("--hist-30t", type=int, default=6000)
    p.add_argument("--hist-1h",  type=int, default=3200)
    p.add_argument("--hist-15t", type=int, default=12000)
    p.add_argument("--hist-5t",  type=int, default=36000)

    # missing cols policy
    p.add_argument("--allow-missing-cols", action="store_true", default=False)
    p.add_argument("--progress-every", type=int, default=50,
                help="log every N iterations inside LIVE loop")

    return p.parse_args()

# ==== Helpers ====
def expect_time_col(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        for cand in ("Time", "timestamp", "datetime", "Date"):
            if cand in df.columns:
                df = df.rename(columns={cand: "time"})
                break
    if "time" not in df.columns:
        raise KeyError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def load_raw_csvs(args) -> Dict[str, pd.DataFrame]:
    paths = {tf: os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP[tf]}.csv") for tf in TF_MAP}
    raw: Dict[str, pd.DataFrame] = {}
    for tf, path in paths.items():
        if not os.path.isfile(path):
            if tf == "30T":
                raise FileNotFoundError(f"Main TF (30T) missing: {path}")
            else:
                continue
        raw[tf] = expect_time_col(pd.read_csv(path))
    if "30T" not in raw:
        raise FileNotFoundError("30T dataframe missing; abort.")
    return raw

def write_tail_csvs(raw_df: Dict[str, pd.DataFrame], symbol: str, cutoff: pd.Timestamp,
                    out_dir: str, hist_rows: Dict[str, int]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for tf, df in raw_df.items():
        sub = df.loc[df["time"] <= cutoff]
        if sub.empty:
            continue
        need = int(hist_rows.get(tf, 200))
        if need > 0 and len(sub) > need:
            sub = sub.tail(need)
        mapped = TF_MAP.get(tf, tf)
        out_path = os.path.join(out_dir, f"{symbol}_{mapped}.csv")
        cols = ["time"] + [c for c in sub.columns if c != "time"]
        sub[cols].to_csv(out_path, index=False)
        out[tf] = out_path
    return out

def _safe_index(s: pd.Series, key) -> Optional[int]:
    if key not in s.index:
        return None
    val = s.loc[key]
    if isinstance(val, (pd.Series, np.ndarray, list)):
        try:
            return int(np.asarray(val)[-1])
        except Exception:
            return int(val.iloc[-1])
    return int(val)

# ==== Model / thresholds I/O ====
def load_model(model_dir: str):
    payload = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    pipeline   = payload["pipeline"]
    window     = int(payload.get("window_size", 1))
    neg_thr    = float(payload.get("neg_thr", 0.5))
    pos_thr    = float(payload.get("pos_thr", 0.5))
    final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    if not isinstance(final_cols, list):
        final_cols = list(final_cols)
    return payload, pipeline, window, neg_thr, pos_thr, final_cols

def load_train_medians(model_dir: str, payload: dict) -> Optional[dict]:
    path = payload.get("train_distribution") or "train_distribution.json"
    if not os.path.isabs(path):
        path = os.path.join(model_dir, path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        src = j.get("columns", j)
        med = {}
        for c, stats in src.items():
            if isinstance(stats, dict):
                if "median" in stats:
                    med[c] = stats["median"]
                elif "q50" in stats:
                    med[c] = stats["q50"]
        return med or None
    except Exception:
        return None

# ==== Core ====
def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

def build_baseline_predict_strict(args, log, pipeline, window, final_cols):
    filepaths = {tf: f"{args.data_dir}/{args.symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=True, strict_disk_feed=True)
    merged = prep.load_data()

    X, _, _, _, t_idx = prep.ready(
        merged, window=window, selected_features=final_cols,
        mode="predict", with_times=True, predict_drop_last=True
    )
    if not X.empty and all(c in X.columns for c in final_cols):
        X = X[final_cols]

    probs = pipeline.predict_proba(X)[:, 1] if len(X) else np.array([], dtype=float)
    preds = np.full(len(probs), -1, dtype=int)
    payload = joblib.load(os.path.join(args.model_dir, "best_model.pkl"))
    neg_thr = float(payload.get("neg_thr", 0.5)); pos_thr = float(payload.get("pos_thr", 0.5))
    preds[probs <= neg_thr] = 0; preds[probs >= pos_thr] = 1

    t_idx = pd.to_datetime(t_idx).reset_index(drop=True) if t_idx is not None else pd.Series([], dtype="datetime64[ns]")
    tcol = "30T_time" if "30T_time" in merged.columns else "time"
    mtimes = pd.to_datetime(merged[tcol]).reset_index(drop=True)
    mclose = merged["30T_close" if "30T_close" in merged.columns else "close"].reset_index(drop=True)

    time_to_idx = {}
    for ii, tt in enumerate(mtimes):
        time_to_idx[tt] = ii

    y_true_list: List[Union[int, float]] = []
    for tt in t_idx:
        mi = time_to_idx.get(tt, None)
        if mi is None or (mi + 1) >= len(mclose):
            y_true_list.append(np.nan)
        else:
            y_true_list.append(1 if float(mclose.iloc[mi + 1]) - float(mclose.iloc[mi]) > 0 else 0)

    valid = ~pd.isna(y_true_list)
    df = pd.DataFrame({
        "time": t_idx[valid].values,
        "prob": probs[valid],
        "pred": preds[valid],
        "true": pd.Series(y_true_list, dtype="float")[valid].astype(int).values
    }).reset_index(drop=True)

    out_csv = os.path.join(args.model_dir, "deploy_baseline.csv")
    df.to_csv(out_csv, index=False)
    log.info("[baseline] saved -> %s (N=%d)", out_csv, len(df))
    return {"df": df, "X": X.reset_index(drop=True).loc[valid, :]}

def _tails_from_base(b30: int, m1h: float, m15: float, m5: float) -> Dict[str, int]:
    return {
        "30T": int(b30),
        "1H":  int(max(32, math.ceil(b30 * m1h))),
        "15T": int(max(64, math.ceil(b30 * m15))),
        "5T":  int(max(128, math.ceil(b30 * m5))),
    }

def auto_tail_search_with_convergence(
    log: logging.Logger,
    raw_df: Dict[str, pd.DataFrame],
    symbol: str,
    final_cols: List[str],
    window: int,
    drop_last: bool,
    base_times: pd.Series,
    base_X: pd.DataFrame,
    *,
    samples: int,
    max30t: int,
    mult_1h: float,
    mult_15t: float,
    mult_5t: float,
    tol_l2: float,
    tol_maxabs: float
) -> Dict[str, int]:
    log.info("[auto-tail] (fixed) start: samples=%d max30t=%d", samples, max30t)
    main = raw_df["30T"]
    N = len(main)

    # robust cutoff sampling even for samples=1
    if samples <= 0:
        samples = 1
    step = max(1, (N - 2) // samples)
    idxs = [max(0, N - 2 - i * step) for i in range(samples)]
    idxs = sorted(set([i for i in idxs if 0 <= i < N - 1]))
    if not idxs:
        idxs = [N - 2]
    cutoffs = [pd.to_datetime(main.loc[i, "time"]) for i in idxs]

    best_needed_30 = 0
    tmp_root = tempfile.mkdtemp(prefix="deploy_autotail_")
    try:
        for ci, cutoff in enumerate(cutoffs, 1):
            # safer lower bound
            low  = max(64, 40 * max(1, int(window)))
            high = min(max30t, N - 2)
            ok_high = None
            base = low
            while base <= high:
                tails = _tails_from_base(base, mult_1h, mult_15t, mult_5t)
                it_dir = os.path.join(tmp_root, f"c{ci}_b{base}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, it_dir, tails)
                if "30T" not in fps:
                    shutil.rmtree(it_dir, ignore_errors=True)
                    base = base * 2 if base > 0 else low * 2
                    continue

                prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                              verbose=False, fast_mode=True, strict_disk_feed=True)
                merged = prep.load_data()
                X, _, _, _, t_idx = prep.ready(
                    merged, window=window, selected_features=final_cols,
                    mode="predict", with_times=True, predict_drop_last=drop_last
                )
                ok = False
                if not X.empty and len(t_idx) > 0:
                    want = list(final_cols)
                    if all(c in X.columns for c in want):
                        X = X[want]
                        t_feat = pd.to_datetime(t_idx.iloc[-1])
                        base_time_to_idx = pd.Series(np.arange(len(base_times), dtype=int), index=base_times)
                        bj = _safe_index(base_time_to_idx, t_feat)
                        if bj is not None and 0 <= bj < len(base_X):
                            v_live = X.tail(1).iloc[0].values.astype(float)
                            v_base = base_X.iloc[bj][want].values.astype(float)
                            diff = v_live - v_base
                            l2 = float(np.linalg.norm(diff))
                            mx = float(np.max(np.abs(diff)))
                            ok = (l2 <= tol_l2) and (mx <= tol_maxabs)

                shutil.rmtree(it_dir, ignore_errors=True)
                if ok:
                    ok_high = base
                    break
                base *= 2

            if ok_high is None:
                log.warning("[auto-tail] cutoff=%s not converged up to max30t=%d", str(cutoff), max30t)
                best_needed_30 = max(best_needed_30, max30t)
                continue

            # binary search to minimize
            lo, hi, best = max(32, ok_high // 2), ok_high, ok_high
            while lo <= hi:
                mid = (lo + hi) // 2
                tails = _tails_from_base(mid, mult_1h, mult_15t, mult_5t)
                it_dir = os.path.join(tmp_root, f"c{ci}_bin{mid}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, it_dir, tails)
                prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                              verbose=False, fast_mode=True, strict_disk_feed=True)
                merged = prep.load_data()
                X, _, _, _, t_idx = prep.ready(
                    merged, window=window, selected_features=final_cols,
                    mode="predict", with_times=True, predict_drop_last=drop_last
                )
                ok = False
                if not X.empty and len(t_idx) > 0 and all(c in X.columns for c in final_cols):
                    X = X[final_cols]
                    t_feat = pd.to_datetime(t_idx.iloc[-1])
                    base_time_to_idx = pd.Series(np.arange(len(base_times), dtype=int), index=base_times)
                    bj = _safe_index(base_time_to_idx, t_feat)
                    if bj is not None and 0 <= bj < len(base_X):
                        v_live = X.tail(1).iloc[0].values.astype(float)
                        v_base = base_X.iloc[bj].values.astype(float)
                        diff = v_live - v_base
                        l2 = float(np.linalg.norm(diff)); mx = float(np.max(np.abs(diff)))
                        ok = (l2 <= tol_l2) and (mx <= tol_maxabs)
                shutil.rmtree(it_dir, ignore_errors=True)
                if ok:
                    best = mid; hi = mid - 1
                else:
                    lo = mid + 1
            log.info("[auto-tail] cutoff=%s -> min30T=%d (converged)", str(cutoff), best)
            best_needed_30 = max(best_needed_30, best)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    tails = _tails_from_base(best_needed_30, mult_1h, mult_15t, mult_5t)
    log.info("[auto-tail] chosen tails: 30T=%d 1H=%d 15T=%d 5T=%d", tails["30T"], tails["1H"], tails["15T"], tails["5T"])
    return tails

def run_live_and_compare(args, log, payload, pipeline, window, final_cols, baseline, tol_l2, tol_maxabs):
    raw_df = load_raw_csvs(args)
    main_df = raw_df["30T"]

    base_df: pd.DataFrame = baseline["df"]
    base_X:  pd.DataFrame = baseline["X"]
    base_times = pd.to_datetime(base_df["time"])
    base_time_to_idx = pd.Series(np.arange(len(base_times), dtype=int), index=base_times)

    # tails
    if args.auto_tail:
        hist_rows = auto_tail_search_with_convergence(
            log, raw_df, args.symbol, final_cols, window, args.predict_drop_last,
            base_times, base_X,
            samples=max(1, int(args.auto_tail_samples)),
            max30t=int(args.auto_tail_max30t),
            mult_1h=float(args.mult_1h), mult_15t=float(args.mult_15t), mult_5t=float(args.mult_5t),
            tol_l2=tol_l2, tol_maxabs=tol_maxabs
        )
    else:
        hist_rows = {"30T": int(args.hist_30t), "1H": int(args.hist_1h), "15T": int(args.hist_15t), "5T": int(args.hist_5t)}
    log.info("[live] tails: %s", hist_rows)

    N = len(main_df)
    end_idx = N - 2
    start_idx = max(0, end_idx - int(args.tail_iters) + 1)
    total = end_idx - start_idx + 1
    log.info("[live] iteration range: start_idx=%d end_idx=%d total=%d", start_idx, end_idx, total)

    neg_thr = float(payload.get("neg_thr", 0.5)); pos_thr = float(payload.get("pos_thr", 0.5))
    rows = []
    tmp_root = tempfile.mkdtemp(prefix="deploy_live_")
    try:
        for k in range(total):
            i = start_idx + k
            cutoff = pd.to_datetime(main_df.loc[i, "time"])
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            fps = write_tail_csvs(raw_df, args.symbol, cutoff, iter_dir, hist_rows)
            if "30T" not in fps:
                shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                          verbose=False, fast_mode=True, strict_disk_feed=True)
            merged = prep.load_data()
            X, _, _, _, t_idx = prep.ready(
                merged, window=window, selected_features=final_cols,
                mode="predict", with_times=True, predict_drop_last=True
            )
            if X.empty:
                shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            want = list(final_cols)
            missing = [c for c in want if c not in X.columns]
            if missing:
                if not args.allow_missing_cols:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    log.error("[live] missing %d train cols @%s e.g. %s", len(missing), str(cutoff), missing[:10])
                    raise SystemExit(2)
                med = load_train_medians(args.model_dir, payload)
                if med is None:
                    log.error("[live] --allow-missing-cols set but train medians missing")
                    raise SystemExit(2)
                X = X.reindex(columns=want, fill_value=np.nan)
                for c in missing:
                    mv = med.get(c, None)
                    if mv is None or not np.isfinite(mv):
                        log.error("[live] no median for column %s", c)
                        raise SystemExit(2)
                    X[c] = float(mv)
                X = X.fillna(0.0)
            else:
                X = X[want]

            t_feat = pd.to_datetime(t_idx.iloc[-1]) if (t_idx is not None and len(t_idx) > 0) else None
            prob_live = float(pipeline.predict_proba(X.tail(1))[:, 1])
            pred_live = decide(prob_live, neg_thr, pos_thr)

            c0 = float(main_df.loc[i, "close"]); c1 = float(main_df.loc[i+1, "close"])
            y_true = 1 if (c1 - c0) > 0 else 0

            base_prob = base_pred = base_true = np.nan
            l2 = maxabs = np.nan
            if t_feat is not None:
                bj = _safe_index(base_time_to_idx, t_feat)
                if bj is not None and 0 <= bj < len(base_df):
                    base_prob = float(base_df.loc[bj, "prob"]); base_pred = int(base_df.loc[bj, "pred"]); base_true = int(base_df.loc[bj, "true"])
                    row_live = X.tail(1).iloc[0].values.astype(float)
                    row_base = base_X.iloc[bj][want].values.astype(float)
                    diff = row_live - row_base
                    l2 = float(np.linalg.norm(diff))
                    maxabs = float(np.max(np.abs(diff)))

            rows.append({
                "iter": k+1, "cutoff": cutoff, "feat_time": t_feat,
                "prob_live": prob_live, "pred_live": int(pred_live), "true": int(y_true),
                "base_prob": float(base_prob) if not np.isnan(base_prob) else np.nan,
                "base_pred": int(base_pred) if not np.isnan(base_prob) else -9,
                "base_true": int(base_true) if not np.isnan(base_prob) else -9,
                "l2_diff": float(l2) if not np.isnan(l2) else np.nan,
                "maxabs_diff": float(maxabs) if not np.isnan(maxabs) else np.nan,
            })
            shutil.rmtree(iter_dir, ignore_errors=True)
            if ((k + 1) % max(1, getattr(args, "progress_every", 50)) == 0) or args.verbose:
                log.info("[live %5d/%5d] cutoff=%s feat_time=%s prob=%.4f -> pred=%s  true=%d  | L2=%.3g MaxAbs=%.3g",
                        k+1, total, str(cutoff), str(t_feat), prob_live,
                        {1:"BUY ",0:"SELL",-1:"NONE"}[pred_live], y_true,
                        (l2 if not np.isnan(l2) else -1.0),
                        (maxabs if not np.isnan(maxabs) else -1.0))

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_live = os.path.join(args.model_dir, "deploy_live.csv")
    live_df = pd.DataFrame(rows)
    live_df.to_csv(out_live, index=False)
    log.info("[live] saved -> %s (N=%d)", out_live, len(live_df))

    common = live_df.dropna(subset=["base_prob"]).copy()
    pred_agree = float((common["base_pred"].values == common["pred_live"].values).mean()) * 100.0 if len(common)>0 else 0.0
    feat_ok = float(((common["l2_diff"].fillna(0) <= tol_l2) & (common["maxabs_diff"].fillna(0) <= tol_maxabs)).mean()) * 100.0 if len(common)>0 else 0.0
    log.info("[parity] pred_agree=%.3f%%  feat_ok=%.3f%% (tol_l2=%.1e, tol_maxabs=%.1e) on %d rows",
             pred_agree, feat_ok, tol_l2, tol_maxabs, len(common))
    return live_df, pred_agree, feat_ok

def replay_train_acc(args, log, payload, window, final_cols) -> float:
    filepaths = {tf: f"{args.data_dir}/{args.symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()
    X, y, _, _ = prep.ready(
        merged, window=window, selected_features=final_cols,
        mode="train", with_times=False, predict_drop_last=False, train_drop_last=False
    )
    if X.empty:
        log.warning("[train] empty X")
        return 0.0
    X = X[final_cols] if all(c in X.columns for c in final_cols) else X
    probs = payload["pipeline"].predict_proba(X)[:, 1]
    neg_thr = float(payload.get("neg_thr", 0.5)); pos_thr = float(payload.get("pos_thr", 0.5))
    preds = np.full(len(probs), -1, dtype=int)
    preds[probs <= neg_thr] = 0; preds[probs >= pos_thr] = 1
    mask = preds != -1
    if mask.sum() == 0:
        return 0.0
    acc = float(((preds == y.values) & mask).sum()) / mask.sum() * 100.0
    log.info("[train] replay acc=%.4f%% (P=%d / N=%d)", acc, int(mask.sum()), len(preds))
    return acc

def main():
    args = parse_args()
    log = setup_logger(os.path.join(args.model_dir, args.log_file), args.verbose)
    log.info("=== deploy_ready_guard (fixed) starting ===")
    log.info("data-dir=%s symbol=%s model-dir=%s tail-iters=%d", args.data_dir, args.symbol, args.model_dir, args.tail_iters)

    payload, pipeline, window, neg_thr, pos_thr, final_cols = load_model(args.model_dir)
    log.info("Model: window=%d thr(neg=%.3f,pos=%.3f) final_cols=%d", window, neg_thr, pos_thr, len(final_cols))

    # 1) Baseline with strict LIVE path
    baseline = build_baseline_predict_strict(args, log, pipeline, window, final_cols)

    # 2) LIVE with L2-converged tails + compare
    live_df, pred_agree, feat_ok = run_live_and_compare(
        args, log, payload, pipeline, window, final_cols, baseline,
        tol_l2=args.tol_l2, tol_maxabs=args.tol_maxabs
    )

    # 3) Optional: compare TRAIN acc vs BASELINE acc
    ok_train = True
    if args.check_train_acc:
        train_acc = replay_train_acc(args, log, payload, window, final_cols)
        base_df = baseline["df"]
        m = base_df["pred"] != -1
        base_acc = float((base_df.loc[m, "pred"] == base_df.loc[m, "true"]).mean()) * 100.0 if m.any() else 0.0
        delta = abs(train_acc - base_acc)
        ok_train = (delta <= args.acc_diff_max)
        log.info("[acc-check] train=%.4f%%  baseline=%.4f%%  |Δ|=%.4f ≤ %.4f ? %s",
                 train_acc, base_acc, delta, args.acc_diff_max, "OK" if ok_train else "FAIL")

    ok_pred = pred_agree >= args.pred_parity_min
    ok_feat = feat_ok    >= args.feat_parity_min
    ready = ok_pred and ok_feat and ok_train

    log.info("[result] pred_parity_ok=%s feat_parity_ok=%s train_acc_ok=%s -> %s",
             ok_pred, ok_feat, ok_train, "READY" if ready else "NOT-READY")
    log.info("=== deploy_ready_guard (fixed) finished ===")

    sys.exit(0 if ready else 2)

if __name__ == "__main__":
    main()


# python3 deploy_ready_guard_fixed.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 4000 \
#   --auto-tail \
#   --auto-tail-samples 3 \
#   --auto-tail-max30t 12000 \
#   --tol-l2 1e-8 --tol-maxabs 1e-8 \
#   --pred-parity-min 99.9 \
#   --feat-parity-min 99.9 \
#   --check-train-acc \
#   --acc-diff-max 0.5 \
#   --progress-every 50 \
#   --verbose
