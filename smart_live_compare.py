#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_live_compare.py
Run 3 modes back-to-back (replay / incremental / live-tail) and log rich diagnostics:
- replay: exact TRAIN-style evaluation (mode="train")
- incremental: stateful live on fully-merged data (mode="predict", drop-last)
- live-tail: file-tail live with auto-tail (optional) + per-iteration diffs vs baseline

Outputs:
- Logs:   smart_live_compare.log
- CSVs:   smart_replay_train.csv, smart_replay_predict.csv, smart_incremental.csv, smart_live_tail.csv
"""

from __future__ import annotations

import os, sys, math, json, shutil, tempfile, argparse, logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

# ==== logging ================================================================

def setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("smart-sim")
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

# ==== joblib compat for _CompatWrapper ======================================
try:
    import model_pipeline as _mp
    if hasattr(_mp, "_CompatWrapper"):
        _CW = _mp._CompatWrapper
        def _cw_safe_getattr(self, item):
            if item.startswith("__") and item.endswith("__"):
                raise AttributeError(item)
            return getattr(object.__getattribute__(self, "_model"), item)
        def _cw_getstate(self):
            return {"_model": getattr(self, "_model", None), "_inner": getattr(self, "_inner", None)}
        def _cw_setstate(self, state):
            self._model = state.get("_model", None)
            self._inner = state.get("_inner", None)
            self.named_steps = getattr(self._inner, "named_steps", {})
            self.steps = getattr(self._inner, "steps", [])
        _CW.__getattr__ = _cw_safe_getattr
        _CW.__getstate__ = _cw_getstate
        _CW.__setstate__ = _cw_setstate
except Exception:
    pass

# ==== project imports ========================================================
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}
ALL_TFS = ("30T", "1H", "15T", "5T")

# ==== CLI ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart live-vs-train comparator (replay + incremental + live-tail).")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9", help="Raw CSV folder.")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol.")
    p.add_argument("--model-dir", default="/home/pouria/gold_project9", help="Folder containing best_model.pkl.")
    p.add_argument("--tail-iters", type=int, default=4000, help="How many last feature rows/iterations to report.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--log-file", default="smart_live_compare.log", help="Log filename.")
    p.add_argument("--allow-missing-cols", action="store_true",
                   help="If live-tail misses columns, fill by train medians (train_distribution.json).")
    p.add_argument("--diff-topk", type=int, default=8, help="Top-K features to log on mismatch (live-tail vs baseline).")

    # live-tail defaults
    p.add_argument("--predict-drop-last", action="store_true",
                   help="Drop last row after feature build in live-tail (recommended).")
    p.add_argument("--hist-30t", type=int, default=480,  help="Manual tail length for 30T if auto-tail is OFF.")
    p.add_argument("--hist-1h",  type=int, default=240,  help="Manual tail length for 1H if auto-tail is OFF.")
    p.add_argument("--hist-15t", type=int, default=960,  help="Manual tail length for 15T if auto-tail is OFF.")
    p.add_argument("--hist-5t",  type=int, default=2880, help="Manual tail length for 5T if auto-tail is OFF.")

    # auto-tail
    p.add_argument("--auto-tail", action="store_true", help="Search safe tails automatically.")
    p.add_argument("--auto-tail-samples", type=int, default=3, help="Cutoffs to probe for auto-tail.")
    p.add_argument("--auto-tail-max30t", type=int, default=6000, help="Max 30T rows to try in auto-tail.")
    p.add_argument("--mult-1h",  type=float, default=0.5, help="Tail multiplier vs 30T for 1H.")
    p.add_argument("--mult-15t", type=float, default=2.0, help="Tail multiplier vs 30T for 15T.")
    p.add_argument("--mult-5t",  type=float, default=6.0, help="Tail multiplier vs 30T for 5T.")
    return p.parse_args()

# ==== helpers ===============================================================

def expect_cols(df: pd.DataFrame) -> pd.DataFrame:
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
    base_csvs = {
        "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
        "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
        "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
        "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
    }
    raw_df: Dict[str, pd.DataFrame] = {}
    for tf, path in base_csvs.items():
        if not os.path.isfile(path):
            if tf == "30T":
                raise FileNotFoundError(f"Main TF (30T) is required. Missing: {path}")
            else:
                continue
        raw_df[tf] = expect_cols(pd.read_csv(path))
    if "30T" not in raw_df:
        raise FileNotFoundError("30T dataframe missing; abort.")
    return raw_df

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

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

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

# ---- Auto-tail -------------------------------------------------------------

def _test_tail_once(filepaths: Dict[str, str], window: int, final_cols: List[str],
                    predict_drop_last: bool) -> Tuple[bool, int]:
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=True, strict_disk_feed=True)
    merged = prep.load_data()
    if merged.empty:
        return (False, 0)
    X, _, _, _, _ = prep.ready(merged, window=window, selected_features=final_cols,
                               mode="predict", with_times=True, predict_drop_last=predict_drop_last)
    if X.empty:
        return (False, len(final_cols))
    miss = list(set(final_cols) - set(X.columns))
    return (len(miss) == 0, len(miss))

def auto_tail_search(log: logging.Logger, raw_df: Dict[str, pd.DataFrame], symbol: str, final_cols: List[str],
                     window: int, predict_drop_last: bool, *,
                     samples: int = 3, max30t: int = 6000,
                     mult_1h: float = 0.5, mult_15t: float = 2.0, mult_5t: float = 6.0) -> Dict[str, int]:
    log.info("[auto-tail] start: samples=%d max30t=%d multipliers(1H=%.2f,15T=%.2f,5T=%.2f)",
             samples, max30t, mult_1h, mult_15t, mult_5t)
    main = raw_df["30T"]
    N = len(main)
    step = max(1, N // (samples + 1))
    idxs = [N - 1 - i * step for i in range(samples)]
    idxs = [i for i in idxs if 0 <= i < N - 1]
    cutoffs = [pd.to_datetime(main.loc[i, "time"]) for i in sorted(set(idxs))]

    def tails_from_base(b30: int) -> Dict[str, int]:
        return {
            "30T": int(b30),
            "1H":  int(max(32, math.ceil(b30 * mult_1h))),
            "15T": int(max(64, math.ceil(b30 * mult_15t))),
            "5T":  int(max(128, math.ceil(b30 * mult_5t))),
        }

    best_needed_30 = 0
    tmp_root = tempfile.mkdtemp(prefix="smart_autotail_")
    try:
        for ci, cutoff in enumerate(cutoffs, 1):
            low, high = 64, min(max30t, N - 2)
            ok_high = None
            base = low
            # exponential search to find an OK bound
            while base <= high:
                tails = tails_from_base(base)
                it_dir = os.path.join(tmp_root, f"c{ci}_b{base}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, it_dir, tails)
                if "30T" not in fps:
                    shutil.rmtree(it_dir, ignore_errors=True)
                    base *= 2
                    continue
                ok, _ = _test_tail_once(fps, window, final_cols, predict_drop_last)
                shutil.rmtree(it_dir, ignore_errors=True)
                if ok:
                    ok_high = base
                    break
                base *= 2
            if ok_high is None:
                log.warning("[auto-tail] cutoff=%s not satisfied up to max30t=%d",
                            str(cutoff), max30t)
                best_needed_30 = max(best_needed_30, max30t)
                continue
            # binary search for minimal OK base
            lo, hi, best = max(32, ok_high // 2), ok_high, ok_high
            while lo <= hi:
                mid = (lo + hi) // 2
                tails = tails_from_base(mid)
                it_dir = os.path.join(tmp_root, f"c{ci}_bin{mid}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, it_dir, tails)
                if "30T" not in fps:
                    shutil.rmtree(it_dir, ignore_errors=True)
                    lo = mid + 1
                    continue
                ok, _ = _test_tail_once(fps, window, final_cols, predict_drop_last)
                shutil.rmtree(it_dir, ignore_errors=True)
                if ok:
                    best = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            log.info("[auto-tail] cutoff=%s → min30T=%d", str(cutoff), best)
            best_needed_30 = max(best_needed_30, best)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    final = tails_from_base(best_needed_30)
    log.info("[auto-tail] chosen tails: 30T=%d 1H=%d 15T=%d 5T=%d",
             final["30T"], final["1H"], final["15T"], final["5T"])
    return final

# ==== core runs ==============================================================

def run_replay_and_predict_baseline(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols):
    data_dir, symbol = args.data_dir, args.symbol
    filepaths = {tf: f"{data_dir}/{symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()

    # --- replay (TRAIN-like)
    X_tr, y_tr, _, _ = prep.ready(
        merged,
        window=window,
        selected_features=final_cols,
        mode="train",
        predict_drop_last=False,
        train_drop_last=False
    )
    if not X_tr.empty and all(c in X_tr.columns for c in final_cols):
        X_tr = X_tr[final_cols]
    probs_tr = pipeline.predict_proba(X_tr)[:, 1] if len(X_tr) else np.array([], dtype=float)
    preds_tr = np.full(len(probs_tr), -1, dtype=int)
    preds_tr[probs_tr <= neg_thr] = 0
    preds_tr[probs_tr >= pos_thr] = 1
    mask_tr = preds_tr != -1
    correct_tr = int(((preds_tr == y_tr.values) & mask_tr).sum()) if len(y_tr) else 0
    acc_tr = (correct_tr / mask_tr.sum() * 100.0) if mask_tr.any() else 0.0

    out_tr = os.path.join(args.model_dir, "smart_replay_train.csv")
    pd.DataFrame({"prob": probs_tr, "pred": preds_tr, "true": (y_tr.values if len(y_tr) else [])}).to_csv(out_tr, index=False)
    log.info("[replay/train] total=%d predicted=%d unpred=%d acc=%.4f%%",
             len(preds_tr), int(mask_tr.sum()), int((~mask_tr).sum()), acc_tr)

    # --- baseline/predict (mode=predict + drop-last)
    Xp, _, _, _, t_idx = prep.ready(
        merged,
        window=window,
        selected_features=final_cols,
        mode="predict",
        with_times=True,
        predict_drop_last=True
    )
    if not Xp.empty and all(c in Xp.columns for c in final_cols):
        Xp = Xp[final_cols]

    probs_p = pipeline.predict_proba(Xp)[:, 1] if len(Xp) else np.array([], dtype=float)
    preds_p = np.full(len(probs_p), -1, dtype=int)
    preds_p[probs_p <= neg_thr] = 0
    preds_p[probs_p >= pos_thr] = 1

    # y_true aligned to feature timestamps:
    t_idx = pd.to_datetime(t_idx).reset_index(drop=True) if t_idx is not None else pd.Series([], dtype="datetime64[ns]")
    tcol = "30T_time" if "30T_time" in merged.columns else "time"
    mtimes = pd.to_datetime(merged[tcol]).reset_index(drop=True)
    mclose = merged["30T_close" if "30T_close" in merged.columns else "close"].reset_index(drop=True)

    # map time -> last index (dedup safe)
    time_to_idx = {}
    for ii, tt in enumerate(mtimes):
        time_to_idx[tt] = ii  # keep last occurrence

    y_true_list: List[Union[int, float]] = []
    for tt in t_idx:
        mi = time_to_idx.get(tt, None)
        if mi is None or (mi + 1) >= len(mclose):
            y_true_list.append(np.nan)
        else:
            y_true_list.append(1 if float(mclose.iloc[mi + 1]) - float(mclose.iloc[mi]) > 0 else 0)

    valid_mask_np = (~pd.isna(y_true_list)).astype(bool)
    t_idx_p = t_idx[valid_mask_np] if len(t_idx) else pd.Series([], dtype="datetime64[ns]")
    probs_p_v = probs_p[valid_mask_np] if len(probs_p) else np.array([], dtype=float)
    preds_p_v = preds_p[valid_mask_np] if len(preds_p) else np.array([], dtype=int)
    y_true_p = pd.Series(y_true_list, dtype="float").dropna().astype(int).reset_index(drop=True)

    mask_p = preds_p_v != -1
    correct_p = int(((preds_p_v == y_true_p.values) & mask_p).sum()) if len(y_true_p) else 0
    acc_p = (correct_p / mask_p.sum() * 100.0) if mask_p.any() else 0.0

    df_p = pd.DataFrame({
        "time": t_idx_p.values,
        "prob": probs_p_v,
        "pred": preds_p_v,
        "true": y_true_p.values
    })
    out_p = os.path.join(args.model_dir, "smart_replay_predict.csv")
    df_p.to_csv(out_p, index=False)

    log.info("[baseline/predict] total=%d predicted=%d unpred=%d acc=%.4f%%",
             len(preds_p_v), int(mask_p.sum()), int((~mask_p).sum()), acc_p)

    return {
        "prep": prep,
        "merged": merged,
        "t_idx": t_idx_p.reset_index(drop=True),
        "Xp": Xp.reset_index(drop=True),
        "probs_p": probs_p_v,
        "preds_p": preds_p_v,
        "y_true_p": y_true_p.reset_index(drop=True),
        "acc_train": acc_tr,
        "acc_predict": acc_p,
        "mask_p": mask_p
    }

def run_incremental(args, log, baseline, pipeline, neg_thr, pos_thr, final_cols):
    t_idx = baseline["t_idx"]
    probs_all = baseline["probs_p"]
    preds_all = baseline["preds_p"]
    y_true_all = baseline["y_true_p"]
    if len(t_idx) == 0:
        log.error("[incremental] empty baseline t_idx; abort.")
        return
    total_feat = len(t_idx)
    start = max(0, total_feat - int(args.tail_iters))
    end   = total_feat - 2
    total = max(0, end - start + 1)
    log.info("[incremental] feature-range: start=%d end=%d total=%d", start, end, total)
    rows = []
    preds = wins = losses = unpred = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0
    for k, j in enumerate(range(start, end + 1), start=1):
        prob = float(probs_all[j])
        pred = decide(prob, neg_thr, pos_thr)
        true = int(y_true_all.iloc[j])
        if pred == -1:
            unpred += 1; none_n += 1; verdict = "UNPRED"
        else:
            preds += 1
            if pred == 1: buy_n += 1
            else: sell_n += 1
            if pred == true:
                wins += 1; verdict = "WIN"
                if true == 1: tp += 1
                else: tn += 1
            else:
                losses += 1; verdict = "LOSS"
                if true == 1: fn += 1
                else: fp += 1
        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        cov = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
        dec_label = {1: "BUY ", 0: "SELL", -1: "NONE"}[pred]
        log.info("[INC %5d/%5d] time=%s prob=%.3f → pred=%s true=%d → %s | P=%d W=%d L=%d U=%d Acc=%.2f%% Cov=%.2f%% | BUY=%d SELL=%d NONE=%d",
                 k, total, str(t_idx.iloc[j]), prob, dec_label, true, verdict,
                 preds, wins, losses, unpred, acc, cov, buy_n, sell_n, none_n)
        rows.append({"iter": k, "time": t_idx.iloc[j], "prob": prob, "pred": int(pred), "true": int(true), "verdict": verdict})
    out_inc = os.path.join(args.model_dir, "smart_incremental.csv")
    pd.DataFrame(rows).to_csv(out_inc, index=False)
    log.info("[incremental] saved → %s", out_inc)

def _safe_get_index_from_series_map(s: pd.Series, key) -> Optional[int]:
    """Return a single int index from a Series map possibly with duplicate index."""
    if key not in s.index:
        return None
    val = s.loc[key]
    if isinstance(val, (pd.Series, np.ndarray, list)):
        # choose the last occurrence to mimic TRAIN's dedup keep="last"
        try:
            return int(np.asarray(val)[-1])
        except Exception:
            return int(val.iloc[-1])
    else:
        return int(val)

def run_live_tail(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols, baseline):
    raw_df = load_raw_csvs(args)
    main_df = raw_df["30T"]

    # tails
    if args.auto_tail:
        tails = auto_tail_search(
            log, raw_df, args.symbol, final_cols, window, args.predict_drop_last,
            samples=max(1, int(args.auto_tail_samples)),
            max30t=int(args.auto_tail_max30t),
            mult_1h=float(args.mult_1h),
            mult_15t=float(args.mult_15t),
            mult_5t=float(args.mult_5t)
        )
        hist_rows = {tf: tails.get(tf, 0) for tf in ALL_TFS}
    else:
        hist_rows = {
            "30T": int(args.hist_30t), "1H": int(args.hist_1h),
            "15T": int(args.hist_15t), "5T": int(args.hist_5t)
        }
    log.info("[live] tails: %s", hist_rows)

    # baseline maps
    base_times: pd.Series = baseline["t_idx"]          # feature timestamps
    base_probs: np.ndarray = baseline["probs_p"]
    base_preds: np.ndarray = baseline["preds_p"]
    base_ytrue: pd.Series = baseline["y_true_p"]
    base_X: pd.DataFrame = baseline["Xp"]

    base_time_to_idx = pd.Series(np.arange(len(base_times), dtype=int), index=base_times)

    N = len(main_df)
    end_idx = N - 2
    start_idx = max(0, end_idx - int(args.tail_iters) + 1)
    total_iters = end_idx - start_idx + 1
    log.info("[live] iteration range: start_idx=%d end_idx=%d total=%d", start_idx, end_idx, total_iters)

    preds = wins = losses = unpred = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0

    rows = []
    tmp_root = tempfile.mkdtemp(prefix="smart_live_")
    train_medians = load_train_medians(args.model_dir, payload)

    try:
        for k in range(total_iters):
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
                merged,
                window=window,
                selected_features=final_cols,
                mode="predict",
                with_times=True,
                predict_drop_last=args.predict_drop_last
            )
            if X.empty:
                shutil.rmtree(iter_dir, ignore_errors=True)
                log.debug("[live/skip] empty X at k=%d cutoff=%s", k + 1, str(cutoff))
                continue

            want = list(final_cols)
            missing = [c for c in want if c not in X.columns]
            filled_with_median = False
            if missing:
                if not args.allow_missing_cols:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    log.error("[live/fatal] missing %d train cols at cutoff=%s; ex: %s",
                              len(missing), str(cutoff), missing[:10])
                    return
                if train_medians is None:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    log.error("[live/fatal] --allow-missing-cols set but train medians file missing.")
                    return
                X = X.reindex(columns=want, fill_value=np.nan)
                for c in missing:
                    mv = train_medians.get(c, None)
                    if mv is None or not np.isfinite(mv):
                        shutil.rmtree(iter_dir, ignore_errors=True)
                        log.error("[live/fatal] no median for column %s", c)
                        return
                    X[c] = float(mv)
                X = X.fillna(0.0)
                filled_with_median = True
            else:
                X = X[want]

            t_feat = pd.to_datetime(t_idx.iloc[-1]) if (t_idx is not None and len(t_idx) > 0) else None

            prob_live = float(pipeline.predict_proba(X.tail(1))[:, 1])
            pred_live = decide(prob_live, neg_thr, pos_thr)
            c0 = float(main_df.loc[i, "close"])
            c1 = float(main_df.loc[i + 1, "close"])
            y_true = 1 if (c1 - c0) > 0 else 0

            base_prob = base_pred = base_true = np.nan
            l2 = max_abs = np.nan
            topk_txt = ""
            if t_feat is not None:
                bj = _safe_get_index_from_series_map(base_time_to_idx, t_feat)
                if bj is not None and 0 <= bj < len(base_probs):
                    base_prob = float(base_probs[bj])
                    base_pred = int(base_preds[bj])
                    base_true = int(base_ytrue.iloc[bj])
                    try:
                        row_live = X.tail(1).iloc[0]
                        row_base = base_X.iloc[bj]
                        v_live = row_live.values.astype(float)
                        v_base = row_base[want].values.astype(float) if all(c in row_base.index for c in want) else row_base.values.astype(float)
                        diff = v_live - v_base
                        l2 = float(np.linalg.norm(diff))
                        max_abs = float(np.max(np.abs(diff)))
                        idxs = np.argsort(-np.abs(diff))[:max(1, int(args.diff_topk))]
                        parts = [f"{want[p]}:{diff[p]:+.3g}" for p in idxs]
                        topk_txt = "; ".join(parts)
                    except Exception:
                        pass

            # update confusion
            if pred_live == -1:
                unpred += 1; none_n += 1; verdict = "UNPRED"
            else:
                preds += 1
                if pred_live == 1: buy_n += 1
                else: sell_n += 1
                if pred_live == y_true:
                    wins += 1; verdict = "WIN"
                    if y_true == 1: tp += 1
                    else: tn += 1
                else:
                    losses += 1; verdict = "LOSS"
                    if y_true == 1: fn += 1
                    else: fp += 1

            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            cov = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
            log.info(
                "[LIVE %5d/%5d] cutoff=%s feat_time=%s prob_live=%.3f → pred=%s true=%d → %s | base_prob=%.3f base_pred=%s base_true=%s | L2=%.3g MaxAbs=%.3g miss_cols=%d filled_median=%s | Acc=%.2f%% Cov=%.2f%%",
                k + 1, total_iters, str(cutoff), str(t_feat), prob_live,
                {1: 'BUY ', 0: 'SELL', -1: 'NONE'}[pred_live], y_true, verdict,
                base_prob, (str(base_pred) if not np.isnan(base_prob) else "NA"),
                (str(base_true) if not np.isnan(base_prob) else "NA"),
                (l2 if not np.isnan(l2) else 0.0), (max_abs if not np.isnan(max_abs) else 0.0),
                len(missing), filled_with_median, acc, cov
            )
            if (not np.isnan(base_prob)) and (int(base_pred) != int(pred_live)):
                log.debug("[LIVE DIFF] top%d: %s", int(args.diff_topk), topk_txt)

            rows.append({
                "iter": k + 1, "cutoff": cutoff, "feat_time": t_feat,
                "prob_live": prob_live, "pred_live": int(pred_live), "true": int(y_true),
                "base_prob": base_prob, "base_pred": (int(base_pred) if not np.isnan(base_prob) else -9),
                "base_true": (int(base_true) if not np.isnan(base_prob) else -9),
                "l2_diff": l2, "maxabs_diff": max_abs,
                "miss_cols": len(missing), "filled_median": filled_with_median
            })

            shutil.rmtree(iter_dir, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_live = os.path.join(args.model_dir, "smart_live_tail.csv")
    pd.DataFrame(rows).to_csv(out_live, index=False)
    log.info("[live] saved → %s", out_live)

# ==== MAIN ==================================================================

def main():
    args = parse_args()
    log = setup_logger(args.log_file, args.verbose)
    log.info("=== smart_live_compare starting ===")
    log.info("data-dir=%s  symbol=%s  model-dir=%s  tail-iters=%d",
             args.data_dir, args.symbol, args.model_dir, args.tail_iters)

    # load model
    payload, pipeline, window, neg_thr, pos_thr, final_cols = load_model(args.model_dir)
    log.info("Model: window=%d thr(neg=%.3f,pos=%.3f) final_cols=%d",
             window, neg_thr, pos_thr, len(final_cols))

    # 1) REPLAY + BASELINE (predict)
    baseline = run_replay_and_predict_baseline(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols)
    log.info("[summary] replay/train acc=%.4f%%  baseline/predict acc=%.4f%%",
             baseline["acc_train"], baseline["acc_predict"])

    # 2) INCREMENTAL
    run_incremental(args, log, baseline, pipeline, neg_thr, pos_thr, final_cols)

    # 3) LIVE-TAIL
    run_live_tail(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols, baseline)

    log.info("=== smart_live_compare finished ===")

if __name__ == "__main__":
    main()




# python3 -u smart_live_compare.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 4000 \
#   --predict-drop-last \
#   --auto-tail \
#   --auto-tail-samples 3 \
#   --auto-tail-max30t 6000 \
#   --mult-1h 0.5 --mult-15t 2.0 --mult-5t 6.0 \
#   --verbose



# خروجی‌ها

# لاگ: smart_live_compare.log

# CSVها (داخل --model-dir):

# smart_replay_train.csv

# smart_replay_predict.csv

# smart_incremental.csv

# smart_live_tail.csv