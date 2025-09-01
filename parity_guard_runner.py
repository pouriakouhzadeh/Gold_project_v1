#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parity_guard_runner.py

هدف: یک «ناظر برابرساز» که تضمین کند مسیر محاسبه‌ی فیچر در LIVE دقیقاً
با مسیر baseline (predict) یکی است؛ اختلاف‌ها را شفاف لاگ می‌کند و در انتها
اعلام می‌کند «READY / NOT-READY» برای دپلوی.

اصول کلیدی:
- «دقت TRAIN» (mode=train) معیار دپلوی نیست؛ به‌جایش «baseline/predict»
  با همان کُد مسیر LIVE معیار است. (train تقریباً همیشه خوش‌بینانه‌تر است.)
- برای برابری بیت‌به‌بیت، baseline و live هر دو باید از یک کُدpath استفاده کنند:
  fast_mode=True و strict_disk_feed=True و predict_drop_last=True.
- Auto-tail حداقل طول‌های امن را می‌یابد تا همه‌ی ستون‌های TRAIN ساخته شوند.
- اگر هنوز اختلاف باشد، فایل لاگ top-K فیچرهایی که اختلاف می‌سازند را چاپ می‌کند
  + نُرم اختلاف (L2/MaxAbs) را می‌دهد.

خروجی‌ها (داخل --model-dir):
- parity_baseline.csv      → baseline/predict_strict با y_true هم‌تراز
- parity_live_tail.csv     → LIVE-tail با همان تنظیمات
- parity_guard.log         → لاگ کامل اجرا و تشخیص آماده‌بودن

Exit code:
- 0 → READY (پاریتی با baseline برقرار است)
- 2 → NOT READY (اختلاف معنادار)
"""
from __future__ import annotations

import os, sys, math, json, shutil, tempfile, argparse, logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

# ==== Project imports ========================================================
# انتظار می‌رود در پروژه شما وجود داشته باشد
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}
ALL_TFS = ("30T", "1H", "15T", "5T")

# ==== Logging ================================================================

def setup_logger(path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("parity-guard")
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

# ==== CLI ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parity Guard: match LIVE with baseline/predict_strict.")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--model-dir", default="/home/pouria/gold_project9")
    p.add_argument("--tail-iters", type=int, default=4000)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", default="parity_guard.log")

    # drop-last و حالت strict برای هر دو مسیر baseline و live روشن می‌ماند
    p.add_argument("--predict-drop-last", action="store_true", default=True)

    # آستانه‌ی اختلاف قابل قبول
    p.add_argument("--tol-l2", type=float, default=1e-6, help="حداکثر L2 بین ردیف baseline و live")
    p.add_argument("--tol-maxabs", type=float, default=1e-6, help="حداکثر قدرمطلق اختلاف یک فیچر")
    p.add_argument("--diff-topk", type=int, default=12, help="Top-K فیچر برای گزارش mismatch")

    # Auto-tail
    p.add_argument("--auto-tail", action="store_true", default=True)
    p.add_argument("--auto-tail-samples", type=int, default=3)
    p.add_argument("--auto-tail-max30t", type=int, default=6000)
    p.add_argument("--mult-1h", type=float, default=0.5)
    p.add_argument("--mult-15t", type=float, default=2.0)
    p.add_argument("--mult-5t", type=float, default=6.0)

    # در صورت نبود برخی ستون‌ها، از میانه‌ی TRAIN پر شود
    p.add_argument("--allow-missing-cols", action="store_true", default=True)

    return p.parse_args()

# ==== Helpers ===============================================================

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

# ---- Auto-tail -------------------------------------------------------------

def _test_tail_once(filepaths: Dict[str, str], window: int, final_cols: List[str], drop_last: bool) -> Tuple[bool, int]:
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=True, strict_disk_feed=True)
    merged = prep.load_data()
    X, _, _, _, _ = prep.ready(
        merged,
        window=window,
        selected_features=final_cols,
        mode="predict",
        with_times=True,
        predict_drop_last=drop_last,
    )
    if X.empty:
        return (False, len(final_cols))
    miss = list(set(final_cols) - set(X.columns))
    return (len(miss) == 0, len(miss))


def auto_tail_search(log: logging.Logger, raw_df: Dict[str, pd.DataFrame], symbol: str,
                     final_cols: List[str], window: int, drop_last: bool,
                     *, samples: int = 3, max30t: int = 6000,
                     mult_1h: float = 0.5, mult_15t: float = 2.0, mult_5t: float = 6.0) -> Dict[str, int]:
    log.info("[auto-tail] start: samples=%d max30t=%d", samples, max30t)
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
    tmp_root = tempfile.mkdtemp(prefix="parity_autotail_")
    try:
        for ci, cutoff in enumerate(cutoffs, 1):
            low, high = 64, min(max30t, N - 2)
            ok_high = None
            base = low
            while base <= high:
                tails = tails_from_base(base)
                it_dir = os.path.join(tmp_root, f"c{ci}_b{base}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, it_dir, tails)
                if "30T" not in fps:
                    shutil.rmtree(it_dir, ignore_errors=True)
                    base *= 2
                    continue
                ok, _ = _test_tail_once(fps, window, final_cols, drop_last)
                shutil.rmtree(it_dir, ignore_errors=True)
                if ok:
                    ok_high = base
                    break
                base *= 2
            if ok_high is None:
                log.warning("[auto-tail] cutoff=%s not satisfied up to max30t=%d", str(cutoff), max30t)
                best_needed_30 = max(best_needed_30, max30t)
                continue
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
                ok, _ = _test_tail_once(fps, window, final_cols, drop_last)
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

    tails = tails_from_base(best_needed_30)
    log.info("[auto-tail] chosen tails: 30T=%d 1H=%d 15T=%d 5T=%d", tails["30T"], tails["1H"], tails["15T"], tails["5T"])
    return tails

# ==== Model IO ==============================================================

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

# ==== Core utils ============================================================

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1


def build_baseline_predict_strict(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols):
    """Baseline بر مبنای همان مسیر LIVE: fast_mode=True, strict_disk_feed=True."""
    filepaths = {tf: f"{args.data_dir}/{args.symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
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
    if not X.empty and all(c in X.columns for c in final_cols):
        X = X[final_cols]

    probs = pipeline.predict_proba(X)[:, 1] if len(X) else np.array([], dtype=float)
    preds = np.full(len(probs), -1, dtype=int)
    preds[probs <= neg_thr] = 0
    preds[probs >= pos_thr] = 1

    # y_true از سری قیمتی merged 
    t_idx = pd.to_datetime(t_idx).reset_index(drop=True) if t_idx is not None else pd.Series([], dtype="datetime64[ns]")
    tcol = "30T_time" if "30T_time" in merged.columns else "time"
    mtimes = pd.to_datetime(merged[tcol]).reset_index(drop=True)
    mclose = merged["30T_close" if "30T_close" in merged.columns else "close"].reset_index(drop=True)

    # map time → آخرین اندیس
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

    valid_mask = ~pd.isna(y_true_list)
    df = pd.DataFrame({
        "time": t_idx[valid_mask].values,
        "prob": probs[valid_mask],
        "pred": preds[valid_mask],
        "true": pd.Series(y_true_list, dtype="float")[valid_mask].astype(int).values,
    }).reset_index(drop=True)

    out_csv = os.path.join(args.model_dir, "parity_baseline.csv")
    df.to_csv(out_csv, index=False)

    mask = df["pred"].values != -1
    acc = (float((df.loc[mask, "pred"].values == df.loc[mask, "true"].values).sum()) / max(1, mask.sum())) * 100.0
    log.info("[baseline/predict_strict] total=%d predicted=%d unpred=%d acc=%.4f%%",
             len(df), int(mask.sum()), int(len(df) - int(mask.sum())), acc)

    return {"df": df, "X": X.reset_index(drop=True), "acc": acc}


def _safe_get_index_from_series_map(s: pd.Series, key) -> Optional[int]:
    if key not in s.index:
        return None
    val = s.loc[key]
    if isinstance(val, (pd.Series, np.ndarray, list)):
        try:
            return int(np.asarray(val)[-1])
        except Exception:
            return int(val.iloc[-1])
    else:
        return int(val)


def run_live_tail_and_compare(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols, baseline):
    raw_df = load_raw_csvs(args)
    main_df = raw_df["30T"]

    # tails
    if args.auto_tail:
        hist_rows = auto_tail_search(
            log, raw_df, args.symbol, final_cols, window, args.predict_drop_last,
            samples=max(1, int(args.auto_tail_samples)),
            max30t=int(args.auto_tail_max30t),
            mult_1h=float(args.mult_1h), mult_15t=float(args.mult_15t), mult_5t=float(args.mult_5t)
        )
    else:
        hist_rows = {"30T": 480, "1H": 240, "15T": 960, "5T": 2880}
    log.info("[live] tails: %s", hist_rows)

    base_df: pd.DataFrame = baseline["df"]
    base_X:  pd.DataFrame = baseline["X"]
    base_times = pd.to_datetime(base_df["time"])  # feature timestamps

    base_time_to_idx = pd.Series(np.arange(len(base_times), dtype=int), index=base_times)

    N = len(main_df)
    end_idx = N - 2
    start_idx = max(0, end_idx - int(args.tail_iters) + 1)
    total = end_idx - start_idx + 1
    log.info("[live] iteration range: start_idx=%d end_idx=%d total=%d", start_idx, end_idx, total)

    rows = []
    preds = wins = losses = unpred = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0

    tmp_root = tempfile.mkdtemp(prefix="parity_live_")
    train_medians = load_train_medians(args.model_dir, payload)

    try:
        for k in range(total):
            i = start_idx + k
            cutoff = pd.to_datetime(main_df.loc[i, "time"])  # تا قبل از این زمان
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
                predict_drop_last=True,
            )
            if X.empty:
                shutil.rmtree(iter_dir, ignore_errors=True)
                log.debug("[live/skip] empty X at k=%d cutoff=%s", k + 1, str(cutoff))
                continue

            want = list(final_cols)
            missing = [c for c in want if c not in X.columns]
            filled_median = False
            if missing:
                if not args.allow_missing_cols:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    log.error("[live/fatal] missing %d train cols @%s e.g. %s", len(missing), str(cutoff), missing[:10])
                    return False
                if train_medians is None:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    log.error("[live/fatal] --allow-missing-cols but train medians not found")
                    return False
                X = X.reindex(columns=want, fill_value=np.nan)
                for c in missing:
                    mv = train_medians.get(c, None)
                    if mv is None or not np.isfinite(mv):
                        shutil.rmtree(iter_dir, ignore_errors=True)
                        log.error("[live/fatal] no median for column %s", c)
                        return False
                    X[c] = float(mv)
                X = X.fillna(0.0)
                filled_median = True
            else:
                X = X[want]

            t_feat = pd.to_datetime(t_idx.iloc[-1]) if (t_idx is not None and len(t_idx) > 0) else None
            prob_live = float(pipeline.predict_proba(X.tail(1))[:, 1])
            pred_live = decide(prob_live, neg_thr, pos_thr)

            # true از قیمت اصلی
            c0 = float(main_df.loc[i, "close"]) ; c1 = float(main_df.loc[i+1, "close"]) 
            y_true = 1 if (c1 - c0) > 0 else 0

            # baseline mapping by time
            base_prob = base_pred = base_true = np.nan
            l2 = maxabs = np.nan
            topk_txt = ""
            if t_feat is not None:
                bj = _safe_get_index_from_series_map(base_time_to_idx, t_feat)
                if bj is not None and 0 <= bj < len(base_df):
                    base_prob = float(base_df.loc[bj, "prob"]) ; base_pred = int(base_df.loc[bj, "pred"]) ; base_true = int(base_df.loc[bj, "true"]) 
                    try:
                        row_live = X.tail(1).iloc[0]
                        row_base = base_X.iloc[bj]
                        v_live = row_live.values.astype(float)
                        v_base = row_base[want].values.astype(float) if all(c in row_base.index for c in want) else row_base.values.astype(float)
                        diff = v_live - v_base
                        l2 = float(np.linalg.norm(diff))
                        maxabs = float(np.max(np.abs(diff)))
                        idxs = np.argsort(-np.abs(diff))[:max(1, int(args.diff_topk))]
                        parts = [f"{want[p]}:{diff[p]:+.3g}" for p in idxs]
                        topk_txt = "; ".join(parts)
                    except Exception:
                        pass

            # update metrics
            if pred_live == -1:
                unpred += 1 ; none_n += 1 ; verdict = "UNPRED"
            else:
                preds += 1
                if pred_live == 1: buy_n += 1
                else: sell_n += 1
                if pred_live == y_true:
                    wins += 1 ; verdict = "WIN"
                    if y_true == 1: tp += 1
                    else: tn += 1
                else:
                    losses += 1 ; verdict = "LOSS"
                    if y_true == 1: fn += 1
                    else: fp += 1

            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            cov = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0

            log.info("[LIVE %5d/%5d] cutoff=%s feat_time=%s prob_live=%.3f → pred=%s true=%d → %s | base_prob=%.3f base_pred=%s base_true=%s | L2=%.3g MaxAbs=%.3g miss_cols=%d filled_median=%s | Acc=%.2f%% Cov=%.2f%%",
                     k+1, total, str(cutoff), str(t_feat), prob_live,
                     {1:"BUY ",0:"SELL",-1:"NONE"}[pred_live], y_true, verdict,
                     base_prob, (str(base_pred) if not np.isnan(base_prob) else "NA"), (str(base_true) if not np.isnan(base_prob) else "NA"),
                     (l2 if not np.isnan(l2) else 0.0), (maxabs if not np.isnan(maxabs) else 0.0),
                     len(missing), filled_median, acc, cov)
            if (not np.isnan(base_prob)) and int(base_pred) != int(pred_live):
                log.debug("[LIVE DIFF] top%d: %s", int(args.diff_topk), topk_txt)

            rows.append({
                "iter": k+1, "cutoff": cutoff, "feat_time": t_feat,
                "prob_live": prob_live, "pred_live": int(pred_live), "true": int(y_true),
                "base_prob": float(base_prob) if not np.isnan(base_prob) else np.nan,
                "base_pred": int(base_pred) if not np.isnan(base_prob) else -9,
                "base_true": int(base_true) if not np.isnan(base_prob) else -9,
                "l2_diff": float(l2) if not np.isnan(l2) else np.nan,
                "maxabs_diff": float(maxabs) if not np.isnan(maxabs) else np.nan,
                "miss_cols": len(missing), "filled_median": filled_median,
            })

            shutil.rmtree(iter_dir, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_live = os.path.join(args.model_dir, "parity_live_tail.csv")
    df_live = pd.DataFrame(rows)
    df_live.to_csv(out_live, index=False)
    log.info("[live] saved → %s", out_live)

    # قضاوت پاریتی: 
    # - فقط ردیف‌هایی که feat_time در baseline پیدا می‌شوند را بررسی می‌کنیم
    common = df_live.dropna(subset=["base_prob"]).copy()
    eq_pred = (common["base_pred"].values == common["pred_live"].values)
    l2_ok = (common["l2_diff"].fillna(0) <= args.tol_l2).values
    mx_ok = (common["maxabs_diff"].fillna(0) <= args.tol_maxabs).values
    parity_mask = eq_pred & l2_ok & mx_ok

    ok_ratio = float(parity_mask.sum()) / max(1, len(parity_mask))
    log.info("[parity-check] matched_rows=%d / %d  (%.2f%%)", parity_mask.sum(), len(parity_mask), ok_ratio*100.0)

    if ok_ratio >= 0.999:  # تقریباً همه باید منطبق باشند
        log.info("[parity-check] READY ✅  → مسیر LIVE با baseline/predict_strict برابر است.")
        return True
    else:
        # نمونه‌هایی از اختلاف را گزارش کنیم
        bad = common.loc[~parity_mask].head(10)
        for _, r in bad.iterrows():
            log.warning("[mismatch] t=%s pred_live=%s base_pred=%s l2=%.3g maxabs=%.3g",
                        str(r["feat_time"]), int(r["pred_live"]), int(r["base_pred"]), r["l2_diff"], r["maxabs_diff"])
        log.error("[parity-check] NOT READY ❌  → اختلاف معنادار بین baseline و LIVE.")
        return False

# ==== MAIN ==================================================================

def main():
    args = parse_args()
    log = setup_logger(os.path.join(args.model_dir, args.log_file), args.verbose)
    log.info("=== parity_guard_runner starting ===")
    log.info("data-dir=%s  symbol=%s  model-dir=%s  tail-iters=%d", args.data_dir, args.symbol, args.model_dir, args.tail_iters)

    # load model
    payload, pipeline, window, neg_thr, pos_thr, final_cols = load_model(args.model_dir)
    log.info("Model: window=%d thr(neg=%.3f,pos=%.3f) final_cols=%d", window, neg_thr, pos_thr, len(final_cols))

    # 1) Baseline (predict_strict) — همان مسیر LIVE
    baseline = build_baseline_predict_strict(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols)
    log.info("[summary] baseline/predict_strict acc=%.4f%%", baseline["acc"]) 

    # 2) LIVE-tail با همان مسیر و مقایسه با baseline
    ok = run_live_tail_and_compare(args, log, payload, pipeline, window, neg_thr, pos_thr, final_cols, baseline)

    log.info("=== parity_guard_runner finished ===  status=%s", "READY" if ok else "NOT-READY")
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()



# python3 -u parity_guard_runner.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 4000 \
#   --verbose
