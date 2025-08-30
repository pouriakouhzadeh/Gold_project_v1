#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-like simulator with:
  • replay       → بازپخشِ دقیق TRAIN (Accuracy = TRAIN)
  • incremental  → لایوِ حالت‌مند روی merged ازپیش‌محاسبه‌شده (stateful, بدون I/O تکراری)
  • live         → لایوِ file-tail (برای دیپلوی/دیباگ). با --auto-tail، طول دم‌ها خودکار تعیین می‌شود.

Auto-Tail (برای mode=live):
  - روی چند cutoff نزدیک انتها تست می‌کند (نمونه‌گیری).
  - یک "پایهٔ 30T" را به صورت افزایشی (doubling) بزرگ می‌کند و سپس باینری‌سرچ می‌زند
    تا کمینهٔ طول دم پیدا شود که «همهٔ ستون‌های TRAIN» ساخته شوند و X خالی نباشد.
  - طول بقیهٔ تایم‌فریم‌ها از روی ضریب‌های امن نسبت به 30T محاسبه می‌شود (قابل‌تنظیم).
  - نتیجهٔ نهایی: hist-30t/1h/15t/5t بهینه برای همان داده/مدل، و نزدیک‌ترین برابری با TRAIN.

نکته‌های کلیدی برای برابری با TRAIN:
  - در replay دقیقاً همان X/y TRAIN ساخته می‌شود.
  - در incremental کل merged مثل TRAIN ساخته می‌شود (fast_mode=False, strict_disk_feed=False)
    سپس X_all با mode="predict" و predict_drop_last=True «یکبار» ساخته شده و قدم‌به‌قدم امتیازدهی می‌شود.
  - برچسب هر قدم y(t) = 1{ close(t+1) > close(t) } از merged اصلی استخراج می‌شود (بدون off-by-one).

"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import json
import math
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# ========================= Logging (console + file) =========================

def setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("live-like")
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

# ========================= Safe _CompatWrapper (joblib) =========================
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

# ========================= Data prep =========================
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}
ALL_TFS = ("30T", "1H", "15T", "5T")

# ========================= CLI =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live-like simulation: replay / incremental / live (+ auto-tail).")
    p.add_argument("--mode", choices=["replay", "incremental", "live"], default="incremental",
                   help="حالت اجرا (replay=برابر TRAIN، incremental=stateful، live=file-tail).")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9", help="پوشهٔ CSVهای خام.")
    p.add_argument("--symbol", default="XAUUSD", help="نماد.")
    p.add_argument("--model-dir", default=".", help="پوشهٔ مدل (best_model.pkl).")
    p.add_argument("--tail-iters", type=int, default=4000, help="تعداد آخرین تکرارها برای گزارش (در incremental/live).")
    p.add_argument("--verbose", action="store_true", help="لاگِ پرجزئیات.")
    p.add_argument("--log-file", default="live_like_sim.log", help="فایل لاگ خروجی.")

    # thresholds & audit
    p.add_argument("--allow-missing-cols", action="store_true",
                   help="اگر برخی ستون‌های TRAIN ساخته نشدند با میانهٔ TRAIN پر شود (ترجیحاً خاموش).")

    # replay / incremental: محدودهٔ زمانی اختیاری
    p.add_argument("--start-time", type=str, default=None, help="فیلتر شروع (مثلاً 2024-01-01).")
    p.add_argument("--end-time",   type=str, default=None, help="فیلتر پایان (مثلاً 2025-06-01).")

    # live (file-tail) only
    p.add_argument("--predict-drop-last", action="store_true",
                   help="(فقط mode=live) بعد از ساخت فیچرها آخرین ردیف حذف شود (پیشنهادی).")
    p.add_argument("--hist-30t", type=int, default=480,  help="(live) tail 30T rows per iter.")
    p.add_argument("--hist-1h",  type=int, default=240,  help="(live) tail 1H rows per iter.")
    p.add_argument("--hist-15t", type=int, default=960,  help="(live) tail 15T rows per iter.")
    p.add_argument("--hist-5t",  type=int, default=2880, help="(live) tail 5T rows per iter.")

    # ---------- Auto-Tail ----------
    p.add_argument("--auto-tail", action="store_true",
                   help="در mode=live طول دم‌ها را به‌صورت خودکار بیاب.")
    p.add_argument("--auto-tail-samples", type=int, default=3,
                   help="تعداد cutoff نمونه برای تخمین طول امن دم.")
    p.add_argument("--auto-tail-max30t", type=int, default=6000,
                   help="حداکثر طول مجاز 30T برای جستجوی خودکار.")
    # ضرایب تبدیل طول 30T به سایر TFها (قابل تنظیم)
    p.add_argument("--mult-1h",  type=float, default=0.5,
                   help="ضریب 30T→1H (به‌طور پیش‌فرض 0.5 یعنی نصف 30T).")
    p.add_argument("--mult-15t", type=float, default=2.0,
                   help="ضریب 30T→15T (پیش‌فرض 2×).")
    p.add_argument("--mult-5t",  type=float, default=6.0,
                   help="ضریب 30T→5T  (پیش‌فرض 6×).")

    return p.parse_args()

# ========================= Helpers =========================

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
                if "median" in stats: med[c] = stats["median"]
                elif "q50" in stats:  med[c] = stats["q50"]
        return med or None
    except Exception:
        return None

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

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
            continue
        raw_df[tf] = expect_cols(pd.read_csv(path))
    if "30T" not in raw_df:
        raise FileNotFoundError("30T dataframe missing; abort.")
    return raw_df

def write_tail_csvs(
    raw_df: Dict[str, pd.DataFrame],
    symbol: str,
    cutoff: pd.Timestamp,
    out_dir: str,
    hist_rows: Dict[str, int],
) -> Dict[str, str]:
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

# ========================= Auto-Tail =========================

def _test_tail_once(
    filepaths: Dict[str, str],
    window: int,
    final_cols: List[str],
    predict_drop_last: bool,
) -> Tuple[bool, int]:
    """
    یک تست سریع: آیا با این فایل‌ها X ساخته می‌شود و تمام ستون‌های TRAIN حضور دارند؟
    برمی‌گرداند: (ok, cols_missing_count)
    """
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths=filepaths, main_timeframe="30T",
        verbose=False, fast_mode=True, strict_disk_feed=True
    )
    merged = prep.load_data()
    if merged.empty:
        return (False, 0)
    X, _, _, _, _ = prep.ready(
        merged,
        window=window,
        selected_features=final_cols,
        mode="predict",
        with_times=True,
        predict_drop_last=predict_drop_last
    )
    if X.empty:
        return (False, len(final_cols))
    live = set(X.columns)
    want = set(final_cols)
    miss = list(want - live)
    return (len(miss) == 0, len(miss))

def auto_tail_search(
    log: logging.Logger,
    raw_df: Dict[str, pd.DataFrame],
    symbol: str,
    final_cols: List[str],
    window: int,
    predict_drop_last: bool,
    *,
    samples: int = 3,
    max30t: int = 6000,
    mult_1h: float = 0.5,
    mult_15t: float = 2.0,
    mult_5t: float = 6.0,
) -> Dict[str, int]:
    """
    روی چند cutoff نزدیک انتهای 30T (samples) جستجو می‌کند و کمینهٔ tailِ امن را می‌یابد.
    خروجی: دیکشنری {'30T': n30, '1H': n1h, '15T': n15, '5T': n5}
    """
    log.info("[auto-tail] starting… samples=%d max30t=%d multipliers(1H=%.2f,15T=%.2f,5T=%.2f)",
             samples, max30t, mult_1h, mult_15t, mult_5t)

    main = raw_df["30T"]
    N = len(main)
    if N < 100:
        # داده خیلی کم
        base = min(N, 256)
        return {
            "30T": base,
            "1H":  max(32, math.ceil(base * mult_1h)),
            "15T": max(64, math.ceil(base * mult_15t)),
            "5T":  max(128, math.ceil(base * mult_5t)),
        }

    # نقاط نمونه: انتهای سری و چند نقطه قبل‌تر
    step = max(1, N // (samples + 1))
    idxs = [N - 1 - i * step for i in range(samples)]
    idxs = [i for i in idxs if 0 <= i < N-1]  # نیاز به c(t+1) هم داریم
    cutoffs = [pd.to_datetime(main.loc[i, "time"]) for i in sorted(set(idxs))]

    # تابع تولید طول‌های tail از روی base30
    def tails_from_base(base30: int) -> Dict[str, int]:
        return {
            "30T": int(base30),
            "1H":  int(max(32, math.ceil(base30 * mult_1h))),
            "15T": int(max(64, math.ceil(base30 * mult_15t))),
            "5T":  int(max(128, math.ceil(base30 * mult_5t))),
        }

    best_needed_30 = 0
    tmp_root = tempfile.mkdtemp(prefix="auto_tail_")

    try:
        for c_idx, cutoff in enumerate(cutoffs, 1):
            # 1) upper bound پیدا کن (doubling)
            low, high = 64, min(max30t, N - 2)
            ok_high = None
            while low <= high:
                base = low
                tails = tails_from_base(base)
                # نوشتن فایل‌های دم
                iter_dir = os.path.join(tmp_root, f"c{c_idx}_b{base}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, iter_dir, tails)
                if "30T" not in fps:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    low = base * 2
                    continue
                ok, miss = _test_tail_once(fps, window, final_cols, predict_drop_last)
                shutil.rmtree(iter_dir, ignore_errors=True)

                if ok:
                    ok_high = base
                    break
                # نه: دم کم است → دو برابر کن
                low = base * 2
                if low > max30t:
                    break

            if ok_high is None:
                # حتی با max30t هم کامل نشد؛ همین max را ثبت کن
                log.warning("[auto-tail] cutoff=%s not satisfied within max30t=%d", str(cutoff), max30t)
                best_needed_30 = max(best_needed_30, max30t)
                continue

            # 2) باینری‌سرچ بین (prev_fail, ok_high)
            lo, hi = max(32, ok_high // 2), ok_high
            best = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                tails = tails_from_base(mid)
                iter_dir = os.path.join(tmp_root, f"c{c_idx}_bin{mid}")
                fps = write_tail_csvs(raw_df, symbol, cutoff, iter_dir, tails)
                if "30T" not in fps:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    lo = mid + 1
                    continue
                ok, miss = _test_tail_once(fps, window, final_cols, predict_drop_last)
                shutil.rmtree(iter_dir, ignore_errors=True)

                if ok:
                    best = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            log.info("[auto-tail] cutoff=%s → min 30T tail=%d", str(cutoff), best)
            best_needed_30 = max(best_needed_30, best)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    final = tails_from_base(best_needed_30)
    log.info("[auto-tail] chosen tails: 30T=%d 1H=%d 15T=%d 5T=%d",
             final.get("30T", 0), final.get("1H", 0), final.get("15T", 0), final.get("5T", 0))
    return final

# ========================= Core runners =========================

def run_incremental(args, log):
    model_path = os.path.join(args.model_dir, "best_model.pkl")
    payload = joblib.load(model_path)
    pipeline   = payload["pipeline"]
    window     = int(payload.get("window_size", 1))
    neg_thr    = float(payload.get("neg_thr", 0.5))
    pos_thr    = float(payload.get("pos_thr", 0.5))
    final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    if not isinstance(final_cols, list): final_cols = list(final_cols)
    log.info("Model loaded. window=%d  thr=(neg=%.3f, pos=%.3f)  final_cols=%d",
             window, neg_thr, pos_thr, len(final_cols))

    # merge مثل TRAIN
    data_dir, symbol = args.data_dir, args.symbol
    filepaths = {tf: f"{data_dir}/{symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    merged = prep.load_data()
    tcol = "30T_time" if "30T_time" in merged.columns else "time"

    if args.start_time:
        merged = merged[pd.to_datetime(merged[tcol]) >= pd.to_datetime(args.start_time)]
    if args.end_time:
        merged = merged[pd.to_datetime(merged[tcol]) <= pd.to_datetime(args.end_time)]
    merged = merged.reset_index(drop=True)

    # یک‌بار همهٔ فیچرها به صورت causal
    X_all, _, _, _, t_idx = prep.ready(
        merged, window=window,
        selected_features=final_cols,
        mode="predict", with_times=True,
        predict_drop_last=True
    )
    if X_all.empty:
        log.error("X_all is empty after READY; increase history or check features."); return
    X_all = X_all[final_cols] if all(c in X_all.columns for c in final_cols) else X_all
    probs_all = pipeline.predict_proba(X_all)[:, 1]
    t_idx = pd.to_datetime(t_idx).reset_index(drop=True)

    mtimes = pd.to_datetime(merged[tcol]).reset_index(drop=True)
    mclose = merged["30T_close" if "30T_close" in merged.columns else "close"].reset_index(drop=True)
    # map time→index (فرض بر یکتایی زمان‌ها بعد از پاک‌سازی)
    time_to_idx = pd.Series(np.arange(len(mtimes), dtype=int), index=mtimes)

    total_feat = len(t_idx)
    if total_feat < 2:
        log.error("Not enough feature rows (need >=2)."); return
    # فقط انتهای سری گزارش شود
    start = max(0, total_feat - int(args.tail_iters))
    end   = total_feat - 2  # باید +1 برای y وجود داشته باشد
    total = max(0, end - start + 1)
    log.info("INCREMENTAL feature-range: start=%d end=%d total=%d", start, end, total)
    if total <= 0:
        log.error("Nothing to evaluate."); return

    preds = wins = losses = unpred = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0

    rows: List[dict] = []

    for k, j in enumerate(range(start, end+1), start=1):
        t_feat = t_idx.iloc[j]
        prob = float(probs_all[j])
        pred = decide(prob, neg_thr, pos_thr)

        if t_feat not in time_to_idx.index:
            continue
        mi = int(time_to_idx.loc[t_feat])
        if mi+1 >= len(mclose):
            continue
        y_true = 1 if float(mclose.iloc[mi+1]) - float(mclose.iloc[mi]) > 0 else 0

        if pred == -1:
            unpred += 1; none_n += 1; verdict = "UNPRED"
        else:
            preds += 1
            if pred == 1: buy_n += 1
            else:         sell_n += 1
            if pred == y_true:
                wins += 1; verdict = "WIN"
                if y_true == 1: tp += 1
                else:           tn += 1
            else:
                losses += 1; verdict = "LOSS"
                if y_true == 1: fn += 1
                else:           fp += 1

        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        cover = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
        dec_label = {1: "BUY ", 0: "SELL", -1: "NONE"}[pred]

        log.info(
            "[%5d/%5d] feat_time=%s  prob=%.3f → pred=%s  true=%d → %s  | P=%d W=%d L=%d U=%d  Acc=%.2f%% Cover=%.2f%%  | BUY=%d SELL=%d NONE=%d",
            k, total, str(t_feat), prob, dec_label, y_true, verdict,
            preds, wins, losses, unpred, acc, cover, buy_n, sell_n, none_n
        )
        rows.append({"iter": k, "feat_time": t_feat, "prob": prob, "pred": int(pred), "true": int(y_true), "verdict": verdict})

    acc = (wins / preds * 100.0) if preds > 0 else 0.0
    cover = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
    print("\n========== SUMMARY (incremental) ==========")
    print(f"Feature rows tested: {total}")
    print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
    print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {cover:.2f}%")
    print(f"Decisions  → BUY: {buy_n}  SELL: {sell_n}  NONE: {none_n}")
    print(f"Confusion  → TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
    print("==========================================\n")

    out_csv = os.path.join(args.model_dir, "sim_results_incremental.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("Incremental results saved → %s", out_csv)


def run_replay(args, log):
    model_path = os.path.join(args.model_dir, "best_model.pkl")
    payload = joblib.load(model_path)
    pipeline   = payload["pipeline"]
    window     = int(payload.get("window_size", 1))
    neg_thr    = float(payload.get("neg_thr", 0.5))
    pos_thr    = float(payload.get("pos_thr", 0.5))
    final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    if not isinstance(final_cols, list): final_cols = list(final_cols)
    log.info("Model loaded. window=%d  thr=(neg=%.3f, pos=%.3f)  final_cols=%d",
             window, neg_thr, pos_thr, len(final_cols))

    data_dir, symbol = args.data_dir, args.symbol
    filepaths = {tf: f"{data_dir}/{symbol}_{TF_MAP[tf]}.csv" for tf in TF_MAP}
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T",
                                  verbose=False, fast_mode=False, strict_disk_feed=False)
    raw = prep.load_data()

    X, y, _, _ = prep.ready(
        raw, window=window,
        selected_features=final_cols,
        mode="train",
        predict_drop_last=False,
        train_drop_last=False
    )
    X = X[final_cols] if all(c in X.columns for c in final_cols) else X
    probs = pipeline.predict_proba(X)[:,1]
    preds = np.full(len(probs), -1, dtype=int)
    preds[probs <= neg_thr] = 0
    preds[probs >= pos_thr] = 1

    mask = preds != -1
    correct = int(((preds == y.values) & mask).sum())
    incorrect = int(mask.sum() - correct)
    unpred = int((~mask).sum())
    acc = (correct / mask.sum() * 100.0) if mask.any() else 0.0

    print("\n========== SUMMARY (replay) ==========")
    print(f"Total={len(preds)}  Predicted={int(mask.sum())}  Unpredicted={unpred}")
    print(f"Accuracy (predicted only): {acc:.2f}%")
    print("======================================\n")


def run_live_tail(args, log):
    # بارگذاری مدل
    payload = joblib.load(os.path.join(args.model_dir, "best_model.pkl"))
    pipeline   = payload["pipeline"]
    window     = int(payload.get("window_size", 1))
    neg_thr    = float(payload.get("neg_thr", 0.5))
    pos_thr    = float(payload.get("pos_thr", 0.5))
    final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    if not isinstance(final_cols, list): final_cols = list(final_cols)
    log.info("Model loaded. window=%d  thr=(neg=%.3f, pos=%.3f)  final_cols=%d",
             window, neg_thr, pos_thr, len(final_cols))

    # خام‌ها
    raw_df = load_raw_csvs(args)
    main_df = raw_df["30T"]

    # اگر auto-tail روشن است، طول دم‌ها را تخمین بزن
    if args.auto_tail:
        tails = auto_tail_search(
            log, raw_df, args.symbol, final_cols, window, args.predict_drop_last,
            samples=max(1, int(args.auto_tail_samples)),
            max30t=int(args.auto_tail_max30t),
            mult_1h=float(args.mult_1h),
            mult_15t=float(args.mult_15t),
            mult_5t=float(args.mult_5t),
        )
        hist_rows = {tf: tails.get(tf, 0) for tf in ALL_TFS}
        log.info("[live] auto-tail applied → %s", hist_rows)
    else:
        hist_rows = {
            "30T": int(args.hist_30t),
            "1H":  int(args.hist_1h),
            "15T": int(args.hist_15t),
            "5T":  int(args.hist_5t),
        }

    N = len(main_df); end_idx = N - 2
    start_idx = max(0, end_idx - int(args.tail_iters) + 1)
    total_iters = end_idx - start_idx + 1
    log.info("LIVE iteration range: start_idx=%d end_idx=%d total=%d", start_idx, end_idx, total_iters)

    # جمع‌کننده‌ها
    preds = wins = losses = unpred = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0

    tmp_root = tempfile.mkdtemp(prefix="live_tail_")

    try:
        for k in range(total_iters):
            i = start_idx + k
            cutoff = pd.to_datetime(main_df.loc[i, "time"])

            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            fps = write_tail_csvs(raw_df, args.symbol, cutoff, iter_dir, hist_rows)
            if "30T" not in fps:
                shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths=fps, main_timeframe="30T",
                verbose=False, fast_mode=True, strict_disk_feed=True
            )
            merged = prep.load_data()
            X, _, _, _, _ = prep.ready(
                merged, window=window,
                selected_features=final_cols,
                mode="predict", with_times=True,
                predict_drop_last=args.predict_drop_last
            )
            if X.empty:
                shutil.rmtree(iter_dir, ignore_errors=True)
                log.debug("[SKIP] empty X at k=%d cutoff=%s", k+1, str(cutoff))
                continue

            # اصرار بر ترتیب ستون‌ها
            want = list(final_cols)
            missing = [c for c in want if c not in X.columns]
            if missing:
                if not args.allow_missing_cols:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    log.error("[FATAL] missing %d train columns, e.g., %s", len(missing), missing[:10])
                    return
                # پر کردن با میانهٔ TRAIN (در صورت اصرار کاربر)
                train_medians = load_train_medians(args.model_dir, payload)
                if train_medians is None:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    log.error("[FATAL] Missing columns but no train medians file."); return
                X = X.reindex(columns=want, fill_value=np.nan)
                for c in missing:
                    mv = train_medians.get(c, None)
                    if mv is None or not np.isfinite(mv):
                        shutil.rmtree(iter_dir, ignore_errors=True)
                        log.error("[FATAL] No median for column: %s", c); return
                    X[c] = float(mv)
                X = X.fillna(0.0)
            else:
                X = X[want]

            prob = float(pipeline.predict_proba(X.tail(1))[:, 1])
            pred = decide(prob, neg_thr, pos_thr)

            c0 = float(main_df.loc[i,   "close"])
            c1 = float(main_df.loc[i+1, "close"])
            y_true = 1 if (c1 - c0) > 0 else 0

            if pred == -1:
                unpred += 1; none_n += 1; verdict = "UNPRED"
            else:
                preds += 1
                if pred == 1: buy_n += 1
                else:         sell_n += 1
                if pred == y_true:
                    wins += 1; verdict = "WIN"
                    if y_true == 1: tp += 1
                    else:           tn += 1
                else:
                    losses += 1; verdict = "LOSS"
                    if y_true == 1: fn += 1
                    else:           fp += 1

            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            cov = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
            dec_label = {1:"BUY ", 0:"SELL", -1:"NONE"}[pred]
            log.info(
                "[%5d/%5d] cutoff=%s  prob=%.3f → pred=%s  true=%d → %s  | P=%d W=%d L=%d U=%d  Acc=%.2f%% Cover=%.2f%%  | BUY=%d SELL=%d NONE=%d",
                k+1, total_iters, str(cutoff), prob, dec_label, y_true, verdict,
                preds, wins, losses, unpred, acc, cov, buy_n, sell_n, none_n
            )
            shutil.rmtree(iter_dir, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    acc = (wins / preds * 100.0) if preds > 0 else 0.0
    cov = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
    print("\n========== SUMMARY (live-tail) ==========")
    print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
    print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {cov:.2f}%")
    print("=========================================\n")

# ========================= Main =========================

def main():
    args = parse_args()
    log = setup_logger(args.log_file, args.verbose)
    log.info("=== sim starting: mode=%s ===", args.mode)
    log.info("data-dir=%s  symbol=%s  model-dir=%s", args.data_dir, args.symbol, args.model_dir)

    if args.mode == "replay":
        run_replay(args, log)
    elif args.mode == "incremental":
        run_incremental(args, log)
    else:
        run_live_tail(args, log)

if __name__ == "__main__":
    main()



# python3 -u live_like_sim.py \
#   --mode replay \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --verbose


# for deploy
# python3 -u live_like_sim.py \
#   --mode incremental \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --tail-iters 4000 \
#   --verbose


# python3 -u live_like_sim.py \
#   --mode live \
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
