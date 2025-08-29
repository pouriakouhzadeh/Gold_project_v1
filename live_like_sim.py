#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-like rolling simulation aligned with TRAIN logic (stable-after-feature-drop).

منطق کلیدی:
- برای پیش‌بینی بازه t→t+Δ (مثال 07:00→07:30)، تا زمان cutoff=t (مثال 07:00) دادهٔ خام به
  PREPARE_DATA_FOR_TRAIN داده می‌شود تا فیچر ساخته شود؛ سپس «بعد از ساخت فیچر» آخرین ردیف
  حذف می‌شود تا ناپایداربودن احتمالی رفع شود و Xِ پایدارِ t-Δ (مثال 06:30) به مدل برسد؛
  اما حقیقت (label) برای همان بازه t→t+Δ (مثال 07:00→07:30) سنجیده می‌شود.

- این اسکریپت فرض می‌کند در TRAIN از منطق زیر استفاده کرده‌اید:
  * train_drop_last=True در PREPARE_DATA_FOR_TRAIN.ready  ➜  برچسب یک گام جلو رفته (y(t+1))
  * بلاک اطمینانِ حذفِ آخرین ردیف در TRAIN موجود است (طبق تغییرات قبلی).
  نتیجه: X(06:30−06:00) ↔ y(07:00→07:30)

- در SIM (این فایل): predict_drop_last=True بطور پیش‌فرض، تا ردیف نهایی پس از ساخت فیچر حذف شود.

خروجی‌ها:
- لاگ روی کنسول و فایل (به‌طور هم‌زمان)
- ساخت CSVهای بریده‌شدهٔ per-iteration (temp) + حذف اتوماتیک مگر --keep-tmp بدهید
- ذخیرهٔ نتایج iterationها در sim_results.csv داخل --model-dir
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import argparse
import logging
import json
from typing import Dict, List, Optional

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

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# ========================= Safe _CompatWrapper (joblib) =========================
# جلوگیری از خطای RecursionError هنگام بارگذاری مدل‌های wrap شده
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

# ========================= CLI =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live-like rolling simulation (stable-after-feature-drop).")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9", help="پوشهٔ CSVهای خام.")
    p.add_argument("--symbol", default="XAUUSD", help="نماد (پیش‌فرض: XAUUSD).")
    p.add_argument("--model-dir", default=".", help="پوشهٔ مدل (حاوی best_model.pkl).")
    p.add_argument("--window-rows", type=int, default=4000, help="طول پنجرهٔ رولینگ بر حسب ردیف‌های 30T.")
    p.add_argument("--tail-iters", type=int, default=4000, help="تعداد آخرین تکرارها برای اجرا.")
    p.add_argument("--keep-tmp", action="store_true", help="حذف‌نکردن فولدرهای موقت per-iteration.")
    p.add_argument("--verbose", action="store_true", help="لاگِ پرجزئیات.")
    p.add_argument("--log-file", default="live_like_sim.log", help="فایل لاگ خروجی.")

    # پیش‌فرض: حذف ردیف آخر پس از ساخت فیچرها در PREDICT (مطابق لایو)
    gx = p.add_mutually_exclusive_group()
    gx.add_argument("--predict-drop-last", dest="predict_drop_last", action="store_true",
                    help="پس از ساخت فیچرها، آخرین ردیف حذف شود (پیشنهادی/پیش‌فرض).")
    gx.add_argument("--no-predict-drop-last", dest="predict_drop_last", action="store_false",
                    help="آخرین ردیف پس از ساخت فیچرها حذف نشود.")
    p.set_defaults(predict_drop_last=True)

    p.add_argument("--align-debug", type=int, default=0,
                   help="تعداد خطوط دیباگ هم‌ترازی (چاپ زمان cutoff و زمان آخرین X).")
    p.add_argument("--allow-missing-cols", action="store_true",
                   help="اگر ستون‌های Train در X لایو نبود، با میانهٔ Train پر کن و ادامه بده.")

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

def write_iter_csvs(raw_df: Dict[str, pd.DataFrame], symbol: str, cutoff, iter_dir: str) -> Dict[str, str]:
    os.makedirs(iter_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for tf, df in raw_df.items():
        sub = df.loc[df["time"] <= cutoff]
        if sub.empty:  # ممکن است در برخی TFها داده‌ای تا این cutoff نباشد
            continue
        # نام فایل سازگار با کلاس PREPARE
        mapped = TF_MAP.get(tf, tf)
        out_path = os.path.join(iter_dir, f"{symbol}_{mapped}.csv")
        cols = ["time"] + [c for c in sub.columns if c != "time"]
        sub[cols].to_csv(out_path, index=False)
        out[tf] = out_path
    return out

# ========================= Main =========================

def main():
    args = parse_args()
    log = setup_logger(args.log_file, args.verbose)

    log.info("=== live_like_sim.py starting ===")
    log.info("data-dir=%s  symbol=%s  model-dir=%s", args.data_dir, args.symbol, args.model_dir)
    log.info("window_rows=%d  tail_iters=%d  predict_drop_last=%s",
             args.window_rows, args.tail_iters, args.predict_drop_last)

    # ---------- Load raw CSVs ----------
    base_csvs = {
        "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
        "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
        "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
        "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
    }
    raw_df: Dict[str, pd.DataFrame] = {}
    for tf, path in base_csvs.items():
        exists = os.path.isfile(path)
        log.info("CSV[%s] => %s (exists=%s)", tf, path, exists)
        if not exists:
            if tf == "30T":
                log.error("Main TF (30T) is required. Missing: %s", path)
                return
            else:
                log.warning("Missing CSV for TF=%s; skipping that TF.", tf)
                continue
        try:
            raw_df[tf] = expect_cols(pd.read_csv(path))
        except Exception as e:
            log.error("Failed to load %s: %s", path, e)
            return
        log.info("Loaded %d rows for TF=%s", len(raw_df[tf]), tf)

    if "30T" not in raw_df:
        log.error("30T dataframe missing; abort.")
        return

    main_df = raw_df["30T"]

    # ---------- Load model payload ----------
    model_path = os.path.join(args.model_dir, "best_model.pkl")
    if not os.path.isfile(model_path):
        log.error("best_model.pkl not found at: %s", model_path)
        return

    log.info("Loading model artefacts: %s", model_path)
    try:
        payload = joblib.load(model_path)
    except RecursionError:
        sys.setrecursionlimit(1_000_000)
        payload = joblib.load(model_path)

    try:
        pipeline   = payload["pipeline"]
        window     = int(payload.get("window_size", 1))
        neg_thr    = float(payload.get("neg_thr", 0.5))
        pos_thr    = float(payload.get("pos_thr", 0.5))
        final_cols = payload.get("train_window_cols") or payload.get("feats") or []
        if not isinstance(final_cols, list): final_cols = list(final_cols)
    except Exception as e:
        log.error("Invalid model payload structure: %s", e); return

    for need in ("predict_proba", "predict"):
        if not hasattr(pipeline, need):
            log.error("Loaded pipeline lacks `%s`.", need); return

    train_medians = load_train_medians(args.model_dir, payload)
    if train_medians is None:
        log.info("Train medians not found. Strict audit still enforced.")

    log.info("Model loaded. window=%d  thr=(neg=%.3f, pos=%.3f)  final_cols=%d",
             window, neg_thr, pos_thr, len(final_cols))

    # ---------- Iteration bounds ----------
    N = len(main_df)
    need_min = args.window_rows + 2  # برای داشتن i و i+1
    if N < need_min:
        log.error("Not enough 30T rows: have=%d, need>=%d", N, need_min)
        return

    base_start = args.window_rows - 1
    end_idx    = N - 2                 # چون حقیقت به i+1 نیاز دارد
    start_idx  = max(base_start, end_idx - args.tail_iters + 1) if (args.tail_iters and args.tail_iters > 0) else base_start
    total_iters = max(0, end_idx - start_idx + 1)

    log.info("Iteration range: start_idx=%d  end_idx=%d  total=%d (N=%d)", start_idx, end_idx, total_iters, N)
    if total_iters <= 0:
        log.error("Nothing to simulate."); return

    # ---------- Accumulators ----------
    preds = wins = losses = unpred = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0

    results_rows: List[dict] = []

    # ---------- Temp root ----------
    tmp_root = tempfile.mkdtemp(prefix="live_like_")
    log.info("Temp root: %s", tmp_root)

    try:
        for k in range(total_iters):
            i = start_idx + k
            cutoff = pd.to_datetime(main_df.loc[i, "time"])

            # 1) Write per-iteration truncated CSVs
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            fps = write_iter_csvs(raw_df, args.symbol, cutoff, iter_dir)
            if "30T" not in fps:
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            # 2) Build features up to cutoff; drop last unstable row AFTER feature construction (predict_drop_last)
            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths=fps, main_timeframe="30T",
                verbose=False, fast_mode=True, strict_disk_feed=True
            )
            merged = prep.load_data()
            if merged.empty:
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            X, _, _, _, t_idx = prep.ready(
                merged,
                window=window,
                selected_features=final_cols,  # دقیقاً همان ستون‌های TRAIN
                mode="predict",
                with_times=True,
                predict_drop_last=args.predict_drop_last
            )
            if X.empty:
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            # 3) STRICT FEATURE AUDIT
            live_cols = list(X.columns)
            want = list(final_cols)
            missing = [c for c in want if c not in live_cols]
            extra   = [c for c in live_cols if c not in want]

            if missing:
                if not args.allow_missing_cols:
                    log.error("[FATAL] %d training columns missing in live X. Example: %s",
                              len(missing), missing[:10])
                    if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                    return
                if train_medians is None:
                    log.error("[FATAL] Missing columns but train medians not available.")
                    if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                    return
                # reindex & fill by medians
                X = X.reindex(columns=want, fill_value=np.nan)
                fill_vals = {}
                for c in missing:
                    mv = train_medians.get(c, None)
                    if mv is None or not np.isfinite(mv):
                        log.error("[FATAL] No median for missing column: %s", c)
                        if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                        return
                    fill_vals[c] = float(mv)
                for c, v in fill_vals.items():
                    X[c] = v
                if X.isna().any().any():
                    n = int(X.isna().sum().sum())
                    log.warning("[WARN] %d NaNs after median fill; filling with 0.0 fallback.", n)
                    X = X.fillna(0.0)
            else:
                # exact set; reorder
                X = X[want]

            # align debug
            if args.align_debug and k < args.align_debug:
                x_time = None
                try:
                    if t_idx is not None and len(t_idx) > 0:
                        x_time = pd.to_datetime(t_idx.iloc[-1])
                except Exception:
                    pass
                log.info("[ALIGN] cutoff=%s | X_last_time=%s | drop_last=%s", cutoff, x_time, args.predict_drop_last)

            # 4) Predict last available row
            X_last = X.tail(1)
            try:
                prob = float(pipeline.predict_proba(X_last)[:, 1])
            except Exception:
                prob = float(pipeline.predict_proba(X_last.values)[:, 1])

            pred = decide(prob, neg_thr, pos_thr)

            # 5) Truth for interval i→i+1 (e.g., 07:00→07:30)
            try:
                c0 = float(main_df.loc[i,   "close"])
                c1 = float(main_df.loc[i+1, "close"])
            except Exception:
                # اگر به هر دلیل i+1 وجود نداشت، این iteration را رد کن
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue
            y_true = 1 if (c1 - c0) > 0 else 0

            # 6) Metrics update
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
            coverage = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
            dec_label = {1: "BUY ", 0: "SELL", -1: "NONE"}[pred]

            log.info(
                "[%5d/%5d] cutoff=%s  prob=%.3f → pred=%s  true=%d → %s  | P=%d W=%d L=%d U=%d  Acc=%.2f%% Cover=%.2f%%  | BUY=%d SELL=%d NONE=%d",
                k+1, total_iters, str(cutoff), prob, dec_label, y_true, verdict,
                preds, wins, losses, unpred, acc, coverage, buy_n, sell_n, none_n
            )

            # 7) Collect iteration row
            x_time_last = None
            try:
                if t_idx is not None and len(t_idx) > 0:
                    x_time_last = pd.to_datetime(t_idx.iloc[-1])
            except Exception:
                pass
            results_rows.append({
                "iter": k+1,
                "cutoff": pd.to_datetime(cutoff),
                "x_time_last": x_time_last,
                "prob": prob,
                "pred": int(pred),
                "true": int(y_true),
                "verdict": verdict
            })

            # 8) cleanup temp
            if not args.keep_tmp:
                shutil.rmtree(iter_dir, ignore_errors=True)

        # ---------- Summary ----------
        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        coverage = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0

        print("\n========== SUMMARY ==========")
        print(f"Cutoffs tested (iterations): {total_iters}")
        print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
        print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {coverage:.2f}%")
        print(f"Decisions  → BUY: {buy_n}  SELL: {sell_n}  NONE: {none_n}")
        print(f"Confusion  → TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
        print("================================\n")

        # Save results CSV
        out_csv = os.path.join(args.model_dir, "sim_results.csv")
        pd.DataFrame(results_rows).to_csv(out_csv, index=False)
        log.info("Iteration results saved → %s", out_csv)

        log.info("=== live_like_sim.py finished ===")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()



# python3 live_like_sim.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --window-rows 4000 \
#   --tail-iters 4000 \
#   --predict-drop-last \
#   --align-debug 5 \
#   --verbose
