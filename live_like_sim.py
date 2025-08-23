#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-like rolling simulation with 4 TFs (5m/15m/30m/1h), aligned to *pre-decision* logic:

در هر تکرار با cutoff = t:
  1) هر ۴ CSV به بازه [start .. t] (شامل t) کات می‌شود.
  2) ویژگی‌ها با PREPARE_DATA_FOR_TRAIN در حالت predict ساخته می‌شوند، **اما ردیف آخر حذف می‌شود**
     (predict_drop_last=True) تا آخرین سطر X مربوط به **t-1** باشد.
  3) پیش‌بینی از آخرین سطر X (t-1) گرفته می‌شود → تصمیم برای بازه [t → t+1].
  4) حقیقت y_true برای [t → t+1] از سری 30T «پاک‌سازیِ دقیقاً مثل Train» ساخته می‌شود.
  5) همه‌چیز روی کنسول استریم می‌شود (پروب، pred، جمع‌آمار، و زمان آخرین فیچر).

اجرا:
python3 live_like_sim.py \
  --data-dir /home/pouria/gold_project9 \
  --symbol XAUUSD \
  --model-dir /home/pouria/gold_project9 \
  --window-rows 4000 \
  --tail-iters 4000 \
  --verbose
"""

from __future__ import annotations
import os, sys, shutil, tempfile, warnings, argparse, logging
from typing import Dict
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.setrecursionlimit(200_000)

# --- ایمنی برای pickleهای قدیمی با Wrapper ---
try:
    import model_pipeline
    if hasattr(model_pipeline, "_CompatWrapper"):
        _CW = model_pipeline._CompatWrapper
        def _cw_safe_getattr(self, item):
            if item.startswith("__") and item.endswith("__"):
                raise AttributeError(item)
            try:
                target = object.__getattribute__(self, "_model")
            except Exception:
                raise AttributeError(item)
            if target is self or target is None:
                raise AttributeError(item)
            return getattr(target, item)
        def _cw_getstate(self):
            return {"_model": getattr(self, "_model", None),
                    "_inner": getattr(self, "_inner", None)}
        def _cw_setstate(self, st):
            self._model = st.get("_model", None)
            self._inner = st.get("_inner", None)
            self.named_steps = getattr(self._inner, "named_steps", {})
            self.steps = getattr(self._inner, "steps", [])
        _CW.__getattr__   = _cw_safe_getattr
        _CW.__getstate__  = _cw_getstate
        _CW.__setstate__  = _cw_setstate
except Exception:
    pass

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from clear_data import ClearData  # برای ساخت Truth همسان با Train

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}

# -------------------- logging --------------------
def build_logger(verbose: bool) -> logging.Logger:
    lg = logging.getLogger("live-like")
    lg.setLevel(logging.DEBUG if verbose else logging.INFO)
    lg.propagate = False
    for h in list(lg.handlers): lg.removeHandler(h)
    h = logging.StreamHandler(stream=sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    lg.addHandler(h)
    return lg

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live-like rolling backtest (pre-decision alignment)")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9",
                   help="Directory with raw CSVs (XAUUSD_M30.csv, ...)")
    p.add_argument("--symbol", default="XAUUSD", help="Symbol file prefix")
    p.add_argument("--model-dir", default=".", help="Folder containing best_model.pkl")
    p.add_argument("--window-rows", type=int, default=4000,
                   help="Rolling window length over 30T (rows).")
    p.add_argument("--tail-iters", type=int, default=4000,
                   help="Run ONLY the last K iterations (K last cutoffs).")
    p.add_argument("--keep-tmp", action="store_true", help="Keep per-iteration temp CSVs")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

# -------------------- helpers --------------------
def expect_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        for cand in ("Time", "timestamp", "datetime", "Date"):
            if cand in df.columns:
                df = df.rename(columns={cand: "time"})
                break
    if "time" not in df.columns:
        raise KeyError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def cut_by_time(df: pd.DataFrame, start, end) -> pd.DataFrame:
    m = (df["time"] >= start) & (df["time"] <= end)  # inclusive
    return df.loc[m].copy()

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0   # SELL/CASH
    if prob >= pos_thr: return 1   # BUY
    return -1                      # NONE

def build_truth_map(raw30: pd.DataFrame) -> pd.Series:
    """
    حقیقت را دقیقاً مثل Train می‌سازیم:
    - ClearData.clean
    - حذف weekend
    - حذف duplicate روی time
    سپس y_true(t) = 1{close(t+1) > close(t)} با ایندکس = time(t).
    """
    df = ClearData().clean(raw30.copy())
    if "time" not in df.columns or "close" not in df.columns:
        raise ValueError("30T CSV must contain 'time' and 'close'")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    # حذف weekend مثل Train.load_data():
    dow = df["time"].dt.dayofweek
    df = df.loc[~dow.isin([5, 6])].copy()
    # حذف duplications:
    df = df.loc[~df["time"].duplicated(keep="last")].reset_index(drop=True)
    # حقیقت:
    y = (df["close"].shift(-1) - df["close"] > 0).astype("Int8")
    y.index = pd.DatetimeIndex(df["time"])
    return y  # ایندکس: time(t)

# -------------------- main --------------------
def main():
    args = parse_args()
    log = build_logger(args.verbose)

    print("=== live_like_sim.py starting ===", flush=True)
    log.info("data-dir=%s | symbol=%s | model-dir=%s", args.data_dir, args.symbol, args.model_dir)
    log.info("window_rows=%d | tail_iters=%d", args.window_rows, args.tail_iters)

    # 1) CSV paths
    base_csvs = {
        "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
        "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
        "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
        "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
    }
    for tf, pth in base_csvs.items():
        log.info("CSV[%s] => %s (exists=%s)", tf, pth, os.path.isfile(pth))

    # 2) Load raw CSVs
    raw_df: Dict[str, pd.DataFrame] = {}
    for tf, pth in base_csvs.items():
        if not os.path.isfile(pth):
            if tf == "30T":
                print(f"[FATAL] Main TF file missing: {pth}", flush=True)
                return
            else:
                log.warning("Missing CSV for %s: %s (skip)", tf, pth)
                continue
        try:
            df = pd.read_csv(pth)
            df = expect_cols(df)
        except Exception as e:
            print(f"[FATAL] Failed to read/parse {pth}: {e}", flush=True)
            return
        raw_df[tf] = df
        log.info("Loaded %d rows for TF=%s", len(df), tf)

    if "30T" not in raw_df:
        print("[FATAL] 30T data not loaded; abort.", flush=True)
        return
    main_df = raw_df["30T"]

    # 3) Truth map (once)
    truth_map = build_truth_map(main_df)
    log.info("Truth map built: keys=%d (cleaned 30T timeline)", truth_map.notna().sum())

    # 4) Load model artefacts
    model_path = os.path.join(args.model_dir, "best_model.pkl")
    log.info("Loading model artefacts: %s", model_path)
    if not os.path.isfile(model_path):
        print(f"[FATAL] best_model.pkl not found at: {model_path}", flush=True)
        try: print("Directory listing:", os.listdir(args.model_dir), flush=True)
        except Exception: pass
        return

    try:
        import sklearn, imblearn  # noqa
        from sklearnex import patch_sklearn
        patch_sklearn(verbose=False)
    except Exception:
        pass

    try:
        payload = joblib.load(model_path)
    except RecursionError as e:
        print(f"[WARN] RecursionError on joblib.load: {e}. Retrying …", flush=True)
        sys.setrecursionlimit(1_000_000)
        payload = joblib.load(model_path)
    except Exception as e:
        print(f"[FATAL] joblib.load failed: {e}", flush=True)
        return

    try:
        pipeline   = payload["pipeline"]
        window     = int(payload.get("window_size", 1))
        neg_thr    = float(payload.get("neg_thr", 0.5))
        pos_thr    = float(payload.get("pos_thr", 0.5))
        final_cols = payload.get("train_window_cols") or payload.get("feats") or []
        if not isinstance(final_cols, list):
            final_cols = list(final_cols)
    except Exception as e:
        print(f"[FATAL] Invalid model payload structure: {e}", flush=True)
        return

    for need in ("predict_proba", "predict"):
        if not hasattr(pipeline, need):
            print(f"[FATAL] Loaded pipeline lacks `{need}` method.", flush=True)
            return

    warmup_rows = (window - 1) + 2  # پنجره‌سازی + (shift(1).diff)
    log.info("Model loaded: window=%d thr=(neg=%.3f,pos=%.3f) final_cols=%d warmup_rows=%d",
             window, neg_thr, pos_thr, len(final_cols), warmup_rows)

    # 5) Iter bounds روی سری خام 30T (فقط برای تعیین t)
    N = len(main_df)
    need_min = args.window_rows + 2
    if N < need_min:
        print(f"[FATAL] Not enough 30T rows: have={N}, need>={need_min}", flush=True)
        return

    base_start = args.window_rows - 1   # inclusive
    end_idx    = N - 2                  # نیاز به i+1 برای حقیقت
    start_idx  = max(base_start, end_idx - (args.tail_iters or (end_idx-base_start+1)) + 1)
    total_iters = end_idx - start_idx + 1
    print(f"[INFO] N={N} | start_idx={start_idx} | end_idx={end_idx} | total_iters={total_iters}", flush=True)
    if total_iters <= 0:
        print("[FATAL] Nothing to simulate. Adjust --tail-iters/--window-rows.", flush=True)
        return

    # 6) Accumulators
    wins = losses = unpred = preds = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0
    skipped_misaligned = 0
    skipped_no_truth   = 0

    # 7) Temp dir
    tmp_root = tempfile.mkdtemp(prefix="live_like_")
    log.info("Temp root: %s", tmp_root)

    try:
        for k in range(total_iters):
            i = start_idx + k
            cutoff_time = pd.to_datetime(main_df.loc[i, "time"])
            start_time  = pd.to_datetime(main_df.loc[i - (args.window_rows - 1), "time"])

            # اگر t در TruthMap نیست (به‌خاطر پاک‌سازی/Weekend)، skip
            if cutoff_time not in truth_map.index or pd.isna(truth_map.loc[cutoff_time]):
                skipped_no_truth += 1
                continue

            # Per-iteration cut (تا t شامل)
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            os.makedirs(iter_dir, exist_ok=True)
            tmp_paths = {}
            for tf, df in raw_df.items():
                sub = cut_by_time(df, start_time, cutoff_time)  # شامل t
                if sub.empty: continue
                out_path = os.path.join(iter_dir, f"{args.symbol}_{tf}.csv")
                cols = ["time"] + [c for c in sub.columns if c != "time"]
                sub[cols].to_csv(out_path, index=False)
                tmp_paths[tf] = out_path

            if "30T" not in tmp_paths:
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            # آماده‌سازی فیچرها با حذف ردیف آخر (تا t-1 تغذیه شود)
            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths=tmp_paths, main_timeframe="30T",
                verbose=False, fast_mode=True, strict_disk_feed=True
            )
            merged = prep.load_data()
            if merged.empty:
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            X_live, _, _, _, t_idx = prep.ready(
                merged,
                window=window,
                selected_features=final_cols,
                mode="predict",
                with_times=True,
                predict_drop_last=True   # ← کلیدی: آخرین ردیف حذف شود → آخرین X = t-1
            )
            if X_live.empty or t_idx is None or len(t_idx) == 0:
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            feat_time = pd.to_datetime(t_idx.iloc[-1])  # باید t-1 باشد
            if pd.isna(feat_time) or not (feat_time < cutoff_time):
                # اگر به هر دلیلی آخرین فیچر >= t بود، این تکرار را امن skip کن
                skipped_misaligned += 1
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            if final_cols:
                X_live = X_live.reindex(columns=final_cols, fill_value=0.0)

            try:
                probs = pipeline.predict_proba(X_live)[:, 1]
            except Exception:
                probs = pipeline.predict_proba(X_live.values)[:, 1]

            p_last = float(probs[-1])
            pred = decide(p_last, neg_thr, pos_thr)

            # حقیقتِ [t → t+1] از TruthMap (همسان Train)
            y_true = int(truth_map.loc[cutoff_time])

            # شمارنده‌ها
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

            # چاپِ زمان آخرین فیچر (باید t-1 باشد) برای اطمینان از هم‌ترازی
            print(
                f"[{k+1:>5}/{total_iters}] @ t={cutoff_time}  X_last={feat_time}  "
                f"p={p_last:.3f} → pred={dec_label}  true={y_true} → {verdict}   "
                f"| cum P={preds} W={wins} L={losses} U={unpred} "
                f"Acc={acc:.2f}% Cover={coverage:.2f}%  "
                f"| buys={buy_n} sells={sell_n} none={none_n}",
                flush=True
            )

            if not args.keep_tmp:
                shutil.rmtree(iter_dir, ignore_errors=True)

        # Summary
        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        coverage = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
        print("\n========== SUMMARY ==========")
        print(f"Cutoffs tested (iterations): {total_iters}")
        print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
        print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {coverage:.2f}%")
        print(f"Decisions  → BUY: {buy_n}  SELL: {sell_n}  NONE: {none_n}")
        print(f"Confusion  → TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
        print(f"Skipped    → misaligned: {skipped_misaligned} | no-truth(after clean): {skipped_no_truth}")
        print("================================\n", flush=True)

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print("=== live_like_sim.py finished ===", flush=True)

if __name__ == "__main__":
    main()



# python3 live_like_sim.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --window-rows 4000 \
#   --tail-iters 4000 \
#   --verbose
