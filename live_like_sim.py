#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-like rolling simulation with 4 TFs (5m/15m/30m/1h).

- در هر تکرار:
  1) کات کردن CSVها تا بازه [start .. cutoff=t] (شامل t)
  2) ساخت فیچرها در حالت predict با منطق بدون لیکیج (shift(1).diff)
  3) ❗️بدون حذف ردیف آخر (predict_drop_last=False) تا ردیف آخر X همان «سیگنال در t» باشد
  4) گرفتن p_last و نگاشت به {BUY/SELL/NONE} با آستانه‌ها
  5) سنجش با y_true = 1{ close(t+1) > close(t) }
  6) چاپ لاگ‌های جزئی و جمع‌بند‌ی

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

# --- اگر _CompatWrapper قدیمی باشد، برای ایمنی پچِ سبک ---
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
        _CW.__getattr__ = _cw_safe_getattr
        _CW.__getstate__ = _cw_getstate
        _CW.__setstate__ = _cw_setstate
except Exception:
    pass

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}

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

def parse_args():
    p = argparse.ArgumentParser(description="Live-like rolling backtest")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--model-dir", default=".")
    p.add_argument("--window-rows", type=int, default=4000,
                   help="طول پنجرهٔ رولتایم روی 30T (ردیف).")
    p.add_argument("--tail-iters", type=int, default=4000,
                   help="فقط K تکرار آخر را اجرا کن.")
    p.add_argument("--keep-tmp", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

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

    # 2) Load raw
    raw_df: Dict[str, pd.DataFrame] = {}
    for tf, pth in base_csvs.items():
        if not os.path.isfile(pth):
            if tf == "30T":
                print(f"[FATAL] Main TF file missing: {pth}", flush=True)
                return
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

    # 3) Load model payload
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

    warmup_rows = (window - 1) + 2  # window-stack + (shift(1).diff)
    log.info("Model loaded: window=%d thr=(neg=%.3f,pos=%.3f) final_cols=%d warmup_rows=%d",
             window, neg_thr, pos_thr, len(final_cols), warmup_rows)

    # 4) Iter bounds
    N = len(main_df)
    need_min = args.window_rows + 2
    if N < need_min:
        print(f"[FATAL] Not enough 30T rows: have={N}, need>={need_min}", flush=True)
        return

    base_start = args.window_rows - 1   # inclusive
    end_idx    = N - 2                  # need i+1 for truth
    start_idx  = max(base_start, end_idx - (args.tail_iters or (end_idx-base_start+1)) + 1)
    total_iters = end_idx - start_idx + 1
    print(f"[INFO] N={N} | start_idx={start_idx} | end_idx={end_idx} | total_iters={total_iters}", flush=True)
    if total_iters <= 0:
        print("[FATAL] Nothing to simulate. Adjust --tail-iters/--window-rows.", flush=True)
        return

    # 5) Accumulators
    wins = losses = unpred = preds = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0

    # 6) Temp dir
    tmp_root = tempfile.mkdtemp(prefix="live_like_")
    log.info("Temp root: %s", tmp_root)

    try:
        for k in range(total_iters):
            i = start_idx + k
            cutoff_time = pd.to_datetime(main_df.loc[i, "time"])
            start_time  = pd.to_datetime(main_df.loc[i - (args.window_rows - 1), "time"])

            # Per-iteration cut
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            os.makedirs(iter_dir, exist_ok=True)
            tmp_paths = {}
            for tf, df in raw_df.items():
                sub = cut_by_time(df, start_time, cutoff_time)
                if sub.empty: continue
                out_path = os.path.join(iter_dir, f"{args.symbol}_{tf}.csv")
                cols = ["time"] + [c for c in sub.columns if c != "time"]
                sub[cols].to_csv(out_path, index=False)
                tmp_paths[tf] = out_path

            if "30T" not in tmp_paths:
                log.warning("No 30T rows at cutoff %s (skip)", cutoff_time)
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            # Prepare (STRICT disk feed), NO drop-last
            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths=tmp_paths, main_timeframe="30T",
                verbose=False, fast_mode=True, strict_disk_feed=True
            )
            merged = prep.load_data()
            if merged.empty:
                log.warning("Merged empty at cutoff %s (skip)", cutoff_time)
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            # with_times=True → برای چک هم‌ترازی
            X_live, _, _, _, t_idx = prep.ready(
                merged,
                window=window,
                selected_features=final_cols,
                mode="predict",
                with_times=True,
                predict_drop_last=False   # ❗️ مهم: آخرین ردیف همان t است
            )
            if X_live.empty or t_idx is None or len(t_idx) == 0:
                log.warning("X_live/t_idx empty at cutoff %s (skip)", cutoff_time)
                if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                continue

            # اطمینان از هم‌ترازی: آخرین زمان فیچر باید = cutoff باشد
            feat_time = pd.to_datetime(t_idx.iloc[-1])
            if pd.isna(feat_time) or feat_time != cutoff_time:
                log.warning("Feature time misaligned (feat=%s vs cutoff=%s). Skipped.",
                            feat_time, cutoff_time)
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

            # Truth for [t → t+1] بر اساس 30T اصلی
            c0 = float(main_df.loc[i,   "close"])
            c1 = float(main_df.loc[i+1, "close"])
            y_true = 1 if (c1 - c0) > 0 else 0

            # Update metrics
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

            print(
                f"[{k+1:>5}/{total_iters}] @ {cutoff_time}  "
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
