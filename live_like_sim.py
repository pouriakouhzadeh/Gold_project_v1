#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-like rolling simulation with 4 TFs (5m/15m/30m/1h) + dual logging (console & file).
- Cuts raw CSVs to [start .. cutoff] (inclusive).
- Builds features in predict mode with predict_drop_last=True (drop unstable last row).
- Predicts last stable row (t-1) as the signal to act for [t -> t+1] per your requested logic.
- Compares vs true next-bar direction on 30T.
- Streams detailed logs to console AND to live_like_sim.log (overwritten each run).

Usage example:
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
from typing import Dict, List, TextIO
import numpy as np
import pandas as pd
import joblib

# -------------------- Dual logging (console + file) --------------------
class _Tee:
    """Duplicate writes to multiple text streams (e.g., console + file)."""
    def __init__(self, *streams: TextIO):
        self._streams = streams
    def write(self, data: str):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

def _install_dual_logging(log_path: str, verbose: bool) -> logging.Logger:
    """
    - Truncates log file on each run.
    - Mirrors EVERYTHING printed to console into the file by tee-ing stdout/stderr.
    - Sets up a logging.Logger that writes to console (which is now tee'd).
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    sys._orig_stdout = getattr(sys, "_orig_stdout", sys.stdout)
    sys._orig_stderr = getattr(sys, "_orig_stderr", sys.stderr)
    sys.stdout = _Tee(sys._orig_stdout, log_file)
    sys.stderr = _Tee(sys._orig_stderr, log_file)

    logger = logging.getLogger("live-like")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    sh = logging.StreamHandler(stream=sys.stdout)  # goes to tee -> console+file
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(sh)

    logger._tee_log_file = log_file  # type: ignore[attr-defined]
    return logger

def _close_dual_logging(logger: logging.Logger) -> None:
    """Restore stdio and close log file cleanly."""
    try:
        if hasattr(logger, "_tee_log_file"):
            logger._tee_log_file.flush()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stdout = getattr(sys, "_orig_stdout", sys.stdout)
        sys.stderr = getattr(sys, "_orig_stderr", sys.stderr)
    finally:
        try:
            if hasattr(logger, "_tee_log_file"):
                logger._tee_log_file.close()  # type: ignore[attr-defined]
        except Exception:
            pass

# -------------------- RecursionError guard for joblib --------------------
sys.setrecursionlimit(200_000)
warnings.filterwarnings("ignore")

# --- Project imports ---
# Patch _CompatWrapper to avoid recursive getattr during unpickle
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

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

TF_MAP = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live-like rolling backtest (dual logging)")
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
    p.add_argument("--log-file", default="live_like_sim.log",
                   help="Path to log file (default: ./live_like_sim.log)")
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

# -------------------- main --------------------
def main():
    args = parse_args()

    # Dual-logging (truncate file, tee stdout/stderr)
    log = _install_dual_logging(args.log_file, args.verbose)
    try:
        log.info("=== live_like_sim.py starting ===")
        log.info("data-dir=%s | symbol=%s | model-dir=%s", args.data_dir, args.symbol, args.model_dir)
        log.info("window_rows=%d | tail_iters=%d | log_file=%s",
                 args.window_rows, args.tail_iters, os.path.abspath(args.log_file))

        # 1) Resolve CSV paths
        base_csvs = {
            "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
            "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
            "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
            "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
        }
        for tf, path in base_csvs.items():
            log.info("CSV[%s] => %s (exists=%s)", tf, path, os.path.isfile(path))

        # 2) Read raw CSVs
        raw_df: Dict[str, pd.DataFrame] = {}
        for tf, path in base_csvs.items():
            if not os.path.isfile(path):
                if tf == "30T":
                    log.error("Main TF file missing: %s", path); return
                else:
                    log.warning("Missing CSV for %s: %s (TF skipped)", tf, path)
                    continue
            try:
                df = pd.read_csv(path)
                df = expect_cols(df)
            except Exception as e:
                log.error("Failed to read/parse %s: %s", path, e); return
            raw_df[tf] = df
            log.info("Loaded %s rows for TF=%s", len(df), tf)

        if "30T" not in raw_df:
            log.error("30T data not loaded; abort."); return
        main_df = raw_df["30T"]

        # 3) Load model artefacts
        model_path = os.path.join(args.model_dir, "best_model.pkl")
        log.info("Loading model artefacts: %s", model_path)
        if not os.path.isfile(model_path):
            log.error("best_model.pkl not found at: %s", model_path)
            try: log.info("Directory listing: %s", os.listdir(args.model_dir))
            except Exception: pass
            return

        try:
            import sklearn  # noqa
            import imblearn # noqa
            from sklearnex import patch_sklearn
            patch_sklearn(verbose=False)
        except Exception:
            pass

        try:
            payload = joblib.load(model_path)
        except RecursionError as e:
            log.warning("RecursionError on joblib.load: %s. Retrying with higher limit…", e)
            sys.setrecursionlimit(1_000_000)
            payload = joblib.load(model_path)
        except Exception as e:
            log.error("joblib.load failed: %s", e); return

        try:
            pipeline   = payload["pipeline"]
            window     = int(payload.get("window_size", 1))
            neg_thr    = float(payload.get("neg_thr", 0.5))
            pos_thr    = float(payload.get("pos_thr", 0.5))
            final_cols = payload.get("train_window_cols") or payload.get("feats") or []
            if not isinstance(final_cols, list):
                final_cols = list(final_cols)
        except Exception as e:
            log.error("Invalid model payload structure: %s", e); return

        for need in ("predict_proba", "predict"):
            if not hasattr(pipeline, need):
                log.error("Loaded pipeline lacks `%s` method.", need); return

        log.info("Model loaded: window=%d thr=(neg=%.3f,pos=%.3f) final_cols=%d",
                 window, neg_thr, pos_thr, len(final_cols))

        # 4) Compute iteration bounds
        N = len(main_df)
        need_min = args.window_rows + 2
        if N < need_min:
            log.error("Not enough 30T rows: have=%d, need>=%d", N, need_min); return

        base_start = args.window_rows - 1     # inclusive
        end_idx    = N - 2                    # need i+1 for truth
        if args.tail_iters and args.tail_iters > 0:
            start_idx = max(base_start, end_idx - args.tail_iters + 1)
        else:
            start_idx = base_start

        total_iters = end_idx - start_idx + 1
        print(f"[INFO] N={N} | start_idx={start_idx} | end_idx={end_idx} | total_iters={total_iters}")
        if total_iters <= 0:
            log.error("Nothing to simulate (total_iters<=0). Reduce --tail-iters or window."); return

        # 5) Accumulators
        wins = losses = unpred = preds = 0
        tp = tn = fp = fn = 0
        buy_n = sell_n = none_n = 0

        # 6) Temp directory
        tmp_root = tempfile.mkdtemp(prefix="live_like_")
        log.info("Temp root: %s", tmp_root)

        try:
            for k in range(total_iters):
                i = start_idx + k
                cutoff_time = pd.to_datetime(main_df.loc[i, "time"])
                start_time  = pd.to_datetime(main_df.loc[i - (args.window_rows - 1), "time"])

                # Per-iteration CSVs up to cutoff (inclusive)
                iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
                os.makedirs(iter_dir, exist_ok=True)
                tmp_paths = {}
                for tf, df in raw_df.items():
                    sub = cut_by_time(df, start_time, cutoff_time)
                    if sub.empty: continue
                    out_name = f"{args.symbol}_{tf}.csv"
                    out_path = os.path.join(iter_dir, out_name)
                    cols = ["time"] + [c for c in sub.columns if c != "time"]
                    sub[cols].to_csv(out_path, index=False)
                    tmp_paths[tf] = out_path

                if "30T" not in tmp_paths:
                    logging.getLogger("live-like").warning("No 30T rows at cutoff %s (skip)", cutoff_time)
                    if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                    continue

                # Prepare features (predict mode, strict feed) and DROP unstable last row (per requested logic)
                prep = PREPARE_DATA_FOR_TRAIN(
                    filepaths=tmp_paths, main_timeframe="30T",
                    verbose=False, fast_mode=True, strict_disk_feed=True
                )
                merged = prep.load_data()
                if merged.empty:
                    logging.getLogger("live-like").warning("Merged empty at cutoff %s (skip)", cutoff_time)
                    if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                    continue

                X_live, _, _, _ = prep.ready(
                    merged,
                    window=window,
                    selected_features=final_cols,
                    mode="predict",
                    predict_drop_last=True
                )
                if X_live.empty:
                    logging.getLogger("live-like").warning("X_live empty at cutoff %s (skip)", cutoff_time)
                    if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                    continue

                if not final_cols:
                    final_cols = list(X_live.columns)
                    logging.getLogger("live-like").warning("final_cols was empty; using X_live columns (%d).", len(final_cols))

                X_live = X_live.reindex(columns=final_cols, fill_value=0.0)

                try:
                    probs = pipeline.predict_proba(X_live)[:, 1]
                except Exception:
                    probs = pipeline.predict_proba(X_live.values)[:, 1]

                p_last = float(probs[-1])
                pred = decide(p_last, neg_thr, pos_thr)

                # Ground truth for next 30T bar
                c0 = float(main_df.loc[i,   "close"])
                c1 = float(main_df.loc[i+1, "close"])
                y_true = 1 if (c1 - c0) > 0 else 0

                # Update counters
                if pred == -1:
                    unpred += 1
                    none_n += 1
                    verdict = "UNPRED"
                else:
                    preds += 1
                    if pred == 1: buy_n += 1
                    else:         sell_n += 1

                    if pred == y_true:
                        wins += 1
                        verdict = "WIN"
                        if y_true == 1: tp += 1
                        else:           tn += 1
                    else:
                        losses += 1
                        verdict = "LOSS"
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
                    f"| buys={buy_n} sells={sell_n} none={none_n}"
                )

                if not args.keep_tmp:
                    shutil.rmtree(iter_dir, ignore_errors=True)

            # Final summary
            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            coverage = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
            print("\n========== SUMMARY ==========")
            print(f"Cutoffs tested (iterations): {total_iters}")
            print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
            print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {coverage:.2f}%")
            print(f"Decisions  → BUY: {buy_n}  SELL: {sell_n}  NONE: {none_n}")
            print(f"Confusion  → TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
            print("================================\n")

        finally:
            if not args.keep_tmp:
                shutil.rmtree(tmp_root, ignore_errors=True)

        log.info("=== live_like_sim.py finished ===")

    finally:
        _close_dual_logging(log)

if __name__ == "__main__":
    main()




# python3 live_like_sim.py \
#   --data-dir /home/pouria/gold_project9 \
#   --symbol XAUUSD \
#   --model-dir /home/pouria/gold_project9 \
#   --window-rows 4000 \
#   --tail-iters 4000 \
#   --verbose
