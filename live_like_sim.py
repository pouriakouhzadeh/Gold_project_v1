#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-like rolling simulation with 4 TFs (5m/15m/30m/1h) + dual logging + strict feature audit.

Key points:
- For each cutoff t, CSVs are cut to <= t (inclusive).
- Features use shift(1).diff → only past is used.
- Default: --no-predict-drop-last  → last X row is at t → predicts [t→t+1] (TRAIN-like).
- Optional: --predict-drop-last    → last X row is t-Δ → predicts [t-Δ→t] (stable-by-construction).
- STRICT FEATURE AUDIT:
    * By default, if any training feature column is missing in live X, we ABORT and list them.
    * If you pass --allow-missing-cols, we try to fill missing columns with TRAIN MEDIANS
      (from train_distribution.json saved during training). If median not found → error.
- Writes everything to console AND to live_like_sim.log (file truncated each run).

Usage example:
python3 live_like_sim.py \
  --data-dir /home/pouria/gold_project9 \
  --symbol XAUUSD \
  --model-dir /home/pouria/gold_project9 \
  --window-rows 4000 \
  --tail-iters 4000 \
  --no-predict-drop-last \
  --align-debug 5 \
  --verbose
"""

from __future__ import annotations
import os, sys, shutil, tempfile, warnings, argparse, logging, json
from typing import Dict, List, TextIO, Optional
import numpy as np
import pandas as pd
import joblib

# -------------------- Dual logging (console + file; truncate) --------------------
class _Tee:
    def __init__(self, *streams: TextIO): self._streams = streams
    def write(self, data: str):
        for s in self._streams:
            try: s.write(data)
            except Exception: pass
        self.flush()
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass

def _install_dual_logging(log_path: str, verbose: bool) -> logging.Logger:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # truncate each run

    sys._orig_stdout = getattr(sys, "_orig_stdout", sys.stdout)
    sys._orig_stderr = getattr(sys, "_orig_stderr", sys.stderr)
    sys.stdout = _Tee(sys._orig_stdout, log_file)
    sys.stderr = _Tee(sys._orig_stderr, log_file)

    logger = logging.getLogger("live-like")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers): logger.removeHandler(h)
    sh = logging.StreamHandler(stream=sys.stdout)  # goes to tee
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(sh)
    logger._tee_log_file = log_file  # type: ignore[attr-defined]
    return logger

def _close_dual_logging(logger: logging.Logger) -> None:
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

# -------------------- RecursionError guard for joblib (_CompatWrapper) --------------------
warnings.filterwarnings("ignore")
sys.setrecursionlimit(200_000)
try:
    import model_pipeline
    if hasattr(model_pipeline, "_CompatWrapper"):
        _CW = model_pipeline._CompatWrapper
        def _cw_safe_getattr(self, item):
            if item.startswith("__") and item.endswith("__"): raise AttributeError(item)
            try: target = object.__getattribute__(self, "_model")
            except Exception: raise AttributeError(item)
            if target is self or target is None: raise AttributeError(item)
            return getattr(target, item)
        def _cw_getstate(self): return {"_model": getattr(self, "_model", None),
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
    p = argparse.ArgumentParser(description="Live-like rolling backtest (strict feature audit)")

    p.add_argument("--data-dir", default="/home/pouria/gold_project9")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--model-dir", default=".")
    p.add_argument("--window-rows", type=int, default=4000,
                   help="Rolling window length over 30T (rows).")
    p.add_argument("--tail-iters", type=int, default=4000,
                   help="Run ONLY the last K iterations (K last cutoffs).")
    p.add_argument("--keep-tmp", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", default="live_like_sim.log")

    # drop-last toggle (mutually exclusive)
    gx = p.add_mutually_exclusive_group()
    gx.add_argument("--predict-drop-last", dest="predict_drop_last", action="store_true",
                    help="Use stable t-Δ row → predicts [t-Δ→t].")
    gx.add_argument("--no-predict-drop-last", dest="predict_drop_last", action="store_false",
                    help="Use last row at t (default) → predicts [t→t+1].")
    p.set_defaults(predict_drop_last=False)

    # alignment & audit helpers
    p.add_argument("--align-debug", type=int, default=0,
                   help="Print first N alignment lines: cutoff, X_last_time, drop_last flag.")
    p.add_argument("--allow-missing-cols", action="store_true",
                   help="If some training columns are missing in live X, fill by TRAIN MEDIAN (from train_distribution.json) instead of aborting.")

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
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

def _load_train_medians(model_dir: str, payload: dict) -> Optional[dict]:
    # payload may contain "train_distribution" path
    path = payload.get("train_distribution", None)
    if not path:
        path = "train_distribution.json"
    if not os.path.isabs(path):
        path = os.path.join(model_dir, path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        # expected formats (support a couple):
        # 1) {"columns": {"feat": {"median": 0.123, ...}, ...}}
        # 2) {"feat": {"median": ...}, ...}
        # 3) {"feat": {"q50": ...}, ...}
        med = {}
        if "columns" in j and isinstance(j["columns"], dict):
            src = j["columns"]
        else:
            src = j
        for c, stats in src.items():
            if isinstance(stats, dict):
                if "median" in stats: med[c] = stats["median"]
                elif "q50" in stats:  med[c] = stats["q50"]
        return med if med else None
    except Exception:
        return None

# -------------------- main --------------------
def main():
    args = parse_args()
    log = _install_dual_logging(args.log_file, args.verbose)

    try:
        log.info("=== live_like_sim.py starting ===")
        log.info("data-dir=%s | symbol=%s | model-dir=%s", args.data_dir, args.symbol, args.model_dir)
        log.info("window_rows=%d | tail_iters=%d | log_file=%s",
                 args.window_rows, args.tail_iters, os.path.abspath(args.log_file))

        # 1) Resolve and load CSVs
        base_csvs = {
            "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
            "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
            "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
            "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
        }
        raw_df: Dict[str, pd.DataFrame] = {}
        for tf, path in base_csvs.items():
            log.info("CSV[%s] => %s (exists=%s)", tf, path, os.path.isfile(path))
            if not os.path.isfile(path):
                if tf == "30T":
                    log.error("Main TF file missing: %s", path); return
                log.warning("Missing CSV for %s; skipping that TF.", tf)
                continue
            try:
                raw_df[tf] = expect_cols(pd.read_csv(path))
            except Exception as e:
                log.error("Failed to read/parse %s: %s", path, e); return
            log.info("Loaded %d rows for TF=%s", len(raw_df[tf]), tf)

        if "30T" not in raw_df:
            log.error("30T data not loaded; abort."); return
        main_df = raw_df["30T"]

        # 2) Load model artefacts
        model_path = os.path.join(args.model_dir, "best_model.pkl")
        log.info("Loading model artefacts: %s", model_path)
        if not os.path.isfile(model_path):
            log.error("best_model.pkl not found at: %s", model_path); return
        try:
            payload = joblib.load(model_path)
        except RecursionError as e:
            log.warning("RecursionError on joblib.load: %s. Retrying with higher limit…", e)
            sys.setrecursionlimit(1_000_000)
            payload = joblib.load(model_path)

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

        train_medians = _load_train_medians(args.model_dir, payload)
        if train_medians is None:
            log.info("Train medians not found; strict audit still enforced.")
        else:
            log.info("Train medians loaded for %d columns.", len(train_medians))

        log.info("Model loaded: window=%d thr=(neg=%.3f,pos=%.3f) final_cols=%d",
                 window, neg_thr, pos_thr, len(final_cols))

        # 3) Compute iteration bounds
        N = len(main_df)
        need_min = args.window_rows + 2
        if N < need_min:
            log.error("Not enough 30T rows: have=%d, need>=%d", N, need_min); return

        base_start = args.window_rows - 1
        end_idx    = N - 2
        start_idx  = max(base_start, end_idx - args.tail_iters + 1) if (args.tail_iters and args.tail_iters > 0) else base_start
        total_iters = end_idx - start_idx + 1
        print(f"[INFO] N={N} | start_idx={start_idx} | end_idx={end_idx} | total_iters={total_iters}")
        if total_iters <= 0:
            log.error("Nothing to simulate (total_iters<=0)."); return

        # 4) accumulators
        wins = losses = unpred = preds = 0
        tp = tn = fp = fn = 0
        buy_n = sell_n = none_n = 0

        # 5) loop
        tmp_root = tempfile.mkdtemp(prefix="live_like_")
        log.info("Temp root: %s", tmp_root)

        try:
            for k in range(total_iters):
                i = start_idx + k
                cutoff = pd.to_datetime(main_df.loc[i, "time"])

                # write per-iteration truncated CSVs to <= cutoff
                iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
                os.makedirs(iter_dir, exist_ok=True)
                fps = {}
                for tf, df in raw_df.items():
                    sub = df.loc[df["time"] <= cutoff]
                    if sub.empty: continue
                    out = os.path.join(iter_dir, f"{args.symbol}_{tf}.csv")
                    cols = ["time"] + [c for c in sub.columns if c != "time"]
                    sub[cols].to_csv(out, index=False)
                    fps[tf] = out
                if "30T" not in fps:
                    if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                    continue

                # build features
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
                    selected_features=final_cols,   # enforce the very same columns
                    mode="predict",
                    with_times=True,
                    predict_drop_last=args.predict_drop_last
                )
                if X.empty:
                    if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                    continue

                # ---- STRICT FEATURE AUDIT ----
                live_cols = list(X.columns)
                want = list(final_cols)
                missing = [c for c in want if c not in live_cols]
                extra   = [c for c in live_cols if c not in want]

                if missing:
                    if not args.allow_missing_cols:
                        print(f"[FATAL] {len(missing)} training columns are missing in live X. "
                              f"First 10: {missing[:10]}")
                        if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                        return
                    # try to fill by train medians
                    if train_medians is None:
                        print(f"[FATAL] Missing columns ({len(missing)}) but train medians not available. Aborting.")
                        if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                        return
                    fill_vals = {}
                    for c in missing:
                        if c in train_medians and np.isfinite(train_medians[c]):
                            fill_vals[c] = float(train_medians[c])
                        else:
                            print(f"[FATAL] No median for missing column: {c}")
                            if not args.keep_tmp: shutil.rmtree(iter_dir, ignore_errors=True)
                            return
                    # reindex with medians for missing
                    X = X.reindex(columns=want, fill_value=np.nan)
                    for c, v in fill_vals.items():
                        X[c] = v
                    # any remaining NaN? fill with 0 but warn
                    if X.isna().any().any():
                        n = int(X.isna().sum().sum())
                        print(f"[WARN] {n} NaNs after median fill; filling with 0 as fallback.")
                        X = X.fillna(0.0)
                else:
                    # exact column set; just reorder
                    X = X[want]

                if args.align_debug and k < args.align_debug:
                    x_time = None
                    try:
                        if t_idx is not None and len(t_idx) > 0:
                            x_time = pd.to_datetime(t_idx.iloc[-1])
                    except Exception:
                        pass
                    print(f"[ALIGN] cutoff={cutoff} | X_last_time={x_time} | drop_last={args.predict_drop_last}")

                # predict last row prob
                try:
                    prob = float(pipeline.predict_proba(X.tail(1))[:, 1])
                except Exception:
                    prob = float(pipeline.predict_proba(X.tail(1).values)[:, 1])

                pred = decide(prob, neg_thr, pos_thr)

                # truth from raw main (cutoff i → compare close[i] vs close[i+1])
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
                coverage = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
                dec_label = {1: "BUY ", 0: "SELL", -1: "NONE"}[pred]

                print(
                    f"[{k+1:>5}/{total_iters}] @cutoff={cutoff}  "
                    f"p={prob:.3f} → pred={dec_label}  true={y_true} → {verdict}   "
                    f"| cum P={preds} W={wins} L={losses} U={unpred} "
                    f"Acc={acc:.2f}% Cover={coverage:.2f}%  "
                    f"| buys={buy_n} sells={sell_n} none={none_n}"
                )

                if not args.keep_tmp:
                    shutil.rmtree(iter_dir, ignore_errors=True)

            # summary
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
