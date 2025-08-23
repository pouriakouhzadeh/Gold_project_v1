#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily half-hour predictions for a given day.
For each cutoff t in that day (30-minute bars), predict the next interval [t -> t+1].

- Cuts 4 TF CSVs to [.. t] inclusive for each t.
- Features are built in predict mode, strict_disk_feed=True (no drift scan).
- Default: --no-predict-drop-last  → X_last_time == t → predicts [t -> t+1].
- Optional: --predict-drop-last    → X_last_time == t-Δ → predicts [t-Δ -> t].

It also prints per-cutoff line and a day summary; if the next 30T bar exists,
it evaluates WIN/LOSS (backtest-style check for that historical day).

Example:
python3 live_day_predict.py \
  --data-dir /home/pouria/gold_project9 \
  --symbol XAUUSD \
  --model-dir /home/pouria/gold_project9 \
  --day 2025-03-05 \
  --no-predict-drop-last \
  --verbose
"""

from __future__ import annotations
import os, sys, tempfile, shutil, argparse, warnings, logging
from typing import Dict, List, TextIO
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.setrecursionlimit(200_000)

# ---------- optional dual logging to file (truncate on each run) ----------
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

def _install_dual_logging(log_path: str | None, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("day-predict")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers): logger.removeHandler(h)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        f = open(log_path, "w", encoding="utf-8", buffering=1)
        sys._orig_stdout = getattr(sys, "_orig_stdout", sys.stdout)
        sys._orig_stderr = getattr(sys, "_orig_stderr", sys.stderr)
        sys.stdout = _Tee(sys._orig_stdout, f)
        sys.stderr = _Tee(sys._orig_stderr, f)
        logger._tee_log_file = f  # type: ignore[attr-defined]

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(sh)
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

# ---------- guard for _CompatWrapper during unpickle ----------
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

def expect_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        for c in ("Time", "timestamp", "datetime", "Date"):
            if c in df.columns:
                df = df.rename(columns={c: "time"}); break
    if "time" not in df.columns:
        raise KeyError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def cut_to(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return df.loc[df["time"] <= cutoff].copy()

def decide(prob: float, neg_thr: float, pos_thr: float) -> int:
    if prob <= neg_thr: return 0
    if prob >= pos_thr: return 1
    return -1

def parse_args():
    p = argparse.ArgumentParser(description="Predict next 30-min interval for each bar in a given day")
    p.add_argument("--data-dir", default="/home/pouria/gold_project9")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--model-dir", default=".")
    p.add_argument("--day", required=True, help="Date like 2025-03-05 (local tz assumed)")
    p.add_argument("--verbose", action="store_true")
    # mutually-exclusive toggle for drop-last
    gx = p.add_mutually_exclusive_group()
    gx.add_argument("--predict-drop-last", dest="predict_drop_last", action="store_true",
                    help="Use X at t-1 → predicts [t-1→t].")
    gx.add_argument("--no-predict-drop-last", dest="predict_drop_last", action="store_false",
                    help="Use X at t (default) → predicts [t→t+1].")
    p.set_defaults(predict_drop_last=False)
    p.add_argument("--log-file", default=None, help="If set, mirror console to this file (overwritten each run).")
    p.add_argument("--out-csv", default=None, help="Optional path to write a CSV of all predictions.")
    return p.parse_args()

def main():
    args = parse_args()
    log = _install_dual_logging(args.log_file, args.verbose)
    try:
        day = pd.to_datetime(args.day).normalize()
        log.info("=== live_day_predict starting for day=%s (drop_last=%s) ===", day.date(), args.predict_drop_last)

        # 1) load full CSVs once
        base = {
            "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
            "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
            "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
            "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
        }
        raw_df: Dict[str, pd.DataFrame] = {}
        for tf, path in base.items():
            if not os.path.isfile(path):
                if tf == "30T":
                    log.error("Main TF missing: %s", path); return
                log.warning("Missing TF %s (%s) – continuing with remaining TFs.", tf, path)
                continue
            raw_df[tf] = expect_cols(pd.read_csv(path))
            log.info("Loaded TF=%s rows=%d", tf, len(raw_df[tf]))

        if "30T" not in raw_df:
            log.error("No main 30T data; abort."); return

        main_df = raw_df["30T"]
        mask_day = main_df["time"].dt.normalize() == day
        cutoffs: List[pd.Timestamp] = list(main_df.loc[mask_day, "time"])
        if not cutoffs:
            log.error("No 30T bars found for the requested day."); return

        # 2) load model artefacts
        model_path = os.path.join(args.model_dir, "best_model.pkl")
        if not os.path.isfile(model_path):
            log.error("best_model.pkl not found at: %s", model_path); return
        try:
            payload = joblib.load(model_path)
        except RecursionError:
            sys.setrecursionlimit(1_000_000)
            payload = joblib.load(model_path)

        pipeline   = payload["pipeline"]
        window     = int(payload.get("window_size", 1))
        neg_thr    = float(payload.get("neg_thr", 0.5))
        pos_thr    = float(payload.get("pos_thr", 0.5))
        final_cols = payload.get("train_window_cols") or payload.get("feats") or []
        if not isinstance(final_cols, list): final_cols = list(final_cols)

        for need in ("predict_proba", "predict"):
            if not hasattr(pipeline, need):
                log.error("Loaded pipeline lacks `%s` method.", need); return

        log.info("Model loaded: window=%d thr=(neg=%.3f,pos=%.3f) final_cols=%d",
                 window, neg_thr, pos_thr, len(final_cols))

        # 3) iterate all cutoffs in the day
        tmp_root = tempfile.mkdtemp(prefix="day_predict_")
        results = []
        buys=sells=none=preds=wins=losses=unpred=0

        try:
            for k, cutoff in enumerate(cutoffs, 1):
                # write per-cutoff truncated CSVs
                iter_dir = os.path.join(tmp_root, f"iter_{k:03d}"); os.makedirs(iter_dir, exist_ok=True)
                fps = {}
                for tf, df in raw_df.items():
                    sub = df.loc[df["time"] <= cutoff]
                    if sub.empty: continue
                    out = os.path.join(iter_dir, f"{args.symbol}_{tf}.csv")
                    cols = ["time"] + [c for c in sub.columns if c != "time"]
                    sub[cols].to_csv(out, index=False)
                    fps[tf] = out
                if "30T" not in fps:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    continue

                prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T",
                                              verbose=False, fast_mode=True, strict_disk_feed=True)
                merged = prep.load_data()
                if merged.empty:
                    shutil.rmtree(iter_dir, ignore_errors=True)
                    continue

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
                    continue

                X = X.reindex(columns=final_cols, fill_value=0.0)
                x_time = pd.to_datetime(t_idx.iloc[-1]) if (t_idx is not None and len(t_idx)>0) else None

                prob = float(pipeline.predict_proba(X.tail(1))[:, 1])
                pred = decide(prob, neg_thr, pos_thr)
                decision = {1: "BUY ", 0: "SELL", -1: "NONE"}[pred]

                # optional truth for historical evaluation
                verdict = "UNPRED"
                y_true = None
                # find this cutoff index in full main_df
                idx = main_df.index[main_df["time"] == cutoff]
                if len(idx) == 1 and (idx[0] + 1) < len(main_df):
                    c0 = float(main_df.loc[idx[0], "close"])
                    c1 = float(main_df.loc[idx[0] + 1, "close"])
                    y_true = 1 if (c1 - c0) > 0 else 0

                if pred == -1:
                    none += 1; unpred += 1
                else:
                    preds += 1
                    if pred == 1: buys += 1
                    else:         sells += 1
                    if y_true is not None:
                        if pred == y_true: wins += 1; verdict = "WIN"
                        else:              losses += 1; verdict = "LOSS"

                acc = (wins / preds * 100.0) if preds > 0 else 0.0
                cov = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0

                print(f"[{k:02d}/{len(cutoffs)}] cutoff={cutoff}  X_last_time={x_time}  "
                      f"prob={prob:.3f} → pred={decision}"
                      + (f"  true={y_true} → {verdict}" if y_true is not None else "")
                      + f"  | cum P={preds} W={wins} L={losses} U={unpred} Acc={acc:.2f}% Cov={cov:.2f}%")

                results.append({
                    "cutoff": cutoff, "X_last_time": x_time,
                    "prob_up": prob, "decision": decision,
                    "y_true": y_true, "verdict": verdict
                })

                shutil.rmtree(iter_dir, ignore_errors=True)

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        # day summary
        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        cov = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
        print("\n========== DAY SUMMARY ==========")
        print(f"Day: {day.date()} | cutoffs seen: {len(cutoffs)}")
        print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
        print(f"BUY: {buys} | SELL: {sells} | NONE: {none}")
        print(f"Accuracy (predicted only): {acc:.2f}% | Coverage: {cov:.2f}%")
        print("=================================\n")

        # optional CSV
        if args.out_csv:
            out_df = pd.DataFrame(results)
            out_df.to_csv(args.out_csv, index=False)
            print(f"Saved predictions to: {os.path.abspath(args.out_csv)}")

        log.info("=== live_day_predict finished ===")

    finally:
        _close_dual_logging(log)

if __name__ == "__main__":
    main()
