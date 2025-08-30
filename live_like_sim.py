#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic REPLAY of TRAIN vs. optional LIVE-like rolling.
-------------------------------------------------------------
● حالت پیش‌فرض: --mode replay
    - همان مسیر TRAIN را دقیقاً بازپخش می‌کند:
      * همان PREPARE_DATA_FOR_TRAIN با تنظیمات TRAIN (بدون fast/strict)
      * همان window و همان ستون‌های پس از پنجره‌بندی (train_window_cols)
      * label از خودِ prep.ready(..., mode="train") می‌آید (نه از close)
      * یکبار predict_proba روی کل X → آستانه‌گذاری → گزارش iteration-by-iteration
    - نتیجه باید با «دقت آموزش» هم‌ارز باشد (مثلاً ~99٪).

● حالت اختیاری: --mode live
    - همان نسخه‌ی tail-only شما (بدون تضمین برابر بودن با TRAIN).
"""

from __future__ import annotations

import os, sys, argparse, logging, json, shutil, tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# ========================= Logging (console + file) =========================

def setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("sim")
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

# ========================= Compat wrapper guard (joblib) =========================
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
    p = argparse.ArgumentParser(description="Replay TRAIN exactly (default) or run LIVE-like rolling.")
    p.add_argument("--mode", choices=["replay", "live"], default="replay",
                   help="replay: بازپخشِ آموزش (پیش‌فرض) · live: شبیه‌ساز تِیل-اُولی.")

    p.add_argument("--data-dir", default="/home/pouria/gold_project9", help="پوشهٔ CSVهای خام.")
    p.add_argument("--symbol", default="XAUUSD", help="نماد.")
    p.add_argument("--model-dir", default=".", help="پوشهٔ مدل (حاوی best_model.pkl).")
    p.add_argument("--log-file", default="live_like_sim.log", help="فایل لاگ خروجی.")
    p.add_argument("--verbose", action="store_true", help="لاگِ پرجزئیات.")

    # فقط برای گزارشِ iteration window/tail:
    p.add_argument("--tail-iters", type=int, default=4000, help="برای گزارش فقط N نمونهٔ انتهایی را لاگ کن.")

    # فقط در حالت live:
    p.add_argument("--predict-drop-last", dest="predict_drop_last", action="store_true")
    p.add_argument("--no-predict-drop-last", dest="predict_drop_last", action="store_false")
    p.set_defaults(predict_drop_last=True)
    p.add_argument("--align-debug", type=int, default=0, help="پرینت هم‌ترازسازی در چند iteration اول (live).")
    p.add_argument("--allow-missing-cols", action="store_true", help="پر کردن ستون‌های مفقود با میانهٔ TRAIN (live).")
    p.add_argument("--hist-30t", type=int, default=480,  help="tail 30T (live).")
    p.add_argument("--hist-1h",  type=int, default=240,  help="tail 1H (live).")
    p.add_argument("--hist-15t", type=int, default=960,  help="tail 15T (live).")
    p.add_argument("--hist-5t",  type=int, default=2880, help="tail 5T (live).")
    return p.parse_args()

# ========================= Helpers =========================

def load_payload(model_dir: str, log: logging.Logger):
    path = os.path.join(model_dir, "best_model.pkl")
    if not os.path.isfile(path):
        log.error("best_model.pkl not found at: %s", path); sys.exit(1)
    try:
        payload = joblib.load(path)
    except RecursionError:
        sys.setrecursionlimit(1_000_000); payload = joblib.load(path)
    need = ["pipeline", "window_size", "neg_thr", "pos_thr"]
    for k in need:
        if k not in payload:
            log.error("Model payload missing key: %s", k); sys.exit(1)
    final_cols = payload.get("train_window_cols") or payload.get("feats") or []
    if not isinstance(final_cols, list): final_cols = list(final_cols)
    return payload, final_cols

def expect_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        for cand in ("Time", "timestamp", "datetime", "Date"):
            if cand in df.columns:
                df = df.rename(columns={cand: "time"}); break
    if "time" not in df.columns:
        raise KeyError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def load_train_medians(model_dir: str, payload: dict) -> Optional[dict]:
    path = payload.get("train_distribution") or "train_distribution.json"
    if not os.path.isabs(path):
        path = os.path.join(model_dir, path)
    if not os.path.isfile(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f: j = json.load(f)
        src = j.get("columns", j); med = {}
        for c, stats in src.items():
            if isinstance(stats, dict):
                if "median" in stats: med[c] = stats["median"]
                elif "q50" in stats:  med[c] = stats["q50"]
        return med or None
    except Exception:
        return None

# ---------- LIVE helpers (tail-only disk slicing) ----------
def write_iter_csvs_tail(raw_df: Dict[str, pd.DataFrame], symbol: str, cutoff, iter_dir: str, hist_rows: Dict[str, int]) -> Dict[str, str]:
    os.makedirs(iter_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for tf, df in raw_df.items():
        sub = df.loc[df["time"] <= cutoff]
        if sub.empty: continue
        need = int(hist_rows.get(tf, 200))
        if need > 0 and len(sub) > need: sub = sub.tail(need)
        mapped = TF_MAP.get(tf, tf); out_path = os.path.join(iter_dir, f"{symbol}_{mapped}.csv")
        cols = ["time"] + [c for c in sub.columns if c != "time"]; sub[cols].to_csv(out_path, index=False)
        out[tf] = out_path
    return out

# ========================= MAIN =========================

def main():
    args = parse_args()
    log = setup_logger(args.log_file, args.verbose)
    log.info("=== sim starting: mode=%s ===", args.mode)
    log.info("data-dir=%s  symbol=%s  model-dir=%s", args.data_dir, args.symbol, args.model_dir)

    payload, final_cols = load_payload(args.model_dir, log)
    pipeline   = payload["pipeline"]
    window     = int(payload.get("window_size", 1))
    neg_thr    = float(payload.get("neg_thr", 0.5))
    pos_thr    = float(payload.get("pos_thr", 0.5))
    train_medians = load_train_medians(args.model_dir, payload)

    log.info("Model loaded. window=%d  thr=(neg=%.3f, pos=%.3f)  final_cols=%d", window, neg_thr, pos_thr, len(final_cols))

    # -------------------- REPLAY TRAIN (default) --------------------
    if args.mode == "replay":
        # همان مسیر TRAIN: بدون fast_mode و بدون strict_disk_feed
        filepaths = {
            "30T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['30T']}.csv"),
            "15T": os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['15T']}.csv"),
            "5T":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['5T']}.csv"),
            "1H":  os.path.join(args.data_dir, f"{args.symbol}_{TF_MAP['1H']}.csv"),
        }
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T", verbose=False, fast_mode=False, strict_disk_feed=False)

        merged = prep.load_data()
        if merged.empty: log.error("Merged is empty."); sys.exit(1)

        # دقیقاً مثل TRAIN: label از ready(mode='train') می‌آید
        X, y, _, price_ser, t_idx = prep.ready(
            merged,
            window=window,
            selected_features=final_cols,     # همان ستون‌های آموزش
            mode="train",
            with_times=True,
            predict_drop_last=False,
            train_drop_last=False
        )
        if isinstance(y, (pd.Series, np.ndarray)):
            y = pd.Series(y).astype(int).reset_index(drop=True)
        else:
            y = pd.Series(y, dtype=int)

        # نظم ستون‌ها را مطابق TRAIN نگه داریم
        X = X[final_cols]

        # پیش‌بینی «یکجا»، سپس لاگ iteration-به-iteration
        try:
            probs = pipeline.predict_proba(X)[:, 1]
        except Exception:
            probs = pipeline.predict_proba(X.values)[:, 1]

        # نگاشت به 3 حالت
        pred = np.full(len(probs), -1, dtype=int)
        pred[probs <= neg_thr] = 0
        pred[probs >= pos_thr] = 1

        # محدودهٔ گزارش: فقط N نمونهٔ انتهایی (tail-iters)
        total = len(pred)
        start = max(0, total - int(args.tail_iters))
        idxs  = range(start, total)

        # انباشت‌گرها
        preds = wins = losses = unpred = 0
        tp = tn = fp = fn = 0
        buy_n = sell_n = none_n = 0

        rows: List[dict] = []

        for k, i in enumerate(idxs, 1):
            cutoff = pd.to_datetime(t_idx.iloc[i]) if t_idx is not None and len(t_idx) > i else None
            p      = float(probs[i])
            pdc    = int(pred[i])
            yt     = int(y.iloc[i])
            if pdc == -1:
                unpred += 1; none_n += 1; verdict = "UNPRED"
            else:
                preds += 1
                if pdc == 1: buy_n += 1
                else:        sell_n += 1
                if pdc == yt:
                    wins += 1; verdict = "WIN"
                    if yt == 1: tp += 1
                    else:       tn += 1
                else:
                    losses += 1; verdict = "LOSS"
                    if yt == 1: fn += 1
                    else:       fp += 1

            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            cover = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
            dec_label = {1:"BUY ",0:"SELL",-1:"NONE"}[pdc]
            log.info(
                "[%5d/%5d] cutoff=%s  prob=%.3f → pred=%s  true=%d → %s  | P=%d W=%d L=%d U=%d  Acc=%.2f%% Cover=%.2f%%  | BUY=%d SELL=%d NONE=%d",
                k, len(range(start,total)), str(cuttoff:=cutoff) if cutoff is not None else "NA", p, dec_label, yt, verdict,
                preds, wins, losses, unpred, acc, cover, buy_n, sell_n, none_n
            )

            rows.append({
                "idx": i,
                "time": cutoff,
                "prob": p,
                "pred": int(pdc),
                "true": int(yt),
                "verdict": verdict
            })

        # خلاصه
        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        cover = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
        print("\n========== SUMMARY (REPLAY) ==========")
        print(f"Rows reported (tail): {len(range(start,total))} of {total}")
        print(f"Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpredicted: {unpred}")
        print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {cover:.2f}%")
        print(f"Decisions  → BUY: {buy_n}  SELL: {sell_n}  NONE: {none_n}")
        print(f"Confusion  → TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
        print("======================================\n")

        # ذخیره نتایج
        out_csv = os.path.join(args.model_dir, "sim_results.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log.info("Replay results saved → %s", out_csv)
        log.info("=== sim finished (replay) ===")
        return

    # -------------------- LIVE-like (tail-only) --------------------
    # (همان منطق قبلی شما؛ دقتش الزماً برابر TRAIN نیست)
    # بارگذاری CSVها
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
                log.error("Main TF (30T) is required. Missing: %s", path); sys.exit(1)
            else:
                log.warning("Missing CSV for TF=%s; skipping.", tf); continue
        raw_df[tf] = expect_cols(pd.read_csv(path))
        log.info("Loaded %d rows for TF=%s", len(raw_df[tf]), tf)
    main_df = raw_df["30T"]

    N = len(main_df)
    end_idx = N - 2
    start_idx = max(0, end_idx - int(args.tail_iters) + 1)
    total_iters = max(0, end_idx - start_idx + 1)
    log.info("LIVE iteration range: start_idx=%d end_idx=%d total=%d", start_idx, end_idx, total_iters)
    if total_iters <= 0: log.error("Nothing to simulate."); sys.exit(1)

    preds = wins = losses = unpred = 0
    tp = tn = fp = fn = 0
    buy_n = sell_n = none_n = 0
    rows: List[dict] = []
    tmp_root = tempfile.mkdtemp(prefix="live_like_")
    hist_rows = {"30T":args.hist_30t, "1H":args.hist_1h, "15T":args.hist_15t, "5T":args.hist_5t}
    try:
        for k in range(total_iters):
            i = start_idx + k
            cutoff = pd.to_datetime(main_df.loc[i, "time"])
            iter_dir = os.path.join(tmp_root, f"iter_{k:06d}")
            fps = write_iter_csvs_tail(raw_df, args.symbol, cutoff, iter_dir, hist_rows)
            if "30T" not in fps:
                shutil.rmtree(iter_dir, ignore_errors=True); continue

            prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T", verbose=False, fast_mode=True, strict_disk_feed=True)
            merged = prep.load_data()
            if merged.empty:
                shutil.rmtree(iter_dir, ignore_errors=True); continue

            X, _, _, _, t_idx = prep.ready(
                merged, window=window, selected_features=final_cols,
                mode="predict", with_times=True, predict_drop_last=args.predict_drop_last
            )
            if X.empty:
                shutil.rmtree(iter_dir, ignore_errors=True)
                if args.align_debug and k < args.align_debug:
                    log.debug("[ALIGN] cutoff=%s → X empty after tail.", cutoff)
                continue

            # ترتیب ستون‌ها
            X = X[final_cols]

            try:
                prob = float(pipeline.predict_proba(X.tail(1))[:, 1])
            except Exception:
                prob = float(pipeline.predict_proba(X.tail(1).values)[:, 1])

            # تصمیم با آستانه‌ها
            pdc = -1
            if prob <= neg_thr: pdc = 0
            elif prob >= pos_thr: pdc = 1

            # حقیقت: از close 30T
            c0 = float(main_df.loc[i,   "close"]); c1 = float(main_df.loc[i+1, "close"])
            yt = 1 if (c1 - c0) > 0 else 0

            if pdc == -1:
                unpred += 1; none_n += 1; verdict = "UNPRED"
            else:
                preds += 1
                if pdc == 1: buy_n += 1
                else:        sell_n += 1
                if pdc == yt:
                    wins += 1; verdict = "WIN"
                    if yt == 1: tp += 1
                    else:       tn += 1
                else:
                    losses += 1; verdict = "LOSS"
                    if yt == 1: fn += 1
                    else:       fp += 1

            acc = (wins / preds * 100.0) if preds > 0 else 0.0
            cover = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
            dec_label = {1:"BUY ",0:"SELL",-1:"NONE"}[pdc]

            # align debug
            if args.align_debug and k < args.align_debug:
                x_last = None
                try:
                    if t_idx is not None and len(t_idx) > 0:
                        x_last = pd.to_datetime(t_idx.iloc[-1])
                except Exception:
                    pass
                log.info("[ALIGN] cutoff=%s | X_last_time=%s | drop_last=%s", cutoff, x_last, args.predict_drop_last)

            log.info(
                "[%5d/%5d] cutoff=%s  prob=%.3f → pred=%s  true=%d → %s  | P=%d W=%d L=%d U=%d  Acc=%.2f%% Cover=%.2f%%  | BUY=%d SELL=%d NONE=%d",
                k+1, total_iters, str(cutoff), prob, dec_label, yt, verdict,
                preds, wins, losses, unpred, acc, cover, buy_n, sell_n, none_n
            )

            rows.append({
                "iter": k+1,
                "cutoff": cutoff,
                "prob": prob,
                "pred": int(pdc),
                "true": int(yt),
                "verdict": verdict
            })

            shutil.rmtree(iter_dir, ignore_errors=True)

        acc = (wins / preds * 100.0) if preds > 0 else 0.0
        cover = (preds / (preds + unpred) * 100.0) if (preds + unpred) > 0 else 0.0
        print("\n========== SUMMARY (LIVE) ==========")
        print(f"Iters: {total_iters}   Predicted: {preds} | Wins: {wins} | Losses: {losses} | Unpred: {unpred}")
        print(f"Accuracy (predicted only): {acc:.2f}%  | Coverage: {cover:.2f}%")
        print(f"Decisions → BUY:{buy_n} SELL:{sell_n} NONE:{none_n}")
        print(f"Confusion → TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
        print("====================================\n")

        out_csv = os.path.join(args.model_dir, "sim_results.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log.info("Live-like results saved → %s", out_csv)
        log.info("=== sim finished (live) ===")

    finally:
        # تمیز کردن ریشهٔ temp اگر در LIVE ساخته شده باشد
        pass

if __name__ == "__main__":
    # اجرای بدون بافر برای دیدن لاگ‌ها لحظه‌ای
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()



# PYTHONUNBUFFERED=1 python3 -u live_like_sim.py \
#   --mode replay \
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
#   --hist-30t 480 --hist-1h 240 --hist-15t 960 --hist-5t 2880 \
#   --verbose
