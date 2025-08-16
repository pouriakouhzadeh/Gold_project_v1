#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic disk-feed live simulator (strict) + batch parity
=========================================================

- در هر گام، چهار CSV (5T/15T/30T/1H) دقیقا تا t_cur روی دیسک وجود دارد.
- همان CSVها لود می‌شوند، Clean/Feature ساخته می‌شود، و «بعد از آماده‌سازی» ردیف ناپایدار حذف می‌شود.
- از ready_incremental برای وارم‌آپ و تولید فقط ردیف پایدار t−1 استفاده می‌شود.
- پیش‌بینی انجام می‌شود و GT از CSV 30T اصلی ساخته می‌شود.
- فایل‌های tmp به‌صورت افزایشی append می‌شوند (یک‌بار پردازش سنگین در هر استپ).
- در پایان، پاریتی با Batch روی همان بازه گزارش می‌شود.

اجرای نمونه (یک خط):
python3 live_like_sim_diskfeed_strict.py --mode real --split tail --n-test 4000 --history-bars 2500 --fast-mode 1 --audit 1 --base-data-dir /home/pouria/gold_project9 --symbol XAUUSD --model /home/pouria/gold_project9/best_model.pkl --use-model-thresholds 1 --tmp-dir _sim_csv --cleanup 1 --save-csv live_like_results.csv --log-file live_like_real.log
"""
from __future__ import annotations

import os, sys, json, argparse, logging, pickle, ast, glob, shutil
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import timedelta

# Optional: sklearn acceleration
try:
    from sklearnex import patch_sklearn  # type: ignore
    patch_sklearn(verbose=False)
except Exception:
    pass

# Project code
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

# --------------------------- Logger ---------------------------

def setup_logger(log_file: Optional[str]) -> logging.Logger:
    log = logging.getLogger("live_like_diskfeed_strict")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log

# --------------------------- Model IO --------------------------

def _smart_load_model(path: str):
    """Try joblib → pickle (+gzip/zlib) → skops."""
    # joblib
    try:
        from joblib import load as joblib_load  # type: ignore
        return joblib_load(path)
    except Exception:
        pass
    # pickle (+gzip/zlib detection)
    try:
        with open(path, "rb") as f:
            head = f.read(2)
        if head == b"\x1f\x8b":  # gzip
            import gzip
            with gzip.open(path, "rb") as g:
                return pickle.load(g)
        if head in (b"x\x9c", b"x\xda"):
            import zlib
            data = open(path, "rb").read()
            try:
                decomp = zlib.decompress(data)
                return pickle.loads(decomp)
            except Exception:
                with open(path, "rb") as f:
                    return pickle.load(f)
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    # skops (optional)
    try:
        from skops.io import load as skops_load  # type: ignore
        return skops_load(path)
    except Exception:
        pass
    raise RuntimeError(f"Could not load model from '{path}'. Save with joblib/pickle or a compatible format.")

_COLS_KEYS = ("train_window_cols", "train_cols", "columns", "features", "feature_names", "cols")
_THR_KEYS  = ("neg_thr", "pos_thr", "negative_threshold", "positive_threshold")
_WIN_KEYS  = ("window_size", "window", "win", "train_window")


def _list_of_strings(x: Any) -> Optional[List[str]]:
    if isinstance(x, list) and x and all(isinstance(i, (str, int)) for i in x):
        return [str(i) for i in x]
    return None


def _find_cols_recursively(obj: Any) -> Optional[List[str]]:
    los = _list_of_strings(obj)
    if los:
        return los
    if isinstance(obj, dict):
        for k in _COLS_KEYS:
            for kk in (k, k.upper(), k.lower()):
                if kk in obj:
                    los = _list_of_strings(obj[kk])
                    if los:
                        return los
                    if isinstance(obj[kk], str):
                        s = obj[kk].strip()
                        for parser in (json.loads, ast.literal_eval):
                            try:
                                v = parser(s)
                                los = _list_of_strings(v)
                                if los:
                                    return los
                            except Exception:
                                pass
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                los = _find_cols_recursively(v)
                if los:
                    return los
    if isinstance(obj, str) and "\n" in obj:
        lines = [ln.strip() for ln in obj.splitlines() if ln.strip()]
        if len(lines) >= 5:
            return lines
    return None


def recover_cols_from_train_distribution(search_dirs: List[str], log: logging.Logger) -> Optional[List[str]]:
    for d in search_dirs:
        p = os.path.join(d or ".", "train_distribution.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            cols = _find_cols_recursively(obj)
            if cols and len(cols) >= 5:
                log.info("Recovered train_window_cols from %s (%d cols)", p, len(cols))
                return cols
        except Exception as e:
            log.warning("Failed to parse %s: %s", p, e)
    return None


def recover_thresholds_and_window(search_dirs: List[str], log: logging.Logger) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    jsons: List[str] = []
    for d in search_dirs:
        jsons.extend(sorted(glob.glob(os.path.join(d or ".", "*.json"))))
    jsons = sorted(jsons, key=lambda p: (0 if os.path.basename(p).lower() in ("model_meta.json", "thresholds.json", "meta.json") else 1, p))
    neg = pos = None
    win = None
    for p in jsons:
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for k in _THR_KEYS:
                for kk in (k, k.upper(), k.lower()):
                    if kk in obj and isinstance(obj[kk], (int, float)):
                        if "neg" in kk or "negative" in kk:
                            neg = float(obj[kk])
                        if "pos" in kk or "positive" in kk:
                            pos = float(obj[kk])
            for k in _WIN_KEYS:
                for kk in (k, k.upper(), k.lower()):
                    if kk in obj and isinstance(obj[kk], int) and obj[kk] > 0:
                        win = int(obj[kk])
        except Exception:
            continue
    if any(v is not None for v in (neg, pos, win)):
        log.info("Recovered thresholds/window from JSONs → neg=%s pos=%s win=%s",
                 ("{:.6f}".format(neg) if neg is not None else "∅"),
                 ("{:.6f}".format(pos) if pos is not None else "∅"),
                 (str(win) if win is not None else "∅"))
    return neg, pos, win


def infer_window_from_cols(cols: List[str]) -> int:
    import re
    r = re.compile(r"_tminus(\d+)$")
    mx = 0
    for c in cols:
        m = r.search(str(c))
        if m:
            mx = max(mx, int(m.group(1)))
    return (mx + 1) if mx > 0 else 1


def coerce_pipeline(obj: Any) -> Any:
    """Handle cases where best_model.pkl is a dict with pipeline & meta."""
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ("pipeline", "model", "estimator", "clf", "sk_pipeline"):
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    raise TypeError("Loaded model object is not a scikit-learn pipeline/classifier with predict_proba.")

# ----------------------- Time helpers ---------------------------

def timeframe_delta(tf: str) -> timedelta:
    tf = tf.upper()
    if tf in ("30T", "M30"):
        return timedelta(minutes=30)
    if tf in ("15T", "M15"):
        return timedelta(minutes=15)
    if tf in ("5T", "M5"):
        return timedelta(minutes=5)
    if tf in ("1H", "H1"):
        return timedelta(hours=1)
    return timedelta(minutes=30)


def compute_gt_from_m30(m30: pd.DataFrame, t_feat: pd.Timestamp) -> Optional[int]:
    if pd.isna(t_feat):
        return None
    t = pd.to_datetime(m30["time"])  # expect MT4 column name
    pos = t.searchsorted(t_feat, side="left")
    if pos >= len(m30):
        return None
    if t.iloc[pos] != t_feat:
        pos = pos - 1
        if pos < 0 or t.iloc[pos] > t_feat:
            return None
    if pos + 1 >= len(m30):
        return None
    c0 = float(m30.iloc[pos]["close"])
    c1 = float(m30.iloc[pos + 1]["close"])
    return 1 if (c1 - c0) > 0 else 0

# ----------------------- Disk-feed helpers ----------------------

def ensure_tmp_dir(tmp_dir: str, cleanup: bool):
    if cleanup and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)


def load_base_csvs(base_dir: str, symbol: str) -> Dict[str, pd.DataFrame]:
    fps = {
        "30T": os.path.join(base_dir, f"{symbol}_M30.csv"),
        "15T": os.path.join(base_dir, f"{symbol}_M15.csv"),
        "5T":  os.path.join(base_dir, f"{symbol}_M5.csv"),
        "1H":  os.path.join(base_dir, f"{symbol}_H1.csv"),
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for tf, p in fps.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"CSV not found for {tf}: {p}")
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise ValueError(f"'time' column missing in {tf} CSV")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        dfs[tf] = df
    return dfs


def write_initial_tmp(tmp_dir: str, dfs: Dict[str, pd.DataFrame], t_hist_start: pd.Timestamp, t0: pd.Timestamp):
    for tf, df in dfs.items():
        mask = (df["time"] >= t_hist_start) & (df["time"] <= t0)
        out = df.loc[mask].copy()
        out.to_csv(os.path.join(tmp_dir, f"XAUUSD_{'M30' if tf=='30T' else ('M15' if tf=='15T' else ('M5' if tf=='5T' else 'H1'))}.csv"), index=False)


def append_new_rows(tmp_dir: str, base_df: pd.DataFrame, tf: str, last_written_time: Optional[pd.Timestamp], t_cur: pd.Timestamp) -> Optional[pd.Timestamp]:
    # compute new rows to append: (last_written_time, t_cur]
    if last_written_time is None:
        mask = base_df["time"] <= t_cur
    else:
        mask = (base_df["time"] > last_written_time) & (base_df["time"] <= t_cur)
    new = base_df.loc[mask]
    if new.empty:
        return last_written_time
    out_path = os.path.join(tmp_dir, f"XAUUSD_{'M30' if tf=='30T' else ('M15' if tf=='15T' else ('M5' if tf=='5T' else 'H1'))}.csv")
    mode = "a" if os.path.exists(out_path) else "w"
    header = not os.path.exists(out_path) if mode == "a" else True
    new.to_csv(out_path, mode=mode, header=header, index=False)
    return pd.Timestamp(new.iloc[-1]["time"])  # last written

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser("Strict disk-feed live simulator + batch parity")
    ap.add_argument("--mode", default="real", choices=["real"])  # reserved
    ap.add_argument("--split", default="tail", choices=["ga", "tail"], help="Anchor selection")
    ap.add_argument("--n-test", type=int, default=4000, help="Used in split=tail")
    ap.add_argument("--history-bars", type=int, default=2500, help="Initial 30T history bars before first anchor")
    ap.add_argument("--fast-mode", type=int, default=1, help="Use PREP fast_mode (skip drift scan)")
    ap.add_argument("--audit", type=int, default=50, help="Log every N steps (1=every step)")
    ap.add_argument("--base-data-dir", default=".")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-cols-json", default=None)
    ap.add_argument("--neg-thr", type=float, default=None)
    ap.add_argument("--pos-thr", type=float, default=None)
    ap.add_argument("--use-model-thresholds", type=int, default=1)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--tmp-dir", default="_sim_csv")
    ap.add_argument("--cleanup", type=int, default=1, help="Remove tmp dir at end")
    ap.add_argument("--save-csv", default="")
    ap.add_argument("--log-file", default=None)
    # اختیاری: بازنویسی کامل فایل‌های tmp و پاکسازی در هر استپ
    ap.add_argument("--rewrite-per-step", type=int, default=0)
    ap.add_argument("--cleanup-per-step", type=int, default=0)
    args = ap.parse_args()

    log = setup_logger(args.log_file)
    log.info("==> Starting live_like_diskfeed_strict with args: %s", vars(args))

    # --- Load model
    obj = _smart_load_model(args.model)
    model_side_cols = None
    if isinstance(obj, dict):
        for k in _COLS_KEYS:
            if k in obj:
                model_side_cols = _list_of_strings(obj[k])
                if model_side_cols:
                    break
    pipeline = coerce_pipeline(obj)

    # --- Columns
    cols = model_side_cols
    if not cols:
        cols = recover_cols_from_train_distribution([os.path.dirname(args.model), os.getcwd()], log)
    if not cols:
        raise ValueError("Could not obtain train_window_cols from model dict or train_distribution.json")
    seen = set()
    train_window_cols = [str(c) for c in cols if not (str(c) in seen or seen.add(str(c)))]

    # --- Window
    window = args.window if (args.window and args.window > 0) else infer_window_from_cols(train_window_cols)

    # --- Thresholds
    neg_thr = args.neg_thr
    pos_thr = args.pos_thr
    if int(args.use_model_thresholds) == 1:
        n2, p2, w2 = recover_thresholds_and_window([os.path.dirname(args.model), os.getcwd()], log)
        if neg_thr is None and n2 is not None:
            neg_thr = float(n2)
        if pos_thr is None and p2 is not None:
            pos_thr = float(p2)
        if (args.window == 0) and (w2 is not None) and (w2 > 0):
            window = int(w2)
    if neg_thr is None:
        neg_thr = 0.005
    if pos_thr is None:
        pos_thr = 0.990

    log.info("Loaded model OK | window=%d | neg_thr=%.6f | pos_thr=%.6f | #cols=%d",
             window, neg_thr, pos_thr, len(train_window_cols))

    # --- Load base CSVs (for anchor & GT)
    base = load_base_csvs(args.base_data_dir, args.symbol)
    m30 = base["30T"]
    all_times_30 = pd.to_datetime(m30["time"]).sort_values()

    if args.split == "ga":
        total = len(all_times_30)
        train_end, thresh_end = int(total * 0.85), int(total * 0.90)
        anchors = list(all_times_30.iloc[thresh_end:])
        log.info("Using GA split anchors → train_end=%d thresh_end=%d test_len=%d", train_end, thresh_end, len(anchors))
    else:
        anchors = list(all_times_30.tail(args.n_test))
        log.info("Using TAIL anchors → n_test=%d", len(anchors))

    # --- Prepare tmp dir & initial seed
    ensure_tmp_dir(args.tmp_dir, cleanup=True)
    dt30 = timeframe_delta("30T")
    t0 = pd.Timestamp(anchors[0])
    t_hist_start = t0 - args.history_bars * dt30

    # seed
    write_initial_tmp(args.tmp_dir, base, t_hist_start, t0)

    # Track last written times per TF
    last_written: Dict[str, Optional[pd.Timestamp]] = {}
    for tf, df in base.items():
        mask = (df["time"] >= t_hist_start) & (df["time"] <= t0)
        last_written[tf] = pd.Timestamp(df.loc[mask].iloc[-1]["time"]) if mask.any() else None

    # --- PREP bound to tmp files (no trimming inside) → strict_disk_feed=True
    tmp_filepaths = {
        "30T": os.path.join(args.tmp_dir, "XAUUSD_M30.csv"),
        "1H":  os.path.join(args.tmp_dir, "XAUUSD_H1.csv"),
        "15T": os.path.join(args.tmp_dir, "XAUUSD_M15.csv"),
        "5T":  os.path.join(args.tmp_dir, "XAUUSD_M5.csv"),
    }
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths=tmp_filepaths, main_timeframe="30T",
        verbose=False, fast_mode=bool(args.fast_mode),
        strict_disk_feed=True
    )


    # --- Loop
    wins = loses = undecided = 0
    TP = TN = FP = FN = 0
    y_true_decided: List[int] = []
    y_pred_decided: List[int] = []
    live_times_decided: List[pd.Timestamp] = []
    rows_csv: List[Dict[str, Any]] = []
    total_steps = len(anchors)

    for step, t_cur in enumerate(anchors, start=1):
        t_cur = pd.Timestamp(t_cur)

        if int(args.rewrite_per_step) == 1:  # بازنویسی کامل فایل‌ها تا t_cur
            ensure_tmp_dir(args.tmp_dir, cleanup=True)
            write_initial_tmp(args.tmp_dir, base, t_hist_start, t_cur)
            last_written = {tf: t_cur for tf in ("30T", "1H", "15T", "5T")}
        else:
            # Append new rows from base to tmp files up to t_cur
            for tf in ("30T", "1H", "15T", "5T"):
                last_written[tf] = append_new_rows(args.tmp_dir, base[tf], tf, last_written.get(tf), t_cur)

        # Build features ONCE for current CSV state (strict realism)
        raw_all = prep.load_data()  # heavy but once per-step (on growing tmp files)

        # Incremental ready: pass only a small tail window; warm-up on first call
        tail_len = max(window + 3, 12)
        data_window = raw_all.tail(tail_len)
        X_tail, feats = prep.ready_incremental(data_window, window=window, selected_features=train_window_cols)
        if X_tail.empty:
            if int(args.audit) == 1 or (args.audit and (step % int(args.audit) == 0)):
                log.info("[WARM] ready_incremental primed at step %d; skipping prediction.", step)
            continue

        X_in = X_tail.reindex(columns=train_window_cols, fill_value=0.0)
        X_in = X_in.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # safety for LR

        try:
            p = float(pipeline.predict_proba(X_in)[:, 1][0])
        except Exception as e:
            undecided += 1
            log.error("[%d/%d] Predict failed at %s: %s → ∅", step, total_steps, str(t_cur)[:16], e)
            continue

        pred = (0 if p <= neg_thr else (1 if p >= pos_thr else -1))
        t_feat = t_cur - dt30  # آخرین ردیف پایدار ورودی مربوط به t−1 است
        gt = compute_gt_from_m30(m30, t_feat)

        outcome = "∅"
        if (gt is None) or (pred == -1):
            undecided += 1
        else:
            if gt == 1 and pred == 1:
                TP += 1; wins += 1; outcome = "WIN"
            elif gt == 0 and pred == 0:
                TN += 1; wins += 1; outcome = "WIN"
            elif gt == 0 and pred == 1:
                FP += 1; loses += 1; outcome = "LOSE"
            elif gt == 1 and pred == 0:
                FN += 1; loses += 1; outcome = "LOSE"
            y_true_decided.append(int(gt))
            y_pred_decided.append(int(pred))
            live_times_decided.append(t_feat)

        decided = wins + loses
        acc = (wins / decided) if decided > 0 else float("nan")
        tpr = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
        tnr = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
        bal_running = ((tpr + tnr) / 2.0) if (not np.isnan(tpr) and not np.isnan(tnr)) else float("nan")
        coverage = decided / (decided + undecided) if (decided + undecided) > 0 else float("nan")

        if int(args.audit) == 1 or (args.audit and (step % int(args.audit) == 0)):
            log.info(
                "[%d/%d] t_cur=%s | t_feat=%s | proba=%.6f | pred=%s | gt=%s → %s | acc=%s | bal_acc=%s | coverage=%.2f | decided=%d | ∅=%d",
                step, total_steps, str(t_cur)[:16], str(t_feat)[:16], p, pred,
                ("?" if gt is None else str(gt)), outcome,
                ("{:.4f}".format(acc) if decided>0 else "n/a"),
                ("{:.4f}".format(bal_running) if decided>0 else "n/a"),
                (coverage if not np.isnan(coverage) else 0.0), decided, undecided,
            )

        if int(args.cleanup_per_step) == 1:
            # حذف فایل‌های tmp پس از هر استپ (دقیقاً معادل دریافت تک‌شات از متاتریدر؛ کندتر است)
            for fn in ("XAUUSD_M30.csv", "XAUUSD_M15.csv", "XAUUSD_M5.csv", "XAUUSD_H1.csv"):
                try:
                    os.remove(os.path.join(args.tmp_dir, fn))
                except FileNotFoundError:
                    pass

        if args.save_csv:
            rows_csv.append({
                "step": step,
                "t_cur": str(t_cur),
                "t_feat": str(t_feat),
                "proba": p,
                "pred": pred,
                "gt": (None if gt is None else int(gt)),
                "outcome": outcome,
                "acc_running": (None if np.isnan(acc) else float(acc)),
                "bal_acc_running": (None if np.isnan(bal_running) else float(bal_running)),
                "coverage_running": (None if np.isnan(coverage) else float(coverage)),
            })

    # --- Final live metrics
    decided = wins + loses
    try:
        from sklearn.metrics import balanced_accuracy_score
        bal_live = balanced_accuracy_score(y_true_decided, y_pred_decided) if decided > 0 else float("nan")
    except Exception:
        bal_live = float("nan")
    acc_live = wins / decided if decided > 0 else float("nan")
    coverage_live = decided / (decided + undecided) if (decided + undecided) > 0 else float("nan")

    log.info("[LIVE] decided=%d (wins=%d, loses=%d) · ∅=%d · acc=%s · bal_acc=%s · coverage=%.2f",
             decided, wins, loses, undecided,
             ("{:.4f}".format(acc_live) if decided>0 else "n/a"),
             ("{:.4f}".format(bal_live) if decided>0 else "n/a"),
             coverage_live)

    # --- Batch parity on exact slice
    t_min = pd.Timestamp(anchors[0])
    t_max = pd.Timestamp(anchors[-1])

    prep_batch = PREPARE_DATA_FOR_TRAIN(
        filepaths={
            "30T": os.path.join(args.base_data_dir, f"{args.symbol}_M30.csv"),
            "1H":  os.path.join(args.base_data_dir, f"{args.symbol}_H1.csv"),
            "15T": os.path.join(args.base_data_dir, f"{args.symbol}_M15.csv"),
            "5T":  os.path.join(args.base_data_dir, f"{args.symbol}_M5.csv"),
        },
        main_timeframe="30T", verbose=False, fast_mode=bool(args.fast_mode)
    )
    raw_all_b = prep_batch.load_data()
    tcol = f"{prep_batch.main_timeframe}_time"
    all_times_b = pd.to_datetime(raw_all_b[tcol])
    mask = (all_times_b >= t_min) & (all_times_b <= t_max)
    test_slice = raw_all_b.loc[mask].copy()

    Xb, yb, _, _, t_idx_b = prep_batch.ready(
        test_slice,
        window=window,
        selected_features=train_window_cols,
        mode="train",
        with_times=True,
        predict_drop_last=False,
    )
    Xb = Xb.reindex(columns=train_window_cols, fill_value=0.0)
    prob_b = pipeline.predict_proba(Xb)[:, 1]
    pred_b = np.full_like(yb, -1, dtype=int)
    pred_b[prob_b <= neg_thr] = 0
    pred_b[prob_b >= pos_thr] = 1

    mask_b = pred_b != -1
    decided_b = int(mask_b.sum())
    correct_b = int((pred_b[mask_b] == yb[mask_b]).sum())
    acc_b = (correct_b / decided_b) if decided_b > 0 else float("nan")
    try:
        from sklearn.metrics import balanced_accuracy_score
        bal_b = balanced_accuracy_score(yb[mask_b], pred_b[mask_b]) if decided_b > 0 else float("nan")
    except Exception:
        bal_b = float("nan")
    coverage_b = decided_b / len(pred_b) if len(pred_b) else float("nan")

    log.info("[BATCH/ALL] rows=%d decided=%d acc=%s bal_acc=%s coverage=%.2f",
             len(pred_b), decided_b,
             ("{:.4f}".format(acc_b) if decided_b>0 else "n/a"),
             ("{:.4f}".format(bal_b) if decided_b>0 else "n/a"),
             coverage_b)

    # Batch parity روی دقیقاً همان زمان‌هایی که لایو تصمیم گرفته
    t_map_b = {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(t_idx_b))}
    idxs = [t_map_b.get(pd.Timestamp(t)) for t in live_times_decided]
    idxs = [i for i in idxs if i is not None]
    if idxs:
        pred_b_sub = pred_b[idxs]
        yb_sub = yb.iloc[idxs]
        mask_sub = pred_b_sub != -1
        decided_b_sub = int(mask_sub.sum())
        correct_b_sub = int((pred_b_sub[mask_sub] == yb_sub[mask_sub]).sum())
        acc_b_sub = (correct_b_sub / decided_b_sub) if decided_b_sub > 0 else float("nan")
        try:
            from sklearn.metrics import balanced_accuracy_score
            bal_b_sub = balanced_accuracy_score(yb_sub[mask_sub], pred_b_sub[mask_sub]) if decided_b_sub > 0 else float("nan")
        except Exception:
            bal_b_sub = float("nan")
        log.info("[BATCH/ON_LIVE_TIMES] decided=%d acc=%s bal_acc=%s",
                 decided_b_sub,
                 ("{:.4f}".format(acc_b_sub) if decided_b_sub>0 else "n/a"),
                 ("{:.4f}".format(bal_b_sub) if decided_b_sub>0 else "n/a"))
        delta_acc = (acc_live if not np.isnan(acc_live) else np.nan) - (acc_b_sub if not np.isnan(acc_b_sub) else np.nan)
        delta_bal = (bal_live if not np.isnan(bal_live) else np.nan) - (bal_b_sub if not np.isnan(bal_b_sub) else np.nan)
        log.info("[DELTA] live_minus_batch_on_live_times → acc=%s bal_acc=%s",
                 ("{:+.4f}".format(delta_acc) if not np.isnan(delta_acc) else "n/a"),
                 ("{:+.4f}".format(delta_bal) if not np.isnan(delta_bal) else "n/a"))
    else:
        log.info("[BATCH/ON_LIVE_TIMES] No overlapping decided timestamps with live.")

    if args.save_csv and rows_csv:
        out = os.path.abspath(args.save_csv)
        pd.DataFrame(rows_csv).to_csv(out, index=False)
        log.info("Saved per-step results to %s", out)

    if int(args.cleanup) == 1:
        try:
            shutil.rmtree(args.tmp_dir, ignore_errors=True)
            log.info("Temporary dir '%s' removed.", args.tmp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
