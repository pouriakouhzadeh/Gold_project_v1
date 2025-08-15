"""
Live‑like simulator (REAL‑Stable) that mirrors GA/batch evaluation.
------------------------------------------------------------------
Goals
- Feed features up to t‑1 and predict the move from t→t+1 (binary 0/1).
- Drop the unstable last prepared row in predict mode.
- Use the *exact* train windowed feature set and thresholds saved at train time.
- Offer GA‑style split (85/5/10) so the evaluated slice matches the GA "Test".
- Be deployable in real‑time (MT4/CSV feed), with clear logs & CSV output.
- NEW: Optional "disk‑feed" mode to emulate MT4 writing CSVs each step, then
  re‑loading, cleaning, feature‑engineering, and using ready_incremental.
"""
from __future__ import annotations

import os, sys, json, argparse, logging, pickle, ast, glob, shutil
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import OrderedDict

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
    log = logging.getLogger("live_like_sim")
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
        if head in (b"x\x9c", b"x\xda"):  # zlib
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

# ----------------------- Column helpers -----------------------
_RESERVED_META_KEYS = {
    "meta","version","bins","n_features","n_samples",
    "train_bins","train_dist","hist","stats","classes_"
}
_COLS_CANDIDATE_KEYS = (
    "train_window_cols", "train_cols", "columns", "features", "feature_names", "cols"
)


def _list_of_strings(x: Any) -> Optional[List[str]]:
    if isinstance(x, list) and x and all(isinstance(i, (str, int)) for i in x):
        return [str(i) for i in x]
    return None


def _find_cols_recursively(obj: Any) -> Optional[List[str]]:
    """Generic search inside JSON-like objects to find a list of columns."""
    los = _list_of_strings(obj)
    if los:
        return los
    if isinstance(obj, dict):
        lower_map = {str(k).lower(): k for k in obj.keys()}
        # Common keys for feature names
        for lk, orig in lower_map.items():
            if lk in _COLS_CANDIDATE_KEYS:
                los = _list_of_strings(obj[orig])
                if los:
                    return los
                if isinstance(obj[orig], str):
                    s = obj[orig].strip()
                    for parser in (json.loads, ast.literal_eval):
                        try:
                            v = parser(s)
                            los = _list_of_strings(v)
                            if los:
                                return los
                        except Exception:
                            pass
        # Search nested
        meta = obj.get("meta")
        if isinstance(meta, dict):
            res = _find_cols_recursively(meta)
            if res:
                return res
        for v in obj.values():
            res = _find_cols_recursively(v)
            if res:
                return res
    if isinstance(obj, str) and "\n" in obj:
        lines = [ln.strip() for ln in obj.splitlines() if ln.strip()]
        if len(lines) >= 10:
            return lines
    return None


def _extract_cols_from_train_distribution_obj(td: Any) -> Optional[List[str]]:
    los = _list_of_strings(td)
    if los:
        return los
    if not isinstance(td, dict):
        return None
    direct = _find_cols_recursively(td)
    if direct and len(direct) >= 5:
        return direct
    for k in ("train_bins", "train_dist"):
        d = td.get(k)
        if isinstance(d, dict) and len(d) >= 5:
            keys = [str(x) for x in d.keys() if isinstance(x, (str, int))]
            if len(keys) >= 5:
                return keys
    if len(td) >= 5:
        keys = [str(k) for k, v in td.items()
                if isinstance(k, (str, int))
                and str(k) not in _RESERVED_META_KEYS
                and isinstance(v, (dict, list, tuple))]
        if len(keys) >= 5:
            return keys
    return None


def parse_train_cols_source(src: Optional[str]) -> Optional[List[str]]:
    """`src` may be a path to JSON/text or inline JSON/list."""
    if not src:
        return None
    text = None
    if os.path.isfile(src):
        try:
            with open(src, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = None
    if text:
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(text)
                cols = _find_cols_recursively(obj)
                if cols:
                    return cols
            except Exception:
                pass
        cols = _find_cols_recursively(text)
        if cols:
            return cols
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(src)
            cols = _find_cols_recursively(obj)
            if cols:
                return cols
        except Exception:
            pass
    return None


def recover_cols_from_train_distribution(search_dirs: List[str], log: logging.Logger) -> Optional[List[str]]:
    for d in search_dirs:
        p = os.path.join(d, "train_distribution.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            cols = _extract_cols_from_train_distribution_obj(obj)
            if cols and len(cols) >= 5:
                log.info("Recovered train_window_cols from %s (%d cols)", p, len(cols))
                return cols
        except Exception as e:
            log.warning("Failed to parse %s: %s", p, e)
            continue
    return None

# ----------------------- Threshold discovery --------------------

THR_CANDIDATE_KEYS = ("neg_thr", "pos_thr", "negative_threshold", "positive_threshold")
WIN_CANDIDATE_KEYS = ("window", "window_size", "win", "train_window")


def _search_thresholds_in_obj(obj: Any) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    neg = pos = None
    window = None
    if isinstance(obj, dict):
        lower_map = {str(k).lower(): k for k in obj.keys()}
        # Direct top-level keys
        for lk, orig in lower_map.items():
            if lk in THR_CANDIDATE_KEYS:
                v = obj[orig]
                if isinstance(v, (int, float)):
                    if "neg" in lk or "negative" in lk:
                        neg = float(v)
                    if "pos" in lk or "positive" in lk:
                        pos = float(v)
            if lk in WIN_CANDIDATE_KEYS:
                v = obj[orig]
                if isinstance(v, int) and v > 0:
                    window = int(v)
        # Meta section
        meta = obj.get("meta")
        if isinstance(meta, dict):
            n2, p2, w2 = _search_thresholds_in_obj(meta)
            neg = neg if (neg is not None) else n2
            pos = pos if (pos is not None) else p2
            window = window if (window is not None) else w2
        # Nested dicts
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                n2, p2, w2 = _search_thresholds_in_obj(v)
                neg = neg if (neg is not None) else n2
                pos = pos if (pos is not None) else p2
                window = window if (window is not None) else w2
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            n2, p2, w2 = _search_thresholds_in_obj(v)
            neg = neg if (neg is not None) else n2
            pos = pos if (pos is not None) else p2
            window = window if (window is not None) else w2
    return neg, pos, window


def recover_thresholds_and_window(search_dirs: List[str], log: logging.Logger) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Scan common JSON sidecars in the model directory and CWD."""
    jsons: List[str] = []
    for d in search_dirs:
        if not d:
            d = "."
        jsons.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    # Prioritize likely names
    jsons = sorted(jsons, key=lambda p: (
        0 if os.path.basename(p).lower() in ("model_meta.json", "meta.json", "thresholds.json") else 1,
        p
    ))
    for p in jsons:
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            neg, pos, win = _search_thresholds_in_obj(obj)
            if any(v is not None for v in (neg, pos, win)):
                log.info("Parsed thresholds/window from %s → neg=%s pos=%s window=%s",
                         os.path.basename(p),
                         ("{:.6f}".format(neg) if neg is not None else "∅"),
                         ("{:.6f}".format(pos) if pos is not None else "∅"),
                         (str(win) if win is not None else "∅"))
                return neg, pos, win
        except Exception as e:
            log.debug("Skipping %s: %s", p, e)
            continue
    return None, None, None

# ----------------------- Data helpers -----------------------------


def _read_times_only(csv_path: str) -> pd.DatetimeIndex:
    df = pd.read_csv(csv_path, usecols=["time"])  # MT4 format
    t = pd.to_datetime(df["time"], errors="coerce")
    return pd.DatetimeIndex(t.dropna().values).sort_values()


def timeframe_delta(main_tf: str) -> timedelta:
    tf = main_tf.upper()
    if tf in ("30T", "M30"):
        return timedelta(minutes=30)
    if tf in ("15T", "M15"):
        return timedelta(minutes=15)
    if tf in ("5T", "M5"):
        return timedelta(minutes=5)
    if tf in ("1H", "H1"):
        return timedelta(hours=1)
    return timedelta(minutes=30)


def compute_gt_from_main(main_30: pd.DataFrame, t_feat: pd.Timestamp) -> Optional[int]:
    if pd.isna(t_feat):
        return None
    tcol = "time" if "time" in main_30.columns else next((c for c in main_30.columns if c.lower()=="time"), None)
    if tcol is None:
        return None
    times = pd.to_datetime(main_30[tcol])
    pos = times.searchsorted(t_feat, side="left")
    if pos >= len(main_30):
        return None
    # If exact match not found, step back one candle <= t_feat
    if times.iloc[pos] != t_feat:
        pos = max(0, pos - 1)
        if times.iloc[pos] > t_feat:
            return None
    if pos + 1 >= len(main_30):
        return None
    c0 = float(main_30.iloc[pos]["close"]) if "close" in main_30.columns else float(main_30.iloc[pos]["Close"])
    c1 = float(main_30.iloc[pos + 1]["close"]) if "close" in main_30.columns else float(main_30.iloc[pos + 1]["Close"])
    return 1 if (c1 - c0) > 0 else 0

# ----------------------- Model/meta loader ------------------------


def infer_window_from_cols(cols: List[str]) -> int:
    import re
    mx = 0
    r = re.compile(r"_tminus(\d+)$")
    for c in cols:
        m = r.search(str(c))
        if m:
            mx = max(mx, int(m.group(1)))
    return (mx + 1) if mx > 0 else 1


def load_model_and_meta(args, log: logging.Logger):
    if not args.model or not os.path.isfile(args.model):
        raise FileNotFoundError(f"--model not found: {args.model}")
    pipeline = _smart_load_model(args.model)

    # 1) Feature columns (windowed) → from --train-cols-json or train_distribution.json
    train_window_cols: Optional[List[str]] = parse_train_cols_source(args.train_cols_json)
    if not train_window_cols:
        search_dirs = list(dict.fromkeys([os.path.dirname(args.model) or ".", os.getcwd()]))
        train_window_cols = recover_cols_from_train_distribution(search_dirs, log)
    if not train_window_cols:
        raise ValueError(
            "Could not obtain train_window_cols. Provide --train-cols-json (list/JSON/text) "
            "or ensure train_distribution.json carries the columns map."
        )

    # 2) Window inference (if not forced)
    window = args.window if (args.window and args.window > 0) else infer_window_from_cols(train_window_cols)
    if window < 1:
        window = 1

    # 3) Thresholds discovery (unless user forces manual)
    neg_thr = float(args.neg_thr) if args.neg_thr is not None else None
    pos_thr = float(args.pos_thr) if args.pos_thr is not None else None

    if int(args.use_model_thresholds) == 1:
        n2, p2, w2 = recover_thresholds_and_window([os.path.dirname(args.model), os.getcwd()], log)
        if neg_thr is None and n2 is not None:
            neg_thr = float(n2)
        if pos_thr is None and p2 is not None:
            pos_thr = float(p2)
        if (args.window == 0) and (w2 is not None) and (w2 > 0):
            window = int(w2)

    # final fallbacks
    if neg_thr is None:
        neg_thr = 0.005
    if pos_thr is None:
        pos_thr = 0.990

    # Column order: unique & keep order
    seen = set()
    train_window_cols = [str(c) for c in train_window_cols if not (str(c) in seen or seen.add(str(c)))]

    # Log a quick dir listing (useful during debugging in prod)
    try:
        lst = os.listdir(os.path.dirname(args.model) or ".")
        log.info("Model dir listing (%s): %s", os.path.dirname(args.model) or ".", ", ".join(lst[:120]))
    except Exception:
        pass

    log.info("Loaded model OK | window=%d | neg_thr=%.6f | pos_thr=%.6f | #cols=%d",
             window, neg_thr, pos_thr, len(train_window_cols))
    log.info("Sample columns: %s%s",
             ", ".join(map(str, train_window_cols[:5])),
             (" ..." if len(train_window_cols) > 5 else ""))

    return pipeline, window, neg_thr, pos_thr, train_window_cols

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser("Live-like REAL simulation (REAL‑Stable) — GA parity + disk‑feed")
    ap.add_argument("--mode", default="real", choices=["real"])  # reserved
    ap.add_argument("--predict-drop-last", type=int, default=1, help="Drop last prepared row in predict (1/0)")
    ap.add_argument("--fast-mode", type=int, default=1, help="Fast prep (skip drift scan; heavy tail trim)")
    ap.add_argument("--split", default="ga", choices=["ga", "tail"],
                    help="Anchor selection: 'ga' = last 10%% (85/5/10 split), 'tail' = last n_test bars")
    ap.add_argument("--audit", type=int, default=50, help="Print AUDIT every N steps")
    ap.add_argument("--base-data-dir", default=".")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--n-test", type=int, default=600, help="Used in split=tail; ignored in split=ga")
    ap.add_argument("--model", required=True, help="Path to model file (joblib/pickle/skops)")
    ap.add_argument("--train-cols-json", required=False,
                    help="Path or inline JSON/text for train_window_cols (optional if train_distribution.json exists)")
    ap.add_argument("--neg-thr", type=float, default=None, help="Override negative threshold; otherwise auto")
    ap.add_argument("--pos-thr", type=float, default=None, help="Override positive threshold; otherwise auto")
    ap.add_argument("--use-model-thresholds", type=int, default=1,
                    help="Try to auto-read thresholds/window from JSON sidecars (1/0)")
    ap.add_argument("--window", type=int, default=0, help="Force window (>0) else infer from columns/JSON")
    ap.add_argument("--save-csv", default="", help="Optional path to write per-step results CSV")
    ap.add_argument("--disk-feed", type=int, default=0,
                    help="1 = emulate MT4 by writing per-step CSVs then re-loading & using ready_incremental")
    ap.add_argument("--tmp-dir", default="_sim_csv_live", help="Temp dir to write per-step CSVs when --disk-feed=1")
    ap.add_argument("--cleanup", type=int, default=1, help="Remove tmp dir at the end (only when --disk-feed=1)")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    log = setup_logger(args.log_file)
    log.info("==> Starting live_like_sim with args: %s", vars(args))

    # --- model/meta
    pipeline, window, neg_thr, pos_thr, train_window_cols = load_model_and_meta(args, log)

    # --- Base CSV paths (original, read-only)
    base_fps = OrderedDict([
        ("30T", os.path.join(args.base_data_dir, f"{args.symbol}_M30.csv")),  # 30T FIRST (prevents KeyError in PREP)
        ("1H",  os.path.join(args.base_data_dir, f"{args.symbol}_H1.csv")),
        ("15T", os.path.join(args.base_data_dir, f"{args.symbol}_M15.csv")),
        ("5T",  os.path.join(args.base_data_dir, f"{args.symbol}_M5.csv")),
    ])
    for tf, p in base_fps.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"CSV not found for {tf}: {p}")
    log.info("CSV files OK: %s", base_fps)

    # Load base raw DataFrames (for slicing & GT)
    base_raw: Dict[str, pd.DataFrame] = {}
    for tf, p in base_fps.items():
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise ValueError(f"'time' column missing in {p}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        base_raw[tf] = df

    # Anchors
    main_30_times = pd.to_datetime(base_raw["30T"]["time"])  # use main TF
    if args.split == "ga":
        total = len(main_30_times)
        train_end, thresh_end = int(total * 0.85), int(total * 0.90)
        anchors = list(main_30_times.iloc[thresh_end:])
        log.info("Using GA split anchors → train_end=%d thresh_end=%d test_len=%d", train_end, thresh_end, len(anchors))
    else:
        anchors = list(main_30_times.tail(args.n_test))
        log.info("Using TAIL anchors → n_test=%d", len(anchors))

    log.info("Prepared %d anchor times. Mode=REAL. Starting simulation …", len(anchors))

    # --- Metrics accumulators
    wins = loses = undecided = 0
    y_true_decided: List[int] = []
    y_pred_decided: List[int] = []

    # Optional CSV rows
    csv_rows: List[Dict[str, Any]] = []

    dt = timeframe_delta("30T")

    if int(args.disk_feed) == 1:
        # Ensure tmp dir exists
        tmp_dir = os.path.abspath(args.tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        # Fixed filenames inside tmp dir (so PREP instance can be reused with warm state)
        tmp_fps = OrderedDict([
            ("30T", os.path.join(tmp_dir, f"{args.symbol}_M30.csv")),
            ("1H",  os.path.join(tmp_dir, f"{args.symbol}_H1.csv")),
            ("15T", os.path.join(tmp_dir, f"{args.symbol}_M15.csv")),
            ("5T",  os.path.join(tmp_dir, f"{args.symbol}_M5.csv")),
        ])
        # One PREP instance that always reads the tmp files
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=tmp_fps, main_timeframe="30T", verbose=False, fast_mode=bool(args.fast_mode))

        step = 0
        warm_started = False
        for t_cur in anchors:
            step += 1
            # 1) Write per‑TF slices up to t_cur
            for tf, df in base_raw.items():
                slice_df = df.loc[df["time"] <= t_cur].copy()
                slice_df.to_csv(tmp_fps[tf], index=False)
            # 2) Build merged + features from tmp files
            merged = prep.load_data()
            # 3) Use ready_incremental (warm‑up on first call)
            if not warm_started:
                # First call just warms the internal prev2 buffer
                _X0, _feats = prep.ready_incremental(merged, window=window, selected_features=train_window_cols)
                warm_started = True
                log.info("[WARM] ready_incremental primed; skipping prediction on first step.")
                continue
            X_one, feats = prep.ready_incremental(merged, window=window, selected_features=train_window_cols)
            if X_one.empty:
                undecided += 1
                log.info("[%d] %s  proba=NaN  pred=-1  gt=?  → ∅   | cum: wins=%d loses=%d undecided=%d decided=%d",
                         step, str(t_cur)[:16], wins, loses, undecided, wins + loses)
                continue
            # 4) Align columns & predict
            X_in = X_one.reindex(columns=train_window_cols, fill_value=0.0)
            try:
                p = float(pipeline.predict_proba(X_in)[:, 1][0])
            except Exception as e:
                undecided += 1
                log.error("[%d] Predict failed at %s: %s  → marking undecided", step, str(t_cur)[:16], e)
                continue
            pred = (0 if p <= neg_thr else (1 if p >= pos_thr else -1))
            # Feature time corresponds to t_cur - dt because we dropped the last prepared row
            t_feat = pd.Timestamp(t_cur) - dt
            gt = compute_gt_from_main(base_raw["30T"], t_feat)
            outcome = "∅"
            if gt is None or pred == -1:
                undecided += 1
            else:
                if pred == gt:
                    wins += 1; outcome = "WIN"
                else:
                    loses += 1; outcome = "LOSE"
                y_true_decided.append(int(gt))
                y_pred_decided.append(int(pred))
            decided = wins + loses
            acc = (wins / decided) if decided > 0 else float("nan")
            if args.audit and (step % int(args.audit) == 0):
                log.info("[AUDIT %d] t_cur=%s | t_feat=%s | GT=%s", args.audit, str(t_cur)[:19], str(t_feat)[:19], ("?" if gt is None else str(gt)))
            t_show = str(t_feat)[:16]
            log.info("[%d] %s  proba=%.6f  pred=%s  gt=%s  → %s   | cum: wins=%d loses=%d undecided=%d decided=%d acc=%s",
                     step, t_show, p, pred, ("?" if gt is None else str(gt)), outcome,
                     wins, loses, undecided, decided,
                     ("{:.4f}".format(acc) if decided > 0 else "n/a"))
            if args.save_csv:
                csv_rows.append({
                    "t_cur": str(t_cur),
                    "t_feat": str(t_feat),
                    "proba": p,
                    "pred": pred,
                    "gt": (None if gt is None else int(gt)),
                    "outcome": outcome
                })
        # Cleanup tmp dir if requested
        if int(args.cleanup) == 1:
            try:
                shutil.rmtree(tmp_dir)
                log.info("Tmp dir removed: %s", tmp_dir)
            except Exception as e:
                log.warning("Failed to remove tmp dir %s: %s", tmp_dir, e)

    else:
        # In‑memory path (faster). Still enforces REAL‑Stable via predict_drop_last.
        # Build fps with 30T first to match PREP's assumption
        fps = OrderedDict([
            ("30T", base_fps["30T"]),
            ("1H",  base_fps["1H"]),
            ("15T", base_fps["15T"]),
            ("5T",  base_fps["5T"]),
        ])
        # Optional FAST pre-trim anchor to speed up (safe big buffer)
        shared_start_override = None
        if int(args.fast_mode) == 1:
            times_30 = pd.to_datetime(base_raw["30T"]["time"])  # main TF
            safety = max(2000, 3 * window * 60)  # generous buffer for heavy indicators/window
            need_main = (args.n_test if args.split == "tail" else max(1000, 2000)) + safety
            if len(times_30) > need_main:
                idx = max(0, len(times_30) - need_main)
                shared_start_override = pd.Timestamp(times_30.iloc[idx])
        # PREP
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=fps, main_timeframe="30T", verbose=False, fast_mode=bool(args.fast_mode))
        if shared_start_override is not None:
            prep.shared_start_date = shared_start_override
        raw_all = prep.load_data()
        main_time_col = f"{prep.main_timeframe}_time"
        # Loop
        step = 0
        for t_cur in anchors:
            step += 1
            raw_slice = raw_all.loc[pd.to_datetime(raw_all[main_time_col]) <= t_cur].copy()
            if raw_slice.empty:
                undecided += 1
                log.info("[%d] %s  proba=NaN  pred=-1  gt=?  → ∅   | cum: wins=%d loses=%d undecided=%d decided=%d",
                         step, str(t_cur)[:16], wins, loses, undecided, wins + loses)
                continue
            X_f, _, _, _price_ser, _t_idx = prep.ready(
                raw_slice,
                window=window,
                selected_features=train_window_cols,
                mode="predict",
                with_times=False,
                predict_drop_last=True,
            )
            if X_f.empty:
                undecided += 1
                log.info("[%d] %s  proba=NaN  pred=-1  gt=?  → ∅   | cum: wins=%d loses=%d undecided=%d decided=%d",
                         step, str(t_cur)[:16], wins, loses, undecided, wins + loses)
                continue
            X_in = X_f.tail(1).reindex(columns=train_window_cols, fill_value=0.0)
            try:
                p = float(pipeline.predict_proba(X_in)[:, 1][0])
            except Exception as e:
                undecided += 1
                log.error("[%d] Predict failed at %s: %s  → marking undecided", step, str(t_cur)[:16], e)
                continue
            pred = (0 if p <= neg_thr else (1 if p >= pos_thr else -1))
            t_feat = pd.Timestamp(t_cur) - dt
            gt = compute_gt_from_main(base_raw["30T"], t_feat)
            outcome = "∅"
            if gt is None or pred == -1:
                undecided += 1
            else:
                if pred == gt:
                    wins += 1; outcome = "WIN"
                else:
                    loses += 1; outcome = "LOSE"
                y_true_decided.append(int(gt))
                y_pred_decided.append(int(pred))
            decided = wins + loses
            acc = (wins / decided) if decided > 0 else float("nan")
            if args.audit and (step % int(args.audit) == 0):
                log.info("[AUDIT %d] t_cur=%s | t_feat=%s | GT=%s", args.audit, str(t_cur)[:19], str(t_feat)[:19], ("?" if gt is None else str(gt)))
            log.info("[%d] %s  proba=%.6f  pred=%s  gt=%s  → %s   | cum: wins=%d loses=%d undecided=%d decided=%d acc=%s",
                     step, str(t_feat)[:16], p, pred, ("?" if gt is None else str(gt)), outcome,
                     wins, loses, undecided, decided,
                     ("{:.4f}".format(acc) if decided > 0 else "n/a"))
            if args.save_csv:
                csv_rows.append({
                    "t_cur": str(t_cur),
                    "t_feat": str(t_feat),
                    "proba": p,
                    "pred": pred,
                    "gt": (None if gt is None else int(gt)),
                    "outcome": outcome
                })

    # --- Final metrics
    decided = wins + loses
    acc = wins / decided if decided > 0 else float("nan")

    try:
        from sklearn.metrics import balanced_accuracy_score
        bal_acc = balanced_accuracy_score(y_true_decided, y_pred_decided) if decided > 0 else float("nan")
    except Exception:
        bal_acc = float("nan")

    coverage = decided / (decided + undecided) if (decided + undecided) > 0 else float("nan")

    log.info("DONE. decided=%d (wins=%d, loses=%d) · undecided=%d · acc=%s · bal_acc=%s · coverage=%.2f",
             decided, wins, loses, undecided,
             ("{:.4f}".format(acc) if decided > 0 else "n/a"),
             ("{:.4f}".format(bal_acc) if decided > 0 else "n/a"),
             coverage)

    if args.save_csv and csv_rows:
        out = os.path.abspath(args.save_csv)
        pd.DataFrame(csv_rows).to_csv(out, index=False)
        log.info("Saved per‑step results to %s", out)


if __name__ == "__main__":
    main()
