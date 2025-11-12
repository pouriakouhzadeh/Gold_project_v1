# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
live_like_sim_v4.py
-------------------
- شبیه‌سازی لایو واقعی با اصل «اول برش، بعد ریسَمپل»
- تلاش برای استفاده از کلاس تولیدی PREPARE_DATA_FOR_TRAIN_PRODUCTION
- فالبک داخلی بدون وابستگی‌های بیرونی در صورت نبود/خرابی کلاس تولیدی
"""

from __future__ import annotations
import os, sys, json, gzip, bz2, lzma, zipfile, pickle, logging, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


LOG_FILE = "live_like_sim_v4.log"


# ============================== Logging ==============================
def setup_logging(verbosity: int = 1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity <= 1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger



def _normalize_freq(freq: str) -> str:
    # 30T -> 30min
    return re.sub(r'(?i)^(\d+)\s*T$', r'\1min', freq.strip())


# ======================== Robust model loader ========================
def _try_pickle(fp: str):
    with open(fp, "rb") as f:
        return pickle.load(f)


def _try_joblib(fp: str):
    try:
        import joblib  # type: ignore
    except Exception as e:
        raise RuntimeError("joblib not available") from e
    return joblib.load(fp)


def _try_gzip(fp: str):
    with gzip.open(fp, "rb") as f:
        return pickle.load(f)


def _try_bz2(fp: str):
    with bz2.open(fp, "rb") as f:
        return pickle.load(f)


def _try_lzma(fp: str):
    with lzma.open(fp, "rb") as f:
        return pickle.load(f)


def _try_zip(fp: str):
    with zipfile.ZipFile(fp, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            with zf.open(name, "r") as f:
                data = f.read()
                # try pickle then json
                try:
                    return pickle.loads(data)
                except Exception:
                    try:
                        return json.loads(data.decode("utf-8"))
                    except Exception:
                        pass
    raise ValueError("No loadable member found in ZIP.")


def _raw_head(fp: str, n: int = 8) -> bytes:
    with open(fp, "rb") as f:
        return f.read(n)


def load_payload_best_model(pkl_path: str = "best_model.pkl") -> dict:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"'best_model.pkl' not found at {p.resolve()}")

    head = _raw_head(pkl_path, 8)
    if head.startswith(b"\x80"):
        loaders = [_try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"\x1f\x8b"):
        loaders = [_try_gzip, _try_pickle, _try_joblib, _try_bz2, _try_lzma, _try_zip]
    elif head.startswith(b"BZh"):
        loaders = [_try_bz2, _try_pickle, _try_joblib, _try_gzip, _try_lzma, _try_zip]
    elif head.startswith(b"\xfd7zXZ\x00"):
        loaders = [_try_lzma, _try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_zip]
    elif head.startswith(b"PK\x03\x04"):
        loaders = [_try_zip, _try_pickle, _try_joblib, _try_gzip, _try_bz2, _try_lzma]
    else:
        loaders = [_try_joblib, _try_pickle, _try_gzip, _try_bz2, _try_lzma, _try_zip]

    last_err = None
    raw = None
    for loader in loaders:
        try:
            raw = loader(pkl_path)
            break
        except Exception as e:
            last_err = e

    if raw is None:
        # maybe plain JSON
        try:
            with open(pkl_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            pass

    if raw is None:
        raise RuntimeError(
            f"Could not load best_model.pkl with multiple strategies. "
            f"First bytes: {head!r}. Last error: {repr(last_err)}"
        )

    payload: dict = {}
    if isinstance(raw, dict):
        model = (
            raw.get("pipeline")
            or raw.get("model")
            or raw.get("estimator")
            or raw.get("clf")
            or raw.get("best_estimator")
        )
        if model is None:
            for k, v in raw.items():
                if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                    model = v
                    break
        if model is None:
            raise ValueError("Loaded dict but no estimator found.")

        payload["pipeline"] = model
        payload["window_size"] = int(raw.get("window_size", 1))
        feats = raw.get("train_window_cols") or raw.get("feats") or []
        payload["train_window_cols"] = list(feats) if isinstance(feats, (list, tuple)) else []
        payload["neg_thr"] = float(raw.get("neg_thr", 0.005))
        payload["pos_thr"] = float(raw.get("pos_thr", 0.995))
        if "scaler" in raw:
            payload["scaler"] = raw["scaler"]
    else:
        payload["pipeline"] = raw
        payload["window_size"] = 1
        payload["train_window_cols"] = []
        payload["neg_thr"] = 0.005
        payload["pos_thr"] = 0.995

    # thresholds override (اختیاری)
    if os.path.exists("train_distribution.json"):
        try:
            with open("train_distribution.json", "r", encoding="utf-8") as jf:
                td = json.load(jf)
            if "neg_thr" in td:
                payload["neg_thr"] = float(td["neg_thr"])
            if "pos_thr" in td:
                payload["pos_thr"] = float(td["pos_thr"])
            logging.info("Train distribution loaded from train_distribution.json")
        except Exception:
            pass

    return payload


# ======================== CSV path resolver ========================
def resolve_timeframe_paths(base_dir: Path, symbol: str) -> Dict[str, str]:
    candidates = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T": [f"{symbol}_5T.csv", f"{symbol}_M5.csv"],
        "1H": [f"{symbol}_1H.csv", f"{symbol}_H1.csv"],
    }
    resolved: Dict[str, str] = {}
    for tf, names in candidates.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists():
                found = str(p)
                break
        if found is None:
            # یک اسکن سبک بین فایل‌های دایرکتوری برای نام‌های هم‌ارزش (حساس‌نبودن به حروف)
            for child in base_dir.iterdir():
                if child.is_file() and any(child.name.lower() == nm.lower() for nm in names):
                    found = str(child)
                    break
        if found:
            resolved[tf] = found

    if "30T" not in resolved:
        raise FileNotFoundError(
            f"Main timeframe '30T' not found under {base_dir}. "
            f"Tried: {', '.join(candidates['30T'])}"
        )
    return resolved


# ====================== Ensure columns order =======================
def ensure_columns(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return X
    X2 = X.copy()
    missing = [c for c in cols if c not in X2.columns]
    for c in missing:
        X2[c] = 0.0
    return X2[cols]


# ================== Optional Production PREP import =================
# اگر کلاس تولیدی موجود باشد از آن استفاده می‌کنیم؛ وگرنه فالبک داخلی داریم
_HAVE_PROD_PREP = True
try:
    from prepare_data_for_train_production import PREPARE_DATA_FOR_TRAIN_PRODUCTION  # type: ignore
except Exception as _e:
    _HAVE_PROD_PREP = False
    logging.getLogger(__name__).warning(
        "prepare_data_for_train_production not found or failed to import; "
        "will use internal fallback PREP class."
    )


# ======================== Internal Fallback PREP ========================
# این کلاس فقط برای اجرا بدون وابستگی است (حداقل‌ها را دارد).
class _FallbackPrep:
    def __init__(self, filepaths: Dict[str, str], main_timeframe: str = "30T", verbose: bool = True):
        if not isinstance(filepaths, dict) or "30T" not in filepaths:
            raise ValueError("filepaths must include '30T'.")
        self.filepaths = dict(filepaths)
        self.main_timeframe = main_timeframe
        self.verbose = bool(verbose)
        self._raw_cache: Dict[str, pd.DataFrame] = {}
        self._loaded = False
        if self.verbose:
            logging.info("[FALLBACK PREP] main_timeframe=%s", self.main_timeframe)

    # ---- utils ----
    @staticmethod
    def _timedelta_to_seconds(df: pd.DataFrame):
        for col in df.columns:
            if pd.api.types.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds()

    @staticmethod
    def _safe_agg_group(key: pd.Timestamp, grp: pd.DataFrame, agg_dict: dict):
        if len(grp) >= 2:
            out = grp.iloc[:-1].agg(agg_dict)
            out.name = key
            return out.to_frame().T
        if len(grp) == 1:
            g = grp.iloc[[0]].copy()
            g.index = pd.DatetimeIndex([key])
            return g
        return None

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # استانداردسازی ستون زمان
        if "time" not in df.columns:
            for c in df.columns:
                if c.lower() in ("date", "datetime", "timestamp"):
                    df.rename(columns={c: "time"}, inplace=True)
                    break
        if "time" not in df.columns:
            raise ValueError("CSV must contain a 'time' column.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")

        # اسامی استاندارد قیمت/حجم
        lowers = {c.lower(): c for c in df.columns}
        for need in ("open", "high", "low", "close", "volume"):
            if need not in df.columns and need in lowers:
                df.rename(columns={lowers[need]: need}, inplace=True)
        return df

    def load_all(self):
        self._raw_cache.clear()
        for tf, fp in self.filepaths.items():
            if not os.path.exists(fp):
                raise FileNotFoundError(f"[{tf}] data file not found: {os.path.abspath(fp)}")
            df = pd.read_csv(fp)
            df = self._clean(df)
            self._raw_cache[tf] = df
        self._loaded = True

    def _engineer_features_for_df(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """حداقل فیچرهای سبک برای سازگاری ابعادی؛ از اندیکاتورهای خارجی خبری نیست."""
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        df.set_index("time", inplace=True)

        prefix = f"{tf}_"

        # چند فیچر سبک
        df[f"{prefix}ma_volume20"] = df["volume"].rolling(20, 1).mean()
        df[f"{prefix}rolling_mean_20"] = df["close"].rolling(20, 1).mean()
        df[f"{prefix}rolling_std_20"] = df["close"].rolling(20, 1).std()

        # تضمین نسخه‌ی prefixed ستون‌های خام
        for base_col in ("open", "high", "low", "close", "volume"):
            pref_col = f"{prefix}{base_col}"
            if pref_col not in df.columns and base_col in df.columns:
                df[pref_col] = df[base_col]

        # ستون‌های زمانی (بدون لیک)
        df.reset_index(inplace=True)
        df.rename(columns={"time": f"{tf}_time"}, inplace=True)
        tcol = f"{tf}_time"
        if df[tcol].notna().any():
            df[f"{prefix}hour"] = pd.to_datetime(df[tcol]).shift(1).dt.hour
            df[f"{prefix}day_of_week"] = pd.to_datetime(df[tcol]).shift(1).dt.dayofweek
            df[f"{prefix}is_weekend"] = df[f"{prefix}day_of_week"].isin([5, 6]).astype(int)

        self._timedelta_to_seconds(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.dropna(how="all", inplace=True)
        return df

    def _resample_to_main(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        if tf == self.main_timeframe:
            return df
        df = df.copy()
        tcol = f"{tf}_time"
        if tcol not in df.columns:
            raise KeyError(f"Expected time column '{tcol}' not found for TF {tf}")
        df.set_index(tcol, inplace=True)

        base_aggs = {
            "open": lambda x: x.iloc[0] if not x.empty else np.nan,
            "high": lambda x: x.expanding().max().shift(1).dropna().iloc[-1] if len(x) > 1 else (x.iloc[0] if not x.empty else np.nan),
            "low": lambda x: x.expanding().min().shift(1).dropna().iloc[-1] if len(x) > 1 else (x.iloc[0] if not x.empty else np.nan),
            "close": lambda x: x.iloc[-2] if len(x) > 1 else np.nan,
            "volume": lambda x: x.iloc[:-1].sum() if len(x) > 1 else 0,
        }

        agg_dict = {}
        for col in df.columns:
            base = None
            if col.endswith("_open"):
                base = base_aggs["open"]
            elif col.endswith("_high"):
                base = base_aggs["high"]
            elif col.endswith("_low"):
                base = base_aggs["low"]
            elif col.endswith("_close"):
                base = base_aggs["close"]
            elif col.endswith("_volume"):
                base = base_aggs["volume"]
            agg_dict[col] = base if base is not None else (lambda x: x.shift(1).iloc[-1] if len(x) > 1 else np.nan)

        resampled_rows = [
            self._safe_agg_group(key, grp, agg_dict)
            for key, grp in df.groupby(pd.Grouper(freq=_normalize_freq(self.main_timeframe)))
        ]
        out = (
            pd.concat([r for r in resampled_rows if r is not None], axis=0)
            if resampled_rows
            else pd.DataFrame(columns=df.columns)
        )
        out = out[~out.index.duplicated(keep="last")]
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out.ffill(inplace=True)
        out.dropna(how="all", inplace=True)
        out.reset_index(inplace=True)
        out.rename(columns={"index": f"{self.main_timeframe}_time"}, inplace=True)
        if f"{self.main_timeframe}_time" not in out.columns:
            out.rename(columns={out.columns[0]: f"{self.main_timeframe}_time"}, inplace=True)
        return out

    def load_data_up_to(self, ts_end: pd.Timestamp) -> pd.DataFrame:
        if not self._loaded:
            self.load_all()

        tfs = list(self.filepaths.keys())
        if self.main_timeframe not in tfs:
            raise KeyError(f"'{self.main_timeframe}' not in provided filepaths")

        processed: Dict[str, pd.DataFrame] = {}
        for tf in tfs:
            raw = self._raw_cache[tf]
            cut_df = raw[raw["time"] <= pd.Timestamp(ts_end)].copy()
            if cut_df.empty:
                processed[tf] = pd.DataFrame(columns=[f"{self.main_timeframe}_time"])
                continue

            feat_df = self._engineer_features_for_df(cut_df, tf)
            final_df = (
                self._resample_to_main(feat_df, tf)
                if tf != self.main_timeframe
                else feat_df.rename(columns={f"{tf}_time": f"{self.main_timeframe}_time"})
            )
            processed[tf] = final_df

        main = processed[self.main_timeframe]
        if main.empty:
            return main

        main = main.copy()
        mt = f"{self.main_timeframe}_time"
        if mt not in main.columns:
            raise KeyError(f"Main time column '{mt}' missing after processing.")
        main.set_index(mt, drop=False, inplace=True)

        for tf, dft in processed.items():
            if tf == self.main_timeframe or dft.empty:
                continue
            tcol = f"{self.main_timeframe}_time"
            if tcol not in dft.columns:
                dft = dft.rename(columns={dft.columns[0]: tcol})
            main = main.join(dft.set_index(tcol, drop=False), how="outer", rsuffix=f"_{tf}")

        main.replace([np.inf, -np.inf], np.nan, inplace=True)
        main.ffill(inplace=True)
        main.dropna(how="all", inplace=True)
        main = main[~main.index.duplicated(keep="last")]
        main.reset_index(drop=True, inplace=True)
        return main

    def ready(
        self,
        data: pd.DataFrame,
        window: int = 1,
        selected_features: Optional[List[str]] = None,
        mode: str = "train",
        with_times: bool = False,
    ):
        close_col = f"{self.main_timeframe}_close"
        if close_col not in data.columns:
            raise ValueError(f"{close_col} missing in merged data.")

        y = (data[close_col].shift(-1) - data[close_col] > 0).astype("float")
        if mode != "train":
            y[:] = 0.0

        time_cols = [
            c
            for c in data.columns
            if c.endswith("_time") or any(tok in c for tok in ["hour", "day_of_week", "is_weekend"])
        ]
        feat_cols = [c for c in data.columns if c not in time_cols + [close_col]]

        base = data[feat_cols].copy()
        df_diff = base.shift(1).diff()
        self._timedelta_to_seconds(df_diff)
        df_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_diff.ffill(inplace=True)
        df_diff.dropna(how="all", inplace=True)
        df_diff.reset_index(drop=True, inplace=True)

        y = y.iloc[:len(df_diff)].reset_index(drop=True)

        if selected_features:
            feats = [f for f in selected_features if f in df_diff.columns]
            if not feats:
                feats = df_diff.columns.tolist()
        else:
            feats = df_diff.columns.tolist()

        X = df_diff[feats].copy()

        if window > 1:
            stacked = np.concatenate([X.shift(i).iloc[window - 1 :].values for i in range(window)], axis=1)
            cols = [f"{c}_tminus{i}" for i in range(window) for c in feats]
            X = pd.DataFrame(stacked, columns=cols)
            y = y.iloc[window - 1 :].reset_index(drop=True)

        tcol = f"{self.main_timeframe}_time" if f"{self.main_timeframe}_time" in data.columns else "time"
        t_idx = pd.to_datetime(data[tcol]).reset_index(drop=True)
        if len(t_idx) > 0:
            t_idx = t_idx.iloc[2:].reset_index(drop=True)
        if window > 1 and len(t_idx) >= window - 1:
            t_idx = t_idx.iloc[window - 1 :].reset_index(drop=True)

        L = min(len(X), len(y), len(t_idx))
        X = X.iloc[:L].reset_index(drop=True)
        y = y.iloc[:L].astype("int64").reset_index(drop=True)
        t_idx = t_idx.iloc[:L].reset_index(drop=True)
        return (X, y, feats, t_idx) if with_times else (X, y, feats)


# ============================== Main ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== live_like_sim_v4 starting (cut BEFORE resample) ===")

    # 1) Load model bundle
    payload = load_payload_best_model("best_model.pkl")
    model = payload["pipeline"]
    cols = payload.get("train_window_cols") or []
    window = int(payload["window_size"])
    neg_thr = float(payload.get("neg_thr", 0.005))
    pos_thr = float(payload.get("pos_thr", 0.995))
    logging.info(
        f"Model loaded | window={window} | feats={len(cols)} | neg_thr={neg_thr:.3f} | pos_thr={pos_thr:.3f}"
    )

    # 2) Resolve CSV paths
    base_dir = Path(args.base_dir).resolve()
    filepaths = resolve_timeframe_paths(base_dir, args.symbol)

    # 3) PREP instance (production or fallback)
    if _HAVE_PROD_PREP:
        try:
            from prepare_data_for_train_production import PREPARE_DATA_FOR_TRAIN_PRODUCTION  # re-import to be safe
            prep = PREPARE_DATA_FOR_TRAIN_PRODUCTION(filepaths=filepaths, main_timeframe="30T", verbose=True)
            prep.load_all()
            logging.info("Using PREPARE_DATA_FOR_TRAIN_PRODUCTION.")
        except Exception as e:
            logging.error(
                "Failed to init/use PREPARE_DATA_FOR_TRAIN_PRODUCTION; switching to fallback. Error: %r", e
            )
            prep = _FallbackPrep(filepaths=filepaths, main_timeframe="30T", verbose=True)
            prep.load_all()
    else:
        prep = _FallbackPrep(filepaths=filepaths, main_timeframe="30T", verbose=True)
        prep.load_all()

    # 4) Build full merged to get tail timestamps
    merged_full = prep.load_data_up_to(pd.Timestamp.max)
    main_t = f"{prep.main_timeframe}_time"
    if merged_full.empty or main_t not in merged_full.columns:
        logging.info("No data available after processing.")
        return

    merged_full[main_t] = pd.to_datetime(merged_full[main_t])
    merged_full.sort_values(main_t, inplace=True)
    merged_full.reset_index(drop=True, inplace=True)

    total = len(merged_full)
    start_idx = max(0, total - args.last_n)
    logging.info(f"Main rows={total} | last_n={args.last_n} | start_idx={start_idx}")

    timestamps: List[pd.Timestamp] = merged_full.loc[start_idx:, main_t].tolist()
    records: List[dict] = []

    # 5) Live-like loop
    for i, ts_end in enumerate(timestamps, start=1):
        if i % 200 == 0:
            logging.info(f"[Live] step {i}/{len(timestamps)} @ {ts_end}")

        merged_end = prep.load_data_up_to(pd.Timestamp(ts_end))
        if merged_end.empty:
            continue

        X, y, _feats = prep.ready(merged_end, window=window, selected_features=cols, mode="train", with_times=False)
        if X.empty:
            continue

        X_last = ensure_columns(X.tail(1).reset_index(drop=True), cols)
        if hasattr(model, "predict_proba"):
            prob1 = float(model.predict_proba(X_last)[:, 1][0])
            if prob1 <= neg_thr:
                yhat = 0
            elif prob1 >= pos_thr:
                yhat = 1
            else:
                yhat = -1
            p_out = prob1
        else:
            yhat = int(model.predict(X_last)[0])
            p_out = np.nan

        y_true = int(y.iloc[-1]) if len(y) > 0 else -1

        records.append(
            {
                "timestamp": ts_end,
                "pred_live": yhat,
                "prob_live": p_out,
                "y_true": y_true,
            }
        )

    live_df = pd.DataFrame.from_records(records)
    if live_df.empty:
        logging.info("No live records produced.")
        return

    # 6) Chimney over the same tail window
    block_tail = merged_full.iloc[start_idx:].copy()
    Xc, yc, _ = prep.ready(block_tail, window=window, selected_features=cols, mode="train", with_times=False)
    if not Xc.empty:
        if hasattr(model, "predict_proba"):
            prob_c = model.predict_proba(ensure_columns(Xc, cols))[:, 1]
            yp = np.full(len(prob_c), -1, dtype=int)
            yp[prob_c <= neg_thr] = 0
            yp[prob_c >= pos_thr] = 1
        else:
            yp = model.predict(ensure_columns(Xc, cols))
            prob_c = np.full(len(yp), np.nan)

        # زمان‌های متناظر برای تراز کردن (با فرض هم‌قدی tail)
        ts_align = block_tail[f"{prep.main_timeframe}_time"].iloc[-len(yp) :].reset_index(drop=True)
        chim_df = pd.DataFrame({"timestamp": ts_align, "pred_chimney": yp})
        live_df = live_df.merge(chim_df, on="timestamp", how="left")
        live_df["agree_pred"] = (live_df["pred_live"] == live_df["pred_chimney"]).astype(int)
    else:
        live_df["pred_chimney"] = -1
        live_df["agree_pred"] = 0

    # 7) Metrics
    def _metrics(df: pd.DataFrame, col: str) -> Tuple[float, float, float]:
        sub = df[df[col] != -1]
        if sub.empty:
            return (float("nan"), float("nan"), 0.0)
        acc = accuracy_score(sub["y_true"], sub[col])
        bacc = balanced_accuracy_score(sub["y_true"], sub[col])
        cover = len(sub) / len(df)
        return (float(acc), float(bacc), float(cover))

    acc_l, bacc_l, cov_l = _metrics(live_df, "pred_live")
    acc_c, bacc_c, cov_c = _metrics(live_df, "pred_chimney")

    logging.info(f"[Final] Live    : acc={acc_l:.3f} bAcc={bacc_l:.3f} cover={cov_l:.3f}")
    logging.info(f"[Final] Chimney : acc={acc_c:.3f} bAcc={bacc_c:.3f} cover={cov_c:.3f}")
    if "agree_pred" in live_df.columns and len(live_df) > 0:
        logging.info(f"[Final] Pred agreement (chimney vs live): {live_df['agree_pred'].mean():.3f}")

    # 8) Save
    out_csv = "predictions_compare_v4.csv"
    live_df.to_csv(out_csv, index=False)
    logging.info(f"Saved: {out_csv}")
    logging.info("=== live_like_sim_v4 completed ===")


if __name__ == "__main__":
    main()
