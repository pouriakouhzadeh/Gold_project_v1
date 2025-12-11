#!/usr/bin/env python3
"""
Full data‑preparation pipeline for GA trainer (legacy‑compatible).
----------------------------------------------------------------
* بازتولید کامل تمام خطوط مهندسی ویژگی نسخهٔ اصلی (≈ 630 سطر)
* ریسامپل ایمن با مدیریت کندل ناقص (first / max / min / last / sum)
* انتخاب ویژگی با TimeSeriesSplit + Mutual Information + حذف همبستگی
* مدیریت داده‌های مرزی + پنجره‌بندی انعطاف‌پذیر
* حذفِ تعطیلات (شنبه/یکشنبه) و رکوردهای تکراری
* بذر تصادفی ثابت (2025) و لاگ‌گیری خلاصه
"""
from __future__ import annotations
import os
import gc
import re
from collections import defaultdict
import logging
import multiprocessing as mp
import warnings
from typing import List, Tuple
from AllIndicatorsNoLeak import AllIndicatorsNoLeak
from custotechIndicators import CustomTechIndicators
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from leakfree_indicators import LeakFreeBatchLive
from joblib import Parallel, delayed
from numba import config as numba_config
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearnex import patch_sklearn
from pathlib import Path
import json
from clear_data import ClearData
from DriftBasedStartDateSuggester import DriftBasedStartDateSuggester
from custom_indicators import (
    CustomCandlestickPattern,
    CustomIchimokuIndicator,
    CustomPivotPointIndicator,
    CustomVolumeRateOfChangeIndicator,
    CustomWilliamsRIndicator,
    KSTCustomIndicator,
    VortexCustomIndicator,
)
from numba_utils import (
    numba_kurtosis,
    numba_last_local_max_idx,
    numba_last_local_min_idx,
    numba_median,
    numba_skew,
    numba_up_count,
)
from time_utils import TimeColumnFixer as TFix
from stable_extra_features import add_stable_extra_features
from strong_feature_selector import StrongFeatureSelector  


patch_sklearn(verbose=False)

# ---------------- LOGGING & WARNINGS ----------------
logging.getLogger("sklearnex").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("daal4py").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.CRITICAL)
numba_config.LOG_LEVEL = "CRITICAL"

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, filename="genetic_algorithm.log", format="%(asctime)s %(levelname)s:%(message)s")

# ---------------- REPRODUCIBILITY ----------------
SEED = 2025
np.random.seed(SEED)


# ---------------- BLACKLIST ----------------

# در بالای prepare_data_for_train.py (جایی که helper قبلی هست)

def _load_feature_blacklist(parity_path: str = "features_parity_summary.csv") -> set[str]:
    """
    لیست فیچرهایی که نباید در آموزش/پیش‌بینی استفاده شوند.
    خروجی: نام فیچرهای *پایه* (بدون suffix های _tminusN).

    منابعی که با هم ادغام می‌شوند:
      1) feature_blacklist.txt
         - هر خط یک نام فیچر پایه؛ مثال: '30T_rsi_14'
      2) feature_blacklist.json
         - JSON array از رشته‌ها
      3) features_compare_summary.csv
         - خروجی compare_feature_feeds / اسکریپت‌های قدیمی
      4) features_parity_summary.csv
         - خروجی batch_vs_live_feature_parity.py
           (ستون‌های feature و ratio_diff / n_diff)
    """

    bl: set[str] = set()

    def _add_base(name: str) -> None:
        """نام فیچر (با یا بدون _tminusN) را به نام پایه تبدیل و در بلاک‌لیست اضافه می‌کند."""
        name = (name or "").strip()
        if not name:
            return
        # حذف suffix مثل _tminus0 یا _tminus12
        base = re.sub(r"_tminus\d+$", "", name)
        if base:
            bl.add(base)

    # ۱) فایل متنی دستی (feature_blacklist.txt)
    txt_path = Path("feature_blacklist.txt")
    if txt_path.exists():
        for ln in txt_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                _add_base(ln)

    # ۲) فایل JSON دستی (feature_blacklist.json)
    json_path = Path("feature_blacklist.json")
    if json_path.exists():
        try:
            arr = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(arr, (list, tuple)):
                for name in arr:
                    if isinstance(name, str) and name.strip():
                        _add_base(name)
        except Exception:
            # اگر JSON خراب بود، کل ورودی را نادیده می‌گیریم
            pass

    # ۳) گزارش قدیمی features_compare_summary.csv
    compare_path = Path("features_compare_summary.csv")
    if compare_path.exists():
        try:
            dfc = pd.read_csv(compare_path)
            if "feature" in dfc.columns:
                if "mismatch_cnt" in dfc.columns:
                    mask = dfc["mismatch_cnt"] > 0
                    col = dfc.loc[mask, "feature"].astype(str)
                else:
                    # اگر ستون mismatch_cnt نبود، کل feature ها را می‌گیریم
                    col = dfc["feature"].astype(str)

                for f in col:
                    _add_base(f)
        except Exception:
            pass

    # ۴) گزارش جدید features_parity_summary.csv (batch vs live)
    parity_path_obj = Path(parity_path)
    if parity_path_obj.exists():
        try:
            dfp = pd.read_csv(parity_path_obj)
            if "feature" in dfp.columns:
                if "ratio_diff" in dfp.columns:
                    # آستانه‌ی قابل تنظیم؛ فعلاً 1e-3
                    mask = dfp["ratio_diff"] > 1e-3
                    col = dfp.loc[mask, "feature"].astype(str)
                else:
                    # اگر ratio_diff نبود، از n_diff استفاده می‌کنیم
                    if "n_diff" in dfp.columns:
                        mask = dfp["n_diff"] > 0
                        col = dfp.loc[mask, "feature"].astype(str)
                    else:
                        # آخرین fallback: همه‌ی feature ها
                        col = dfp["feature"].astype(str)

                for f in col:
                    _add_base(f)
        except Exception:
            pass

    return bl

# ---------------- HELPERS ----------------

def _timedelta_to_seconds(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()

# ---------------- SAFE RESAMPLE ----------------

def _safe_agg_group(key: pd.Timestamp, grp: pd.DataFrame, agg_dict: dict[str, callable]):
    if len(grp) >= 2:
        out = grp.iloc[:-1].agg(agg_dict)
        out.name = key                # اندیس = کلید گروه (لبهٔ چپ با فرکانس 30T)
        return out.to_frame().T
    if len(grp) == 1:
        g = grp.iloc[[0]].copy()
        g.index = pd.DatetimeIndex([key])
        return g
    return None

# ---------------- MAIN CLASS ----------------
class PREPARE_DATA_FOR_TRAIN:
        # ------------------------------------------------------------------
    # LIVE incremental internal state  (keep last two raw rows)
    # ------------------------------------------------------------------
    _live_prev2: pd.DataFrame | None = None       # آخرین دو ردیف خام
    _live_prev_time: pd.Timestamp | None = None   # فقط اگر خواستی ترتیب را چک کنی

        # --------- NEW ---------
    bad_cols_tf: dict[str, set[str]] = defaultdict(set)   # {"30T": {"colA", ...}, "1H": {...}}
    allow_regex = re.compile(r"(?:is_weekend|day_of_week|hour)$", re.I)
        # ---------- NEW: helpers to unify batch & live -----------------
    def _compute_diff(self, data: pd.DataFrame,
                      feat_cols: list[str],
                      strict_cols: bool) -> pd.DataFrame:
        """
        واحد مرکزی diff/shift + پاک‌سازی؛ همه جا فقط این را صدا می‌زنیم.
        """
        df = data[feat_cols].shift(1).diff()
        _timedelta_to_seconds(df)           # تبدیل timedelta به ثانیه

        if not strict_cols and self.bad_cols_tf:
            bad_union = set().union(*self.bad_cols_tf.values())
            df.drop(columns=[c for c in bad_union if c in df.columns],
                    inplace=True, errors="ignore")

        # ستون‌های تقریباً صفر (غیرباینری)
        is_bin = (df.nunique() <= 2).to_dict()
        zero_like = [c for c in df.columns
                    if df[c].abs().max() < 1e-12 and not is_bin.get(c, False)]
        if not strict_cols:                       # ← فقط وقتی strict نیست حذف کن
            df.drop(columns=zero_like, inplace=True, errors="ignore")


        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.dropna(how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _apply_window(self, X_f: pd.DataFrame, y: pd.Series,
                      feats: list[str], window: int,
                      selected_features: list[str]|None,
                      has_tminus: bool):
        """
        تولید ستون‌های _tminus فقط یک‌جا، برای batch و live.
        """
        if window <= 1:
            return X_f, y, feats

        if len(X_f) < window:
            logging.warning("Not enough rows for window=%d", window)
            return pd.DataFrame(), pd.Series(dtype=int), feats

        stacked = np.concatenate(
            [X_f.shift(i).iloc[window-1:].values for i in range(window)],
            axis=1
        )
        X_f = pd.DataFrame(
            stacked,
            columns=[f"{c}_tminus{i}" for i in range(window) for c in feats]
        )
        y = y.iloc[window-1:].reset_index(drop=True)

        if has_tminus and selected_features:
            X_f = X_f[[c for c in selected_features if c in X_f.columns]]

        return X_f, y, feats
    # ---------- END NEW ---------------------------------------------------

    def _detect_bad_cols_tf(
        self,
        df: pd.DataFrame,
        tf: str,
        *,
        windows: tuple[int, ...] = (1,2,3,4,5,6,9,12,20,24,30,34),
        stride: int = 75,
        ratio_thr: float = 0.12,     # ← حتی 12 ٪ خطا کافی است
        min_fail:  int   = 8,        # ← یا حداقل 8 بار خرابی
    ) -> None:
        """
        • bad_zero_nan  : صفر/NaN در آخر پنجره
        • bad_forward   : تغییر ردیف ماقبل‌آخر پس از ورود رکورد جدید
        ستون حذف می‌شود اگر  (fails / tests  >= ratio_thr)  «یا»  fails >= min_fail
        """

        # ─── شمارنده‌های ستونی ─────────────────────────────────
        zero_nan_fail   : Counter[str] = Counter()
        forward_fail    : Counter[str] = Counter()
        zero_nan_tests  : Counter[str] = Counter()
        forward_tests   : Counter[str] = Counter()

        n_segments = 8
        seg_len    = len(df) // n_segments or len(df)

        for seg_idx in range(n_segments):
            seg = df.iloc[seg_idx*seg_len : (seg_idx+1)*seg_len]

            for win in windows:
                if len(seg) <= win:  # پنجره‌ جا نمی‌شود
                    continue

                for start in range(0, len(seg) - win, stride):
                    # ── (1) 0 / NaN در آخر پنجره ──────────────────────
                    last = seg.iloc[start+win-1]
                    bad  = last.isna() | np.isclose(last, 0.0, atol=1e-12)

                    for col in last.index:            # تست برای هر ستون
                        zero_nan_tests[col] += 1
                        if bad[col]:
                            zero_nan_fail[col] += 1

                    # ── (2) forward-looking ───────────────────────────
                    if start + win >= len(seg):
                        continue                      # رکورد اضافه نداریم

                    sub1 = seg.iloc[start:start+win]
                    sub2 = seg.iloc[start:start+win+1]

                    pen_old = sub1.iloc[-1]
                    pen_new = sub2.iloc[-2]

                    changed = (
                        pen_old.notna()
                        & pen_new.notna()
                        & (~np.isclose(pen_old, pen_new,
                                    rtol=1e-6, atol=1e-12))
                    )

                    for col in pen_old.index:
                        forward_tests[col] += 1
                        if changed[col]:
                            forward_fail[col] += 1

        # ─── تصمیم نهایی برای هر ستون ───────────────────────────
        bad_cols: set[str] = set()

        for col in zero_nan_tests:
            # نسبتِ‌ خطا
            r = zero_nan_fail[col] / zero_nan_tests[col]
            if (r >= ratio_thr) or (zero_nan_fail[col] >= min_fail):
                bad_cols.add(col)

        for col in forward_tests:
            r = forward_fail[col] / forward_tests[col]
            if (r >= ratio_thr) or (forward_fail[col] >= min_fail):
                bad_cols.add(col)

        # ─── حذف ستون‌های مجاز (binary / تقویمی) ────────────────
        bad_cols = {c for c in bad_cols if not self.allow_regex.search(c)}

        # ─── به دیکشنری کلاس اضافه کن ───────────────────────────
        self.bad_cols_tf[tf].update(bad_cols)

        if self.verbose:
            z_bad = len({c for c in bad_cols if c in zero_nan_fail})
            f_bad = len({c for c in bad_cols if c in forward_fail})
            print(f"[DETECT-{tf}] ↑{len(bad_cols)} cols  "
                f"(0/NaN ≥{ratio_thr:.0%} or ≥{min_fail} →{z_bad}, "
                f"fwd ≥{ratio_thr:.0%} or ≥{min_fail} →{f_bad})")

    def __init__(self, filepaths: dict[str, str] | None = None, main_timeframe="30T",
                verbose=True, fast_mode: bool = False, strict_disk_feed: bool = False):
        self.main_timeframe = main_timeframe
        self.verbose = verbose
        self.fast_mode = bool(fast_mode)
        self.strict_disk_feed = bool(strict_disk_feed)
        self.train_columns_after_window: List[str] = []

        # ⬅️ دیفالت امن برای مسیرها
        if filepaths is None:
            base = os.environ.get("BASE_DATA_DIR", ".")
            symbol = os.environ.get("SYMBOL", "XAUUSD")
            self.filepaths = {
                "30T": f"{base}/{symbol}_30T.csv",
                "15T": f"{base}/{symbol}_15T.csv",
                "5T":  f"{base}/{symbol}_5T.csv",
                "1H":  f"{base}/{symbol}_1H.csv",
            }
        else:
            self.filepaths = filepaths

        # فقط در حالت معمول (Train) drift-scan شود؛ در fast_mode خاموش
        self.shared_start_date = None
        if (not fast_mode) and (not strict_disk_feed):
            self.drift_finder = DriftBasedStartDateSuggester(self.filepaths)
            self.shared_start_date = self.drift_finder.find_shared_start_date()

            if verbose:
                print(f"📅 Shared drift-aware training start date: {self.shared_start_date}")

        if verbose:
            print("[PREP] Initialised for", main_timeframe)
        logging.info("[INIT] main_timeframe=%s", self.main_timeframe)

        # فقط وقتی drift-scan انجام شده باشد، آن را چاپ کن
        if (not self.fast_mode) and (self.shared_start_date is not None):
            print(f"📅 Shared drift-aware training start date: {self.shared_start_date}")

    # ---------------- EXTRA FEATURES (stable) ----------------
    def _windows_for_tf(self, tf: str) -> tuple[int, ...]:
        """
        نگاشت پنجره‌ها به ازای هر TF (قابل تنظیم).
        """
        if tf in ("5T",):
            return (8, 16, 32)
        if tf in ("15T",):
            return (6, 12, 24)
        if tf in ("30T",):
            return (5, 10, 20)
        if tf in ("1H", "60T"):
            return (4, 8, 16)
        # پیش‌فرض
        return (5, 10, 20)

    def add_extra_features(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        اضافه‌کردن فیچرهای پایدار به «همه‌ی تایم‌فریم‌ها».
        (هیچ look-forward ندارد و سطر آخر ناپایدار تولید نمی‌کند)
        """
        wins = self._windows_for_tf(tf)
        # اگر خواستی لاگ بگیری:
        # if self.verbose: print(f"[{tf}] add_stable_extra_features windows={wins}")
        return add_stable_extra_features(df, tf=tf, windows=wins, use_log_price=True)

        # ================= 1) LOAD & FEATURE ENGINEER =================
    def load_and_process_timeframe(self, tf: str, filepath: str) -> pd.DataFrame:
        # print("Load and process time frame start ...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[{tf}] Data file not found: {os.path.abspath(filepath)}")
        df = ClearData().clean(pd.read_csv(filepath))
        if "time" not in df.columns:
            raise ValueError("'time' column missing in CSV")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)

        if self.shared_start_date and (not self.strict_disk_feed):
            df = df[df["time"] >= self.shared_start_date]
            print(f"[{tf}] ⏳ Trimmed data from {self.shared_start_date.date()}")

        df.set_index("time", inplace=True)
        if self.verbose:
            print(f"len df {tf} = {len(df)}")
        prefix = f"{tf}_"

        # ---------------- SIMPLE ROLLING ----------------
        # df[f"{prefix}ma20"] = df["close"].rolling(20, 1).mean()
        # df[f"{prefix}ma50"] = df["close"].rolling(50, 1).mean()
        df[f"{prefix}ma_volume20"] = df["volume"].rolling(20, 1).mean()
        # df[f"{prefix}return_difference"] = df["close"].diff()
        # df[f"{prefix}roc"] = df["close"].pct_change() * 100
        df[f"{prefix}rolling_mean_20"] = df["close"].rolling(20, 1).mean()
        df[f"{prefix}rolling_std_20"] = df["close"].rolling(20, 1).std()
        df[f"{prefix}rolling_skew_20"] = df["close"].rolling(20, 1).apply(numba_skew, raw=True)
        df[f"{prefix}rolling_kurt_20"] = df["close"].rolling(20, 1).apply(numba_kurtosis, raw=True)
        df[f"{prefix}rolling_median_20"] = df["close"].rolling(20, 1).apply(numba_median, raw=True)
        df[f"{prefix}rolling_up_count_20"] = df["close"].rolling(20, 1).apply(numba_up_count, raw=True)

        # ---------------- TA FEATURES ----------------
        ta_builder = AllIndicatorsNoLeak(
            df.copy(),           # you can still pass a copy if you like
            o="open", h="high", l="low", c="close", v="volume",
            prefix=prefix        # same prefix you used before
        )
        df = ta_builder.add_features(inplace=True) 
        df = df.loc[:, ~df.columns.duplicated()]
        
        # ---------- LEAK‑FREE replacements for KCP/DCP/BBP/CCI/FI/OBV/ADI/Stoch‑RSI/Pivot … ----------
        safe_ind = LeakFreeBatchLive(df, prefix=prefix,
                                    o="open", h="high", l="low", c="close", v="volume")
        df = pd.concat([df, safe_ind.build()], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]  
        
        # ---------------- MANUAL & CUSTOM INDICATORS ----------------
        # df: دیتافریم خام، prefix همان متغیر قبلی
        ind = CustomTechIndicators(df, prefix=prefix,
                                o="open", h="high", l="low", c="close", v="volume")
        df = ind.add_features(inplace=True)   # همهٔ ستون‌های جدید با یک فراخوانی
        df = df.loc[:, ~df.columns.duplicated()]

        # ---- CUSTOM INDICATORS ----
        kst = KSTCustomIndicator(df["close"], 10, 15, 20, 30, 10, 10, 10, 15, 9, fillna=True)
        df[f"{prefix}kst_main"] = kst.kst(); df[f"{prefix}kst_signal"] = kst.kst_signal(); df[f"{prefix}kst_diff"] = kst.kst_diff()
        vtx = VortexCustomIndicator(df["high"], df["low"], df["close"], 14, fillna=True)
        df[f"{prefix}vortex_pos"] = vtx.vortex_pos(); df[f"{prefix}vortex_neg"] = vtx.vortex_neg()
        ichi = CustomIchimokuIndicator(df["high"], df["low"], df["close"], 9, 26, 52)
        df[f"{prefix}ichimoku_conversion_line"] = ichi.ichimoku_conversion_line()
        df[f"{prefix}ichimoku_base_line"] = ichi.ichimoku_base_line()
        df[f"{prefix}ichimoku_a"] = (df[f"{prefix}ichimoku_conversion_line"] + df[f"{prefix}ichimoku_base_line"])/2
        df[f"{prefix}ichimoku_b"] = (df["high"].rolling(52).max() + df["low"].rolling(52).min())/2
        df[f"{prefix}williams_r"] = CustomWilliamsRIndicator(df["high"], df["low"], df["close"], 14).williams_r()
        df[f"{prefix}vroc"] = CustomVolumeRateOfChangeIndicator(df["volume"], 20).volume_rate_of_change()
        piv = CustomPivotPointIndicator(df["high"], df["low"], df["close"], 5)
        # df[f"{prefix}pivot"] = piv.pivot(); df[f"{prefix}support_1"] = piv.support_1(); df[f"{prefix}support_2"] = piv.support_2(); df[f"{prefix}resistance_1"] = piv.resistance_1(); df[f"{prefix}resistance_2"] = piv.resistance_2()
        candle = CustomCandlestickPattern(df["open"], df["high"], df["low"], df["close"])
        df[f"{prefix}engulfing"] = candle.engulfing(); df[f"{prefix}doji"] = candle.doji()
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4; ha_open = ha_close.shift(1).ffill()
        df[f"{prefix}heikin_ashi_open"] = ha_open; df[f"{prefix}heikin_ashi_close"] = ha_close
        df[f"{prefix}range_close_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        df[f"{prefix}bull_power"] = (df["close"] - df["low"]) / ((df["high"] - df["low"]) + 1e-9)
        df[f"{prefix}bars_from_local_max_20"] = df["close"].rolling(20, 1).apply(numba_last_local_max_idx, raw=True)
        df[f"{prefix}bars_from_local_min_20"] = df["close"].rolling(20, 1).apply(numba_last_local_min_idx, raw=True)
        df[f"{prefix}rsi_macd"] = df[f"{prefix}rsi_14"] * df[f"{prefix}macd"]
        # df[f"{prefix}ma20_ma50_ratio"] = df[f"{prefix}ma20"] / (df[f"{prefix}ma50"] + 1e-9)
        # ---------- اطمینان از وجود ستون‌های خامِ پیشونددار ----------
        for base_col in ("open", "high", "low", "close", "volume"):
            pref_col = f"{prefix}{base_col}"
            if pref_col not in df.columns:
                df[pref_col] = df[base_col]

        df.replace([np.inf, -np.inf], np.nan, inplace=True); df.ffill(inplace=True); df.dropna(how="all", inplace=True)
        
        # ---- NEW: detect bad cols for this timeframe BEFORE resample ----
        if not getattr(self, "fast_mode", False):
            self._detect_bad_cols_tf(df, tf)

        # ---------------- SAFE RESAMPLE ----------------
        # print("Safe resample start ...")
        if tf != self.main_timeframe:
            base_aggs = {
                "open": lambda x: x.iloc[0] if not x.empty else np.nan,
                "high": lambda x: x.expanding().max().shift(1).dropna().iloc[-1] if len(x) > 1 else x.iloc[0],
                "low": lambda x: x.expanding().min().shift(1).dropna().iloc[-1] if len(x) > 1 else x.iloc[0],
                "close": lambda x: x.iloc[-2] if len(x) > 1 else np.nan,
                "volume": lambda x: x.iloc[:-1].sum() if len(x) > 1 else 0,
            }
            agg_dict = {
                col: (base_aggs[col] if col in base_aggs else (lambda x: x.shift(1).iloc[-1] if len(x) > 1 else np.nan))
                for col in df.columns
            }
            resampled_rows = [
            _safe_agg_group(key, grp, agg_dict)
            for key, grp in df.groupby(pd.Grouper(freq=self.main_timeframe))
            ]
            df = pd.concat([r for r in resampled_rows if r is not None]) if resampled_rows else pd.DataFrame(columns=df.columns)
            df = df[~df.index.duplicated(keep="last")]
            df.replace([np.inf, -np.inf], np.nan, inplace=True); df.ffill(inplace=True); df.dropna(how="all", inplace=True)
            if self.verbose:
                print(f"[{tf}] after resample → rows={len(df)}, cols={df.shape[1]}")

        # print("Safe resample finished")
        # ---------------- LOG SCALE VOLUME ----------------
        heavy_regex = r"(?:^|_)(?:volume|obv|vpt|adi|nvi|eom|vr)(?:_|$)"
        heavy_cols = df.columns[df.columns.str.contains(heavy_regex, regex=True, case=False)]
        df[heavy_cols] = np.sign(df[heavy_cols]) * np.log1p(np.abs(df[heavy_cols]))

        # ---------------- add extra stable features (for ALL TFs) ---------------
        df = self.add_extra_features(df, tf=tf)

        # ---------------- CALENDAR COLUMNS ----------------
        df.reset_index(inplace=True)
        if "time" not in df.columns:
            df.rename(columns={df.columns[0]: "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.rename(columns={"time": f"{tf}_time"}, inplace=True)
        if df[f"{tf}_time"].notna().any():
            df[f"{prefix}hour"] = df[f"{tf}_time"].shift(1).dt.hour
            df[f"{prefix}day_of_week"] = df[f"{tf}_time"].shift(1).dt.dayofweek
            df[f"{prefix}is_weekend"] = df[f"{prefix}day_of_week"].isin([5, 6]).astype(int)
        _timedelta_to_seconds(df)

        # print("Load and process time frame finished")
        return df

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 300,
        n_splits: int = 3,
    ) -> List[str]:
        """Time series aware feature selection (Variance→Corr filter→Mutual Info).

        * فقط ستون‌های عددی استفاده می‌شود (datetime / object حذف می‌شود).
        * NaN / Inf قبل از VarianceThreshold و MI با میانه‌ی ستون پر می‌شود.
        """
        logger = logging.getLogger(__name__)

        # --- فقط ستون‌های عددی ---
        num_cols = X.select_dtypes(include=[np.number]).columns
        dropped = [c for c in X.columns if c not in num_cols]
        if dropped:
            logger.info("[select_features] dropping non-numeric columns: %s", dropped)
        if len(num_cols) == 0:
            return []

        X = X[num_cols].copy()

        tscv = TimeSeriesSplit(n_splits=n_splits)
        pool: List[str] = []

        for tr_idx, _ in tscv.split(X):
            X_tr = X.iloc[tr_idx].copy()
            y_tr = y.iloc[tr_idx]

            # --- پاک‌سازی NaN / Inf در این فولد ---
            X_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
            if X_tr.isna().any().any():
                X_tr.fillna(X_tr.median(), inplace=True)

            if X_tr.shape[1] == 0:
                continue

            # 1) Variance threshold
            vt = VarianceThreshold(0.01)
            try:
                vt.fit(X_tr)
            except Exception as e:
                logger.warning("[select_features] VarianceThreshold failed: %s", e)
                continue

            cols_var = X_tr.columns[vt.get_support()]
            if len(cols_var) == 0:
                continue
            X_var = X_tr[cols_var]

            # 2) Correlation filter
            corr = X_var.corr().abs()
            if corr.empty:
                continue

            mask_corr = np.triu(np.ones_like(corr, bool), k=1)
            upper = corr.where(mask_corr)
            drop_cols = [c for c in upper.columns if any(upper[c] > 0.9)]
            X_corr = X_var.drop(columns=drop_cols, errors="ignore")
            if X_corr.empty:
                continue

            # 3) Mutual information (روی داده‌ی scale شده و بدون NaN)
            X_filled = X_corr.replace([np.inf, -np.inf], np.nan)
            if X_filled.isna().any().any():
                X_filled = X_filled.fillna(X_filled.median())

            X_scaled = MinMaxScaler().fit_transform(X_filled)

            mask_y = pd.Series(y_tr).notna().to_numpy()
            if mask_y.sum() < 2:
                continue

            y_arr = pd.Series(y_tr).loc[mask_y].astype(np.int64).to_numpy(copy=False)
            mi = mutual_info_classif(X_scaled[mask_y], y_arr, random_state=SEED)

            pool.extend(
                pd.Series(mi, index=X_corr.columns)
                .nlargest(top_k)
                .index
                .tolist()
            )

        if not pool:
            return []

        counts = pd.Series(pool).value_counts()
        return counts[counts >= n_splits].index[:top_k].tolist()

    # ================= 3) READY (X, y, WINDOW) =================
    def ready(
            self,
            data: pd.DataFrame,
            window: int = 1,
            selected_features: List[str] | None = None,
            mode: str = "train",
            with_times: bool = False,
            predict_drop_last: bool = False,
            train_drop_last: bool = False,
            apply_strong_fs: bool = False,
            strong_fs_max_features: int = 300,
        ):
        """
        نسخه‌ی هم‌معنا با «پیش‌بینی کندل بعدی»:

        - در TRAIN:
            * برای هر کندل با زمان t، برچسب y(t) = 1{ close(t+1) > close(t) }.
            * آخرین کندل (که close(t+1) ندارد) به طور خودکار حذف می‌شود.
        - در PREDICT:
            * اصلاً تارگت واقعی استفاده نمی‌شود؛ فقط فیچرها ساخته می‌شود.
            * آخرین کندل (t_now) هم نگه داشته می‌شود تا مدل روی آن t→t+1 را پیش‌بینی کند.
        - پنجره‌بندی (window>1):
            * سطر خروجی با زمان t همان ستونی است که از کندل‌های t, t-1, ..., t-window+1 ساخته شده،
              و همچنان y(t) = 1{ close(t+1) > close(t) } را هدف می‌گیرد.
        """

        close_col = f"{self.main_timeframe}_close"
        tcol = (
            f"{self.main_timeframe}_time"
            if f"{self.main_timeframe}_time" in data.columns
            else "time"
        )
        if close_col not in data.columns:
            raise ValueError(f"{close_col} missing")

        # --- کپی امن و سری قیمت/زمان ---
        df = data.copy().reset_index(drop=True)
        close = df[close_col].astype(float).reset_index(drop=True)
        t_idx = pd.to_datetime(df[tcol]).reset_index(drop=True)

        # ----------------- برچسب‌دهی -----------------
        if mode == "train":
            # y(t) = 1{ close(t+1) > close(t) }  با NaN برای آخرین ردیف
            close_next = close.shift(-1)
            y = (close_next > close).astype("float64")
            y[close_next.isna()] = np.nan  # آخرین ردیف، تارگت ندارد

            valid = y.notna()
            df = df.loc[valid].reset_index(drop=True)
            t_idx = t_idx.loc[valid].reset_index(drop=True)
            close = close.loc[valid].reset_index(drop=True)
            y = y.loc[valid].reset_index(drop=True)

            # --- PATCH: حذف روزهای «تعطیل‌مانند» با تارگت ثابت و نوسان بسیار کم ---
            # تعریف «تعطیل‌مانند»:
            #   1) در آن روز y فقط یک مقدار دارد (همه 0 یا همه 1)
            #   2) رِنج close آن روز بسیار کوچک است (بازار تقریباً فلت)
            day_idx = t_idx.dt.normalize()
            grp_df = pd.DataFrame({"y": y, "close": close, "day": day_idx})

            bad_days = []
            for d, g in grp_df.groupby("day"):
                if g["y"].nunique(dropna=True) <= 1:
                    c_range = float(g["close"].max() - g["close"].min())
                    c_level = float(max(1.0, g["close"].mean()))
                    # رِنج کمتر از 0.01% سطح قیمت ⇒ بازار تقریباً بسته است
                    if c_range <= 1e-4 * c_level:
                        bad_days.append(d)

            if bad_days:
                mask = ~day_idx.isin(bad_days)
                df = df.loc[mask].reset_index(drop=True)
                y = y.loc[mask].reset_index(drop=True)
                t_idx = t_idx.loc[mask].reset_index(drop=True)
                close = close.loc[mask].reset_index(drop=True)

        else:
            # در پیش‌بینی، y واقعی لازم نداریم؛ بعداً صفر می‌کنیم
            y = pd.Series(np.zeros(len(df), dtype=np.int8))

        # ----------------- انتخاب ستون‌های فیچر پایه -----------------

        # ----------------- انتخاب ستون‌های فیچر پایه -----------------
        time_tokens = ("hour", "day_of_week", "is_weekend")
        time_cols = [
            c
            for c in df.columns
            if c.endswith("_time")
            or c == "time"
            or any(tok in c for tok in time_tokens)
        ]
        base_candidates = [c for c in df.columns if c not in time_cols + [close_col]]

        # فقط ستون‌های عددی (حذف datetime / object، مثل index با dtype عجیب)
        num_cols = (
            df[base_candidates]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )
        dropped_non_numeric = [c for c in base_candidates if c not in num_cols]
        if dropped_non_numeric:
            logging.getLogger(__name__).info(
                "[ready] dropping non-numeric base feature columns: %s",
                dropped_non_numeric,
            )
        base_candidates = num_cols

        # بلاک‌لیست فیچرها (از فایل‌ها)
        black = _load_feature_blacklist()
        if black:
            base_candidates = [c for c in base_candidates if c not in black]


        # ----------------- تفسیر selected_features -----------------
        import re as _re

        tminus_regex = _re.compile(r"_tminus\d+$")
        has_tminus = bool(
            selected_features
            and any(tminus_regex.search(str(f)) for f in selected_features)
        )

        if has_tminus:
            # selected_features لیست فیچرهای پنجره‌ای است (با _tminusN)
            # این‌جا پایه‌ی آنها را در df پیدا می‌کنیم
            base_from_sel = {tminus_regex.sub("", str(f)) for f in selected_features}
            feats_base = [c for c in base_candidates if c in base_from_sel]
        else:
            if selected_features is None:
                # فقط در TRAIN انتخاب فیچر انجام شود
                if mode == "train":
                    feats_base = self.select_features(df[base_candidates], y)
                else:
                    feats_base = base_candidates
            elif selected_features == []:
                # "[]": یعنی همه‌ی فیچرهای موجود
                feats_base = base_candidates
            else:
                # لیست اسامی فیچرهای پایه
                feats_base = [c for c in selected_features if c in base_candidates]

        if not feats_base:
            logger = logging.getLogger(__name__)
            # سناریوی اصلی مشکل‌ساز: TRAIN + selected_features=None
            # یعنی select_features هیچ ستونی برنگردانده؛
            # برای این‌که فاز نهایی GA نخوابد، روی تمام base_candidates برمی‌گردیم.
            if (selected_features is None) and base_candidates:
                logger.warning(
                    "[ready] feats_base empty (mode=%s) – falling back to all %d numeric base features",
                    mode,
                    len(base_candidates),
                )
                feats_base = base_candidates
            else:
                # در سایر حالت‌ها اگر همچنان فیچری نداریم، ناچاریم خروجی خالی بدهیم
                if with_times:
                    return (
                        pd.DataFrame(),
                        pd.Series(dtype="int64"),
                        [],
                        pd.Series(dtype=float),
                        pd.Series(dtype="datetime64[ns]"),
                    )
                else:
                    return (
                        pd.DataFrame(),
                        pd.Series(dtype="int64"),
                        [],
                        pd.Series(dtype=float),
                    )
        X_base = df[feats_base].copy()

        # ----------------- پنجره‌بندی (window > 1) -----------------
        if window > 1:
            if len(X_base) < window:
                logging.warning(
                    "Not enough rows (%d) for window=%d", len(X_base), window
                )
                if with_times:
                    return (
                        pd.DataFrame(),
                        pd.Series(dtype="int64"),
                        [],
                        pd.Series(dtype=float),
                        pd.Series(dtype="datetime64[ns]"),
                    )
                else:
                    return (
                        pd.DataFrame(),
                        pd.Series(dtype="int64"),
                        [],
                        pd.Series(dtype=float),
                    )

            mats = [X_base.shift(i) for i in range(window)]
            Xw = (
                pd.concat(mats, axis=1)
                .iloc[window - 1 :]
                .reset_index(drop=True)
            )
            col_names = [
                f"{c}_tminus{i}" for i in range(window) for c in feats_base
            ]
            Xw.columns = col_names[: Xw.shape[1]]

            # هم‌تراز کردن y, زمان و قیمت با پنجره
            y = y.iloc[window - 1 :].reset_index(drop=True)
            t_idx = t_idx.iloc[window - 1 :].reset_index(drop=True)
            close = close.iloc[window - 1 :].reset_index(drop=True)

            if selected_features and len(selected_features) > 0:
                # فقط فیچرهای خواسته شده را نگه دار (ترتیب همان selected_features)
                cols_keep = [c for c in selected_features if c in Xw.columns]
                X_f = Xw[cols_keep]
            else:
                X_f = Xw
        else:
            X_f = X_base

        # ----------------- هم‌قد کردن نهایی -----------------
        L = min(len(X_f), len(y), len(t_idx), len(close))
        X_f = X_f.iloc[:L].reset_index(drop=True)
        y = y.iloc[:L].reset_index(drop=True)
        t_idx = t_idx.iloc[:L].reset_index(drop=True)
        close = close.iloc[:L].reset_index(drop=True)

        # ----------------- drop-last اختیاری -----------------
        if mode == "train" and train_drop_last and len(X_f) > 0:
            X_f = X_f.iloc[:-1].reset_index(drop=True)
            y = y.iloc[:-1].reset_index(drop=True)
            t_idx = t_idx.iloc[:-1].reset_index(drop=True)
            close = close.iloc[:-1].reset_index(drop=True)

        # ----------------- StrongFeatureSelector (فقط TRAIN نهایی) -----------------
        if (
            mode == "train"
            and apply_strong_fs
            and selected_features is None    # یعنی نه CV در GA، نه کال با لیست خاص
            and X_f.shape[1] > int(strong_fs_max_features)
        ):
            fs_logger = logging.getLogger(__name__)
            MAX_FEATS = int(strong_fs_max_features)

            fs_logger.info(
                "[ready] StrongFeatureSelector input shape: rows=%d, cols=%d",
                X_f.shape[0],
                X_f.shape[1],
            )

            selector = StrongFeatureSelector(
                max_features=MAX_FEATS,
                pre_selection_factor=3,
                random_state=SEED,
                n_estimators=256,
                n_jobs=1,          # برای جلوگیری از oversubscription در GA
                corr_n_jobs=1,
            )

            try:
                X_selected = selector.fit_transform(X_f, y)
                selected_cols = list(X_selected.columns)

                if len(selected_cols) == 0:
                    fs_logger.warning(
                        "[ready] StrongFeatureSelector returned 0 columns – "
                        "falling back to first %d features (out of %d).",
                        MAX_FEATS,
                        X_f.shape[1],
                    )
                    # fallback: truncate به  MAX_FEATS
                    X_f = X_f.iloc[:, :MAX_FEATS].copy()
                else:
                    fs_logger.info(
                        "[ready] StrongFeatureSelector reduced features from %d to %d",
                        X_f.shape[1],
                        len(selected_cols),
                    )
                    X_f = X_selected

            except Exception as e:
                fs_logger.warning(
                    "[ready] StrongFeatureSelector failed (%s); "
                    "falling back to first %d features (out of %d).",
                    e,
                    MAX_FEATS,
                    X_f.shape[1],
                )
                if X_f.shape[1] > MAX_FEATS:
                    X_f = X_f.iloc[:, :MAX_FEATS].copy()

        if mode != "train":
            if predict_drop_last and len(X_f) > 0:
                X_f = X_f.iloc[:-1].reset_index(drop=True)
                t_idx = t_idx.iloc[:-1].reset_index(drop=True)
                close = close.iloc[:-1].reset_index(drop=True)
            # در پیش‌بینی، y مصرف نمی‌شود → صفر نگه می‌داریم
            y = np.zeros(len(X_f), dtype=np.int64)

        # ----------------- خروجی نهایی -----------------
        feats_final = list(X_f.columns)

        if mode == "train":
            y = y.astype("int64")
            # ذخیره‌ی ستون‌های نهایی بعد از پنجره برای سازگاری با live / threshold / test
            self.train_columns_after_window = feats_final

        if with_times:
            return X_f, y, feats_final, close, t_idx
        else:
            return X_f, y, feats_final, close

    # ================= 4) READY_INCREMENTAL =================
    def ready_incremental(
        self,
        data_window: pd.DataFrame,
        window: int = 1,
        selected_features: List[str] | None = None,
        with_times: bool = False,
        predict_drop_last: bool = False,   # ❗ دیفالت جدید = False
    ):
        if not hasattr(self, "_live_prev2") or self._live_prev2 is None:
            self._live_prev2 = data_window.iloc[-2:].copy()
            return (pd.DataFrame(), [], None) if with_times else (pd.DataFrame(), [])

        concat = pd.concat([self._live_prev2, data_window], ignore_index=True)

        X_full, _, feats, price_raw, t_idx = self.ready(
            concat,
            window=window,
            selected_features=selected_features,
            mode="predict",
            with_times=True,
            predict_drop_last=predict_drop_last,
        )

        self._live_prev2 = data_window.iloc[-2:].copy()

        if X_full.empty:
            return (pd.DataFrame(), feats, None) if with_times else (pd.DataFrame(), feats)

        X_last = X_full.tail(1).reset_index(drop=True)
        t_feat = (
            pd.to_datetime(t_idx.iloc[-1])
            if (t_idx is not None and len(t_idx) > 0)
            else None
        )
        return (X_last, feats, t_feat) if with_times else (X_last, feats)

    # ================= 5) LOAD & MERGE =================
    def load_data(self) -> pd.DataFrame:
        if not self.filepaths or not isinstance(self.filepaths, dict):
            raise ValueError("[load_data] filepaths not provided or invalid")

        logging.info("[load_data] parallel load %d timeframes", len(self.filepaths))

        # ⬅️ فقط فایل‌های موجود را نگه داریم؛ نبودن 30T = خطای فوری
        existing = {tf: fp for tf, fp in self.filepaths.items() if os.path.exists(fp)}
        missing  = {tf: fp for tf, fp in self.filepaths.items() if tf not in existing}

        for tf, fp in missing.items():
            print(f"⚠️ File not found: {os.path.abspath(fp)}")

        if self.main_timeframe not in existing:
            raise FileNotFoundError(f"[load_data] Main timeframe '{self.main_timeframe}' file is missing: "
                                    f"{os.path.abspath(self.filepaths.get(self.main_timeframe, ''))}")

        # اگر فقط 30T داری، همین کافی‌ست؛ بقیه تایم‌فریم‌ها اختیاری هستند
        self.filepaths = existing
        logging.info("[load_data] using %d existing timeframes (%s)",
                    len(self.filepaths), ", ".join(sorted(self.filepaths.keys())))

        # ---------- 1) موازی-خوانی و مهندسی هر تایم‌فریم ----------
        dfs = Parallel(
            n_jobs=min(mp.cpu_count(), len(self.filepaths)),
            backend="loky"
        )(
            delayed(self.load_and_process_timeframe)(tf, fp)
            for tf, fp in self.filepaths.items()
        )

        # ---------- 2) ادغام روی تایم‌فریم اصلی ----------
        main_tf  = self.main_timeframe
        tfs = list(self.filepaths.keys())

        # پیدا کردن ایندکس تایم‌فریم اصلی
        try:
            idx_main = tfs.index(self.main_timeframe)
        except ValueError:
            raise KeyError(f"Main timeframe '{self.main_timeframe}' not in filepaths")

        df0 = dfs[idx_main]
        main_tf = self.main_timeframe

        if f"{main_tf}_time" not in df0.columns:
            if "time" in df0.columns:
                df0[f"{main_tf}_time"] = pd.to_datetime(df0["time"], errors="coerce")
            else:
                raise KeyError(f"Missing '{main_tf}_time' and 'time' columns in main timeframe dataframe.")
        main_df = df0.set_index(f"{main_tf}_time", drop=False)

        # join بقیه‌ی تایم‌فریم‌ها
        for j, tf in enumerate(tfs):
            if j == idx_main:
                continue
            dfj = dfs[j]
            main_df = main_df.join(
                dfj.set_index(f"{tf}_time", drop=False),
                how="outer",
                rsuffix=f"_{tf}",
            )

        # ---------- 3) حذف ستون‌های «ناپایدار» که قبلاً برای هر TF کشف شده ----------
        for tf, bad_set in self.bad_cols_tf.items():
            if not bad_set:
                continue
            cols_to_drop = [c for c in main_df.columns if c in bad_set]
            if cols_to_drop:
                main_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
                if self.verbose:
                    print(f"[DROP] Removed {len(cols_to_drop)} unstable cols from {tf}")

        # ---------- 4) پاک‌سازی NaN/Inf و فوروارد-پرکردن ----------
        main_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        main_df.ffill(inplace=True)
        main_df.dropna(how="all", inplace=True)

        # ---------- 5) حذف سطرهای تعطیلات (شنبه/یکشنبه) ----------
        if isinstance(main_df.index, pd.DatetimeIndex):
            main_df = main_df[~main_df.index.dayofweek.isin([5, 6])]
            main_df.ffill(inplace=True)

        # ---------- 6) نهایی‌سازی اندیس / حذف duplications ----------
        main_df.reset_index(drop=False, inplace=True)
        tcol = f"{main_tf}_time"
        if tcol not in main_df.columns:             # ایمنی اگر ستون جابه‌جا شد
            main_df.rename(columns={main_df.columns[0]: tcol}, inplace=True)
        main_df = main_df.loc[~main_df[tcol].duplicated(keep="last")]

        logging.info("[load_data] Final shape=%s", main_df.shape)
        gc.collect()
        return main_df

    # ================= 6) OUTER INTERFACE =================
    def get_prepared_data(self, window=1, mode="train") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        لایه‌ی نهایی آماده‌سازی داده برای استفاده در مدل‌های ML / DL.

        - تمام منطق مهندسی ویژگی و پنجره‌بندی در ready انجام می‌شود.
        - در حالت TRAIN، اگر apply_strong_fs=True باشد، یک مرحله‌ی
          StrongFeatureSelector روی ماتریس نهایی X (بعد از window) اجرا می‌شود
          و حداکثر strong_fs_max_features ستون نگه داشته می‌شود.
        """
        merged = self.load_data()
        X, y, feats, _ = self.ready(
            merged,
            window=window,
            mode=mode,
            apply_strong_fs=(mode == "train"),
            strong_fs_max_features=300,
        )

        if mode == "train":
            # ready خودش train_columns_after_window را ست می‌کند،
            # ولی برای اطمینان sync می‌کنیم
            self.train_columns_after_window = list(feats)

        return X, y, feats



if __name__ == "__main__":
    prep = PREPARE_DATA_FOR_TRAIN(verbose=True)
    X, y, f = prep.get_prepared_data(window=1, mode="train")
    print("Shapes:", X.shape, y.shape, len(f))

