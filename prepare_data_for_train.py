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

import gc
import re
from collections import defaultdict
import logging
import multiprocessing as mp
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import ta
from joblib import Parallel, delayed
from numba import config as numba_config
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearnex import patch_sklearn
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

# ---------------- HELPERS ----------------

def _timedelta_to_seconds(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()

# ---------------- SAFE RESAMPLE ----------------

def _safe_agg_group(grp: pd.DataFrame, agg_dict: dict[str, callable]):
    if len(grp) >= 2:
        return grp.iloc[:-1].agg(agg_dict).to_frame().T
    if len(grp) == 1:
        return grp.iloc[[0]]
    return None

# ---------------- MAIN CLASS ----------------
class PREPARE_DATA_FOR_TRAIN:
        # --------- NEW ---------
    bad_cols_tf: dict[str, set[str]] = defaultdict(set)   # {"30T": {"colA", ...}, "1H": {...}}
    allow_regex = re.compile(r"(?:is_weekend|day_of_week|hour)$", re.I)

    def _detect_bad_cols_tf(
        self,
        df: pd.DataFrame,
        tf: str,
        windows: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        stride: int = 500,
    ) -> None:
        """
        ستون‌هایی را که در «سطرِ آخر» یک پنجره مقدار NaN یا 0 می‌گیرند
        شناسایی می‌کند و در  self.bad_cols_tf[tf]  ذخیره می‌کند.
        اسکن روی چهار قطعه‌ی مستقل از کل سری انجام می‌شود تا ناحیه‌ای
        خاص از داده پنهان نماند.
        """
        bad: set[str] = set()
        n_segments = 8
        seg_len = len(df) // n_segments or len(df)

        for seg_idx in range(n_segments):
            segment = df.iloc[seg_idx * seg_len : (seg_idx + 1) * seg_len]

            for win in windows:
                if len(segment) <= win:
                    continue
                for start in range(0, len(segment) - win, stride):
                    # ردیفِ پایانیِ پنجره
                    last_row = segment.iloc[start + win - 1]

                    # مقادیر NaN یا «تقریباً صفر» معیوب‌اند
                    mask_bad = last_row.isna() | np.isclose(last_row, 0, atol=1e-12)

                    if mask_bad.any():
                        bad.update(last_row.index[mask_bad])


        # ستون‌هایی که ذاتاً ۰/۱ یا نشانگر زمان هستند استثناء می‌شوند
        bad_final = {c for c in bad if not self.allow_regex.search(c)}
        self.bad_cols_tf[tf].update(bad_final)

        if self.verbose:
            print(f"[DETECT-{tf}] found {len(bad_final)} unstable columns")

    def __init__(self, filepaths: dict[str, str] | None = None, main_timeframe="30T", verbose=True):
        defaults = {"30T": "XAUUSD_M30.csv", "1H": "XAUUSD_H1.csv", "15T": "XAUUSD_M15.csv", "5T": "XAUUSD_M5.csv"}
        self.filepaths = filepaths or defaults
        self.main_timeframe = main_timeframe
        self.verbose = verbose
        self.train_columns_after_window: List[str] = []
        self.drift_finder = DriftBasedStartDateSuggester(self.filepaths)
        self.shared_start_date = self.drift_finder.find_shared_start_date()

        if verbose:
            print("[PREP] Initialised for", main_timeframe)
        logging.info("[INIT] main_timeframe=%s", self.main_timeframe)

        print(f"📅 Shared drift-aware training start date: {self.shared_start_date}")

    # ================= 1) LOAD & FEATURE ENGINEER =================
    def load_and_process_timeframe(self, tf: str, filepath: str) -> pd.DataFrame:
        # print("Load and process time frame start ...")
        df = ClearData().clean(pd.read_csv(filepath))
        if "time" not in df.columns:
            raise ValueError("'time' column missing in CSV")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)

        if self.shared_start_date:
            df = df[df["time"] >= self.shared_start_date]
            print(f"[{tf}] ⏳ Trimmed data from {self.shared_start_date.date()}")
        df.set_index("time", inplace=True)
        print(f"len df {tf} = {len(df)}")
        prefix = f"{tf}_"

        # ---------------- SIMPLE ROLLING ----------------
        df[f"{prefix}ma20"] = df["close"].rolling(20, 1).mean()
        df[f"{prefix}ma50"] = df["close"].rolling(50, 1).mean()
        df[f"{prefix}ma_volume20"] = df["volume"].rolling(20, 1).mean()
        df[f"{prefix}return_difference"] = df["close"].diff()
        df[f"{prefix}roc"] = df["close"].pct_change() * 100
        df[f"{prefix}rolling_mean_20"] = df["close"].rolling(20, 1).mean()
        df[f"{prefix}rolling_std_20"] = df["close"].rolling(20, 1).std()
        df[f"{prefix}rolling_skew_20"] = df["close"].rolling(20, 1).apply(numba_skew, raw=True)
        df[f"{prefix}rolling_kurt_20"] = df["close"].rolling(20, 1).apply(numba_kurtosis, raw=True)
        df[f"{prefix}rolling_median_20"] = df["close"].rolling(20, 1).apply(numba_median, raw=True)
        df[f"{prefix}rolling_up_count_20"] = df["close"].rolling(20, 1).apply(numba_up_count, raw=True)

        # ---------------- TA FEATURES ----------------
        ta_feats = ta.add_all_ta_features(df.copy(), open="open", high="high", low="low", close="close", volume="volume", fillna=False).add_prefix(prefix)
        ta_feats.drop(columns=[c for c in ta_feats if "ichimoku" in c.lower()], inplace=True, errors="ignore")
        df = pd.concat([df, ta_feats], axis=1)

        # ---------------- MANUAL & CUSTOM INDICATORS ----------------
        from ta.trend import PSARIndicator
        df[f"{prefix}parabolic_sar"] = PSARIndicator(df["high"], df["low"], df["close"], step=0.018, max_step=0.2).psar()
        df[f"{prefix}momentum_14"] = df["close"].diff(14)
        df[f"{prefix}trix_15"] = ta.trend.TRIXIndicator(df["close"], window=15).trix()
        df[f"{prefix}ultimate_osc"] = ta.momentum.UltimateOscillator(df["high"], df["low"], df["close"], 7, 14, 28).ultimate_oscillator()
        df[f"{prefix}daily_range"] = df["high"] - df["low"]
        log_ret = np.log(df["close"] / (df["close"].shift(1) + 1e-9))
        df[f"{prefix}hv_20"] = log_ret.rolling(20).std() * np.sqrt(24 * 365)
        try:
            hl = np.log(df["high"] / (df["low"] + 1e-9))
            co = np.log(df["close"] / (df["open"] + 1e-9))
            df[f"{prefix}garman_klass"] = (0.5 * hl**2 - np.sqrt(2 * co**2)).rolling(20).mean()
        except Exception:
            df[f"{prefix}garman_klass"] = np.nan
        try:
            ln_hl = np.log(df["high"] / (df["low"] + 1e-9))
            df[f"{prefix}parkinson_20"] = np.sqrt((ln_hl**2).rolling(20).sum() / (4 * np.log(2) * 20))
        except Exception:
            df[f"{prefix}parkinson_20"] = np.nan
        df[f"{prefix}ulcer_index_14"] = ta.volatility.UlcerIndex(df["close"], 14).ulcer_index()
        df[f"{prefix}mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], 14).money_flow_index()
        df[f"{prefix}eom_14"] = ta.volume.EaseOfMovementIndicator(df["high"], df["low"], df["volume"], 14).sma_ease_of_movement()
        df[f"{prefix}dpo_20"] = df["close"] - df["close"].rolling(11, 1).mean()
        df[f"{prefix}macd"] = ta.trend.MACD(df["close"]).macd()
        df[f"{prefix}rsi_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        bb = ta.volatility.BollingerBands(df["close"], 20, 2)
        df[f"{prefix}bollinger_high"] = bb.bollinger_hband(); df[f"{prefix}bollinger_low"] = bb.bollinger_lband(); df[f"{prefix}bollinger_width"] = bb.bollinger_wband()
        df[f"{prefix}sma10"] = ta.trend.SMAIndicator(df["close"], 10).sma_indicator(); df[f"{prefix}sma50"] = ta.trend.SMAIndicator(df["close"], 50).sma_indicator()
        df[f"{prefix}sma10_sma50_diff"] = df[f"{prefix}sma10"] - df[f"{prefix}sma50"]
        df[f"{prefix}ema10"] = ta.trend.EMAIndicator(df["close"], 10).ema_indicator(); df[f"{prefix}ema50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df[f"{prefix}ema10_50_diff"] = df[f"{prefix}ema10"] - df[f"{prefix}ema50"]
        df[f"{prefix}atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df[f"{prefix}bollinger_width_ratio"] = df[f"{prefix}bollinger_width"] / (df[f"{prefix}rolling_std_20"] + 1e-9)
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
        df[f"{prefix}pivot"] = piv.pivot(); df[f"{prefix}support_1"] = piv.support_1(); df[f"{prefix}support_2"] = piv.support_2(); df[f"{prefix}resistance_1"] = piv.resistance_1(); df[f"{prefix}resistance_2"] = piv.resistance_2()
        candle = CustomCandlestickPattern(df["open"], df["high"], df["low"], df["close"])
        df[f"{prefix}engulfing"] = candle.engulfing(); df[f"{prefix}doji"] = candle.doji()
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4; ha_open = ha_close.shift(1).ffill()
        df[f"{prefix}heikin_ashi_open"] = ha_open; df[f"{prefix}heikin_ashi_close"] = ha_close
        df[f"{prefix}range_close_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        df[f"{prefix}bull_power"] = (df["close"] - df["low"]) / ((df["high"] - df["low"]) + 1e-9)
        df[f"{prefix}bars_from_local_max_20"] = df["close"].rolling(20, 1).apply(numba_last_local_max_idx, raw=True)
        df[f"{prefix}bars_from_local_min_20"] = df["close"].rolling(20, 1).apply(numba_last_local_min_idx, raw=True)
        df[f"{prefix}rsi_macd"] = df[f"{prefix}rsi_14"] * df[f"{prefix}macd"]
        df[f"{prefix}ma20_ma50_ratio"] = df[f"{prefix}ma20"] / (df[f"{prefix}ma50"] + 1e-9)

        df.replace([np.inf, -np.inf], np.nan, inplace=True); df.ffill(inplace=True); df.dropna(how="all", inplace=True)
        
        # ---- NEW: detect bad cols for this timeframe BEFORE resample ----
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
            resampled_rows = [_safe_agg_group(grp, agg_dict) for _, grp in df.groupby(pd.Grouper(freq=self.main_timeframe))]
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

    # ================= 2) FEATURE SELECTION =================
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 300,
        n_splits: int = 3,
    ) -> List[str]:
        """Time‑series aware feature selection (Variance→Corr‑filter→Mutual‑Info).

        * تمام NaN/Inf پیش از MI با **میانه** ستون پر می‌شود تا خطای "contains NaN" رفع شود.
        """
        # print("Feature selection start ...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        pool: List[str] = []
        for tr_idx, _ in tscv.split(X):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            # 1) Variance threshold
            vt = VarianceThreshold(0.01)
            cols_var = X_tr.columns[vt.fit(X_tr).get_support()]
            if cols_var.empty:
                continue
            X_var = X_tr[cols_var]
            # 2) Correlation filter
            corr = X_var.corr().abs(); mask = np.triu(np.ones_like(corr, bool), k=1)
            upper = corr.where(mask)
            drop_cols = [c for c in upper.columns if any(upper[c] > 0.9)]
            X_corr = X_var.drop(columns=drop_cols, errors="ignore")
            if X_corr.empty:
                continue
            # ---- FILL NA BEFORE MI ----
            X_filled = X_corr.replace([np.inf, -np.inf], np.nan).fillna(X_corr.median())
            # 3) Mutual information
            X_scaled = MinMaxScaler().fit_transform(X_filled)
            mi = mutual_info_classif(X_scaled, y_tr, random_state=SEED)
            pool.extend(pd.Series(mi, index=X_corr.columns).nlargest(top_k).index.tolist())
        counts = pd.Series(pool).value_counts()
        # print("Feature selection finished")
        return counts[counts >= n_splits].index[:top_k].tolist()

    # ================= 3) READY (X, y, WINDOW) =================
    def ready(
        self,
        data: pd.DataFrame,
        window: int = 1,
        selected_features: List[str] | None = None,
        mode: str = "train",
    ) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]:
        strict_cols = bool(selected_features)
        # ---------------- Basic checks ----------------
        close_col = f"{self.main_timeframe}_close"
        if close_col not in data.columns:
            raise ValueError(f"{close_col} missing")

        # ---------------- Target ----------------
        y = ((data[close_col].shift(-1) - data[close_col]) > 0).astype(int)
        if mode == "train":
            data, y = data.iloc[:-1], y.iloc[:-1]
        else:
            y.iloc[:] = 0  # dummy for predict

        # ---------------- Diff Features ----------------
        time_cols = [
            c
            for c in data.columns
            if any(tok in c for tok in ["hour", "day_of_week", "is_weekend"])
        ]
        feat_cols = [c for c in data.columns if c not in time_cols + [close_col]]
        
        # ------------------------------------------------------------
        # --- NEW ▸ drop RAW columns that are constant-zero در کل دیتاست ---
        # ------------------------------------------------------------
        if not strict_cols:
            # هر ستونی که در کل data مقدارش تقریباً صفر است و باینری واقعی نیست
            is_binary_raw = (data[feat_cols].nunique() <= 2).to_dict()
            zero_raw_cols = [
                c for c in feat_cols
                if pd.api.types.is_numeric_dtype(data[c])          # فقط ستون‌های عددی
                and data[c].abs().max() < 1e-12
                and not is_binary_raw.get(c, False)
            ]
            if zero_raw_cols:
                if self.verbose:
                    print(f"[FILTER] drop {len(zero_raw_cols)} constant-zero RAW cols")
                # هم از دیتافریم و هم از فهرست فیچرها حذف می‌کنیم
                data      = data.drop(columns=zero_raw_cols, errors="ignore")
                feat_cols = [c for c in feat_cols if c not in zero_raw_cols]

        # ------------------------------------------------------------
        # NEW ▸ ffill روی صفرهای پُرکنندهٔ 1H_* (غیرباینری) 
        # ------------------------------------------------------------
        h1_cols = [
            c for c in feat_cols
            if c.startswith("1H_") and not self.allow_regex.search(c)
        ]
        if h1_cols:
            # صفرهای «جای خالی» را به NaN تبدیل کن، سپس ffill
            data[h1_cols] = data[h1_cols].mask(data[h1_cols] == 0).ffill()
        # ------------------------------------------------------------

        df_diff = data[feat_cols].shift(1).diff()
        _timedelta_to_seconds(df_diff)
        # --- 1) drop globally detected bad columns ---
        if self.bad_cols_tf:
            bad_union = set().union(*self.bad_cols_tf.values())
            df_diff.drop(
                columns=[c for c in bad_union if c in df_diff.columns],
                inplace=True,
                errors="ignore",
            )

        # --- 2) drop columns whose *last* value ≈ 0 (ولی ستون باینری نیست) ---
        last_row_abs = df_diff.tail(1).abs().T
        numeric_col  = last_row_abs.select_dtypes(include=[np.number]).columns[0]
        is_binary    = (df_diff.nunique() <= 2).to_dict()
        bad_last_zero = last_row_abs[last_row_abs[numeric_col] < 1e-12].index
        first_col    = last_row_abs.columns[0]                     # ← ستونِ واقعی
        is_binary    = (df_diff.nunique() <= 2).to_dict()          # ستون‌های 0/1
        bad_last_zero = last_row_abs[last_row_abs[first_col] < 1e-12].index
        cols_lastzero_drop = [
            c for c in bad_last_zero if not is_binary.get(c, False)
        ]
        
        if (not strict_cols) and cols_lastzero_drop:
            if self.verbose:
                print(f"[FILTER] drop {len(cols_lastzero_drop)} cols with trailing 0")
            df_diff.drop(columns=cols_lastzero_drop, inplace=True, errors="ignore")
            
        # --- 3) drop columns that are constant-zero across ENTIRE df_diff ---
        if not strict_cols:
            # ستون‌هایی که حداکثر قدرمطلق‌شان تقریباً صفر است
            zero_like_cols = [
                c for c in df_diff.columns
                if df_diff[c].abs().max() < 1e-12
            ]
            # استثنا: ستون‌های باینری (۰/۱) واقعی
            zero_like_cols = [
                c for c in zero_like_cols
                if not is_binary.get(c, False)
            ]
            if zero_like_cols:
                if self.verbose:
                    print(f"[FILTER] drop {len(zero_like_cols)} constant-zero cols")
                df_diff.drop(columns=zero_like_cols,
                             inplace=True,
                             errors="ignore")
    

        # --------- Clean NA / Inf ----------
        df_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_diff.ffill(inplace=True)
        df_diff.dropna(how="all", inplace=True)
        df_diff.reset_index(drop=True, inplace=True)
        y = y.iloc[: len(df_diff)].reset_index(drop=True)
        

        # ---------------- Feature Selection ----------------
        if selected_features is None:
            feats = self.select_features(df_diff, y)
        elif selected_features == []:
            feats = df_diff.columns.tolist()
        else:
            feats = [f for f in selected_features if f in df_diff]

        # --- حذف فیچرهای معیوبِ شناسایی‌شده ---
        if (not strict_cols) and self.bad_cols_tf:
            bad_union = set().union(*self.bad_cols_tf.values())
            feats = [f for f in feats if f not in bad_union]

        X_f = df_diff[feats].copy()

        # ---------------- Windowing ----------------
        if window > 1:
            if len(X_f) < window:
                logging.warning("Not enough rows for window=%d", window)
                return (
                    pd.DataFrame(),
                    pd.Series(dtype=int),
                    feats,
                    pd.Series(dtype=float),
                )

            stacked = np.concatenate(
                [X_f.shift(i).iloc[window - 1 :].values for i in range(window)],
                axis=1,
            )
            X_f = pd.DataFrame(
                stacked,
                columns=[f"{c}_tminus{i}" for i in range(window) for c in feats],
            )
            y = y.iloc[window - 1 :].reset_index(drop=True)

        # ---------------- Final Clean ----------------
        X_f.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_f = X_f.fillna(X_f.median())

        if mode == "train":
            self.train_columns_after_window = X_f.columns.tolist()

        price_raw = data[close_col].iloc[: len(df_diff)].reset_index(drop=True)
        if window > 1:
            price_raw = price_raw.iloc[window - 1 :].reset_index(drop=True)

        return X_f.reset_index(drop=True), y.reset_index(drop=True), feats, price_raw

    # ================= 4) READY_INCREMENTAL =================) READY_INCREMENTAL =================
    def ready_incremental(self, data_window: pd.DataFrame, window=1, selected_features: List[str] | None = None):
        # print("Ready incremental start ...")
        X, _, feats, _ = self.ready(data_window, window, selected_features, mode="predict")
        # print("Ready incremental finished")
        return X.tail(1).reset_index(drop=True), feats

    # ================= 5) LOAD & MERGE =================
    def load_data(self) -> pd.DataFrame:
        logging.info("[load_data] parallel load %d timeframes", len(self.filepaths))

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
        main_df  = dfs[0].set_index(f"{main_tf}_time", drop=False)
        for (tf, _), df in zip(list(self.filepaths.items())[1:], dfs[1:]):
            main_df = main_df.join(
                df.set_index(f"{tf}_time", drop=False),
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
        merged = self.load_data()
        X, y, feats, _ = self.ready(merged, window=window, mode=mode)
        return X, y, feats



if __name__ == "__main__":
    prep = PREPARE_DATA_FOR_TRAIN(verbose=True)
    X, y, f = prep.get_prepared_data(window=1, mode="train")
    print("Shapes:", X.shape, y.shape, len(f))
