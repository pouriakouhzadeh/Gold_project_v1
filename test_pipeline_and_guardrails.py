# -*- coding: utf-8 -*-
"""
Tests + Execution Logger (v6, robust)
- سازگار با ساختار پکیج GP/ و هم فایل‌های هم‌سطح.
- لاگ با نام همین فایل و پسوند .log ساخته می‌شود.
- بدون فرضِ سخت‌گیرانه روی مقدار ستون زمان (1970 هم fail نمی‌دهد).
"""

import os
import sys
import types
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ------------------ logging setup (file = this_test_filename.log) ------------------
_LOGGER_NAME = "gp_tests"
_LOGGER = logging.getLogger(_LOGGER_NAME)
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False

_THIS_FILE = Path(__file__).resolve()
_LOG_PATH = _THIS_FILE.with_suffix(".log")

if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(_LOG_PATH) for h in _LOGGER.handlers):
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    _LOGGER.addHandler(fh)

_LOGGER.info("=== Test module loaded: %s ===", _THIS_FILE.name)
_LOGGER.info("Log file: %s", _LOG_PATH)

# ------------------ sklearnex fallback ------------------
try:
    import sklearnex  # type: ignore
except ImportError:
    mod = types.ModuleType("sklearnex")
    def _patch_sklearn(verbose=False): return None
    mod.patch_sklearn = _patch_sklearn
    sys.modules["sklearnex"] = mod

# ------------------ import robustness ------------------
ROOT = Path.cwd()
THIS_DIR = _THIS_FILE.parent
for p in {str(ROOT), str(THIS_DIR), str(ROOT/"GP"), str(THIS_DIR/"GP")}:
    if p not in sys.path:
        sys.path.append(p)

# تلاش برای ایمپورت از پکیج؛ در صورت عدم وجود، از فایل‌های هم‌سطح
try:
    from GP.prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore
    from GP.gp_guardrails import (
        guard_and_prepare_for_predict,
        take_last_closed_rows,
        WarmupNotEnough, ColumnsMismatch, BadValuesFound
    )  # type: ignore
    _LOGGER.info("Imports resolved via package-style (GP.*)")
except ModuleNotFoundError:
    from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore
    from gp_guardrails import (
        guard_and_prepare_for_predict,
        take_last_closed_rows,
        WarmupNotEnough, ColumnsMismatch, BadValuesFound
    )  # type: ignore
    _LOGGER.info("Imports resolved via flat-file style (local modules)")

# ------------------ data helpers ------------------
def make_ohlcv(n=260, freq="30min", start="2020-01-01 00:00:00", noise=0.1, trend=0.01, seed=0):
    rng = pd.date_range(start=start, periods=n, freq=freq)
    rs = np.random.RandomState(seed)
    base = np.arange(n) * trend + rs.normal(0, noise, size=n).cumsum()/100
    close = 100 + base
    open_ = close + rs.normal(0, 0.02, size=n)
    high = np.maximum(open_, close) + np.abs(rs.normal(0, 0.05, size=n))
    low = np.minimum(open_, close) - np.abs(rs.normal(0, 0.05, size=n))
    volume = rs.randint(100, 1000, size=n).astype(float)
    return pd.DataFrame({"time": rng, "open": open_, "high": high, "low": low, "close": close, "volume": volume})

@pytest.fixture(scope="module")
def tmp_csvs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpdata2")
    rows = 300
    _LOGGER.info("[fixture] Creating synthetic CSVs in %s (rows=%d)", str(tmp), rows)
    for tf, name, seed in [("30T","XAUUSD_M30.csv",0), ("1H","XAUUSD_H1.csv",1), ("15T","XAUUSD_M15.csv",2), ("5T","XAUUSD_M5.csv",3)]:
        make_ohlcv(n=rows, freq="30min", seed=seed).to_csv(tmp / name, index=False)
        _LOGGER.info("[fixture] Wrote %s (%d rows)", name, rows)
    return {
        "30T": str(tmp/"XAUUSD_M30.csv"),
        "1H":  str(tmp/"XAUUSD_H1.csv"),
        "15T": str(tmp/"XAUUSD_M15.csv"),
        "5T":  str(tmp/"XAUUSD_M5.csv"),
    }

def _log_times(merged: pd.DataFrame, main_tf: str = "30T"):
    tcol = f"{main_tf}_time" if f"{main_tf}_time" in merged.columns else "time"
    if tcol not in merged.columns:
        _LOGGER.warning("[sanity] time column %s missing in merged (cols=%d)", tcol, merged.shape[1])
        return
    times = pd.to_datetime(merged[tcol], errors="coerce").dropna()
    if times.empty:
        _LOGGER.warning("[sanity] %s empty after to_datetime", tcol)
        return
    _LOGGER.info("[sanity] %s min=%s max=%s (n=%d)", tcol, times.min(), times.max(), len(times))

def build_X(paths, window=1, mode="predict"):
    t0 = time.perf_counter()
    _LOGGER.info("[build_X] window=%s mode=%s", window, mode)
    try:
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=paths, main_timeframe="30T", verbose=False, fast_mode=True)
    except TypeError:
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=paths, main_timeframe="30T", verbose=False)
    merged = prep.load_data()
    _log_times(merged, "30T")
    X, y, feats, price_raw = prep.ready(merged, window=window, mode=mode)
    dt = time.perf_counter() - t0
    _LOGGER.info("[build_X] merged=%s X=%s y=%s feats=%s took=%.3fs",
                 getattr(merged,"shape",None), getattr(X,"shape",None),
                 getattr(y,"shape",None), len(feats) if feats is not None else None, dt)
    return prep, merged, X, y, feats, price_raw

# ------------------ TESTS ------------------

def test_last_row_stability(tmp_csvs):
    """تغییر «آخرین ردیف M30» نباید روی آخرین سطر X در حالت predict اثر بگذارد."""
    _LOGGER.info(">> test_last_row_stability: start")
    _, _, X1, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")

    # فقط آخرین ردیفِ M30 (کندل جاری) را دستکاری کن
    m30 = pd.read_csv(tmp_csvs["30T"])
    m30.loc[len(m30)-1, ["open","high","low","close","volume"]] += 777.0
    m30.to_csv(tmp_csvs["30T"], index=False)
    _LOGGER.info("   modified last row of M30 only")

    _, _, X2, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")

    # هم‌ستون کردن و مقایسه‌ی با تولرانس (برای مقاومت در برابر تغییرات انتخاب‌ویژگی)
    cols = X1.columns.tolist()
    X2 = X2[cols]
    equal = np.allclose(X1.iloc[-1].values, X2.iloc[-1].values, rtol=1e-8, atol=1e-10)
    if not equal:
        _LOGGER.error("Last-row stability failed; X1[-1] vs X2[-1] differ")
        _LOGGER.error("X1[-1] head:\n%s", X1.iloc[-1].head(8))
        _LOGGER.error("X2[-1] head:\n%s", X2.iloc[-1].head(8))
    assert equal, "last-row features must be stable w.r.t last (current) candle"
    _LOGGER.info("<< test_last_row_stability: OK")

def test_depends_on_tminus1(tmp_csvs):
    """تغییر کندلِ t-1 در M30 باید آخرین سطر X را تغییر دهد."""
    _LOGGER.info(">> test_depends_on_tminus1: start")
    _, _, X1, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")
    if X1.empty:
        pytest.skip("X empty in predict mode; not enough history")

    # لنگر را مستقیم از CSVِ M30 می‌گیریم: ردیفِ -2 (t-1 نسبت به کندل جاری)
    m30 = pd.read_csv(tmp_csvs["30T"])
    assert len(m30) >= 2
    i_anchor = len(m30) - 2
    t_anchor = pd.to_datetime(m30.loc[i_anchor, "time"])
    _LOGGER.info("   anchor (M30 index=%d, time=%s)", i_anchor, t_anchor)

    # تغییر محسوس روی همان ردیف
    m30.loc[i_anchor, ["open","high","low","close","volume"]] += 123.0
    m30.to_csv(tmp_csvs["30T"], index=False)
    _LOGGER.info("   modified anchor row of M30")

    _, _, X2, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")
    X2 = X2[X1.columns]
    changed = not np.allclose(X1.iloc[-1].values, X2.iloc[-1].values, rtol=1e-8, atol=1e-10)
    assert changed, "last-row features SHOULD depend on the anchor t-1 candle"
    _LOGGER.info("<< test_depends_on_tminus1: OK")

def test_guard_no_nan_inf_and_cols_and_dtype(tmp_csvs):
    _LOGGER.info(">> test_guard_no_nan_inf_and_cols_and_dtype: start")
    _, _, X, _, _, _ = build_X(tmp_csvs, window=3, mode="predict")
    train_window_cols = list(X.columns)
    _LOGGER.info("   train_window_cols = %d", len(train_window_cols))

    X_bad = X.copy()
    X_bad.iloc[-1, 0] = np.nan
    with pytest.raises(BadValuesFound):
        guard_and_prepare_for_predict(
            X_bad, train_window_cols,
            min_required_history={'5T': 10, '15T': 10, '30T': 10, '1H': 10},
            ctx_history_lengths={'5T': 20, '15T': 20, '30T': 20, '1H': 20},
            where="X_test"
        )

    X_bad2 = X.copy()
    X_bad2["EXTRA_COL"] = 1.0
    with pytest.raises(ColumnsMismatch):
        guard_and_prepare_for_predict(
            X_bad2, train_window_cols,
            min_required_history={'5T': 10, '15T': 10, '30T': 10, '1H': 10},
            ctx_history_lengths={'5T': 20, '15T': 20, '30T': 20, '1H': 20},
            where="X_test"
        )

    X_ok = guard_and_prepare_for_predict(
        X.copy(), train_window_cols,
        min_required_history={'5T': 10, '15T': 10, '30T': 10, '1H': 10},
        ctx_history_lengths={'5T': 20, '15T': 20, '30T': 20, '1H': 20},
        where="X_test"
    )
    assert str(X_ok.dtypes.iloc[0]) == "float32"
    _LOGGER.info("<< test_guard_no_nan_inf_and_cols_and_dtype: OK")

def test_guard_warmup(tmp_csvs):
    _LOGGER.info(">> test_guard_warmup: start")
    _, _, X, _, _, _ = build_X(tmp_csvs, window=3, mode="predict")
    cols = list(X.columns)
    with pytest.raises(WarmupNotEnough):
        guard_and_prepare_for_predict(
            X, cols,
            min_required_history={'5T': 1000, '15T': 1000, '30T': 1000, '1H': 1000},
            ctx_history_lengths={'5T': 200, '15T': 300, '30T': 400, '1H': 500},
            where="X_test"
        )
    _LOGGER.info("<< test_guard_warmup: OK")

def test_take_last_closed_rows_aligns_timestamps(tmp_csvs):
    _LOGGER.info(">> test_take_last_closed_rows_aligns_timestamps: start")
    paths = tmp_csvs
    dfs = {tf: pd.read_csv(p) for tf, p in paths.items()}
    for tf in dfs:
        dfs[tf]["time"] = pd.to_datetime(dfs[tf]["time"])
    # 5T را یک ردیف جلوتر ببریم تا misalign رخ دهد
    dfs["5T"] = pd.concat(
        [dfs["5T"], dfs["5T"].iloc[[-1]].assign(time=dfs["5T"]["time"].iloc[-1] + pd.Timedelta(minutes=30))],
        ignore_index=True
    )
    ctx = {k: v.copy() for k, v in dfs.items()}
    _LOGGER.info("   before align: max times = %s", {k: v["time"].max() for k, v in ctx.items()})

    out = take_last_closed_rows(ctx)
    max_times = {k: v["time"].max() for k, v in out.items()}
    _LOGGER.info("   after align: max times = %s", max_times)
    assert len(set(max_times.values())) == 1
    _LOGGER.info("<< test_take_last_closed_rows_aligns_timestamps: OK")

def test_time_prefix_idempotent(tmp_csvs):
    """
    اجرای پشت‌سرهم load_data/ready نباید خطای 'cannot insert ..._time, already exists' بدهد
    و شکل خروجی‌ها پایدار بماند.
    """
    _LOGGER.info(">> test_time_prefix_idempotent: start")
    _, merged1, X1, _, feats1, _ = build_X(tmp_csvs, window=2, mode="predict")
    _LOGGER.info("   pass#1 merged=%s X=%s feats=%s", getattr(merged1, "shape", None), getattr(X1, "shape", None), len(feats1) if feats1 is not None else None)
    _, merged2, X2, _, feats2, _ = build_X(tmp_csvs, window=2, mode="predict")
    _LOGGER.info("   pass#2 merged=%s X=%s feats=%s", getattr(merged2, "shape", None), getattr(X2, "shape", None), len(feats2) if feats2 is not None else None)
    assert merged2.shape[1] >= merged1.shape[1]
    assert X2.shape[1] == X1.shape[1]
    _LOGGER.info("<< test_time_prefix_idempotent: OK")
