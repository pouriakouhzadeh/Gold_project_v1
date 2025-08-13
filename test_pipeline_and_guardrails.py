# -*- coding: utf-8 -*-
"""
Tests + Execution Logger (v2)
- لاگ با نام همین فایل و پسوند .log ساخته می‌شود.
- رفع FutureWarning freq='T' → استفاده از '30min'
- تست اضافه: idempotent بودن ستون‌های زمان و اجرای پشت‌سرهم build_X
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

# ------------------ fallback برای sklearnex ------------------
try:
    import sklearnex  # type: ignore
except ImportError:
    mod = types.ModuleType("sklearnex")
    def _patch_sklearn(verbose=False): return None
    mod.patch_sklearn = _patch_sklearn
    sys.modules["sklearnex"] = mod

# مسیر پروژه برای ایمپورت ماژول‌ها
sys.path.append(".")

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from gp_guardrails import (
    guard_and_prepare_for_predict,
    take_last_closed_rows,
    WarmupNotEnough, ColumnsMismatch, BadValuesFound, TimestampNotAligned
)

# ------------------ pytest hooks برای گزارش اجرای تست‌ها در همین لاگ ------------------
_SESSION_START_TS = None
_TEST_RESULTS = []

def pytest_sessionstart(session):
    global _SESSION_START_TS
    _SESSION_START_TS = time.time()
    _LOGGER.info("=== pytest session START ===")
    _LOGGER.info("Platform: %s | Python: %s", sys.platform, sys.version.replace("\n"," "))
    _LOGGER.info("CWD: %s", os.getcwd())

def pytest_runtest_logreport(report):
    if report.when == "call":
        outcome = "PASS" if report.passed else ("SKIP" if report.skipped else "FAIL")
        _TEST_RESULTS.append({
            "nodeid": report.nodeid,
            "outcome": outcome,
            "duration": getattr(report, "duration", 0.0),
        })
        if outcome == "FAIL":
            _LOGGER.error("[FAIL] %s (%.3fs)", report.nodeid, report.duration)
            try:
                longrepr = str(report.longrepr)
                _LOGGER.error("Traceback (tail):\n%s", "\n".join(longrepr.splitlines()[-20:]))
            except Exception:
                _LOGGER.error("Traceback capture failed.")
        elif outcome == "SKIP":
            _LOGGER.warning("[SKIP] %s (%.3fs) reason=%s", report.nodeid, report.duration, getattr(report, "longrepr", ""))
        else:
            _LOGGER.info("[PASS] %s (%.3fs)", report.nodeid, report.duration)

def pytest_sessionfinish(session, exitstatus):
    elapsed = time.time() - (_SESSION_START_TS or time.time())
    total = len(_TEST_RESULTS)
    passed = sum(1 for r in _TEST_RESULTS if r["outcome"] == "PASS")
    failed = sum(1 for r in _TEST_RESULTS if r["outcome"] == "FAIL")
    skipped = sum(1 for r in _TEST_RESULTS if r["outcome"] == "SKIP")
    _LOGGER.info("=== pytest session FINISH (%.3fs) ===", elapsed)
    _LOGGER.info("Summary: total=%d | passed=%d | failed=%d | skipped=%d", total, passed, failed, skipped)
    for r in _TEST_RESULTS:
        _LOGGER.info(" - %s  →  %s (%.3fs)", r["nodeid"], r["outcome"], r["duration"])

# ------------------ توابع کمکی دادهٔ مصنوعی و بیلد X ------------------
def make_ohlcv(n=260, freq="30min", start="2020-01-01 00:00:00", noise=0.1, trend=0.01, seed=0):
    rng = pd.date_range(start=start, periods=n, freq=freq)
    rs = np.random.RandomState(seed)
    base = np.arange(n) * trend + rs.normal(0, noise, size=n).cumsum()/100
    close = 100 + base
    open_ = close + rs.normal(0, 0.02, size=n)
    high = np.maximum(open_, close) + np.abs(rs.normal(0, 0.05, size=n))
    low = np.minimum(open_, close) - np.abs(rs.normal(0, 0.05, size=n))
    volume = rs.randint(100, 1000, size=n).astype(float)
    df = pd.DataFrame({"time": rng, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    return df

@pytest.fixture(scope="module")
def tmp_csvs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpdata2")
    rows = 300
    _LOGGER.info("[fixture] Creating synthetic CSVs in %s (rows=%d)", str(tmp), rows)
    for tf, name, seed in [("30T","XAUUSD_M30.csv",0), ("1H","XAUUSD_H1.csv",1), ("15T","XAUUSD_M15.csv",2), ("5T","XAUUSD_M5.csv",3)]:
        df = make_ohlcv(n=rows, freq="30min", seed=seed)
        df.to_csv(tmp / name, index=False)
        _LOGGER.info("[fixture] Wrote %s (%d rows)", name, len(df))
    return {
        "30T": str(tmp/"XAUUSD_M30.csv"),
        "1H":  str(tmp/"XAUUSD_H1.csv"),
        "15T": str(tmp/"XAUUSD_M15.csv"),
        "5T":  str(tmp/"XAUUSD_M5.csv"),
    }

def build_X(paths, window=1, mode="predict"):
    t0 = time.perf_counter()
    _LOGGER.info("[build_X] window=%s mode=%s", window, mode)
    try:
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=paths, main_timeframe="30T", verbose=False, fast_mode=True)
    except TypeError:
        prep = PREPARE_DATA_FOR_TRAIN(filepaths=paths, main_timeframe="30T", verbose=False)
    merged = prep.load_data()
    _LOGGER.info("[build_X] merged shape: %s | cols=%d", getattr(merged, "shape", None), merged.shape[1] if hasattr(merged, "shape") else -1)
    X, y, feats, price_raw = prep.ready(merged, window=window, mode=mode)
    dt = time.perf_counter() - t0
    _LOGGER.info("[build_X] X=%s | y=%s | feats=%s | took=%.3fs", getattr(X, "shape", None), getattr(y, "shape", None), len(feats) if feats is not None else None, dt)
    return prep, merged, X, y, feats, price_raw

# ------------------ TESTS ------------------
def test_last_row_stability(tmp_csvs):
    _LOGGER.info(">> test_last_row_stability: start")
    _, _, X1, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")
    for tf, path in tmp_csvs.items():
        df = pd.read_csv(path)
        df.loc[len(df)-1, ["open","high","low","close","volume"]] += 777.0
        df.to_csv(path, index=False)
        _LOGGER.info("   modified last row for %s (%s)", tf, path)
    _, _, X2, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")
    pd.testing.assert_series_equal(X1.iloc[-1], X2.iloc[-1], check_names=False)
    _LOGGER.info("<< test_last_row_stability: OK (last row features stable)")

def test_depends_on_tminus1(tmp_csvs):
    _LOGGER.info(">> test_depends_on_tminus1: start")
    _, _, X1, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")
    for tf, path in tmp_csvs.items():
        df = pd.read_csv(path)
        idx = len(df)-2
        df.loc[idx, ["open","high","low","close","volume"]] += 123.0
        df.to_csv(path, index=False)
        _LOGGER.info("   modified t-1 row for %s (idx=%d)", tf, idx)
    _, _, X2, _, _, _ = build_X(tmp_csvs, window=1, mode="predict")
    changed = not np.allclose(X1.iloc[-1].values, X2.iloc[-1].values)
    _LOGGER.info("   last-row changed after t-1 tweak? %s", changed)
    assert changed, "Last-row features SHOULD depend on t-1"
    _LOGGER.info("<< test_depends_on_tminus1: OK (depends on t-1)")

def test_guard_no_nan_inf_and_cols_and_dtype(tmp_csvs):
    _LOGGER.info(">> test_guard_no_nan_inf_and_cols_and_dtype: start")
    _, _, X, _, _, _ = build_X(tmp_csvs, window=3, mode="predict")
    train_window_cols = list(X.columns)
    _LOGGER.info("   train_window_cols = %d", len(train_window_cols))

    # NaN → BadValuesFound
    X_bad = X.copy()
    X_bad.iloc[-1, 0] = np.nan
    _LOGGER.info("   injecting NaN at last-row, col0")
    with pytest.raises(BadValuesFound):
        guard_and_prepare_for_predict(
            X_bad, train_window_cols,
            min_required_history={'5T': 10, '15T': 10, '30T': 10, '1H': 10},
            ctx_history_lengths={'5T': 20, '15T': 20, '30T': 20, '1H': 20},
            where="X_test"
        )

    # ستون اضافه → ColumnsMismatch
    X_bad2 = X.copy()
    X_bad2["EXTRA_COL"] = 1.0
    _LOGGER.info("   injecting EXTRA_COL")
    with pytest.raises(ColumnsMismatch):
        guard_and_prepare_for_predict(
            X_bad2, train_window_cols,
            min_required_history={'5T': 10, '15T': 10, '30T': 10, '1H': 10},
            ctx_history_lengths={'5T': 20, '15T': 20, '30T': 20, '1H': 20},
            where="X_test"
        )

    # حالت درست → float32
    X_ok = guard_and_prepare_for_predict(
        X.copy(), train_window_cols,
        min_required_history={'5T': 10, '15T': 10, '30T': 10, '1H': 10},
        ctx_history_lengths={'5T': 20, '15T': 20, '30T': 20, '1H': 20},
        where="X_test"
    )
    _LOGGER.info("   dtype of first column: %s", str(X_ok.dtypes.iloc[0]))
    assert str(X_ok.dtypes.iloc[0]) == "float32"
    _LOGGER.info("<< test_guard_no_nan_inf_and_cols_and_dtype: OK")

def test_guard_warmup(tmp_csvs):
    _LOGGER.info(">> test_guard_warmup: start")
    _, _, X, _, _, _ = build_X(tmp_csvs, window=3, mode="predict")
    cols = list(X.columns)
    _LOGGER.info("   current columns count=%d", len(cols))
    with pytest.raises(WarmupNotEnough):
        guard_and_prepare_for_predict(
            X, cols,
            min_required_history={'5T': 1000, '15T': 1000, '30T': 1000, '1H': 1000},
            ctx_history_lengths={'5T': 200, '15T': 300, '30T': 400, '1H': 500},
            where="X_test"
        )
    _LOGGER.info("<< test_guard_warmup: OK (WarmupNotEnough triggered)")

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
    _LOGGER.info("<< test_take_last_closed_rows_aligns_timestamps: OK (aligned)")

def test_time_prefix_idempotent(tmp_csvs):
    """
    اجرای پشت‌سرهم load_data/ready نباید خطای "cannot insert ..._time, already exists" بدهد.
    این تست مخصوص پچ idempotent برای ستون‌های زمان است.
    """
    _LOGGER.info(">> test_time_prefix_idempotent: start")
    # بار اول
    prep1, merged1, X1, y1, feats1, _ = build_X(tmp_csvs, window=2, mode="predict")
    _LOGGER.info("   pass#1 merged=%s X=%s feats=%s", getattr(merged1, "shape", None), getattr(X1, "shape", None), len(feats1) if feats1 is not None else None)
    # بار دوم (بدون تغییر ورودی) — اگر insert تکراری باشد، اینجا قبلاً خطا می‌داد
    prep2, merged2, X2, y2, feats2, _ = build_X(tmp_csvs, window=2, mode="predict")
    _LOGGER.info("   pass#2 merged=%s X=%s feats=%s", getattr(merged2, "shape", None), getattr(X2, "shape", None), len(feats2) if feats2 is not None else None)
    # چند چک ساده
    assert merged2.shape[1] >= merged1.shape[1]
    assert X2.shape[1] == X1.shape[1]
    _LOGGER.info("<< test_time_prefix_idempotent: OK (no duplicate time insert)")
