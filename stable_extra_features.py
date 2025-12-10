#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stable_extra_features.py — FINAL FULL VERSION (WITH MICROSTRUCTURE + FFT + CONFIDENCE)

• بدون data-leakage
• بدون look-forward
• ۱۰۰٪ پایدار در batch و live
• استفاده فقط از داده‌ی lagged
"""

from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    pywt = None
    _HAS_PYWT = False



# ================================================================
# ----------------------- BASIC HELPERS ---------------------------
# ================================================================

def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce").astype(float)

    def slope(arr):
        n = arr.shape[0]
        if n < 2:
            return np.nan
        x = np.arange(n, dtype=float)
        xm, ym = x.mean(), arr.mean()
        vx = np.mean((x - xm)**2)
        if vx == 0:
            return 0.0
        cov = np.mean((x - xm) * (arr - ym))
        return cov / vx

    return y.rolling(window, min_periods=2).apply(slope, raw=True)


def _realized_vol(ret, window):
    r2 = pd.to_numeric(ret, errors="coerce").astype(float)**2
    return r2.rolling(window, min_periods=1).sum().pow(0.5)


def _up_ratio(ret, window):
    up = (pd.to_numeric(ret, errors="coerce").astype(float) > 0).astype(float)
    return up.rolling(window, min_periods=1).mean()


def _binary_entropy(p):
    p = pd.to_numeric(p, errors="coerce").astype(float).clip(1e-6, 1-1e-6)
    return -(p*np.log(p) + (1-p)*np.log(1-p))


def _rolling_corr(x, y, window):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)
    return x.rolling(window, min_periods=2).corr(y)


def _zscore(x, window):
    x = pd.to_numeric(x, errors="coerce")
    m = x.rolling(window, min_periods=2).mean()
    s = x.rolling(window, min_periods=2).std(ddof=0)
    return (x - m) / (s + 1e-12)


def _pos_in_channel(x, window):
    x = pd.to_numeric(x, errors="coerce")
    lo = x.rolling(window, min_periods=2).min()
    hi = x.rolling(window, min_periods=2).max()
    rng = (hi - lo).replace(0, np.nan)
    return (x - lo) / (rng + 1e-12)


def _semi_vol(ret, window, side="down"):
    r = pd.to_numeric(ret, errors="coerce").astype(float)
    if side == "down":
        r = np.minimum(r, 0)
    else:
        r = np.maximum(r, 0)
    return (r**2).rolling(window, min_periods=1).sum().pow(0.5)


# ================================================================
# ------------------  MICROSTRUCTURE FEATURES  -------------------
# ================================================================

def _tick_imbalance(c_lag):
    """ sign(close_t - close_(t-1)) """
    diff = c_lag.diff()
    return np.sign(diff).fillna(0.0)


def _volume_imbalance(v_lag):
    """ signed volume: sign(return) * volume """
    return v_lag * np.sign(v_lag.diff().fillna(0))


def _orderflow_imbalance(c_lag, v_lag, w):
    """ (Σ signed_volume) / Σ volume   → classic OFI """
    signed = np.sign(c_lag.diff()) * v_lag
    volsum = v_lag.rolling(w, min_periods=1).sum()
    return signed.rolling(w, min_periods=1).sum() / (volsum + 1e-9)


def _microprice(c_lag):
    """ microprice approximated as close + delta(close)/2 """
    d = c_lag.diff().fillna(0)
    return c_lag + d * 0.5


def _efficient_price(c_lag, w):
    """ cumulative signed return / total absolute return """
    r = c_lag.diff().fillna(0)
    num = r.rolling(w).sum()
    den = np.abs(r).rolling(w).sum()
    return num / (den + 1e-12)


def _vpin(v_lag, ret, w):
    """ simplified VPIN = (sum(abs(signed_volume)) / sum(volume)) """
    signed = np.sign(ret) * v_lag
    vol = v_lag.rolling(w).sum()
    order_diff = np.abs(signed).rolling(w).sum()
    return order_diff / (vol + 1e-9)


# ================================================================
# ------------------  FFT & WAVELET FEATURES ---------------------
# ================================================================

def _fft_energy_ratio(c_lag, w):
    def calc(arr):
        if len(arr) < 4:
            return np.nan
        fft_vals = np.fft.rfft(arr - arr.mean())
        power = np.abs(fft_vals)**2
        if power.sum() == 0:
            return 0.0
        return (power[1:].sum()) / power.sum()
    return c_lag.rolling(w, min_periods=4).apply(calc, raw=True)


def _fft_dom_freq(c_lag, w):
    def calc(arr):
        if len(arr) < 4:
            return np.nan
        fft_vals = np.fft.rfft(arr - arr.mean())
        mag = np.abs(fft_vals)
        idx = np.argmax(mag[1:]) + 1
        return float(idx)
    return c_lag.rolling(w, min_periods=4).apply(calc, raw=True)


def _spectral_entropy(c_lag, w):
    def calc(arr):
        if len(arr) < 4:
            return np.nan
        fft_vals = np.fft.rfft(arr - arr.mean())
        p = np.abs(fft_vals)**2
        p /= p.sum() + 1e-12
        return -(p * np.log(p + 1e-12)).sum()
    return c_lag.rolling(w, min_periods=4).apply(calc, raw=True)


def _wavelet_energy(c_lag, w):
    """
    Wavelet انرژی روی پنجره‌ی w.
    * برای جلوگیری از خطای pandas، min_periods همیشه <= window است.
    * اگر طول arr < 8 باشد، خروجی NaN می‌شود (برای wavelet طول خیلی کم معنی ندارد).
    """
    # اگر pywt نصب یا import نشده، به‌صورت امن فقط NaN برمی‌گردانیم
    if not _HAS_PYWT:
        return pd.Series(index=c_lag.index, dtype=float)

    # تضمین: min_periods <= window
    # برای w < 8، min_periods را خود w می‌گذاریم، ولی داخل calc اگر len<8 بود → NaN
    minp = min(8, w)

    def calc(arr: np.ndarray) -> float:
        if arr.shape[0] < 8:
            return np.nan
        cA, cD = pywt.dwt(arr, "db2")
        return float(np.sum(cD ** 2))

    return c_lag.rolling(window=w, min_periods=minp).apply(calc, raw=True)


# ================================================================
# -------------------  CONFIDENCE FEATURES ------------------------
# ================================================================

def _signal_consistency(ret, w):
    """ fraction of returns with same sign """
    s = np.sign(ret)
    pos = (s > 0).rolling(w).mean()
    neg = (s < 0).rolling(w).mean()
    return (pos - neg).fillna(0)


def _vol_norm_disagreement(ret, w):
    """ std(return) normalized disagreement """
    std = ret.rolling(w).std()
    mean = ret.rolling(w).mean()
    return (std / (np.abs(mean) + 1e-12)).fillna(0)


def _stability_confidence(c_lag, w):
    """ slope stability measure """
    slope = _rolling_slope(c_lag, w)
    return 1 / (1 + np.abs(slope))


# ================================================================
# ------------------  FINAL MERGED FEATURE MAKER -----------------
# ================================================================

def add_stable_extra_features(
    df: pd.DataFrame,
    tf: str,
    windows: Iterable[int] = (5, 10, 20),
    use_log_price: bool = True,
) -> pd.DataFrame:

    out = df.copy()
    prefix = f"{tf}_"

    # base series (lag only)
    c = pd.to_numeric(out["close"], errors="coerce")
    v = pd.to_numeric(out["volume"], errors="coerce")

    c_lag = c.shift(1)
    v_lag = v.shift(1)

    if use_log_price:
        p = np.log(c.replace(0, np.nan)).ffill()
        p_lag = p.shift(1)
        ret = p.diff().shift(1)
    else:
        p_lag = c_lag
        ret = c.pct_change().shift(1).fillna(0)
    # --- MICROSTRUCTURE فیچرهای مستقل از w (یکبار محاسبه شوند) ---
    out[f"{prefix}tick_imbalance"] = _tick_imbalance(c_lag)
    out[f"{prefix}volume_imbalance"] = _volume_imbalance(v_lag)
    out[f"{prefix}microprice"] = _microprice(c_lag)

    for w in windows:

        # ============== (A) PROTOTYPE + STABLE FEATURES ===============
        out[f"{prefix}trend_slope_{w}"] = _rolling_slope(p_lag, w)
        out[f"{prefix}realized_vol_{w}"] = _realized_vol(ret, w)
        up_r = _up_ratio(ret, w)
        out[f"{prefix}up_ratio_{w}"] = up_r
        out[f"{prefix}dir_entropy_{w}"] = _binary_entropy(up_r)
        out[f"{prefix}ret_vol_corr_{w}"] = _rolling_corr(ret, v_lag, w)

        out[f"{prefix}pos_in_channel_{w}"] = _pos_in_channel(c_lag, w)
        out[f"{prefix}zscore_close_{w}"] = _zscore(c_lag, w)
        out[f"{prefix}zscore_volume_{w}"] = _zscore(v_lag, w)
        out[f"{prefix}semi_up_vol_{w}"] = _semi_vol(ret, w, "up")
        out[f"{prefix}semi_down_vol_{w}"] = _semi_vol(ret, w, "down")

        # ============== (B) MICROSTRUCTURE FEATURES ===============
        # این سه تا مستقل از w هستند؛ بهتر است خارج از حلقه محاسبه شوند
        out[f"{prefix}orderflow_imbalance_{w}"] = _orderflow_imbalance(c_lag, v_lag, w)
        out[f"{prefix}efficient_price_{w}"] = _efficient_price(c_lag, w)
        out[f"{prefix}vpin_{w}"] = _vpin(v_lag, ret, w)

        # ============== (C) FFT & WAVELET FEATURES ===============
        out[f"{prefix}fft_energy_ratio_{w}"] = _fft_energy_ratio(c_lag, w)
        out[f"{prefix}fft_dom_freq_{w}"] = _fft_dom_freq(c_lag, w)
        out[f"{prefix}spectral_entropy_{w}"] = _spectral_entropy(c_lag, w)
        out[f"{prefix}wavelet_energy_{w}"] = _wavelet_energy(c_lag, w)

        # ============== (D) CONFIDENCE FEATURES ===============
        out[f"{prefix}signal_consistency_{w}"] = _signal_consistency(ret, w)
        out[f"{prefix}vol_norm_disagreement_{w}"] = _vol_norm_disagreement(ret, w)
        out[f"{prefix}stability_confidence_{w}"] = _stability_confidence(c_lag, w)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.ffill(inplace=True)

    return out
