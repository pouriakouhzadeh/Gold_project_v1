#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extra_features_prototype.py

ساخت یک‌سری فیچر اضافه برای تایم‌فریم اصلی (مثلاً M30)
بدون دیتالیک و پایدار نسبت به batch/live.
"""

from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd


def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """
    شیب خط رگرسیون y روی هر پنجره‌ی rolling.
    از اندیس عددی (0,1,2,...) به عنوان x استفاده می‌کنیم.
    """
    y = y.astype(float)

    def slope(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 2 or np.allclose(arr, arr[0]):
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = arr.mean()
        cov = np.mean((x - x_mean) * (arr - y_mean))
        var = np.mean((x - x_mean) ** 2)
        if var == 0:
            return 0.0
        return cov / var

    return y.rolling(window=window, min_periods=2).apply(slope, raw=True)


def _rolling_realized_vol(ret: pd.Series, window: int) -> pd.Series:
    """
    Realized volatility = sqrt(sum(ret^2)) روی پنجره‌ی N.
    """
    ret2 = ret.astype(float) ** 2
    return ret2.rolling(window=window, min_periods=1).sum().pow(0.5)


def _rolling_up_ratio(ret: pd.Series, window: int) -> pd.Series:
    """
    نسبت کندل‌های مثبت در پنجره‌ی N.
    """
    up = (ret > 0).astype(float)
    return up.rolling(window=window, min_periods=1).mean()


def _binary_entropy(p: pd.Series) -> pd.Series:
    """
    انتروپی باینری: -p log p - (1-p) log (1-p)
    """
    p_clip = p.clip(1e-6, 1 - 1e-6)
    return -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))


def _rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Rolling correlation بین x و y.
    """
    x = x.astype(float)
    y = y.astype(float)
    return x.rolling(window=window, min_periods=2).corr(y)


def add_extra_features(
    df: pd.DataFrame,
    prefix: str = "30T_",
    windows: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    df باید حداقل ستون‌های 'close' و 'volume' را داشته باشد.
    این تابع فقط از گذشته (shift(1)) استفاده می‌کند، پس دیتالیک ندارد.
    """

    out = df.copy()

    close = out["close"].astype(float)
    volume = out.get("volume", pd.Series(index=out.index, dtype=float)).astype(float)

    # بازده لگاریتمی با یک کندل تاخیر
    log_price = np.log(close.replace(0, np.nan)).ffill()
    ret = log_price.diff().shift(1)   # فقط تا کندل t-1 استفاده می‌کنیم

    for w in windows:
        # 1) شیب روند
        out[f"{prefix}trend_slope_{w}"] = _rolling_slope(log_price.shift(1), window=w)

        # 2) نوسان تحقق‌یافته
        out[f"{prefix}realized_vol_{w}"] = _rolling_realized_vol(ret, window=w)

        # 3) نسبت کندل‌های مثبت
        up_ratio = _rolling_up_ratio(ret, window=w)
        out[f"{prefix}up_ratio_{w}"] = up_ratio

        # 4) انتروپی جهت
        out[f"{prefix}dir_entropy_{w}"] = _binary_entropy(up_ratio)

        # 5) همبستگی بازده و حجم
        vol_lag = volume.shift(1)
        out[f"{prefix}ret_vol_corr_{w}"] = _rolling_corr(ret, vol_lag, window=w)

    # پاک‌سازی اولیه
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.ffill(inplace=True)
    out.dropna(how="all", inplace=True)

    return out
