# GP/time_utils.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional, Dict

class TimeColumnFixer:
    """
    ابزار ایمن برای ساخت ستون‌های زمانی با پیشوند TF (مثل '30T_time') به صورت idempotent.
    - اگر ستون قبلاً وجود داشته باشد، فقط dtype را درست می‌کند.
    - اگر وجود نداشته باشد و 'time' یا DatetimeIndex موجود باشد، می‌سازد.
    - از insert استفاده نمی‌کند تا خطای 'already exists' رخ ندهد.
    """

    @staticmethod
    def ensure_prefixed_time_column(df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
        col = f"{tf_label}_time"
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        # منبع زمان
        if "time" in df.columns:
            src = df["time"]
        elif isinstance(df.index, pd.DatetimeIndex):
            src = df.index.to_series()
        else:
            return df
        out = df.copy()
        out[col] = pd.to_datetime(src, errors="coerce")
        # اگر حتماً می‌خواهی ستون در ابتدای DataFrame باشد، این دو خط را باز کن:
        # new_order = [col] + [c for c in out.columns if c != col]
        # out = out[new_order]
        return out

    @staticmethod
    def ensure_all_time_columns(df: pd.DataFrame, tfs: Iterable[str]) -> pd.DataFrame:
        out = df
        for tf in tfs:
            out = TimeColumnFixer.ensure_prefixed_time_column(out, tf)
        return out

    @staticmethod
    def coerce_time_if_exists(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """اگر ستونی با نام col هست، نوعش را datetime می‌کند (بی‌خطر)."""
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
