# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prepare_data_for_train_production.py
------------------------------------
نسخه Production که تضمین می‌کند *Cut-Before-Resample* رعایت شود و
دقیقاً همان پایپ‌لاین کلاس اصلی (PREPARE_DATA_FOR_TRAIN) در Train/Batch تکرار گردد.

ایده‌ی کلیدی:
- برای هر ts_end هر تایم‌فریم را تا <= ts_end کات می‌کنیم،
- این کات‌ها را موقتاً داخل یک پوشه‌ی temp ذخیره می‌کنیم،
- سپس یک نمونه از PREPARE_DATA_FOR_TRAIN با همین فایل‌های کات‌شده می‌سازیم
  و همان متد load_data() خودِ کلاسِ اصلی را صدا می‌زنیم.
=> به‌این‌ترتیب، هر اندیکاتور/ری‌سمپل/ادغام کاملاً مثل Train اجرا می‌شود.
"""

from __future__ import annotations
import os, shutil, tempfile, logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # کلاس اصلی شما

class PREPARE_DATA_FOR_TRAIN_PRODUCTION:
    def __init__(
        self,
        filepaths: Optional[Dict[str, str]] = None,
        main_timeframe: str = "30T",
        verbose: bool = True,
        fast_mode: bool = True,
        strict_disk_feed: bool = True,
        from_memory: Optional[Dict[str, pd.DataFrame]] = None,
        disable_drift: bool = True,
    ):
        self.filepaths = {k: str(v) for k, v in (filepaths or {}).items()}
        self.main_timeframe = main_timeframe
        self.verbose = verbose
        self.fast_mode = fast_mode
        self.strict_disk_feed = strict_disk_feed
        self._raw_memory = None

        if from_memory is not None:
            mem_norm: Dict[str, pd.DataFrame] = {}
            for tf, df in from_memory.items():
                d = df.copy()
                if "time" not in d.columns:
                    if d.columns.size > 0:
                        d = d.rename(columns={d.columns[0]: "time"})
                    else:
                        raise ValueError(f"[from_memory:{tf}] empty dataframe without 'time'")
                d["time"] = pd.to_datetime(d["time"], errors="coerce")
                d = d.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
                mem_norm[tf] = d
            self._raw_memory = mem_norm

        self._raw_cache: Dict[str, pd.DataFrame] = {}
        self._tmp_dir: Optional[Path] = None

        # drift در Production غیرفعال (Parity با لایو)
        self._disable_drift = bool(disable_drift)

    # --------- کمک‌متدها ---------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def load_all(self):
        """خواندن کامل CSV خام یا حافظه‌ای و کش کردن آن‌ها."""
        self._raw_cache.clear()
        if self._raw_memory is not None:
            self._raw_cache.update(self._raw_memory)
            for tf, df in self._raw_cache.items():
                self._log(f"[raw/mem] {tf}: {len(df)} rows")
            return

        if not self.filepaths:
            raise ValueError("[production] filepaths or from_memory is required")
        for tf, fp in self.filepaths.items():
            p = Path(fp)
            if not p.exists():
                raise FileNotFoundError(f"[production] missing file for '{tf}': {p}")
            df = pd.read_csv(p)
            if "time" not in df.columns:
                raise ValueError(f"[{tf}] 'time' column missing in CSV")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            self._raw_cache[tf] = df
            self._log(f"[raw/disk] {tf}: {len(df)} rows -> {p.name}")

    def _ensure_loaded(self):
        if not self._raw_cache:
            self.load_all()

    def _write_cut_csvs(self, ts_now: pd.Timestamp) -> Dict[str, str]:
        """
        برای هر TF تا ts_now کات می‌زند و فایل‌های موقتی می‌سازد.
        خروجی: دیکشنری از مسیر فایل‌های موقت.
        """
        self._ensure_loaded()
        tmp_dir = Path(tempfile.mkdtemp(prefix="prod_cut_"))
        self._tmp_dir = tmp_dir

        out_paths: Dict[str, str] = {}
        for tf, df in self._raw_cache.items():
            cut = df[df["time"] <= ts_now].copy()
            if cut.empty:
                # فایل خالی ولی با سرستون‌ها
                cut = df.head(0).copy()
            out_fp = tmp_dir / f"{tf}.csv"
            cut.to_csv(out_fp, index=False)
            out_paths[tf] = str(out_fp)
        return out_paths

    def _cleanup_tmp(self):
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = None

    # --------- API اصلی Production ---------
    def load_data_up_to(self, ts_now) -> pd.DataFrame:
        """
        1) روی هر TF تا ts_now کات می‌زند،
        2) PREPARE_DATA_FOR_TRAIN را با همان فایل‌های کات‌شده instantiate می‌کند،
        3) همان load_data() کلاس اصلی را فراخوانی می‌کند،
        4) دیتافریم ادغام‌شده‌ی نهایی را برمی‌گرداند.
        """
        ts_now = pd.to_datetime(ts_now)
        cut_paths = self._write_cut_csvs(ts_now)
        try:
            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths=cut_paths,
                main_timeframe=self.main_timeframe,
                verbose=self.verbose,
                fast_mode=self.fast_mode,
                strict_disk_feed=self.strict_disk_feed,
            )
            # drift را در Production حذف می‌کنیم تا parity با لایو باشد
            if self._disable_drift:
                if hasattr(prep, "shared_start_date"):
                    setattr(prep, "shared_start_date", None)
                if hasattr(prep, "drift_finder"):
                    try:
                        delattr(prep, "drift_finder")
                    except Exception:
                        pass

            merged = prep.load_data()
            # پاک‌سازی‌های سبک
            tcol = f"{self.main_timeframe}_time"
            if tcol in merged.columns:
                merged[tcol] = pd.to_datetime(merged[tcol], errors="coerce")
                merged = merged.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
            merged.replace([np.inf, -np.inf], np.nan, inplace=True)
            merged.ffill(inplace=True)
            merged.dropna(how="all", inplace=True)
            return merged
        finally:
            self._cleanup_tmp()

    def ready(
        self,
        merged: pd.DataFrame,
        selected_features: Optional[List[str]] = None,
        make_labels: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        """
        هیچ فیچر جدیدی نمی‌سازد؛ فرض این است که همانند Train، اندیکاتورها/ستون‌ها
        قبلاً توسط کلاس اصلی اضافه شده‌اند. فقط:
          - ترتیب/زیرمجموعه ستون‌ها را مطابق selected_features فیکس می‌کند،
          - در صورت نیاز y را طبق تعریف استاندارد می‌سازد: y(t)=1 اگر close_{t+1}>close_t.
        """
        df = merged.copy()
        tmain = self.main_timeframe
        close_col = f"{tmain}_close"

        if selected_features and len(selected_features) > 0:
            # ستون‌های جاافتاده را بعداً با صفر پر می‌کنیم (در predictor)
            feat_list = list(selected_features)
            X = df[[c for c in feat_list if c in df.columns]].copy()
        else:
            # اگر کاربر feature list نداده، همه‌ی ستون‌های عددی را استفاده کن (اما زمان را حذف کن)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feat_list = [c for c in num_cols if not c.endswith("_time")]
            X = df[feat_list].copy()

        y = None
        if make_labels:
            if close_col not in df.columns:
                raise ValueError(f"Missing '{close_col}' for label building.")
            yy = (df[close_col].shift(-1) > df[close_col]).astype("Int64")
            # سطر آخر لیبل ندارد
            yy = yy.iloc[:-1].astype("Int64")
            X = X.iloc[:-1, :].copy()
            y = yy.astype(int)

        return X, y, feat_list
