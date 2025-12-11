#!/usr/bin/env python3
"""
strong_feature_selector.py

Feature selector نهایی برای مدل‌های کلاسیفیکیشن (باینری 0/1) که:
    1) فقط ستون‌های عددی را در نظر می‌گیرد.
    2) فیچرها را بر اساس |corr(feature, target)| مرتب می‌کند
       و فقط بهترین‌ها را نگه می‌دارد.
    3) روی فیچرهای فیلتر شده یک RandomForestClassifier
       آموزش می‌دهد و حداکثر max_features فیچرِ با اهمیت
       بیشتر را نگه می‌دارد.

نکته مهم:
    - این کلاس فقط ستون‌ها را کم می‌کند (هیچ سطری و هیچ مقدار
      داده‌ای را در X اصلی تغییر نمی‌دهد).
    - خروجی برای هر نوع مدل ML/DL که ورودی عددی می‌گیرد مناسب است.
"""

from __future__ import annotations

from typing import List, Optional
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed


class StrongFeatureSelector:
    """
    Final, model-agnostic feature selector for binary classification.

    Pipeline:
      1) Keep only numeric features.
      2) Rank features by absolute correlation with the binary target y (0/1).
         Keep at most `pre_selection_factor * max_features`.
      3) Train a RandomForestClassifier on the reduced set and keep
         at most `max_features` features with the highest importances.

    پارالل:
      - RandomForest با n_jobs اجرا می‌شود (thread-level).
      - محاسبه‌ی correlation اگر corr_n_jobs > 1 باشد، به صورت block-wise
        با joblib موازی می‌شود.
    """

    def __init__(
        self,
        max_features: int = 300,
        pre_selection_factor: int = 3,
        random_state: int = 2025,
        n_estimators: int = 256,
        n_jobs: int = 1,
        corr_n_jobs: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        max_features : int
            حداکثر تعداد فیچر نهایی که نگه می‌داریم.
        pre_selection_factor : int
            چند برابر max_features را در مرحله‌ی correlation نگه داریم.
        random_state : int
            جهت reproducibility برای RandomForest.
        n_estimators : int
            تعداد درخت‌های RandomForest.
        n_jobs : int
            تعداد تردهای RandomForest (مانند sklearn، -1 یعنی همه‌ی CPUها).
        corr_n_jobs : int
            تعداد job برای محاسبه‌ی corrwith به صورت موازی.
            1 یا کمتر → تک‌تردی. مقادیر منفی مثل joblib رفتار می‌کند.
        """
        self.max_features = int(max_features)
        self.pre_selection_factor = int(max(1, pre_selection_factor))
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.n_jobs = int(n_jobs)
        self.corr_n_jobs = int(corr_n_jobs)

        self.selected_features_: List[str] = []
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # کمک‌تابع: correlation به صورت (امکاناً) موازی
    # ------------------------------------------------------------------ #
    def _corr_with_target(
        self,
        X: pd.DataFrame,
        y_float: pd.Series,
    ) -> pd.Series:
        """
        محاسبه‌ی corrwith به صورت تک‌تردی یا block-wise موازی.
        خروجی: |corr| با fillna(0).
        """
        # تک‌تردی یا دیتافریم کوچک → روش ساده
        if self.corr_n_jobs <= 1 or X.shape[1] <= 1:
            return X.corrwith(y_float).abs().fillna(0.0)

        try:
            # محاسبه‌ی n_jobs مؤثر
            if self.corr_n_jobs < 0:
                try:
                    cpu = mp.cpu_count()
                except Exception:
                    cpu = 1
                n_jobs = max(1, cpu + self.corr_n_jobs + 1)
            else:
                n_jobs = self.corr_n_jobs

            n_jobs = max(1, int(n_jobs))
            cols = list(X.columns)
            n_blocks = min(len(cols), n_jobs * 4)  # چند بلاک بیشتر از تردها برای بالانس

            if n_blocks <= 1:
                return X.corrwith(y_float).abs().fillna(0.0)

            blocks = np.array_split(cols, n_blocks)

            def _block_corr(col_block: List[str]) -> pd.Series:
                if not col_block:
                    return pd.Series(dtype=float)
                return X[col_block].corrwith(y_float)

            parts = Parallel(n_jobs=n_jobs)(
                delayed(_block_corr)(block.tolist())
                for block in blocks
                if len(block) > 0
            )

            corr = pd.concat(parts)
            return corr.abs().fillna(0.0)

        except Exception as e:
            self.logger.warning(
                "[StrongFeatureSelector] parallel corr failed (%s); falling back to single-thread.",
                e,
            )
            return X.corrwith(y_float).abs().fillna(0.0)

    # ------------------------------------------------------------------ #
    # core API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StrongFeatureSelector":
        """
        X : DataFrame
            ماتریس فیچرها (فقط ستون‌ها بررسی می‌شوند، index دست نمی‌خورد).
        y : Series / array-like
            تارگت باینری 0/1.

        فقط ستون‌های عددی در نظر گرفته می‌شوند؛
        هیچ سطری حذف یا جابه‌جا نمی‌شود.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y)
        # به float برای corr (0/1) → (0.0/1.0)
        y_float = y.astype(float)

        # --- فقط ستون‌های عددی ---
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            self.logger.warning(
                "[StrongFeatureSelector] No numeric features; nothing selected."
            )
            self.selected_features_ = []
            return self

        # --- پاکسازی داخلی برای محاسبات (بدون تغییر X اصلی) ---
        X_work = X_num.replace([np.inf, -np.inf], np.nan)
        if X_work.isna().any().any():
            X_work = X_work.fillna(X_work.median())

        # --- sanity target ---
        unique_y = np.unique(y_float[~np.isnan(y_float)])
        if unique_y.size < 2:
            # تارگت دژنره (مثلاً همه 0 یا همه 1) → فقط truncate
            self.logger.warning(
                "[StrongFeatureSelector] Target has <2 unique values; "
                "skipping model-based selection."
            )
            self.selected_features_ = list(X_num.columns[: self.max_features])
            return self

        # ------------------------------------------------------------------
        # 1) پیش‌انتخاب بر اساس correlation با تارگت
        # ------------------------------------------------------------------
        corr = self._corr_with_target(X_work, y_float)

        if corr.empty:
            self.logger.warning(
                "[StrongFeatureSelector] Correlation computation failed; "
                "falling back to first %d numeric features.",
                self.max_features,
            )
            self.selected_features_ = list(X_num.columns[: self.max_features])
            return self

        total_feats = X_work.shape[1]
        # حداکثر تعداد فیچرهایی که بعد از مرحله‌ی corr نگه می‌داریم
        k_pre = min(
            max(self.max_features * self.pre_selection_factor, self.max_features),
            total_feats,
        )

        corr_sorted = corr.sort_values(ascending=False)
        cols_corr = corr_sorted.index[:k_pre].tolist()
        X_corr = X_work[cols_corr]

        # اگر همین حالا تعداد فیچرها کمتر/برابر max_features است، دیگر RF لازم نیست
        if X_corr.shape[1] <= self.max_features:
            self.selected_features_ = list(X_corr.columns)
            return self

        # ------------------------------------------------------------------
        # 2) RandomForestClassifier برای importance و انتخاب نهایی
        # ------------------------------------------------------------------
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced_subsample",
        )

        rf.fit(X_corr.values, y_float.values)

        importances = pd.Series(rf.feature_importances_, index=X_corr.columns)
        importances_sorted = importances.sort_values(ascending=False)

        # فقط فیچرهای با importance غیر صفر را در اولویت قرار بده
        non_zero = importances_sorted[importances_sorted > 0]

        if non_zero.empty:
            # اگر همه صفر شد، همان top-k_pre بر اساس corr را truncate می‌کنیم
            self.logger.warning(
                "[StrongFeatureSelector] All feature importances are zero; "
                "falling back to correlation-only top %d.",
                self.max_features,
            )
            self.selected_features_ = cols_corr[: self.max_features]
        else:
            self.selected_features_ = list(non_zero.index[: self.max_features])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        فقط ستون‌های انتخاب‌شده را نگه می‌دارد.
        هیچ سطری و هیچ مقداری تغییر نمی‌کند.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.selected_features_:
            # اگر هیچ فیچری انتخاب نشده → DataFrame خالی با همان index
            return X.iloc[:, 0:0].copy()

        cols = [c for c in self.selected_features_ if c in X.columns]
        return X[cols].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        shortcut = fit + transform
        """
        return self.fit(X, y).transform(X)

    def get_support(self) -> List[str]:
        """
        نام ستون‌های انتخاب‌شده (برای لاگ/استفاده در جاهای دیگر).
        """
        return list(self.selected_features_)
