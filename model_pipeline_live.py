#!/usr/bin/env python3
# model_pipeline_live.py
# ──────────────────────
# لایو / بک‌تست بدون SMOTE  –  ضد-NaN  –  سازگار با کُدهای قدیمی

from __future__ import annotations
from typing import Any, Dict
import logging, time

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

__all__ = ["ModelPipelineLive"]


class ModelPipelineLive:
    """
    Imputer ➜ StandardScaler ➜ LogisticRegression
            (+CalibratedClassifierCV اختیاری)

    - هیچ SMOTEای در لایو وجود ندارد.
    - خاصیت `base_pipe` برای سازگاری با اسکریپت‌های قدیمی باقی مانده است.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        hyperparams : Dict[str, Any] | None = None,
        *,
        calibrate   : bool = True,
        calib_method: str  = "sigmoid",
    ) -> None:
        self.hyperparams  = hyperparams or {}
        self.calibrate    = calibrate
        self.calib_method = calib_method

        # ---------- ۱)  لوجستیک رگرشن آماده-ی آموزش ----------
        hp = self.hyperparams
        lr = LogisticRegression(
            C             = hp.get("C", 1.0),
            max_iter      = hp.get("max_iter", 5000),
            tol           = hp.get("tol", 1e-3),
            penalty       = hp.get("penalty", "l2"),
            solver        = hp.get("solver", "lbfgs"),
            fit_intercept = hp.get("fit_intercept", True),
            class_weight  = hp.get("class_weight", None),
            random_state  = 42,
            n_jobs        = -1,
            **({"multi_class": hp["multi_class"]}
               if hp.get("multi_class", "auto") != "auto" else {}),
        )

        # ---------- ۲)  پایپ‌لاین «خام» (برای patch-شدن در build_live_est) ----------
        self.base_pipe: Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # ضد-NaN
            ("scaler",  StandardScaler()),
            ("clf",     lr),
        ])

        # ---------- ۳)  پس از fit() اگر calibrate=True ----------
        self._calibrator: CalibratedClassifierCV | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """
        – `self.base_pipe` آموزش می‌بیند.  
        – اگر `calibrate` فعال باشد، یک CalibratedClassifierCV
          روی همان پایپ‌لاین فیت می‌کنیم.
        """
        t0 = time.perf_counter()
        self.base_pipe.fit(X, y)

        if self.calibrate:
            self._calibrator = CalibratedClassifierCV(
                estimator = self.base_pipe,
                method    = self.calib_method,
                cv        = TimeSeriesSplit(n_splits=5),
                n_jobs    = -1,
            ).fit(X, y)

        LOGGER.info("✅ live-pipe fit %.2fs | calibrate=%s",
                    time.perf_counter() - t0, self.calibrate)
        return self

    # ------------------------------------------------------------------ #
    #                    پیش‌بینی / دو آستانه                           #
    # ------------------------------------------------------------------ #
    def _delegate(self):
        return self._calibrator if (self.calibrate and self._calibrator) else self.base_pipe

    def predict_proba(self, X):
        return self._delegate().predict_proba(X)

    def predict(self, X):
        return self._delegate().predict(X)

    @staticmethod
    def apply_thresholds(prob, neg_thr, pos_thr):
        y = np.full(prob.shape[0], -1, dtype=int)
        y[prob <= neg_thr] = 0
        y[prob >= pos_thr] = 1
        return y

    def decide(self, X, *, neg_thr, pos_thr):
        return self.apply_thresholds(self.predict_proba(X)[:, 1], neg_thr, pos_thr)

    # ------------------------------------------------------------------ #
    # دسترسی کمکی
    # ------------------------------------------------------------------ #
    @property
    def steps(self):           # برای سازگاری با کدهای قدیمی
        return self.base_pipe.steps

    def __getattr__(self, item):
        # اگر به چیزی مثل coef_ نیاز باشد
        try:
            return getattr(self._delegate(), item)
        except AttributeError as exc:
            raise AttributeError(item) from exc
