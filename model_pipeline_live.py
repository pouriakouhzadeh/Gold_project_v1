#!/usr/bin/env python3
# model_pipeline_live.py  –  نسخهٔ ضد-NaN (بدون SMOTE)
# -----------------------------------------------------
from __future__ import annotations
from typing import Any, Dict
import logging, time

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer          # ← جدید
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

__all__ = ["ModelPipelineLive"]


class ModelPipelineLive:
    """
    StandardScaler ➜ LogisticRegression (+ اختیارى Calibrator)
    *بدون* SMOTE و با Imputer میانگین/میانه
    """
    # -------------------------------------------------- #
    def __init__(
        self,
        hyperparams: Dict[str, Any] | None = None,
        *,
        calibrate: bool = True,
        calib_method: str = "sigmoid",
    ) -> None:
        self.hyperparams  = hyperparams or {}
        self.calibrate    = calibrate
        self.calib_method = calib_method

        # 1) Base LR
        hp = self.hyperparams
        base_lr = LogisticRegression(
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

        # 2) Base pipeline   (Imputer → Scaler → LR)
        self.base_pipe: Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),   # ← جدید
            ("scaler",  StandardScaler()),
            ("clf",     base_lr),
        ])

        self._calibrator: CalibratedClassifierCV | None = None

    # -------------------------------------------------- #
    def fit(self, X, y):
        t0 = time.perf_counter()
        self.base_pipe.fit(X, y)

        if self.calibrate:
            self._calibrator = CalibratedClassifierCV(
                estimator=self.base_pipe,
                method=self.calib_method,
                cv=TimeSeriesSplit(n_splits=5),
                n_jobs=-1,
            ).fit(X, y)

        LOGGER.info("✅ live-pipe fit %.2fs | calibrate=%s",
                    time.perf_counter() - t0, self.calibrate)
        return self

    # --------------- inference ------------------------ #
    def _delegate(self):
        return self._calibrator if (self.calibrate and self._calibrator) else self.base_pipe

    def predict_proba(self, X):
        return self._delegate().predict_proba(X)

    def predict(self, X):
        return self._delegate().predict(X)

    # -------- two-threshold helper ------------------- #
    @staticmethod
    def apply_thresholds(proba: np.ndarray,
                         neg_thr: float,
                         pos_thr: float) -> np.ndarray:
        y_pred = np.full(proba.shape[0], -1, dtype=int)
        y_pred[proba <= neg_thr] = 0
        y_pred[proba >= pos_thr] = 1
        return y_pred

    def decide(self, X, *, neg_thr: float, pos_thr: float):
        return self.apply_thresholds(self.predict_proba(X)[:, 1], neg_thr, pos_thr)

    # -------------------------------------------------- #
    @property
    def steps(self):
        return self.base_pipe.steps

    def __getattr__(self, name):
        try:
            return getattr(self._delegate(), name)
        except AttributeError as exc:
            raise AttributeError(name) from exc
