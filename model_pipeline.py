"""Leak-free calibrated logistic regression pipeline (compat-safe)

Changes vs previous version:
- Moves StandardScaler + SMOTE *inside* the estimator used by CalibratedClassifierCV
  so that resampling/scaling happen **only** on training folds during calibration (no leakage).
- Keeps backward compatibility: `model.pipeline.named_steps.get("scaler")` still works
  via a thin compatibility wrapper around the underlying estimator.
"""

from __future__ import annotations

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

import logging
logging.getLogger("sklearnex").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("daal4py").setLevel(logging.WARNING)

from typing import Any, Dict, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


class _CompatWrapper:
    """
    Thin wrapper to maintain `.named_steps` access for compatibility with old code:
    - Delegates fit/predict/predict_proba to `model` (CalibratedClassifierCV or Pipeline).
    - Exposes `.named_steps` from the *inner* estimator pipeline (the one with scaler/smote).
    - Exposes `.steps` similarly if available.
    """
    def __init__(self, model, inner_estimator: Optional[ImbPipeline]) -> None:
        self._model = model
        self._inner = inner_estimator
        self.named_steps = getattr(self._inner, "named_steps", {})
        self.steps = getattr(self._inner, "steps", [])

    def fit(self, X, y):
        return self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    # Provide limited sklearn-like API passthrough if needed
    def __getattr__(self, item):
        return getattr(self._model, item)


class ModelPipeline:
    """
    Pipeline = (CalibratedClassifierCV | ImbPipeline[Scaler+SMOTE+LR]) with no leakage.

    If `calibrate=True` (default), uses:
        CalibratedClassifierCV(estimator=ImbPipeline([Scaler, SMOTE, LR]), cv=TimeSeriesSplit(5))

    If `calibrate=False`, uses:
        ImbPipeline([Scaler, SMOTE, LR])

    In both cases, `self.pipeline` is a *compat wrapper* exposing:
        - fit / predict / predict_proba
        - named_steps (from the inner estimator, so .get("scaler") still works)
        - steps (if available)
    """

    def __init__(
        self,
        hyperparams: Dict[str, Any] | None = None,
        calibrate: bool = True,
        calib_method: str = "sigmoid",
    ) -> None:
        self.hyperparams  = hyperparams or {}
        self.calibrate    = bool(calibrate)
        self.calib_method = calib_method
        self.is_calibrated = self.calibrate

        # ---- Extract hyperparameters with safe defaults ----
        C             = self.hyperparams.get('C', 1.0)
        max_iter      = self.hyperparams.get('max_iter', 300)
        tol           = self.hyperparams.get('tol', 3e-4)
        penalty       = self.hyperparams.get('penalty', 'l2')
        solver        = self.hyperparams.get('solver', 'lbfgs')
        fit_intercept = self.hyperparams.get('fit_intercept', True)
        class_weight  = self.hyperparams.get('class_weight', None)
        multi_class   = self.hyperparams.get('multi_class', 'auto')

        # Safety: liblinear does not support multinomial
        if solver == "liblinear" and multi_class == "multinomial":
            multi_class = "auto"

        # extra_multi = {}
        # if multi_class != "auto":
        #     extra_multi["multi_class"] = multi_class

        base_lr = LogisticRegression(
            C=C,
            max_iter=max_iter,
            tol=tol,
            penalty=penalty,
            solver=solver,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            warm_start=True,
            random_state=42,
            n_jobs=-1,
            # هیچ multi_classـی پاس نده
        )
        # --- Inner estimator with preprocessing INSIDE (no leakage) ---
        inner_estimator = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote',  SMOTE(random_state=42)),
            ('lr',     base_lr),
        ])

        if self.calibrate:
            model = CalibratedClassifierCV(
                estimator=inner_estimator,
                method=self.calib_method,
                cv=TimeSeriesSplit(n_splits=5),
                n_jobs=-1
            )
            self._model = model
            self._inner = inner_estimator
        else:
            self._model = inner_estimator
            self._inner = inner_estimator

        # ---- Public handle with backward-compatible attributes ----
        self.pipeline = _CompatWrapper(self._model, self._inner)

    # ------------------------------------------------------------------
    # Proxy methods
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        Fit the model. Kept minimal and robust (no touching internal CV iterator)
        to avoid unexpected behavior across sklearn versions.
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Training started with hyperparams: %s", self.hyperparams)
        self.pipeline.fit(X, y)
        logger.info("✅ Training finished.")
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    # ------------------------------------------------------------------
    # Optional helpers for compatibility/introspection
    # ------------------------------------------------------------------
    def get_scaler(self):
        """Return the StandardScaler if present (None otherwise)."""
        try:
            return self._inner.named_steps.get("scaler")
        except Exception:
            return None

    @property
    def estimator_(self):
        """Access the underlying trained estimator (CalibratedClassifierCV or Pipeline)."""
        return self._model
