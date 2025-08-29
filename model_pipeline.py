# model_pipeline.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class _CompatWrapper:
    """
    Wrapper سبک برای سازگاری با پیکل/جاب‌لیب و کد شبیه‌ساز.
    - predict / predict_proba پاس می‌دهد.
    - get_scaler() در حالت کالیبره هم اسکیلرِ fitted را برمی‌گرداند.
    """
    def __init__(self, inner):
        self._model = inner
        self._inner = inner
        # برای سازگاری با کدی که شاید به named_steps/steps نگاه کند:
        self.named_steps = getattr(inner, "named_steps", {})
        self.steps = getattr(inner, "steps", [])

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_scaler(self):
        """
        تلاش برای پیدا کردن StandardScalerِ فیت‌شده:
        - اگر CalibratedClassifierCV است: base_estimator_ معمولاً Pipeline است.
        - اگر Pipeline است: مستقیماً از named_steps بردار.
        """
        base = getattr(self._model, "base_estimator_", None) or getattr(self._model, "base_estimator", None)
        if isinstance(base, Pipeline):
            sc = base.named_steps.get("scaler")
            if sc is not None:
                return sc

        if isinstance(self._model, Pipeline):
            return self._model.named_steps.get("scaler")

        return None


class ModelPipeline:
    """
    StandardScaler + LogisticRegression (+ optional calibration)
    - سازگار با GA و شبیه‌ساز
    - SMOTE فقط در مرحلهٔ fit و اختیاری (پیش‌فرض OFF)
    """
    def __init__(
        self,
        hyperparams: Dict[str, Any],
        *,
        calibrate: bool = True,
        calib_method: str = "sigmoid",
        use_smote_in_fit: bool = False,         # ← پیش‌فرض خاموش
        random_state: Optional[int] = 2025
    ):
        self.hyperparams = dict(hyperparams or {})
        self.calibrate = bool(calibrate)
        self.calib_method = str(calib_method)
        self.use_smote_in_fit = bool(use_smote_in_fit)
        self.random_state = random_state

        # پایپ‌لاین پایه
        clf = LogisticRegression(**self.hyperparams)
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        # کالیبراسیون اختیاری
        if self.calibrate:
            model = CalibratedClassifierCV(base, method=self.calib_method, cv=3)
        else:
            model = base

        # رَپر سازگار برای ذخیره و شبیه‌ساز
        self.pipeline = _CompatWrapper(model)

    # ---------------- public API ----------------
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        """
        اگر use_smote_in_fit=True باشد فقط در این متد بازنمونه‌گیری انجام می‌شود
        و خودِ pipeline ذخیره‌شده شامل SMOTE نمی‌شود ⇒ در لایو مشکلی نیست.
        """
        X_fit, y_fit = X, y

        if self.use_smote_in_fit:
            try:
                from imblearn.over_sampling import SMOTE
                # n_jobs=1 برای جلوگیری از oversubscription
                sm = SMOTE(random_state=self.random_state, n_jobs=1)
                X_fit, y_fit = sm.fit_resample(X, y)
            except Exception:
                # اگر imblearn در محیط نبود، بی‌سر و صدا ادامه بده
                X_fit, y_fit = X, y

        # آموزش مدل (اگر کالیبره باشد، روی داده‌های resampled/train انجام می‌شود)
        self.pipeline._model.fit(X_fit, y_fit)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def get_scaler(self):
        return self.pipeline.get_scaler()
