# model_pipeline.py
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class _CompatWrapper:
    """
    یک Wrapper سبک برای سازگاری با joblib/pickle و کدهای جانبی (مثل شبیه‌ساز).
    - predict / predict_proba را شفاف پاس می‌دهد.
    - get_scaler(): در هر دو حالت Pipeline و CalibratedClassifierCV اسکیلر fitted را برمی‌گرداند.
    - __getstate__/__setstate__: ذخیره/بازیابی امن بدون بازگشتِ بی‌نهایت.
    """
    def __init__(self, inner):
        self._model = inner          # می‌تواند Pipeline یا CalibratedClassifierCV باشد
        self._inner = inner
        # برای سازگاری با کدهایی که احتمالاً به این فیلدها مراجعه می‌کنند:
        self.named_steps = getattr(inner, "named_steps", {})
        self.steps = getattr(inner, "steps", [])

    # جلوگیری از recursion روی dunderها
    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return getattr(self._model, item)

    def __getstate__(self):
        # فقط کمینه‌ی لازم را ذخیره کن
        return {"_model": getattr(self, "_model", None), "_inner": getattr(self, "_inner", None)}

    def __setstate__(self, state):
        self._model = state.get("_model", None)
        self._inner = state.get("_inner", None)
        # بازتزریق فیلدهای کمکی (اگر موجود باشد)
        self.named_steps = getattr(self._inner, "named_steps", {})
        self.steps = getattr(self._inner, "steps", [])

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_scaler(self):
        """
        تلاش برای پیدا کردن StandardScalerِ فیت‌شده:
        - اگر CalibratedClassifierCV باشد: به base_estimator_ (یا estimator) سر بزن.
        - اگر Pipeline باشد: از named_steps بردار.
        """
        # حالت CalibratedClassifierCV (بعد از fit عموماً base_estimator_ ست می‌شود)
        for attr in ("base_estimator_", "estimator", "base_estimator"):
            base = getattr(self._model, attr, None)
            if isinstance(base, Pipeline):
                sc = base.named_steps.get("scaler")
                if sc is not None:
                    return sc

        # حالت Pipeline مستقیم
        if isinstance(self._model, Pipeline):
            return self._model.named_steps.get("scaler")

        return None


class ModelPipeline:
    """
    StandardScaler + LogisticRegression (+ Calibration اختیاری)

    - پارامترها:
        hyperparams         : دیکت تنظیمات LogisticRegression (مانند C, penalty, solver, ...)
        calibrate           : اگر True باشد، از CalibratedClassifierCV استفاده می‌شود.
        calib_method        : 'sigmoid' یا 'isotonic'
        use_smote_in_fit    : اگر True باشد، فقط در TRAIN هنگام fit بازنمونه‌گیری انجام می‌شود.
                              (در مسیر inference/لایو اثری ندارد)
        smote_kwargs        : دیکت اختیاری برای SMOTE (sampling_strategy, k_neighbors, ...)
        random_state        : بذر تصادفی برای تکرارپذیری

    - توجه: هیچ import از imblearn در سطح ماژول انجام نشده تا اگر imblearn نصب نباشد،
            لایو/پیش‌بینی بی‌خطا بماند. تنها هنگام fit و فقط اگر use_smote_in_fit=True باشد،
            SMOTE به‌صورت دینامیک import می‌شود.
    """
    def __init__(
        self,
        hyperparams: Dict[str, Any],
        *,
        calibrate: bool = True,
        calib_method: str = "sigmoid",
        use_smote_in_fit: bool = False,
        smote_kwargs: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = 2025,
    ):
        self.hyperparams = dict(hyperparams or {})
        self.calibrate = bool(calibrate)
        self.calib_method = str(calib_method)
        self.use_smote_in_fit = bool(use_smote_in_fit)
        # اگر SMOTE در fit فعال است، balanced را خنثی کن تا دوباره‌وزن‌دهی رخ ندهد
        if self.use_smote_in_fit and self.hyperparams.get("class_weight") == "balanced":
            self.hyperparams["class_weight"] = None

        self.smote_kwargs = dict(smote_kwargs or {})
        self.random_state = random_state

        # پایپ‌لاین پایه: StandardScaler سپس LogisticRegression
        clf = LogisticRegression(**self.hyperparams)
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        # در صورت نیاز، کالیبراسیون احتمالات
        model = CalibratedClassifierCV(base, method=self.calib_method, cv=3) if self.calibrate else base

        # رَپر سازگار برای ذخیره و شبیه‌ساز
        self.pipeline = _CompatWrapper(model)

    # ---------------- public API ----------------
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        """
        اگر use_smote_in_fit=True باشد:
        - SMOTE فقط روی داده‌ی TRAIN در همین متد اعمال می‌شود (نه در inference).
        - مقدار k_neighbors به‌صورت امن بر اساس اندازه‌ی کلاس اقلیت تنظیم می‌شود.
        - اگر imblearn نصب نباشد یا کلاس اقلیت خیلی کوچک باشد، بدون SMOTE ادامه می‌دهیم.
        """
        X_fit, y_fit = X, y

        if self.use_smote_in_fit:
            try:
                # import تنبل تا اگر imblearn نصب نباشد، لایو از کار نیفتد
                from imblearn.over_sampling import SMOTE
                from collections import Counter

                # y را یک‌بعدی کن
                y_arr = np.asarray(y).ravel()

                # شمارش کلاس‌ها برای تنظیم امن k_neighbors
                cnt = Counter(y_arr.tolist()) if len(y_arr) else Counter()
                min_class = min(cnt.values()) if cnt else 0

                # مقدار پیش‌فرض/درخواستی کاربر برای k
                k_default = int(self.smote_kwargs.get("k_neighbors", 5))
                # k امن: حداکثر min_class - 1 و حداقل 1
                k_safe = max(1, min(k_default, max(1, min_class - 1)))

                sm_params = dict(self.smote_kwargs)
                sm_params["k_neighbors"] = k_safe
                # برای جلوگیری از oversubscription
                sm = SMOTE(random_state=self.random_state, **sm_params)

                X_res, y_res = sm.fit_resample(
                    X if isinstance(X, np.ndarray) else np.asarray(X),
                    y_arr
                )

                # اگر X اولیه DataFrame بوده، ستون‌ها را حفظ کنیم
                if isinstance(X, pd.DataFrame):
                    X_fit = pd.DataFrame(X_res, columns=X.columns)
                else:
                    X_fit = X_res
                y_fit = y_res

            except Exception:
                # هر خطایی (مثلاً نصب نبودن imblearn یا کوچک‌بودن کلاس اقلیت) ⇒ بدون SMOTE ادامه بده
                X_fit, y_fit = X, y

        # آموزش مدل (در صورت کالیبراسیون، روی داده‌های resampled/Train انجام می‌شود)
        self.pipeline._model.fit(X_fit, y_fit)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def get_scaler(self):
        return self.pipeline.get_scaler()
