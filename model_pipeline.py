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
        algo: str = "logreg",
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
        self.algo = algo
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
                # NEW:
                if "sampling_strategy" not in sm_params:
                    sm_params["sampling_strategy"] = 0.8  # بجای 1.0
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
    
    
    def _build_base_estimator(self):
        """
        بر اساس self.algo مدل پایه را می‌سازد.
        """
        algo = self.algo.lower()
        params = dict(self.hyperparams)

        if algo == "logreg":
            return LogisticRegression(**params)

        elif algo == "svm":
            from sklearn.svm import SVC
            # پارامترهای ساده: C, kernel, gamma
            default = dict(C=1.0, kernel="rbf", gamma="scale", probability=True)
            default.update(params)
            return SVC(**default)

        elif algo == "rf":
            from sklearn.ensemble import RandomForestClassifier
            default = dict(
                n_estimators=params.pop("n_estimators", 200),
                max_depth=params.pop("max_depth", None),
                min_samples_leaf=params.pop("min_samples_leaf", 2),
                n_jobs=-1,
                random_state=self.random_state,
            )
            return RandomForestClassifier(**default)

        elif algo == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            default = dict(
                learning_rate=params.pop("learning_rate", 0.05),
                max_depth=params.pop("max_depth", None),
                max_leaf_nodes=params.pop("max_leaf_nodes", 31),
                max_iter=params.pop("max_iter", 200),
                random_state=self.random_state,
            )
            return HistGradientBoostingClassifier(**default)

        else:
            raise ValueError(f"Unknown algo={algo}")

class EnsembleModel:
    """
    نگه‌دارنده‌ی چند ModelPipeline و انجام رای‌گیری/تجمیع احتمالات.
    """

    def __init__(self, models: list[ModelPipeline], vote_k: int = 3):
        if not models:
            raise ValueError("EnsembleModel needs at least one ModelPipeline")
        self.models = models
        self.vote_k = vote_k

    def predict_proba(self, X):
        """
        احتمال نهایی = میانگین احتمال کلاس ۱ بین همه‌ی مدل‌ها.
        """
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        avg = np.mean(np.vstack(probs), axis=0)
        # برگرداندن در قالب (n_samples, 2) مثل اسکیک‌لرن
        p1 = avg
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict_actions(self, X, neg_thrs: list[float], pos_thrs: list[float]) -> np.ndarray:
        """
        اعمال آستانه‌ی جداگانه برای هر مدل و رای‌گیری:
        - اگر حداقل vote_k مدل BUY بگویند → BUY
        - اگر حداقل vote_k مدل SELL بگویند → SELL
        - در غیر این صورت → NONE
        """
        n = X.shape[0]
        votes = np.zeros((len(self.models), n), dtype=int)  # 1=BUY, 0=SELL, -1=NONE

        for i, m in enumerate(self.models):
            p = m.predict_proba(X)[:, 1]
            neg, pos = neg_thrs[i], pos_thrs[i]
            v = np.full(n, -1, dtype=int)
            v[p <= neg] = 0
            v[p >= pos] = 1
            votes[i, :] = v

        final = np.full(n, -1, dtype=int)
        # BUY اگر >= vote_k رای BUY
        buy_mask = (votes == 1).sum(axis=0) >= self.vote_k
        # SELL اگر >= vote_k رای SELL
        sell_mask = (votes == 0).sum(axis=0) >= self.vote_k

        final[buy_mask] = 1
        final[sell_mask] = 0
        # اگر هم BUY هم SELL شد (خیلی بعید) می‌توانی ترجیحی بگذاری؛ فعلاً اولویت را BUY بگذاریم:
        conflict_mask = buy_mask & sell_mask
        final[conflict_mask] = 1

        return final

    def get_scaler(self):
        # از اولین مدل اسکیلر را بگیر (برای DriftChecker / compatibility)
        return self.models[0].get_scaler()
