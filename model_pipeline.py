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
    Pipeline عمومی برای چند الگوریتم (logreg / rf / hgb / svm / xgb)
    - در صورت نیاز StandardScaler را به‌عنوان مرحله‌ی اول اضافه می‌کند.
    - کالیبراسیون اختیاری با CalibratedClassifierCV.
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
        self.algo = (algo or "logreg").lower()

        # اگر SMOTE در fit فعال است، class_weight='balanced' را خنثی کن
        if self.use_smote_in_fit and self.hyperparams.get("class_weight") == "balanced":
            self.hyperparams["class_weight"] = None

        self.smote_kwargs = dict(smote_kwargs or {})
        self.random_state = random_state

        # مدل پایه بر اساس self.algo
        base_clf = self._build_base_estimator()

        # فقط برای مدل‌هایی که به اسکیل حساس‌اند، StandardScaler می‌گذاریم
        if self.algo in ("logreg", "svm"):
            base = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", base_clf),
                ]
            )
        else:
            base = Pipeline([("clf", base_clf)])

        # کالیبراسیون احتمالات (برای مدل نهایی)
        if self.calibrate:
            model = CalibratedClassifierCV(base, method=self.calib_method, cv=3)
        else:
            model = base

        # Wrapper سازگار
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
        پشتیبانی‌شده: logreg, rf, hgb, svm, xgb
        """
        algo = (self.algo or "logreg").lower()
        params = dict(self.hyperparams)

        if algo == "logreg":
            params.setdefault("max_iter", 10_000)
            return LogisticRegression(**params)

        elif algo == "svm":
            from sklearn.svm import SVC
            base = dict(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
            )
            base.update(params)
            return SVC(**base)

        elif algo == "rf":
            from sklearn.ensemble import RandomForestClassifier

            base = dict(
                n_estimators=params.pop("n_estimators", 300),
                max_depth=params.pop("max_depth", None),
                min_samples_leaf=params.pop("min_samples_leaf", 1),
                max_features=params.pop("max_features", "sqrt"),
                class_weight=params.pop("class_weight", None),
                n_jobs=-1,
                random_state=self.random_state,
            )
            base.update(params)
            return RandomForestClassifier(**base)

        elif algo == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier

            base = dict(
                learning_rate=params.pop("learning_rate", 0.05),
                max_depth=params.pop("max_depth", None),
                max_leaf_nodes=params.pop("max_leaf_nodes", 31),
                max_iter=params.pop("max_iter", 200),
                random_state=self.random_state,
            )
            base.update(params)
            return HistGradientBoostingClassifier(**base)

        elif algo == "xgb":
            try:
                from xgboost import XGBClassifier  # type: ignore
            except Exception as e:  # اگر xgboost نصب نباشد
                raise ImportError(
                    "algo='xgb' انتخاب شده اما کتابخانه xgboost نصب نیست (pip install xgboost)."
                ) from e

            base = dict(
                n_estimators=params.pop("n_estimators", 300),
                learning_rate=params.pop("learning_rate", 0.05),
                max_depth=params.pop("max_depth", 5),
                subsample=params.pop("subsample", 0.8),
                colsample_bytree=params.pop("colsample_bytree", 0.8),
                reg_lambda=params.pop("reg_lambda", 1.0),
                n_jobs=-1,
                eval_metric="logloss",
                random_state=self.random_state,
            )
            base.update(params)
            return XGBClassifier(**base)

        else:
            raise ValueError(f"Unknown algo={algo}")

class EnsembleModel:
    """
    نگه‌دارنده‌ی چند ModelPipeline و انجام vote / میانگین.
    """

    def __init__(
        self,
        models: list[ModelPipeline],
        names: Optional[list[str]] = None,
        vote_k: int = 3,
    ):
        if not models:
            raise ValueError("EnsembleModel needs at least one ModelPipeline")
        self.models = models
        self.vote_k = int(vote_k)
        if names is None:
            names = [getattr(m, "algo", f"model_{i}") for i, m in enumerate(models)]
        self.names = names

    def predict_proba(self, X):
        """
        احتمال نهایی = میانگین احتمال کلاس ۱ بین همه‌ی مدل‌ها.
        خروجی: (n_samples, 2)
        """
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        avg = np.mean(np.vstack(probs), axis=0)
        p1 = avg
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X) -> np.ndarray:
        """
        پیش‌بینی کلاس نهایی با رأی‌گیری ساده روی آستانه‌ی ۰.۵
        (برای داخل GA و بدون آستانه‌های اختصاصی).
        """
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        votes = (np.vstack(probs) >= 0.5).astype(int)  # (n_models, n_samples)
        avg_vote = votes.mean(axis=0)
        return (avg_vote >= 0.5).astype(int)

    def predict_actions(
        self,
        X,
        neg_thrs: list[float],
        pos_thrs: list[float],
        *,
        min_conf_models: int = 3,
        buy_ratio: float = 0.7,
        sell_ratio: float = 0.3,
    ) -> np.ndarray:
        """
        رأی‌گیری با آستانه‌ی جداگانه برای هر مدل (الهام‌گرفته از اسکریپت تست):

        - هر مدل:
            p <= neg_thr[i]  → SELL (0)
            p >= pos_thr[i]  → BUY  (1)
            در غیر این صورت → NONE (-1)

        - اگر حداقل min_conf_models مدل رأی داده باشند و
            نسبت BUYها >= buy_ratio  → سیگنال BUY
          اگر نسبت BUYها <= sell_ratio → سیگنال SELL
          وگرنه → NONE
        """
        n_models = len(self.models)
        if len(neg_thrs) != n_models or len(pos_thrs) != n_models:
            raise ValueError(
                f"len(neg_thrs) / len(pos_thrs) باید برابر تعداد مدل‌ها ({n_models}) باشد."
            )

        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        probs_arr = np.vstack(probs)  # (n_models, n_samples)
        n_samples = probs_arr.shape[1]

        votes = np.full((n_models, n_samples), -1, dtype=int)

        for i in range(n_models):
            p = probs_arr[i]
            v = votes[i]
            v[p <= float(neg_thrs[i])] = 0
            v[p >= float(pos_thrs[i])] = 1

        confident = (votes != -1).sum(axis=0).astype(int)
        buy_count = (votes == 1).sum(axis=0).astype(int)

        actions = np.full(n_samples, -1, dtype=int)

        safe_conf = confident.astype(float)
        safe_conf[safe_conf == 0] = np.nan
        vote_ratio = buy_count / safe_conf

        buy_cond = (confident >= min_conf_models) & (vote_ratio >= buy_ratio)
        sell_cond = (confident >= min_conf_models) & (vote_ratio <= sell_ratio)

        actions[buy_cond] = 1
        actions[sell_cond] = 0

        return actions

    def get_scaler(self):
        """
        برای DriftChecker: اسکیلر مدل اول (معمولاً logreg) را برمی‌گرداند.
        """
        if not self.models:
            return None
        m0 = self.models[0]
        if hasattr(m0, "get_scaler"):
            return m0.get_scaler()
        return None
