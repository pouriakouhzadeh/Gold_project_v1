"""Model pipeline with standard scaling and calibrated logistic regression

This replaces the previous implementation that used plain LogisticRegression.
The pipeline now calibrates predicted probabilities with Platt scaling (sigmoid)
via CalibratedClassifierCV, yielding better‑spread probability estimates.
"""

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

import logging
logging.getLogger("sklearnex").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("daal4py").setLevel(logging.WARNING)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from typing import Any, Dict
from sklearn.model_selection import TimeSeriesSplit
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time

class ModelPipeline:
    """Pipeline = StandardScaler ➜ Calibrated Logistic Regression.

    Parameters
    ----------
    hyperparams : dict
        Hyper‑parameters for the underlying LogisticRegression.
    """

    def __init__(
        self,
        hyperparams: Dict[str, Any] | None = None,
        calibrate: bool = True,
        calib_method: str = "sigmoid"      # ← NEW
    ) -> None:
        self.hyperparams  = hyperparams or {}
        self.calibrate    = calibrate
        self.calib_method = calib_method

        # base logistic regression (not yet calibrated)
        # استخراج همهٔ هایپرپارامترها
        C              = self.hyperparams.get('C', 1.0)
        max_iter       = self.hyperparams.get('max_iter', 300)
        tol            = self.hyperparams.get('tol', 3e-4)
        penalty        = self.hyperparams.get('penalty', 'l2')
        solver         = self.hyperparams.get('solver', 'lbfgs')
        fit_intercept  = self.hyperparams.get('fit_intercept', True)
        class_weight   = self.hyperparams.get('class_weight', None)
        multi_class    = self.hyperparams.get('multi_class', 'auto')   # ← اضافه شد

        # اگر ترکیبِ غیرمجاز باشد، به‌صورت ایمن به 'auto' برمی‌گردیم
        if solver == "liblinear" and multi_class == "multinomial":
            multi_class = "auto"

        multi_class = self.hyperparams.get('multi_class', 'auto')

        # اگر auto است پارامتر را حذف می‌کنیم تا به LogisticRegression فرستاده نشود
        extra_multi_params = {}
        if multi_class != "auto":          # فقط حالت‌های ovr / multinomial
            extra_multi_params["multi_class"] = multi_class

        base_lr = LogisticRegression(
            C=C,
            max_iter=max_iter,
            tol=tol,
            penalty=penalty,
            solver=solver,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            warm_start=True,               # نکتهٔ کاهش ConvergenceWarning
            random_state=42,
            n_jobs=-1,
            **extra_multi_params,          # ← این سطر اضافه شد
        )
        # calibrated classifier (Platt sigmoid, 5‑fold CV)
        calibrated_lr = CalibratedClassifierCV(
            estimator=base_lr,
            method=self.calib_method,          # ← از پارامتر ورودی می‌گیرد
            cv=TimeSeriesSplit(n_splits=5),
            n_jobs=-1
        )

        # full sklearn pipeline
        self.pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', calibrated_lr if self.calibrate else base_lr),
        ])


    # ------------------------------------------------------------------
    # proxy methods
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the pipeline and log training progress (safe for GA usage)."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting training with hyperparameters:\n%s", self.hyperparams)

        start_time = time.time()

        # Optional progress display for TimeSeriesSplit (does NOT affect training)
        try:
            classifier = self.pipeline.named_steps['classifier']
            if isinstance(classifier.cv, TimeSeriesSplit):
                splits = list(classifier.cv.split(X, y))
                # tqdm only runs here – it does NOT change the actual CV logic
                _ = list(tqdm(splits, desc="⏳ Preparing CV splits", leave=False))
                # Use the same splits for training – structurally identical
                classifier.cv = splits
            else:
                logger.info("CV splitter is not TimeSeriesSplit → skipping progress bar.")
        except Exception as e:
            logger.warning("Could not wrap CV for progress display: %s", e)

        self.pipeline.fit(X, y)

        duration = time.time() - start_time
        logger.info("✅ Training finished in %.2f seconds.", duration)
        return self


    def predict_proba(self, X):
        """Predict class probabilities (takes calibrated estimator)."""
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        """Predict class labels."""
        return self.pipeline.predict(X)

    @property
    def steps(self):
        """Expose pipeline steps (useful for slicing e.g. pipeline[:-1])."""
        return self.pipeline.steps

    def __getattr__(self, item):
        """Delegate unknown attributes to the underlying sklearn Pipeline."""
        return getattr(self.pipeline, item)
