# threshold_finder.py

import numpy as np
from sklearn.metrics import f1_score

class ThresholdFinder:
    def __init__(self, steps=600, min_predictions_ratio=0.90):
        """
        Parameters:
        -----------
        steps : int
            تعداد گام‌هایی که برای جستجوی آستانه‌ها طی می‌شود.
        min_predictions_ratio : float
            حداقل نسبت نمونه‌هایی که باید با اطمینان طبقه‌بندی شوند.
        """
        self.steps = steps
        self.min_predictions_ratio = min_predictions_ratio

    def find_best_thresholds(self, y_proba, y_true):
        """
        یافتن بهترین آستانه‌های مثبت و منفی بر اساس F1-score.

        Parameters:
        -----------
        y_proba : array-like
            احتمالات پیش‌بینی شده برای کلاس مثبت.
        y_true : array-like
            برچسب‌های واقعی.

        Returns:
        --------
        neg_threshold : float
            آستانه منفی.
        pos_threshold : float
            آستانه مثبت.
        acc : float
            دقت مدل با استفاده از این آستانه‌ها.
        w1, l1 : int
            تعداد نمونه‌های صحیح و نادرست پیش‌بینی شده.
        """
        best_f1 = -1
        best_neg = 0.0
        best_pos = 1.0
        best_acc = 0.0
        best_w1 = best_l1 = 0

        thresholds = np.linspace(0, 1, self.steps)
        for neg in thresholds:
            for pos in thresholds:
                if neg >= pos:
                    continue  # اطمینان از اینکه آستانه منفی کمتر از آستانه مثبت است
                y_pred = np.full_like(y_true, -1)  # -1 نشان‌دهنده نامطمئن بودن
                y_pred[y_proba <= neg] = 0
                y_pred[y_proba >= pos] = 1

                # محاسبه نسبت پیش‌بینی‌های مطمئن
                confident_ratio = np.sum(y_pred != -1) / len(y_true)
                if confident_ratio < self.min_predictions_ratio:
                    continue

                # محاسبه F1-score برای نمونه‌های مطمئن
                mask = y_pred != -1
                if np.sum(mask) == 0:
                    continue
                f1 = f1_score(y_true[mask], y_pred[mask], average='binary')

                if f1 > best_f1:
                    best_f1 = f1
                    best_neg = neg
                    best_pos = pos
                    # محاسبه دقت
                    acc = np.mean(y_pred[mask] == y_true[mask])
                    best_acc = acc
                    best_w1 = np.sum((y_pred == 1) & (y_true == 1))
                    best_l1 = np.sum((y_pred == 0) & (y_true == 1))

        return best_neg, best_pos, best_acc, best_w1, best_l1
