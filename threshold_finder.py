# threshold_finder.py

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score


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
        یافتن بهترین آستانه‌های مثبت و منفی با درنظرگرفتن:
        - F1 کلاس مثبت
        - Balanced Accuracy
        - داشتن بازه NONE کافی (pos - neg >= 0.05)
        - جلوگیری از نسبت‌های بسیار نامتوازن BUY (بین 20% و 80%)
        """
        best_score = -1.0
        best_f1 = 0.0
        best_neg = 0.0
        best_pos = 1.0
        best_acc = 0.0
        best_w1 = best_l1 = 0

        thresholds = np.linspace(0, 1, self.steps)
        for neg in thresholds:
            for pos in thresholds:
                if neg >= pos:
                    continue  # آستانه منفی باید کمتر از مثبت باشد

                # --- PATCH: حداقل فاصله بین دو آستانه برای داشتن ناحیه NONE ---
                if (pos - neg) < 0.05:
                    continue

                y_pred = np.full_like(y_true, -1)  # -1 = نامطمئن (NONE)
                y_pred[y_proba <= neg] = 0
                y_pred[y_proba >= pos] = 1

                # نسبت نمونه‌های دارای پیش‌بینی مطمئن
                mask = (y_pred != -1)
                confident_ratio = np.mean(mask)
                if confident_ratio < self.min_predictions_ratio or np.sum(mask) == 0:
                    continue

                # هر دو کلاس باید در نمونه‌های مطمئن حضور داشته باشند
                if np.unique(y_true[mask]).size < 2:
                    continue

                # --- PATCH: جلوگیری از BUY-only بودن ---
                pos_ratio = np.mean(y_pred[mask] == 1)
                if (pos_ratio < 0.20) or (pos_ratio > 0.80):
                    # خیلی کم یا خیلی زیاد BUY → این ترکیب آستانه را رد کن
                    continue

                # معیارها
                f1 = f1_score(y_true[mask], y_pred[mask], average="binary")
                bal_acc = balanced_accuracy_score(y_true[mask], y_pred[mask])

                # تابع هدف: ترکیب F1 و Balanced Accuracy
                score = 0.5 * f1 + 0.5 * bal_acc

                if score > best_score:
                    best_score = score
                    best_f1 = f1
                    best_neg = neg
                    best_pos = pos
                    best_acc = bal_acc  # خروجی acc اکنون = BalAcc است
                    best_w1 = int(np.sum((y_pred == 1) & (y_true == 1)))
                    best_l1 = int(np.sum((y_pred == 0) & (y_true == 1)))

        return best_neg, best_pos, best_acc, best_w1, best_l1
