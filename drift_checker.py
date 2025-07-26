import numpy as np
import pandas as pd
import json

class DriftChecker:
    """
    کلاسی برای محاسبه PSI (Population Stability Index)
    و مقایسه توزیع داده Train با داده جدید.
    """
    def __init__(self, quantile: bool = True, verbose: bool = False):
        self.train_bins = None
        self.train_dist = None
        self.quantile   = quantile
        self.verbose    = verbose


    def _compute_hist(self, col_data, bins=10, quantile=False):          # ★ quantile
        col_data = col_data.dropna()
        if col_data.empty:
            return None, None

        if quantile:                                                     # ★
            edges = np.quantile(col_data, np.linspace(0, 1, bins + 1))   # ★
        else:                                                            # ★
            edges = np.linspace(col_data.min(), col_data.max(), bins + 1)# ★

        counts, _ = np.histogram(col_data, bins=edges)
        total = counts.sum()
        if total == 0:
            return edges, [0] * bins
        freqs = counts / total
        return edges, freqs

    def fit_on_train(self, X_train: pd.DataFrame, bins=10, quantile=True):
        self.quantile = quantile
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        self.train_bins = {}
        self.train_dist = {}

        for col in numeric_cols:
            edges, freqs = self._compute_hist(X_train[col], bins=bins,quantile=quantile)
            if edges is not None:
                self.train_bins[col] = edges
                self.train_dist[col] = freqs.tolist()
        return self

    def psi_for_column(self, train_freqs, test_freqs):
        """
        فرمول PSI: sum( (p_i - q_i) * ln(p_i / q_i) ).
        اگر p_i یا q_i=0 باشد، مقدار را خیلی کوچک درنظر می‌گیریم.
        """
        psi_val = 0.0
        for p, q in zip(train_freqs, test_freqs):
            p = max(p, 1e-9)
            q = max(q, 1e-9)
            psi_val += (p - q) * np.log(p / q)
        return psi_val

    def compare_live(self, X_live: pd.DataFrame, bins=10) -> float:
        """
        برای هر ستون numeric، هیستوگرام جدید می‌گیرد و PSI را با توزیع Train می‌سنجد.
        خروجی یک عدد PSI کلی است (میانگین از همه ستون‌ها).
        """
        if self.train_bins is None or self.train_dist is None:
            print("[DriftChecker] Train distribution not found => call fit_on_train first.")
            return 0.0

        numeric_cols = X_live.select_dtypes(include=[np.number]).columns
        psi_list = []

        for col in numeric_cols:
            if col not in self.train_bins:
                continue  # در Train چنین ستونی نبود
            edges = self.train_bins[col]
            train_freqs = self.train_dist[col]

            # histogram داده لایو روی همان edges
            col_data = X_live[col].dropna()
            if col_data.empty:
                continue
            counts, _ = np.histogram(col_data, bins=edges)
            total = counts.sum()
            if total==0:
                continue
            test_freqs = (counts / total).tolist()

            psi_val = self.psi_for_column(train_freqs, test_freqs)
            psi_list.append(psi_val)

        if len(psi_list)==0:
            return 0.0

        if getattr(self, "verbose", False):
            worst = sorted(zip(numeric_cols, psi_list),
                        key=lambda x: x[1], reverse=True)[:10]
            for col, val in worst:
                print(f"[Drift] {col:35s} PSI={val:.3f}")

        return np.mean(psi_list) 

    def save_train_distribution(self, filepath="train_distribution.json"):
        """
        ذخیره histogram و فراوانی Train به شکل JSON
        """
        save_dict = {
            'train_bins': {},
            'train_dist': {}
        }
        if self.train_bins and self.train_dist:
            for col in self.train_bins:
                save_dict['train_bins'][col] = self.train_bins[col].tolist()
            for col in self.train_dist:
                save_dict['train_dist'][col] = self.train_dist[col]
            with open(filepath, 'w') as f:
                json.dump(save_dict, f)
        else:
            print("No train_bins or train_dist to save.")

    def load_train_distribution(self, filepath="train_distribution.json"):
        """
        لود histogram و فراوانی Train از فایل JSON
        """
        import os
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.train_bins = {}
            self.train_dist = {}
            for col in data['train_bins']:
                self.train_bins[col] = np.array(data['train_bins'][col])
            for col in data['train_dist']:
                self.train_dist[col] = data['train_dist'][col]
        else:
            print(f"No file found at {filepath}.")
