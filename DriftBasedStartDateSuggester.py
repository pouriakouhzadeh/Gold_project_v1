import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import ks_2samp
from pathlib import Path
from typing import Optional

class DriftBasedStartDateSuggester:
    def __init__(self, filepaths: dict[str, str], max_acceptable_drift: float = 0.35,
                 min_days_required: int = 90, step_days: int = 30):
        self.filepaths = filepaths
        self.max_acceptable_drift = max_acceptable_drift
        self.min_days_required = min_days_required
        self.step_days = step_days
        self.shared_best_date: Optional[datetime] = None

    def _compute_drift(self, ref_data: pd.DataFrame, cur_data: pd.DataFrame) -> float:
        ref_values = ref_data.select_dtypes(include='number')
        cur_values = cur_data.select_dtypes(include='number')
        scores = []
        for col in ref_values.columns.intersection(cur_values.columns):
            stat, _ = ks_2samp(ref_values[col].dropna(), cur_values[col].dropna())
            scores.append(stat)
        return np.mean(scores) if scores else 1.0

    def _suggest_date_for_single_file(self, csv_path: str, timeframe: str) -> Optional[datetime]:
        try:
            df = pd.read_csv(csv_path, parse_dates=["time"])
            df.dropna(subset=["time"], inplace=True)
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)

            end_date = df["time"].max()
            start_date = df["time"].min()
            min_train_start = end_date - timedelta(days=self.min_days_required)

            candidate_dates = pd.date_range(start=start_date, end=min_train_start,
                                            freq=f"{self.step_days}D")[::-1]

            for candidate_date in candidate_dates:
                ref = df[df["time"] < candidate_date]
                cur = df[(df["time"] >= candidate_date) & (df["time"] <= end_date)]

                if len(cur) < 1000 or len(ref) < 1000:
                    continue

                drift_score = self._compute_drift(ref, cur)
                if drift_score < self.max_acceptable_drift:
                    print(f"[{timeframe}] ✅ Acceptable drift from: {candidate_date.date()} (score={drift_score:.3f})")
                    return candidate_date

            print(f"[{timeframe}] No acceptable date found.")
            return None

        except Exception as e:
            print(f"Error processing {timeframe}: {e}")
            return None

    def find_shared_start_date(self) -> Optional[datetime]:
        print("🔍 Finding shared start date across all timeframes based on drift...")
        best_dates = []

        for tf, path in self.filepaths.items():
            path_obj = Path(path)
            if not path_obj.exists():
                print(f"⚠️ File not found: {path}")
                continue

            best_date = self._suggest_date_for_single_file(str(path_obj), tf)
            if best_date:
                best_dates.append(best_date)

        if best_dates:
            # Choose the latest acceptable date among all
            self.shared_best_date = max(best_dates)
            print(f"\n📌 Final shared start date: {self.shared_best_date.date()}")
        else:
            print("No shared date could be determined.")

        return self.shared_best_date
