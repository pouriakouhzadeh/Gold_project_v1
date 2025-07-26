#!/usr/bin/env python3
import time
import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.exceptions import NotFittedError

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from drift_checker import DriftChecker
from dynamic_threshold_adjuster import DynamicThresholdAdjuster  # Three-state threshold adjuster

logging.basicConfig(
    filename='production.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def remove_files_if_exists():
    try:
        if os.path.exists("XAUUSD30.acn"):
            os.remove("XAUUSD30.acn")
        if os.path.exists("Answer.txt"):
            os.remove("Answer.txt")
        print("Answer.txt and XAUUSD30.acn deleted")
    except:
        pass

class LivePredictor:
    def __init__(self):
        saved_data = joblib.load("best_model.pkl")
        self.pipeline         = saved_data['pipeline']
        self.neg_thr          = saved_data['neg_thr']
        self.pos_thr          = saved_data['pos_thr']
        self.scaler           = saved_data['scaler']
        self.window_size      = saved_data['window_size']
        self.feats            = saved_data.get('feats', None)
        self.train_window_cols = saved_data.get('train_window_cols', [])
        self.train_raw_window  = saved_data.get('train_raw_window', None)

        if self.window_size > 1 and self.train_raw_window is None:
            raise ValueError("Missing train_raw_window in model file.")

        self.filepaths = {
            '30T': 'XAUUSD.F_M30_live.csv',
            '1H':  'XAUUSD.F_H1_live.csv',
            '15T': 'XAUUSD.F_M15_live.csv',
            '5T':  'XAUUSD.F_M5_live.csv'
        }

        self.prep = PREPARE_DATA_FOR_TRAIN(filepaths=self.filepaths, main_timeframe='30T')
        self.answer_file = "Answer.txt"
        self.drift_checker = DriftChecker()
        self.drift_checker.load_train_distribution("train_distribution.json")

        self.threshold_adjuster = DynamicThresholdAdjuster(
            atr_high=2.0,
            vol_low=500,
            shift=0.05
        )

    def check_and_predict(self):
        for tf in self.filepaths:
            if not os.path.exists(self.filepaths[tf]):
                print(f"[Live] File {self.filepaths[tf]} not found. Waiting...")
                return

        try:
            merged_df = self.prep.load_data()
            psi_val = self.drift_checker.compare_live(merged_df)
            logging.info(f"[Live] PSI on new data = {psi_val:.4f}")
            if psi_val > 0.05:
                logging.info("[Live] Data drift detected (>5%). Retraining may be required.")

            needed_rows = self.window_size + 1
            if self.train_raw_window is not None:
                combined_window = pd.concat([self.train_raw_window, merged_df], ignore_index=True).tail(needed_rows)
            else:
                combined_window = merged_df.tail(needed_rows)

            if self.window_size == 1:
                X_live, _, _ = self.prep.ready(combined_window, window=1, selected_features=self.feats, mode='predict')
            else:
                X_live, _, _ = self.prep.ready_incremental(combined_window, window=self.window_size, selected_features=self.feats)

            if X_live.empty:
                print("[Live] X_live is empty. Skipping prediction.")
                logging.info("[Live] X_live is empty. Skipping prediction.")
                self.clean_files()
                return

            last_row = X_live.iloc[-1]
            if last_row.isna().any():
                print("[Live] ⚠️ Last row contains NaN values. Skipping prediction.")
                logging.warning("[Live] Last row contains NaN values. Skipping prediction.")
                self.clean_files()
                return

            if not self._check_timeframes_alignment(merged_df):
                print("[Live] ⚠️ Timeframes are not aligned. Skipping prediction.")
                logging.warning("[Live] Timeframes are not aligned. Skipping prediction.")
                self.clean_files()
                return

            last_atr = merged_df['30T_ATR_14'].iloc[-1] if '30T_ATR_14' in merged_df.columns else 1.0
            last_vol = merged_df['30T_volume'].iloc[-1] if '30T_volume' in merged_df.columns else 1000.0
            dynamic_neg_thr, dynamic_pos_thr = self.threshold_adjuster.adjust(self.neg_thr, self.pos_thr, last_atr, last_vol)

            X_live.columns = [str(c) for c in X_live.columns]
            X_live = X_live.reindex(columns=self.train_window_cols, fill_value=0).astype(float)
            X_input = X_live.to_numpy()[-1].reshape(1, -1)

            try:
                if self.scaler is not None:
                    X_input = self.scaler.transform(X_input)
                proba_array = self.pipeline.predict_proba(X_input)
                proba = proba_array[:, 1][0]
            except Exception as e:
                logging.error(f"[Live] Prediction failed => {e}")
                print(f"[Live] Prediction failed => {e}")
                self.clean_files()
                return

            if proba <= dynamic_neg_thr:
                pred = 0  # SELL
            elif proba >= dynamic_pos_thr:
                pred = 1  # BUY
            else:
                pred = -1  # uncertain

            txt_line = f"NAN,{proba:.4f}" if pred == -1 else ("BUY," if pred == 1 else "SEL,") + f"{proba:.4f}"
            with open(self.answer_file, 'w') as f:
                f.write(txt_line + "\n")
            logging.info(f"[Live] Prediction => {txt_line}")
            print(f"[Live] Prediction => {txt_line}")
            self.clean_files()

        except Exception as e:
            logging.error(f"[Live] General Exception => {e}")
            print(f"[Live] General Exception => {e}")

    def _check_timeframes_alignment(self, merged_df: pd.DataFrame) -> bool:
        if '30T_time' not in merged_df:
            return True
        last_30 = pd.to_datetime(merged_df['30T_time'].dropna().iloc[-1])

        def is_misaligned(tf_name, tolerance_minutes=0):
            col = merged_df.get(f"{tf_name}_time")
            if col is None or col.dropna().empty:
                return True
            last = pd.to_datetime(col.dropna().iloc[-1])
            if tolerance_minutes:
                return last < last_30 - pd.Timedelta(minutes=tolerance_minutes)
            return last != last_30

        return not (is_misaligned("1H", 30) or is_misaligned("15T") or is_misaligned("5T"))

    def clean_files(self):
        for tf in self.filepaths:
            if os.path.exists(self.filepaths[tf]):
                os.remove(self.filepaths[tf])

def main_loop():
    predictor = LivePredictor()
    while True:
        predictor.check_and_predict()
        time.sleep(1)
        remove_files_if_exists()

if __name__ == "__main__":
    main_loop()
