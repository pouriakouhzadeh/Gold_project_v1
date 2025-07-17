# diagnostic_live_predictor.py
import time
import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.exceptions import NotFittedError
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from drift_checker import DriftChecker
from dynamic_threshold_adjuster import DynamicThresholdAdjuster

# Setup diagnostic logging
LOG_FILE = "logs/live_diagnostics.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DiagnosticLivePredictor:
    def __init__(self):
        saved_data = joblib.load("best_model.pkl")
        self.pipeline = saved_data['pipeline']
        self.neg_thr = saved_data['neg_thr']
        self.pos_thr = saved_data['pos_thr']
        self.scaler = saved_data['scaler']
        self.window_size = saved_data['window_size']
        self.feats = saved_data.get('feats', None)
        self.train_window_cols = saved_data.get('train_window_cols', [])
        self.train_raw_window = saved_data.get('train_raw_window', None)

        self.filepaths = {
            '30T': 'XAUUSD.F_M30_live.csv',
            '1H':  'XAUUSD.F_H1_live.csv',
            '15T': 'XAUUSD.F_M15_live.csv',
            '5T':  'XAUUSD.F_M5_live.csv'
        }

        self.prep = PREPARE_DATA_FOR_TRAIN(filepaths=self.filepaths, main_timeframe='30T')
        self.drift_checker = DriftChecker()
        self.drift_checker.load_train_distribution("train_distribution.json")

        self.threshold_adjuster = DynamicThresholdAdjuster(atr_high=2.0, vol_low=500, shift=0.05)
        self.answer_file = "Answer.txt"

    def predict(self):
        try:
            for tf in self.filepaths:
                if not os.path.exists(self.filepaths[tf]):
                    logging.warning(f"Missing file: {self.filepaths[tf]}")
                    return

            merged_df = self.prep.load_data()
            psi_val = self.drift_checker.compare_live(merged_df)

            needed_rows = self.window_size + 1
            if self.train_raw_window is not None:
                combined_window = pd.concat([self.train_raw_window, merged_df], ignore_index=True).tail(needed_rows)
            else:
                combined_window = merged_df.tail(needed_rows)

            # Retrieve last 10 rows for deeper inspection
            X_live, _ = self.prep.ready_incremental(combined_window, window=self.window_size, selected_features=self.feats)

            if X_live.empty:
                logging.warning("X_live is empty after preparation.")
                return

            logging.info(f"X_live.columns: {list(X_live.columns)}")

            # Log last 10 rows of features
            logging.debug("Last 10 rows of X_live:")
            logging.debug(X_live.tail(10).to_string())

            # Log critical indicators from raw merged_df
            indicator_names = ['30T_RSI_14', '30T_MACD', '30T_ATR_14', '30T_Bollinger_Width', '30T_Vortex_Pos', '30T_Vortex_Neg']
            last_ind_vals = {col: merged_df[col].iloc[-1] if col in merged_df.columns else "MISSING" for col in indicator_names}
            logging.info(f"Last indicators: {last_ind_vals}")

            last_atr = merged_df['30T_ATR_14'].iloc[-1] if '30T_ATR_14' in merged_df.columns else 1.0
            last_vol = merged_df['30T_volume'].iloc[-1] if '30T_volume' in merged_df.columns else 1000.0

            d_neg, d_pos = self.threshold_adjuster.adjust(self.neg_thr, self.pos_thr, last_atr, last_vol)
            logging.info(f"Dynamic Thresholds => NEG: {d_neg:.4f}, POS: {d_pos:.4f}, PSI: {psi_val:.4f}")

            X_live.columns = [str(c) for c in X_live.columns]
            X_live = X_live.reindex(columns=self.train_window_cols, fill_value=0).astype(float)
            X_input = X_live.to_numpy()[-1].reshape(1, -1)

            if self.scaler is not None:
                X_input = self.scaler.transform(X_input)
            logging.debug(f"Normalized input vector: {X_input.tolist()}")

            proba = self.pipeline.predict_proba(X_input)[:, 1][0]
            if proba <= d_neg:
                pred = "SEL"
            elif proba >= d_pos:
                pred = "BUY"
            else:
                pred = "NAN"

            logging.info(f"Prediction => {pred}, proba={proba:.4f}")

            with open(self.answer_file, 'w') as f:
                f.write(f"{pred},{proba:.4f}\n")

        except Exception as e:
            logging.exception(f"[Predict] Exception: {e}")

if __name__ == "__main__":
    predictor = DiagnosticLivePredictor()
    while True:
        predictor.predict()
        time.sleep(1)
        if os.path.exists("XAUUSD30.acn"):
            os.remove("XAUUSD30.acn")
        if os.path.exists("Answer.txt"):
            os.remove("Answer.txt")
