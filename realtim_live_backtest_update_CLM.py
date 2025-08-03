#!/usr/bin/env python3
"""
Fixed data preparation pipeline ensuring consistency between training and live environments
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import re

class ConsistentDataPrep:
    def __init__(self, main_timeframe="30T", max_window=52):
        self.main_timeframe = main_timeframe
        self.max_window = max_window  # Max window size for any indicator
        self.feature_state = {}  # Stores historical data for consistent feature calculation
        self.training_features = None  # Features used during training
        self.scaler = None  # Fitted scaler from training
        
    def prepare_training_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training with proper feature engineering"""
        # 1. Clean and sort data
        df = self._clean_data(raw_data)
        time_col = f"{self.main_timeframe}_time"
        df[time_col] = pd.to_datetime(df[time_col])
        df.sort_values(time_col, inplace=True)
        
        # 2. Engineer features with proper shifting
        df = self._engineer_features(df, is_training=True)
        
        # 3. Create target variable
        close_col = f"{self.main_timeframe}_close"
        y = (df[close_col].shift(-1) > df[close_col]).astype(int)
        df, y = df.iloc[:-1], y.iloc[:-1]  # Drop last row with no target
        
        # 4. Store feature state for live inference
        self._store_feature_state(df)
        
        # 5. Store training features
        self.training_features = df.columns.tolist()
        
        return df, y
    
    def prepare_live_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare live data using the same feature engineering as training"""
        # 1. Clean and sort new data
        df = self._clean_data(new_data)
        time_col = f"{self.main_timeframe}_time"
        df[time_col] = pd.to_datetime(df[time_col])
        df.sort_values(time_col, inplace=True)
        
        # 2. Combine with historical state
        combined_df = self._combine_with_state(df)
        
        # 3. Engineer features consistently
        combined_df = self._engineer_features(combined_df, is_training=False)
        
        # 4. Update state with new data
        self._update_feature_state(combined_df)
        
        # 5. Return only new data points with features
        new_features = combined_df.tail(len(df))
        
        # 6. Ensure same features as training
        if self.training_features:
            # Add missing features with NaN
            for feat in self.training_features:
                if feat not in new_features.columns:
                    new_features[feat] = np.nan
            
            # Select only training features
            new_features = new_features[self.training_features]
        
        return new_features
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.ffill().bfill()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Consistent feature engineering for both training and live"""
        # 1. Time-based features
        time_col = f"{self.main_timeframe}_time"
        df[f"{self.main_timeframe}_hour"] = df[time_col].dt.hour
        df[f"{self.main_timeframe}_day_of_week"] = df[time_col].dt.dayofweek
        df[f"{self.main_timeframe}_is_weekend"] = (df[f"{self.main_timeframe}_day_of_week"] >= 5).astype(int)
        
        # 2. Price-based features with proper shifting
        close_col = f"{self.main_timeframe}_close"
        df[f"{self.main_timeframe}_returns"] = df[close_col].pct_change()
        df[f"{self.main_timeframe}_log_returns"] = np.log(df[close_col] / df[close_col].shift(1))
        
        # 3. Rolling features (use consistent window)
        for window in [5, 10, 20]:
            df[f"{self.main_timeframe}_sma_{window}"] = df[close_col].rolling(window).mean().shift(1)
            df[f"{self.main_timeframe}_std_{window}"] = df[close_col].rolling(window).std().shift(1)
            df[f"{self.main_timeframe}_min_{window}"] = df[close_col].rolling(window).min().shift(1)
            df[f"{self.main_timeframe}_max_{window}"] = df[close_col].rolling(window).max().shift(1)
        
        # 4. Technical indicators (simplified example)
        df[f"{self.main_timeframe}_rsi"] = self._calculate_rsi(df[close_col]).shift(1)
        df[f"{self.main_timeframe}_macd"] = self._calculate_macd(df[close_col]).shift(1)
        
        # 5. Volume-based features
        vol_col = f"{self.main_timeframe}_volume"
        if vol_col in df.columns:
            df[f"{self.main_timeframe}_volume_sma"] = df[vol_col].rolling(20).mean().shift(1)
            df[f"{self.main_timeframe}_volume_ratio"] = (df[vol_col] / df[vol_col].rolling(20).mean()).shift(1)
        
        # 6. Drop rows with NaN from feature calculation
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI consistently"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD consistently"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _store_feature_state(self, df: pd.DataFrame):
        """Store historical data needed for consistent feature calculation"""
        # Keep last max_window rows for each timeframe
        for tf in ["5T", "15T", "30T", "1H"]:
            tf_cols = [col for col in df.columns if col.startswith(f"{tf}_")]
            if tf_cols:
                self.feature_state[tf] = df[tf_cols].tail(self.max_window).copy()
    
    def _combine_with_state(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Combine new data with historical state"""
        combined = []
        
        # Combine data for each timeframe
        for tf in ["5T", "15T", "30T", "1H"]:
            tf_cols = [col for col in new_data.columns if col.startswith(f"{tf}_")]
            if not tf_cols:
                continue
                
            # Get historical state for this timeframe
            state_data = self.feature_state.get(tf, pd.DataFrame())
            
            # Combine state with new data
            tf_combined = pd.concat([state_data, new_data[tf_cols]])
            combined.append(tf_combined)
        
        # Combine all timeframes
        if combined:
            result = pd.concat(combined, axis=1)
            # Remove duplicate columns
            result = result.loc[:, ~result.columns.duplicated()]
            return result
        
        return new_data
    
    def _update_feature_state(self, df: pd.DataFrame):
        """Update feature state with new data"""
        for tf in ["5T", "15T", "30T", "1H"]:
            tf_cols = [col for col in df.columns if col.startswith(f"{tf}_")]
            if tf_cols:
                # Keep last max_window rows
                self.feature_state[tf] = df[tf_cols].tail(self.max_window).copy()

# Usage example
if __name__ == "__main__":
    # Initialize with same parameters as training
    prep = ConsistentDataPrep(main_timeframe="30T", max_window=52)
    
    # During training
    # training_data = pd.read_csv("training_data.csv")
    # X_train, y_train = prep.prepare_training_data(training_data)
    
    # During live inference
    # live_data = pd.read_csv("live_data.csv")
    # X_live = prep.prepare_live_data(live_data)
    
    print("Consistent data preparation pipeline initialized")