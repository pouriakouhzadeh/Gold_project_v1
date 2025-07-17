# feature_engineer.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import ta
from scipy.signal import lfilter

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_method: str = 'mean'):
        """
        یک کلاس مهندسی ویژگی که اندیکاتورهای فنی مختلف را روی داده اعمال می‌کند.

        پارامتر:
        - fill_method (str): روش جایگزینی مقدار NaN (مثلاً 'zero' یا 'mean').
          در صورت تمایل می‌توانید گام ایمپیوت را حذف کرده و در Pipeline از SimpleImputer استفاده کنید.
        """
        self.fill_method = fill_method
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        محاسبات مهندسی ویژگی را انجام می‌دهد و سپس مقادیر ∞ را به NaN تبدیل کرده
        و با توجه به استراتژی انتخاب‌شده، جایگزین می‌کند یا اجازه می‌دهد در Pipeline مدیریت شوند.
        """
        # کپی داده اصلی
        data = X.copy()

        # اگر ستون time داریم، حذفش می‌کنیم (یا نگه می‌داریم اگر لازم دارید)
        if 'time' in data.columns:
            data = data.drop(columns=['time'])

        # 1) VWAP
        # جلوگیری از تقسیم بر صفر برای volume.cumsum()
        cumsum_vol = data['volume'].cumsum()
        cumsum_vol = cumsum_vol.replace(0, np.nan)  # اگر جایی صفر بود، NaN شود
        data['vwap'] = ((data['close'] * data['volume']).cumsum()) / cumsum_vol

        # 2) OBV
        # np.sign(...) ممکن است NaN تولید نکند ولی اختلاف price اگر NaN باشد، باید مراقب باشیم
        data['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()

        # 3) MACD
        # از کتابخانه ta استفاده می‌کنیم تا داخل آن چک‌ها را انجام دهد
        data['MACD'] = ta.trend.macd(data['close'], fillna=False)
        data['MACD_Signal'] = ta.trend.macd_signal(data['close'], fillna=False)
        data['MACD_Diff'] = ta.trend.macd_diff(data['close'], fillna=False)

        # 4) Stochastic
        data['Stochastic_Oscillator'] = ta.momentum.stoch(
            data['high'], data['low'], data['close'], fillna=False
        )

        # 5) Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=data['close'], window=20, window_dev=2, fillna=False
        )
        data['Bollinger_High'] = bollinger.bollinger_hband()
        data['Bollinger_Low'] = bollinger.bollinger_lband()
        data['Bollinger_Middle'] = bollinger.bollinger_mavg()

        # 6) CCI
        data['CCI'] = ta.trend.cci(
            data['high'], data['low'], data['close'], window=20, fillna=False
        )

        # 7) Williams %R
        data['Williams_%R'] = ta.momentum.williams_r(
            data['high'], data['low'], data['close'], lbp=14, fillna=False
        )

        # 8) RSI
        data['RSI'] = ta.momentum.RSIIndicator(
            close=data['close'], window=14, fillna=False
        ).rsi()
        data['RSI_volume'] = ta.momentum.RSIIndicator(
            close=data['volume'], window=14, fillna=False
        ).rsi()

        # 9) Percentage Change
        shift_close = data['close'].shift(1).replace(0, np.nan)  # پیشگیری از تقسیم بر صفر
        data['Percentage_Change'] = (data['close'] - shift_close) / shift_close * 100

        # 10) ROC
        data['ROC'] = ta.momentum.ROCIndicator(
            close=data['close'], window=1, fillna=False
        ).roc()

        # 11) Return Difference
        data['ReturnDifference'] = data['close'].diff()

        # 12) Trading Volume Ratio
        rolling_vol_10 = data['volume'].rolling(window=10).mean().replace(0, np.nan)
        data['TradingVolumeRatio'] = data['volume'] / rolling_vol_10

        # 13) Price to Moving Average Ratio
        rolling_close_20 = data['close'].rolling(window=20).mean().replace(0, np.nan)
        data['PriceToMovingAverageRatio'] = data['close'] / rolling_close_20

        # 14) Candle heights
        data['candle_height'] = data['high'] - data['low']
        data['candle_body'] = data['close'] - data['open']
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']

        # 15) Moving Averages
        # با توجه به تعداد زیاد (3 تا 150) باید اوایل داده را هم در نظر گرفت
        for i in range(3, 150):
            ma_col = data['close'].rolling(window=i).mean()
            ma_vol_col = data['volume'].rolling(window=i).mean()
            data[f'ma{i}'] = ma_col
            data[f'ma_volume{i}'] = ma_vol_col

        # 16) فیلتر FIR
        N = 5  # طول فیلتر
        b = np.ones(N) / N
        a = 1
        data['Filtered'] = lfilter(b, a, data['close'])

        # 17) اضافه کردن همهٔ اندیکاتورهای کتابخانه ta (در صورت نیاز)
        all_indicators = ta.add_all_ta_features(
            data.copy(), "open", "high", "low", "close", "volume", fillna=False
        )
        data = pd.concat([data, all_indicators], axis=1)
        data = data.loc[:, ~data.columns.duplicated()]

        # 18) حذف ردیف‌های اولیه برای جلوگیری از NaN ناشی از rolling زیاد
        # (مثلاً اگر تا 150 رولینگ داریم، 150 سطر اول تا حد زیادی ممکن است ناقص باشند)
        data = data[151:].copy()

        # --- پیش از بازگشت، مدیریت ∞ و NaN ---
        # ابتدا ∞ را تبدیل به NaN می‌کنیم
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # بسته به راهبرد ایمپیوت (fill_method) مقادیر NaN را پر می‌کنیم:
        if self.fill_method == 'zero':
            data.fillna(0, inplace=True)
        elif self.fill_method == 'mean':
            data.fillna(data.mean(numeric_only=True), inplace=True)
            # اگر در بعضی ستون‌ها همه NaN باشد یا عددی وجود نداشته باشد، همچنان NaN می‌ماند
            # برای این حالت می‌توان "0" گذاشت:
            data.fillna(0, inplace=True)
        else:
            # مثلاً drop یا هر کار دیگر
            data.dropna(axis=0, inplace=True)

        # در این مرحله داده باید تا حد زیادی از NaN و ∞ خالی باشد.
        return data
