# custom_indicators.py

import numpy as np
import pandas as pd

class CustomWilliamsRIndicator:
    def __init__(self, high, low, close, window=14):
        """
        Initialize the Williams %R Indicator.

        Parameters:
            high (pd.Series): High price data.
            low (pd.Series): Low price data.
            close (pd.Series): Close price data.
            window (int): Period for calculation (default 14).
        """
        self.high = high
        self.low = low
        self.close = close
        self.window = window

    def williams_r(self):
        """
        Calculate the Williams %R indicator.

        Returns:
            pd.Series: Williams %R values.
        """
        highest_high = self.high.rolling(window=self.window).max()
        lowest_low = self.low.rolling(window=self.window).min()
        williams_r = -100 * (highest_high - self.close) / (highest_high - lowest_low + 1e-9)
        return williams_r

class VortexCustomIndicator:
    """
    محاسبه اندیکاتور Vortex به‌صورت سفارشی، بدون وابستگی به کتابخانه ta.

    پارامترها:
    -----------
    high   : pd.Series
        سری مقادیر High
    low    : pd.Series
        سری مقادیر Low
    close  : pd.Series
        سری مقادیر Close
    window : int
        بازهٔ زمانی (دوره) برای Rolling
    fillna : bool
        اگر True باشد، مقادیر NaN در خروجی را با 0 پر می‌کند.

    متدها:
    ------
    vortex_pos() -> pd.Series
        برمی‌گرداند +VI (Vortex مثبت)
    vortex_neg() -> pd.Series
        برمی‌گرداند -VI (Vortex منفی)
    """

    def __init__(self, high, low, close, window=14, fillna=True):
        self.high = high.astype(float)
        self.low = low.astype(float)
        self.close = close.astype(float)
        self.window = window
        self.fillna = fillna

        # پیش‌محاسبه
        self._tr = self._true_range()
        self._vmp = (self.high - self.low.shift(1)).abs()
        self._vmm = (self.low - self.high.shift(1)).abs()

        self._sum_tr = self._tr.rolling(window=self.window, min_periods=1).sum()
        self._sum_vmp = self._vmp.rolling(window=self.window, min_periods=1).sum()
        self._sum_vmm = self._vmm.rolling(window=self.window, min_periods=1).sum()

        self._vip = self._sum_vmp / (self._sum_tr + 1e-9)
        self._vin = self._sum_vmm / (self._sum_tr + 1e-9)

        if self.fillna:
            self._vip = self._vip.fillna(0)
            self._vin = self._vin.fillna(0)

    def _true_range(self) -> pd.Series:
        """
        محاسبه‌ی True Range (TR) به شیوهٔ کلاسیک:
            TR = max(
              high(t) - low(t),
              |high(t) - close(t-1)|,
              |low(t) - close(t-1)|
            )
        """
        shift_close = self.close.shift(1)
        method1 = self.high - self.low
        method2 = (self.high - shift_close).abs()
        method3 = (self.low - shift_close).abs()
        tr = pd.DataFrame({"m1": method1, "m2": method2, "m3": method3}).max(axis=1)
        return tr

    def vortex_pos(self) -> pd.Series:
        """
        برگرداندن Vortex +VI
        """
        return self._vip

    def vortex_neg(self) -> pd.Series:
        """
        برگرداندن Vortex -VI
        """
        return self._vin

class CustomVolumeRateOfChangeIndicator:
    def __init__(self, volume, window=20):
        """
        Initialize the Volume Rate of Change (VROC) Indicator.

        Parameters:
            volume (pd.Series): Volume data.
            window (int): Period for calculation (default 20).
        """
        self.volume = volume
        self.window = window

    def volume_rate_of_change(self):
        """
        Calculate the Volume Rate of Change (VROC).

        Returns:
            pd.Series: VROC values.
        """
        vroc = (self.volume - self.volume.shift(self.window)) / (self.volume.shift(self.window) + 1e-9) * 100
        return vroc

class CustomPivotPointIndicator:
    def __init__(self, high, low, close, window=5):
        """
        Initialize Pivot Point Indicator.

        Parameters:
            high (pd.Series): High price data.
            low (pd.Series): Low price data.
            close (pd.Series): Close price data.
            window (int): Number of periods for calculation (default 5).
        """
        self.high = high
        self.low = low
        self.close = close
        self.window = window

    def pivot(self):
        """
        Calculate Pivot Point (PP).
        """
        return (self.high + self.low + self.close) / 3

    def support_1(self):
        """
        Calculate Support Level 1 (S1).
        """
        pivot = self.pivot()
        return (2 * pivot) - self.high

    def support_2(self):
        """
        Calculate Support Level 2 (S2).
        """
        pivot = self.pivot()
        return pivot - (self.high - self.low)

    def resistance_1(self):
        """
        Calculate Resistance Level 1 (R1).
        """
        pivot = self.pivot()
        return (2 * pivot) - self.low

    def resistance_2(self):
        """
        Calculate Resistance Level 2 (R2).
        """
        pivot = self.pivot()
        return pivot + (self.high - self.low)

class KSTCustomIndicator:
    """
    محاسبه‌ی KST (Know Sure Thing) بدون وابستگی به کتابخانه ta.

    فرمول کلی:
        KST = (ROCMA1) + 2*(ROCMA2) + 3*(ROCMA3) + 4*(ROCMA4)
    که در آن:
        ROCMAi = میانگین متحرکِ (Rate of Change در بازه roc_i)

    پارامترها:
    ----------
    close : pd.Series
        قیمت نهایی (Close) کندل‌ها.
    roc1, roc2, roc3, roc4 : int
        طول پنجره‌ی محاسبه‌ی ROC (نرخ تغییر) برای هر بخش.
    sma1, sma2, sma3, sma4 : int
        طول پنجره‌ی میانگین متحرک ساده برای صاف کردن (Smoothing) هر ROC.
    signal : int
        طول پنجره میانگین متحرک سیگنال (KST Signal).
    fillna : bool
        در صورت True بودن، مقادیر تهی را با Forward Fill پر می‌کند.
    """

    def __init__(
            self,
            close: pd.Series,
            roc1=10, roc2=15, roc3=20, roc4=30,
            sma1=10, sma2=10, sma3=10, sma4=15,
            signal=9,
            fillna=True
    ):
        self.close = close.astype(float)
        self.roc1 = roc1
        self.roc2 = roc2
        self.roc3 = roc3
        self.roc4 = roc4
        self.sma1 = sma1
        self.sma2 = sma2
        self.sma3 = sma3
        self.sma4 = sma4
        self.signal = signal
        self.fillna = fillna

        self._rocma1 = self._rocma(self.roc1, self.sma1)
        self._rocma2 = self._rocma(self.roc2, self.sma2)
        self._rocma3 = self._rocma(self.roc3, self.sma3)
        self._rocma4 = self._rocma(self.roc4, self.sma4)

        self._kst = self._calc_kst()
        self._kst_sig = self._calc_kst_signal()
        self._kst_diff = self._kst - self._kst_sig

        if self.fillna:
            self._kst.fillna(method='ffill', inplace=True)
            self._kst_sig.fillna(method='ffill', inplace=True)
            self._kst_diff.fillna(method='ffill', inplace=True)

    def _roc(self, series: pd.Series, period: int) -> pd.Series:
        shifted = series.shift(period)
        roc = (series - shifted) / (shifted.replace(0, np.nan) + 1e-9) * 100.0
        return roc

    def _sma(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    def _rocma(self, roc_period: int, sma_period: int) -> pd.Series:
        roc_serie = self._roc(self.close, roc_period)
        return self._sma(roc_serie, sma_period)

    def _calc_kst(self) -> pd.Series:
        return (
            self._rocma1
            + 2.0 * self._rocma2
            + 3.0 * self._rocma3
            + 4.0 * self._rocma4
        )

    def _calc_kst_signal(self) -> pd.Series:
        return self._sma(self._kst, self.signal)

    def kst(self) -> pd.Series:
        return self._kst

    def kst_signal(self) -> pd.Series:
        return self._kst_sig

    def kst_diff(self) -> pd.Series:
        return self._kst_diff

class CustomIchimokuIndicator:
    def __init__(self, high, low, close, window1=9, window2=26, window3=52):
        """
        Initialize Ichimoku Indicator with required data and parameters.

        Parameters:
            high (pd.Series): High price data.
            low (pd.Series): Low price data.
            close (pd.Series): Close price data (optional, used for compatibility).
            window1 (int): Conversion line period (default 9).
            window2 (int): Base line period (default 26).
            window3 (int): Leading span B period (default 52).
        """
        self.high = high
        self.low = low
        self.close = close
        self.window1 = window1
        self.window2 = window2
        self.window3 = window3

    def ichimoku_conversion_line(self):
        """
        Calculate the Conversion Line (Tenkan-sen).
        """
        return (
            self.high.rolling(window=self.window1).max() +
            self.low.rolling(window=self.window1).min()
        ) / 2

    def ichimoku_base_line(self):
        """
        Calculate the Base Line (Kijun-sen).
        """
        return (
            self.high.rolling(window=self.window2).max() +
            self.low.rolling(window=self.window2).min()
        ) / 2

    def ichimoku_a(self):
        """
        Calculate the Leading Span A (Senkou Span A).

        در ایچیموکوی استاندارد، این مقدار 26 کندل جلو می‌رود (shift(26)).
        ولی برای اجتناب از دیتالیک، شیفت را حذف کرده‌ایم.
        """
        conversion_line = self.ichimoku_conversion_line()
        base_line = self.ichimoku_base_line()
        # === Original code (shift):
        # return ((conversion_line + base_line) / 2).shift(self.window2)
        # === New code (no future shift):
        return (conversion_line + base_line) / 2

    def ichimoku_b(self):
        """
        Calculate the Leading Span B (Senkou Span B).

        در ایچیموکوی استاندارد، این مقدار هم 26 کندل جلو می‌رود.
        ولی برای اجتناب از دیتالیک، شیفت را حذف کرده‌ایم.
        """
        # === Original code (shift):
        # return (
        #     (self.high.rolling(window=self.window3).max() +
        #      self.low.rolling(window=self.window3).min()) / 2
        # ).shift(self.window2)

        # === New code (no future shift):
        return (
            self.high.rolling(window=self.window3).max() +
            self.low.rolling(window=self.window3).min()
        ) / 2

class CustomCandlestickPattern:
    def __init__(self, open_, high, low, close):
        """
        Initialize Candlestick Pattern Detector.

        Parameters:
            open_ (pd.Series): Open price data.
            high (pd.Series): High price data.
            low (pd.Series): Low price data.
            close (pd.Series): Close price data.
        """
        self.open = open_
        self.high = high
        self.low = low
        self.close = close

    def engulfing(self):
        """
        Detect Engulfing candlestick pattern.

        Returns:
            pd.Series: 1 for bullish engulfing, -1 for bearish engulfing, 0 otherwise.
        """
        previous_close = self.close.shift(1)
        previous_open = self.open.shift(1)

        bullish = (self.close > self.open) & (self.open < previous_close) & (self.close > previous_open)
        bearish = (self.close < self.open) & (self.open > previous_close) & (self.close < previous_open)

        return bullish.astype(int) - bearish.astype(int)

    def doji(self, threshold=0.1):
        """
        Detect Doji candlestick pattern.

        Parameters:
            threshold (float): Percentage difference to consider a candle as a Doji.

        Returns:
            pd.Series: 1 if a Doji pattern is detected, 0 otherwise.
        """
        body = abs(self.close - self.open)
        range_ = self.high - self.low
        doji = (body / (range_ + 1e-9)) < threshold

        return doji.astype(int)
