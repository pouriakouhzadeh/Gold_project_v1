import pandas as pd
import numpy as np

class KST_custom_Indicator:
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

        # پیش‌محاسبه‌ی چهار ROC و SMA مربوطه
        self._rocma1 = self._rocma(self.roc1, self.sma1)
        self._rocma2 = self._rocma(self.roc2, self.sma2)
        self._rocma3 = self._rocma(self.roc3, self.sma3)
        self._rocma4 = self._rocma(self.roc4, self.sma4)

        # محاسبه‌ی KST، سیگنال، و اختلافشان
        self._kst = self._calc_kst()
        self._kst_sig = self._calc_kst_signal()
        self._kst_diff = self._kst - self._kst_sig

        # پر کردن NaNها در صورت تمایل
        if self.fillna:
            self._kst.fillna(method='ffill', inplace=True)
            self._kst_sig.fillna(method='ffill', inplace=True)
            self._kst_diff.fillna(method='ffill', inplace=True)

    def _roc(self, series: pd.Series, period: int) -> pd.Series:
        """
        نرخ تغییر (Rate of Change) در بازه‌ی period، به درصد:
            ROC = ((Close - Close.shift(period)) / Close.shift(period)) * 100
        """
        shifted = series.shift(period)
        roc = (series - shifted) / (shifted.replace(0, np.nan)) * 100.0
        return roc

    def _sma(self, series: pd.Series, window: int) -> pd.Series:
        """
        میانگین متحرک ساده با پنجره‌ی window.
        """
        return series.rolling(window=window, min_periods=1).mean()

    def _rocma(self, roc_period: int, sma_period: int) -> pd.Series:
        """
        ابتدا ROC را در پنجره‌ی roc_period محاسبه می‌کند،
        سپس آن را در بازه‌ی sma_period هموار (Smooth) می‌کند.
        """
        roc_serie = self._roc(self.close, roc_period)
        return self._sma(roc_serie, sma_period)

    def _calc_kst(self) -> pd.Series:
        """
        محاسبه‌ی KST اصلی بر اساس جمع وزنی ROCها:
        KST = (ROCMA1) + 2*(ROCMA2) + 3*(ROCMA3) + 4*(ROCMA4)
        """
        return (
            self._rocma1
            + 2.0 * self._rocma2
            + 3.0 * self._rocma3
            + 4.0 * self._rocma4
        )

    def _calc_kst_signal(self) -> pd.Series:
        """
        سیگنال KST = میانگین متحرک ساده‌ی KST در پنجره‌ی signal
        """
        return self._sma(self._kst, self.signal)

    def kst(self) -> pd.Series:
        """
        خروجی KST اصلی.
        """
        return self._kst

    def kst_signal(self) -> pd.Series:
        """
        خط سیگنال KST.
        """
        return self._kst_sig

    def kst_diff(self) -> pd.Series:
        """
        اختلاف بین KST اصلی و خط سیگنال.
        """
        return self._kst_diff
