import numpy as np
import pandas as pd

class Vortex_custom_Indicator:
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
        self._vmp = (self.high - self.low.shift(1)).abs()  # یا high(t) - low(t-1)
        self._vmm = (self.low - self.high.shift(1)).abs()  # یا low(t) - high(t-1)

        # محاسبه‌ی rolling sum
        self._sum_tr = self._tr.rolling(window=self.window, min_periods=1).sum()
        self._sum_vmp = self._vmp.rolling(window=self.window, min_periods=1).sum()
        self._sum_vmm = self._vmm.rolling(window=self.window, min_periods=1).sum()

        # محاسبه‌ی Vortex مثبت و منفی
        self._vip = self._sum_vmp / (self._sum_tr + 1e-9)
        self._vin = self._sum_vmm / (self._sum_tr + 1e-9)

        # در صورت نیاز، پر کردن NaN
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
