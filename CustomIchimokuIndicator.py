import pandas as pd

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
        """
        conversion_line = self.ichimoku_conversion_line()
        base_line = self.ichimoku_base_line()
        return ((conversion_line + base_line) / 2).shift(self.window2)

    def ichimoku_b(self):
        """
        Calculate the Leading Span B (Senkou Span B).
        """
        return (
            (self.high.rolling(window=self.window3).max() +
             self.low.rolling(window=self.window3).min()) / 2
        ).shift(self.window2)
