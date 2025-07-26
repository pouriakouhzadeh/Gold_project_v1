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
        williams_r = -100 * (highest_high - self.close) / (highest_high - lowest_low)
        return williams_r
