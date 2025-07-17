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
        doji = (body / range_) < threshold

        return doji.astype(int)
