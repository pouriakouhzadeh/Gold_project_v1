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
