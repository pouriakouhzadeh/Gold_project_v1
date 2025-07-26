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
        vroc = (self.volume - self.volume.shift(self.window)) / self.volume.shift(self.window) * 100
        return vroc
