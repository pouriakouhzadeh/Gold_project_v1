import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from the file."""
        self.data = pd.read_csv(self.file_path , parse_dates=['time'])

    def remove_noise(self):
        """Removes noise and invalid values from the data."""
        # Replace inf and -inf with NaN
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop duplicate rows
        self.data.drop_duplicates(inplace=True)

    def fix_missing_values(self):
        """Fills or removes missing values in the data."""
        # Fill numerical columns with the mean
        for col in self.data.select_dtypes(include=['float64', 'int64']).columns:
            self.data[col].fillna(self.data[col].mean(), inplace=True)

        # Fill categorical columns with the mode
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

    def clean_data(self):
        """Executes the entire cleaning process."""
        self.load_data()
        self.remove_noise()
        self.fix_missing_values()
        return self.data

