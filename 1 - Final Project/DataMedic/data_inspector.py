"""
data_inspector.py
Handles dataset inspection: missing values, duplicates, and outliers.
"""

import pandas as pd

class DataInspector:
    """
    Scans a dataset and identifies common data quality issues.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the inspector with a pandas DataFrame.
        """
        self._data = dataframe
        self._issues = {}

    def inspect(self):
        """
        Runs all detection methods and stores results in _issues.
        """
        self._issues['missing'] = self.detect_missing()
        self._issues['duplicates'] = self.detect_duplicates()
        self._issues['outliers'] = self.detect_outliers()
        return self._issues

    def detect_missing(self):
        """
        Detect missing values in the dataset.
        """
        return self._data.isnull().sum()

    def detect_duplicates(self):
        """
        Detect duplicated rows.
        """
        return self._data.duplicated().sum()

    def detect_outliers(self):
        """
        Basic IQR outlier detection for numeric columns.
        """
        outliers = {}
        numeric_cols = self._data.select_dtypes(include="number").columns

        for col in numeric_cols:
            Q1 = self._data[col].quantile(0.25)
            Q3 = self._data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            count = self._data[(self._data[col] < lower) | (self._data[col] > upper)].shape[0]
            outliers[col] = count

        return outliers

    def get_summary(self):
        """
        Returns the collected issues.
        """
        return self._issues
