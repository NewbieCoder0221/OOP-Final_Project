import pandas as pd
import numpy as np
from scipy import stats

class DataInspector:
    """
    A class for inspecting datasets and detecting common data issues.

    Attributes
    ----------
    _data : pd.DataFrame
        The dataset to inspect.
    _issues : dict
        A record of detected data problems.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataInspector with a given dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset to inspect.
        """
        self._data = dataframe.copy()
        self._issues = {}

    def inspect(self):
        """Analyze dataset for missing values, duplicates, and outliers."""
        self.detect_missing()
        self.detect_duplicates()
        self.detect_outliers()
        return self._issues

    def detect_missing(self):
        """Detect missing values in the dataset."""
        missing = self._data.isnull().sum()
        self._issues["missing_values"] = missing[missing > 0]
        return self._issues["missing_values"]

    def detect_duplicates(self):
        """Detect duplicate rows in the dataset."""
        duplicates = self._data.duplicated().sum()
        self._issues["duplicates"] = duplicates
        return duplicates

    def detect_outliers(self):
        """Detect outliers using Z-score method."""
        numeric_data = self._data.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(numeric_data))
        outliers = (z_scores > 3).sum().sum()
        self._issues["outliers"] = int(outliers)
        return outliers

    def get_summary(self):
        """Return a summary of all detected issues."""
        return self._issues

    def __repr__(self):
        return f"DataInspector(summary={len(self._issues)} issues detected)"
