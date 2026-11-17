"""
Preprocessing pipeline for tabular sensor data (N, P, K, temperature, humidity,
moisture, pH). Handles missing values, outlier removal, scaling, and simple
feature engineering. Used both during training and at inference time so the
same transformations are applied consistently.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)

NUMERIC_COLUMNS_DEFAULT = ["N", "P", "K", "temperature", "humidity", "moisture", "ph"]


class SensorPreprocessor:
    """
    Fit/transform style preprocessor for sensor readings, mirroring the
    scikit-learn API so it composes cleanly with model pipelines.
    """

    def __init__(self, numeric_columns: Optional[List[str]] = None):
        self.numeric_columns = numeric_columns or NUMERIC_COLUMNS_DEFAULT
        self.scaler = StandardScaler()
        self._column_medians: dict = {}
        self._is_fitted = False

    # ---- missing values -------------------------------------------------

    def handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            if fit:
                median_val = df[col].median()
                self._column_medians[col] = median_val
            fill_val = self._column_medians.get(col, df[col].median())
            df[col] = df[col].fillna(fill_val)
        return df

    # ---- outliers ---------------------------------------------------------

    def remove_outliers_iqr(self, df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
        """Remove rows with values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
        df = df.copy()
        mask = pd.Series(True, index=df.index)
        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            mask &= df[col].between(lower, upper)
        removed = (~mask).sum()
        if removed:
            logger.info(f"Removed {removed} outlier rows from sensor dataset.")
        return df[mask].reset_index(drop=True)

    # ---- feature engineering ----------------------------------------------

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if {"N", "P", "K"}.issubset(df.columns):
            df["npk_sum"] = df["N"] + df["P"] + df["K"]
            df["n_p_ratio"] = df["N"] / df["P"].replace(0, np.nan)
            df["n_p_ratio"] = df["n_p_ratio"].fillna(0)
        return df

    # ---- scaling ------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = self.handle_missing_values(df, fit=True)
        df = self.remove_outliers_iqr(df)
        df = self.engineer_features(df)

        present_cols = [c for c in self.numeric_columns if c in df.columns]
        df[present_cols] = self.scaler.fit_transform(df[present_cols])
        self._is_fitted = True
        logger.info(f"Sensor preprocessor fitted on {len(df)} rows.")
        return df, present_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("SensorPreprocessor must be fit before calling transform().")
        df = self.handle_missing_values(df, fit=False)
        df = self.engineer_features(df)
        present_cols = [c for c in self.numeric_columns if c in df.columns]
        df[present_cols] = self.scaler.transform(df[present_cols])
        return df
