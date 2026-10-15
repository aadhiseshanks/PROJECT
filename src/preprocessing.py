"""
Preprocessing module for clinical diabetes data.
Implements custom Scikit-Learn transformers to handle clinical zeros as missing values,
robust median imputation, feature scaling, and stratified splitting to guarantee zero data leakage.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.data_loader import ZERO_AS_MISSING_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


class ZeroToNanTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that replaces zero values with NaN
    for designated clinical features where zero is biologically implausible
    (e.g., Glucose, BloodPressure, SkinThickness, Insulin, BMI).
    Works on both pandas DataFrames and numpy ndarrays.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns if columns is not None else ZERO_AS_MISSING_COLUMNS
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            for col in self.columns:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].replace(0, np.nan)
            return X_copy
        elif isinstance(X, np.ndarray):
            X_copy = np.array(X, dtype=float, copy=True)
            # If feature names are known, map columns to indices
            if self.feature_names_in_ is not None:
                for col in self.columns:
                    if col in self.feature_names_in_:
                        idx = self.feature_names_in_.index(col)
                        X_copy[X_copy[:, idx] == 0, idx] = np.nan
            else:
                # Fallback: assume default FEATURE_COLUMNS order
                for col in self.columns:
                    if col in FEATURE_COLUMNS:
                        idx = FEATURE_COLUMNS.index(col)
                        X_copy[X_copy[:, idx] == 0, idx] = np.nan
            return X_copy
        else:
            raise TypeError(f"Unsupported data type for X: {type(X)}")

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_, dtype=object)
        return np.array(FEATURE_COLUMNS, dtype=object)


def create_preprocessor_pipeline(
    scaler_type: str = "robust",
    imputer_strategy: str = "median"
) -> Pipeline:
    """
    Build Scikit-learn Pipeline for end-to-end preprocessing:
    1. ZeroToNanTransformer: replaces invalid clinical zeros with NaN.
    2. SimpleImputer: imputes missing values using median (robust to skewed insulin/skin thickness).
    3. Scaler: scales features using RobustScaler (default) or StandardScaler.
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_type: '{scaler_type}'. Choose 'robust' or 'standard'.")

    pipeline = Pipeline(steps=[
        ("zero_to_nan", ZeroToNanTransformer(columns=ZERO_AS_MISSING_COLUMNS)),
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", scaler)
    ])
    return pipeline


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to preserve target class proportions
    and avoid data leakage.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compute Interquartile Range (IQR) outlier boundaries for exploratory analysis.
    Does not delete observations, respecting clinical context where extreme values
    (e.g., morbid obesity or high insulin) are medically genuine.
    """
    if columns is None:
        columns = [c for c in FEATURE_COLUMNS if c in df.columns]

    outlier_report = {}
    for col in columns:
        # Filter valid non-zero values if zero represents missing data
        data = df[col]
        if col in ZERO_AS_MISSING_COLUMNS:
            data = data[data > 0]

        q25 = float(np.percentile(data.dropna(), 25))
        q75 = float(np.percentile(data.dropna(), 75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_report[col] = {
            "q25": round(q25, 2),
            "median": round(float(data.median()), 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(outliers),
            "outlier_percentage": round((len(outliers) / len(df)) * 100, 2)
        }
    return outlier_report
