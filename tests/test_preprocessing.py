"""
Unit tests for data loader, zero-to-NaN transformer, and preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset, validate_dataset, ZERO_AS_MISSING_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import ZeroToNanTransformer, create_preprocessor_pipeline, split_data


def test_dataset_loading_and_columns():
    """Verify that dataset loads properly and contains all required features."""
    df = load_dataset("diabetes.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert TARGET_COLUMN in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_dataset_validation_metrics():
    """Verify validate_dataset outputs correct statistics and zero counts."""
    df = load_dataset("diabetes.csv")
    val_report = validate_dataset(df)
    assert val_report["total_rows"] == 768
    assert val_report["total_features"] == 8
    assert "zero_statistics" in val_report
    assert val_report["zero_statistics"]["Glucose"]["is_missing_flag"] is True
    assert val_report["zero_statistics"]["Pregnancies"]["is_missing_flag"] is False


def test_zero_to_nan_transformer():
    """Verify that clinical zeros are converted to NaN while legitimate zeros are preserved."""
    test_data = pd.DataFrame({
        "Pregnancies": [0, 2],
        "Glucose": [0, 120],
        "BloodPressure": [70, 0],
        "SkinThickness": [0, 30],
        "Insulin": [0, 100],
        "BMI": [0.0, 28.5],
        "DiabetesPedigreeFunction": [0.5, 0.4],
        "Age": [25, 45]
    })

    transformer = ZeroToNanTransformer(columns=ZERO_AS_MISSING_COLUMNS)
    transformer.fit(test_data)
    transformed = transformer.transform(test_data)

    # Pregnancies zero should remain 0
    assert transformed["Pregnancies"].iloc[0] == 0

    # Glucose, BloodPressure, SkinThickness, Insulin, BMI zeros must be converted to NaN
    assert np.isnan(transformed["Glucose"].iloc[0])
    assert transformed["Glucose"].iloc[1] == 120
    assert np.isnan(transformed["BloodPressure"].iloc[1])
    assert np.isnan(transformed["SkinThickness"].iloc[0])
    assert np.isnan(transformed["Insulin"].iloc[0])
    assert np.isnan(transformed["BMI"].iloc[0])


def test_preprocessor_pipeline_no_nans():
    """Verify that full preprocessor pipeline transforms raw data with zeros into imputed, scaled matrix."""
    df = load_dataset("diabetes.csv")
    X = df.drop(columns=[TARGET_COLUMN])

    preprocessor = create_preprocessor_pipeline(scaler_type="robust", imputer_strategy="median")
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)

    assert not np.isnan(X_trans).any(), "Transformed output must not contain NaNs"
    assert X_trans.shape == X.shape


def test_stratified_split_preserves_class_ratio():
    """Verify stratified split maintains class balance and prevents leakage."""
    df = load_dataset("diabetes.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    total_pos_ratio = df[TARGET_COLUMN].mean()
    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()

    # Stratification ensures ratios are nearly identical
    assert abs(train_pos_ratio - total_pos_ratio) < 0.02
    assert abs(test_pos_ratio - total_pos_ratio) < 0.02
    assert len(X_train) + len(X_test) == len(df)
