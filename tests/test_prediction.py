"""
Unit tests for model loading, prediction inference, input validation, and risk stratification.
"""

import pytest
import pandas as pd
from src.prediction import (
    load_prediction_pipeline,
    load_model_metadata,
    validate_patient_input,
    predict_diabetes_risk
)
from src.data_loader import FEATURE_COLUMNS


@pytest.fixture
def trained_pipeline():
    return load_prediction_pipeline("models/best_model.joblib")


@pytest.fixture
def valid_patient_sample():
    return {
        "Pregnancies": 2,
        "Glucose": 138.0,
        "BloodPressure": 76.0,
        "SkinThickness": 28.0,
        "Insulin": 115.0,
        "BMI": 31.4,
        "DiabetesPedigreeFunction": 0.42,
        "Age": 38
    }


def test_pipeline_loading(trained_pipeline):
    """Verify serialized model pipeline loads correctly."""
    assert trained_pipeline is not None
    assert hasattr(trained_pipeline, "predict")
    assert hasattr(trained_pipeline, "predict_proba")


def test_metadata_loading():
    """Verify model metadata contains evaluation results and feature names."""
    meta = load_model_metadata("models/model_metadata.joblib")
    assert "best_model_name" in meta
    assert "models_evaluation" in meta
    assert meta["feature_names"] == FEATURE_COLUMNS


def test_valid_input_validation(valid_patient_sample):
    """Verify clinical bounds validator passes valid inputs."""
    is_valid, errors = validate_patient_input(valid_patient_sample)
    assert is_valid is True
    assert len(errors) == 0


def test_invalid_input_validation(valid_patient_sample):
    """Verify clinical bounds validator flags invalid inputs."""
    invalid_sample = valid_patient_sample.copy()
    invalid_sample["Glucose"] = -50.0  # Invalid negative
    invalid_sample["Age"] = 250        # Implausible age

    is_valid, errors = validate_patient_input(invalid_sample)
    assert is_valid is False
    assert len(errors) >= 2


def test_predict_diabetes_risk_format(trained_pipeline, valid_patient_sample):
    """Verify end-to-end prediction returns expected schema, probabilities, and risk tiers."""
    res = predict_diabetes_risk(valid_patient_sample, pipeline=trained_pipeline)

    assert "predicted_class" in res
    assert res["predicted_class"] in [0, 1]
    assert "probability_diabetic" in res
    assert 0.0 <= res["probability_diabetic"] <= 1.0
    assert 0.0 <= res["probability_non_diabetic"] <= 1.0
    assert abs(res["probability_diabetic"] + res["probability_non_diabetic"] - 1.0) < 1e-4
    assert "risk_tier" in res
    assert "clinical_disclaimer" in res
    assert isinstance(res["input_df"], pd.DataFrame)
