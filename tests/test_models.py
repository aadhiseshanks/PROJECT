"""
Unit tests for Explainable AI (SHAP and LIME) and health benchmarking modules.
"""

import pytest
import pandas as pd
from src.prediction import load_prediction_pipeline, load_model_metadata
from src.explainability import (
    compute_local_shap_explanation,
    compute_global_shap_values,
    compute_lime_explanation
)
from src.health_insights import benchmark_health_parameter, generate_personalized_guidance


@pytest.fixture
def test_setup():
    pipe = load_prediction_pipeline("models/best_model.joblib")
    meta = load_model_metadata("models/model_metadata.joblib")
    sample_df = pd.DataFrame([{
        "Pregnancies": 3,
        "Glucose": 145.0,
        "BloodPressure": 82.0,
        "SkinThickness": 33.0,
        "Insulin": 170.0,
        "BMI": 33.5,
        "DiabetesPedigreeFunction": 0.52,
        "Age": 45
    }])
    return pipe, meta, sample_df


def test_local_shap_explanation(test_setup):
    """Verify local SHAP returns contributions, directions, and plain-English text."""
    pipe, meta, sample_df = test_setup
    shap_res = compute_local_shap_explanation(pipe, sample_df)

    assert "feature_contributions" in shap_res
    assert len(shap_res["feature_contributions"]) == 8
    assert "plain_english_summary" in shap_res
    assert len(shap_res["plain_english_summary"]) > 0

    # Ensure top risk increasers and reducers are populated
    for item in shap_res["feature_contributions"]:
        assert "feature" in item
        assert "shap_value" in item
        assert "direction" in item


def test_lime_explanation(test_setup):
    """Verify LIME explanation generates feature weights for the patient sample."""
    pipe, meta, sample_df = test_setup
    lime_exp = compute_lime_explanation(pipe, meta["X_train"][:50], sample_df, num_features=4)

    assert isinstance(lime_exp, list)
    assert len(lime_exp) == 4
    for feat_rule, weight in lime_exp:
        assert isinstance(feat_rule, str)
        assert isinstance(weight, float)


def test_health_guidelines_benchmarking():
    """Verify clinical benchmarking adheres to ADA and WHO ranges."""
    # Diabetic range glucose
    bench_high = benchmark_health_parameter("Glucose", 160.0)
    assert bench_high["status"] == "Elevated (Diabetic Range)"
    assert bench_high["badge"] == "danger"

    # Normal glucose
    bench_norm = benchmark_health_parameter("Glucose", 88.0)
    assert bench_norm["status"] == "Normal"
    assert bench_norm["badge"] == "success"

    # Obese BMI
    bench_bmi = benchmark_health_parameter("BMI", 32.5)
    assert bench_bmi["status"] == "Obese Range"


def test_personalized_guidance_generation():
    """Verify guidance generates tailored recommendations based on risk score."""
    sample_inputs = {"Glucose": 150.0, "BMI": 32.0, "BloodPressure": 85.0}
    prediction_result = {"probability_diabetic": 0.75}

    guidance = generate_personalized_guidance(sample_inputs, prediction_result)
    assert len(guidance) >= 4
    categories = [g["category"] for g in guidance]
    assert "Glycemic Nutrition & Diet" in categories
    assert "Weight & Metabolic Fitness" in categories
    assert "Clinical Follow-up" in categories
