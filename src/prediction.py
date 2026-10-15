"""
Prediction inference module.
Provides input validation against clinical physiological limits, runs inference
through the trained Scikit-learn Pipeline, and computes calibrated class probabilities
with documented clinical risk tier interpretations.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import joblib

from src.data_loader import FEATURE_COLUMNS, FEATURE_METADATA, TARGET_COLUMN


def load_prediction_pipeline(model_path: str = "models/best_model.joblib"):
    """Load the serialized Scikit-learn trained pipeline from disk."""
    if not os.path.exists(model_path):
        # Try relative to project root
        alt_paths = [
            model_path,
            os.path.join(os.getcwd(), model_path),
            os.path.join(os.path.dirname(__file__), "..", model_path)
        ]
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                break
        else:
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python -m src.train_model' first."
            )
    return joblib.load(model_path)


def load_model_metadata(metadata_path: str = "models/model_metadata.joblib") -> Dict[str, Any]:
    """Load model training evaluation metadata and test partitions."""
    if not os.path.exists(metadata_path):
        alt_paths = [
            metadata_path,
            os.path.join(os.getcwd(), metadata_path),
            os.path.join(os.path.dirname(__file__), "..", metadata_path)
        ]
        for p in alt_paths:
            if os.path.exists(p):
                metadata_path = p
                break
        else:
            raise FileNotFoundError(
                f"Metadata file not found at '{metadata_path}'. Please train models first."
            )
    return joblib.load(metadata_path)


def validate_patient_input(input_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate patient inputs against clinical physiological bounds.
    Returns: (is_valid, list_of_error_or_warning_messages)
    """
    errors = []
    for col in FEATURE_COLUMNS:
        if col not in input_dict:
            errors.append(f"Missing required parameter: '{col}'")
            continue

        val = input_dict[col]
        meta = FEATURE_METADATA.get(col, {})
        min_v = meta.get("min_valid", 0.0)
        max_v = meta.get("max_valid", 9999.0)

        try:
            val_float = float(val)
            if val_float < min_v or val_float > max_v:
                errors.append(
                    f"Parameter '{meta.get('display_name', col)}' value {val} is outside clinical boundary [{min_v}, {max_v}] {meta.get('unit', '')}"
                )
        except (ValueError, TypeError):
            errors.append(f"Parameter '{col}' must be a valid numerical value.")

    return len(errors) == 0, errors


def predict_diabetes_risk(
    input_dict: Dict[str, Any],
    pipeline=None
) -> Dict[str, Any]:
    """
    Perform end-to-end inference on user-entered patient health parameters.

    Returns:
    - predicted_class: int (0 or 1)
    - predicted_label: str
    - probability_diabetic: float
    - probability_non_diabetic: float
    - clinical_tier: str (Documented probability stratification)
    - interpretation_note: str
    - input_df: pd.DataFrame (single-row DataFrame formatted for pipeline)
    """
    is_valid, validation_errors = validate_patient_input(input_dict)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")

    if pipeline is None:
        pipeline = load_prediction_pipeline()

    # Create ordered single-row DataFrame matching training feature columns
    input_df = pd.DataFrame([{col: float(input_dict[col]) for col in FEATURE_COLUMNS}])

    # Pipeline handles ZeroToNan, Imputation, Scaling, and Prediction
    pred_class = int(pipeline.predict(input_df)[0])

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        prob_non_diabetic = float(probabilities[0])
        prob_diabetic = float(probabilities[1])
    else:
        prob_diabetic = float(pred_class)
        prob_non_diabetic = float(1.0 - pred_class)

    # Documented Statistical Stratification Tier (transparent mapping)
    if prob_diabetic < 0.35:
        risk_tier = "Low Statistical Probability"
        tier_color = "#10b981"  # Emerald green
        tier_guidance = (
            "Model parameters align primarily with non-diabetic profile patterns. "
            "Continue standard preventative lifestyle practices and routine annual check-ups."
        )
    elif prob_diabetic <= 0.65:
        risk_tier = "Moderate / Borderline Probability"
        tier_color = "#f59e0b"  # Amber
        tier_guidance = (
            "Model parameters indicate intermediate markers near decision boundaries. "
            "Formal confirmatory testing (fasting glucose or HbA1c) by a physician is prudent."
        )
    else:
        risk_tier = "Elevated Statistical Probability"
        tier_color = "#ef4444"  # Red
        tier_guidance = (
            "Model parameters exhibit significant concordance with diabetic cohort patterns. "
            "A comprehensive clinical evaluation and laboratory diagnosis are strongly recommended."
        )

    label_str = "Diabetic (Outcome = 1)" if pred_class == 1 else "Non-Diabetic (Outcome = 0)"

    return {
        "predicted_class": pred_class,
        "predicted_label": label_str,
        "probability_diabetic": round(prob_diabetic, 4),
        "probability_non_diabetic": round(prob_non_diabetic, 4),
        "percentage_diabetic": round(prob_diabetic * 100, 1),
        "percentage_non_diabetic": round(prob_non_diabetic * 100, 1),
        "risk_tier": risk_tier,
        "tier_color": tier_color,
        "tier_guidance": tier_guidance,
        "input_df": input_df,
        "clinical_disclaimer": (
            "Note: The predicted probability reflects the algorithmic output of a supervised machine learning model "
            "trained on the Pima Indians observational cohort. It is NOT a clinical diagnosis or medical risk guarantee. "
            "Clinical diagnosis requires diagnostic laboratory assays (HbA1c, OGTT) interpreted by licensed medical doctors."
        )
    }
