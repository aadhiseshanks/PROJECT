"""
Explainable AI (XAI) module.
Implements SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)
for global model transparency and individual patient prediction interpretability.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import FEATURE_COLUMNS, FEATURE_METADATA


def get_explainer_for_pipeline(pipeline, background_data: Optional[np.ndarray] = None):
    """
    Instantiate appropriate SHAP explainer based on pipeline classifier architecture.
    Uses TreeExplainer for tree-based models, and general Explainer for others.
    """
    classifier = pipeline.named_steps["classifier"]
    clf_type = type(classifier).__name__

    if "RandomForest" in clf_type or "GradientBoosting" in clf_type or "DecisionTree" in clf_type:
        return shap.TreeExplainer(classifier)
    else:
        if background_data is None:
            raise ValueError("Background data is required for non-tree explainers.")
        return shap.Explainer(classifier.predict_proba, background_data)


def compute_global_shap_values(
    pipeline,
    X_samples: pd.DataFrame,
    max_samples: int = 150
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute global SHAP values over sample dataset.
    Returns:
    - shap_values_positive_class: ndarray (N, features)
    - transformed_X: ndarray (N, features)
    - feature_names: List[str]
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    if len(X_samples) > max_samples:
        X_sub = X_samples.sample(n=max_samples, random_state=42)
    else:
        X_sub = X_samples

    X_trans = preprocessor.transform(X_sub)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    elif isinstance(X_trans, pd.DataFrame):
        X_trans = X_trans.values

    explainer = get_explainer_for_pipeline(pipeline, background_data=X_trans[:30])
    shap_vals = explainer.shap_values(X_trans)

    # Standardize SHAP output structure for positive class (Index 1)
    if isinstance(shap_vals, list):
        # List of [class_0, class_1]
        shap_pos = np.array(shap_vals[1])
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        # (N, features, 2)
        shap_pos = shap_vals[:, :, 1]
    else:
        shap_pos = np.array(shap_vals)

    return shap_pos, X_trans, FEATURE_COLUMNS


def compute_local_shap_explanation(
    pipeline,
    input_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate local instance SHAP explanations for a single patient record.
    Returns feature contributions, direction (risk-increasing vs risk-reducing),
    and plain-English summary.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform patient input using pipeline preprocessor
    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    elif isinstance(X_trans, pd.DataFrame):
        X_trans = X_trans.values

    explainer = get_explainer_for_pipeline(pipeline)
    raw_shap = explainer.shap_values(X_trans)

    if isinstance(raw_shap, list):
        local_shap = raw_shap[1][0]
    elif isinstance(raw_shap, np.ndarray) and len(raw_shap.shape) == 3:
        local_shap = raw_shap[0, :, 1]
    else:
        local_shap = raw_shap[0]

    # Base value (expected value)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_val = float(base_val)

    # Map SHAP values to feature details
    contributions = []
    for i, col in enumerate(FEATURE_COLUMNS):
        sv = float(local_shap[i])
        raw_val = float(input_df[col].iloc[0])
        unit = FEATURE_METADATA.get(col, {}).get("unit", "")
        display = FEATURE_METADATA.get(col, {}).get("display_name", col)

        direction = "Risk-Increasing (+)" if sv > 0 else "Risk-Reducing (-)"
        contributions.append({
            "feature": col,
            "display_name": display,
            "patient_value": raw_val,
            "unit": unit,
            "shap_value": round(sv, 4),
            "abs_shap": abs(sv),
            "direction": direction,
            "is_risk_increasing": sv > 0
        })

    # Sort by absolute SHAP impact
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

    # Generate plain-English explanation
    plain_english = generate_plain_english_explanation(contributions)

    return {
        "base_value": round(base_val, 4),
        "feature_contributions": contributions,
        "plain_english_summary": plain_english,
        "top_risk_increasers": [c for c in contributions if c["is_risk_increasing"]][:3],
        "top_risk_reducers": [c for c in contributions if not c["is_risk_increasing"]][:3]
    }


def compute_lime_explanation(
    pipeline,
    X_train: pd.DataFrame,
    input_df: pd.DataFrame,
    num_features: int = 6
) -> List[Tuple[str, float]]:
    """
    Generate LIME tabular explanation for the given patient instance.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Preprocess training data for background representation
    X_train_trans = preprocessor.transform(X_train)
    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()
    elif isinstance(X_train_trans, pd.DataFrame):
        X_train_trans = X_train_trans.values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_trans),
        feature_names=FEATURE_COLUMNS,
        class_names=["Non-Diabetic", "Diabetic"],
        mode="classification",
        random_state=42
    )

    sample_trans = preprocessor.transform(input_df)
    if hasattr(sample_trans, "toarray"):
        sample_trans = sample_trans.toarray()
    elif isinstance(sample_trans, pd.DataFrame):
        sample_trans = sample_trans.values

    exp = explainer.explain_instance(
        sample_trans[0],
        classifier.predict_proba,
        num_features=num_features
    )
    return exp.as_list()


def generate_plain_english_explanation(contributions: List[Dict[str, Any]]) -> List[str]:
    """
    Synthesize accessible, clinician- and patient-friendly explanations
    describing the specific drivers of the model's prediction.
    """
    points = []
    increasers = [c for c in contributions if c["is_risk_increasing"]]
    reducers = [c for c in contributions if not c["is_risk_increasing"]]

    if increasers:
        top_inc = increasers[0]
        points.append(
            f"**Primary Risk Driver:** The patient's **{top_inc['display_name']}** ({top_inc['patient_value']} {top_inc['unit']}) "
            f"contributed strongest positive weight (+{top_inc['shap_value']:.3f} SHAP score), shifting the statistical assessment towards the diabetic cohort."
        )

    if len(increasers) > 1:
        second_inc = increasers[1]
        points.append(
            f"**Secondary Elevated Marker:** **{second_inc['display_name']}** ({second_inc['patient_value']} {second_inc['unit']}) "
            f"also exerted an upward risk pull (+{second_inc['shap_value']:.3f} SHAP score)."
        )

    if reducers:
        top_red = reducers[0]
        points.append(
            f"**Protective / Mitigating Factor:** The patient's **{top_red['display_name']}** ({top_red['patient_value']} {top_red['unit']}) "
            f"exerted a counteracting protective influence ({top_red['shap_value']:.3f} SHAP score), reducing the model's risk estimate."
        )

    points.append(
        "*Methodological Note:* These SHAP score values quantify the mathematical contribution of each parameter "
        "within the trained model. They reflect associative observational patterns in the dataset, not biological causation."
    )

    return points


def generate_shap_summary_figure(shap_pos: np.ndarray, X_samples: pd.DataFrame) -> plt.Figure:
    """Generate a clean SHAP summary plot as a Matplotlib Figure styled for dark mode."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0b1120")
    ax.set_facecolor("#0b1120")
    shap.summary_plot(shap_pos, X_samples, feature_names=FEATURE_COLUMNS, show=False)
    ax.tick_params(colors="#e2e8f0", which="both")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.2))
    plt.tight_layout()
    return fig

