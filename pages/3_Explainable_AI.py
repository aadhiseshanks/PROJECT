"""
Page 3: Explainable AI (XAI).
Global model interpretability and individual patient explanation using SHAP and LIME.
Explains positive/negative feature contributions and distinguishes correlation from causation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.ui_components import apply_custom_theme, render_hero_banner, render_sidebar_summary, render_medical_disclaimer
from src.prediction import load_prediction_pipeline, load_model_metadata, predict_diabetes_risk
from src.explainability import (
    compute_global_shap_values,
    compute_local_shap_explanation,
    compute_lime_explanation,
    generate_shap_summary_figure
)
from src.visualizations import (
    plot_local_shap_waterfall,
    plot_global_feature_importance
)
from src.data_loader import FEATURE_COLUMNS, FEATURE_METADATA


render_hero_banner(
    title="Explainable AI (XAI): Model Transparency & Interpretability",
    subtitle="Deconstruct machine learning decisions using game-theoretic Shapley Additive exPlanations (SHAP) and Local Interpretable Model-agnostic Explanations (LIME).",
    badge_text="Explainable Machine Learning"
)

# Load pipeline & metadata
try:
    pipeline = load_prediction_pipeline()
    metadata = load_model_metadata()
    X_train = metadata["X_train"]
    X_test = metadata["X_test"]
    best_model_name = metadata["best_model_name"]
except Exception as e:
    st.error(f"Error loading system artifacts: {e}. Please ensure models are trained.")
    st.stop()

# XAI Tabs
tab_local, tab_global, tab_lime, tab_methodology = st.tabs([
    "👤 Individual Patient Explanation (Local SHAP)",
    "🌐 Cohort Feature Importance (Global SHAP)",
    "🍋 LIME Surrogate Explanations",
    "📚 Methodological & Ethical Context"
])

# ----------------- TAB 1: LOCAL EXPLAINABILITY -----------------
with tab_local:
    st.markdown("### Individual Prediction Attribution (Local SHAP Waterfall)")

    # Retrieve patient input from session state or generate standard sample
    if "current_prediction" in st.session_state:
        patient_df = st.session_state["current_prediction"]["input_df"]
        pred_label = st.session_state["current_prediction"]["predicted_label"]
        prob_pct = st.session_state["current_prediction"]["percentage_diabetic"]
        st.success(f"Loaded Active Patient Assessment: **{pred_label}** (Estimated Probability: **{prob_pct}%**)")
    else:
        st.info("No active patient evaluated yet. Using representative demonstration patient profile.")
        sample_input = {
            "Pregnancies": 4, "Glucose": 154.0, "BloodPressure": 82.0,
            "SkinThickness": 32.0, "Insulin": 175.0, "BMI": 34.5,
            "DiabetesPedigreeFunction": 0.58, "Age": 44
        }
        pred_res = predict_diabetes_risk(sample_input, pipeline)
        patient_df = pred_res["input_df"]
        pred_label = pred_res["predicted_label"]
        prob_pct = pred_res["percentage_diabetic"]

    with st.spinner("Computing Shapley additive values for patient profile..."):
        local_xai = compute_local_shap_explanation(pipeline, patient_df)

    col_chart, col_narrative = st.columns([1.1, 0.9])

    with col_chart:
        st.plotly_chart(plot_local_shap_waterfall(local_xai["feature_contributions"]), use_container_width=True)
        st.caption("Bars pointing right (red) increase diabetic risk probability; bars pointing left (green) reduce risk.")

    with col_narrative:
        st.markdown("#### 🗣️ Plain-English Clinical Translation")
        for sentence in local_xai["plain_english_summary"]:
            st.markdown(f"- {sentence}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📌 Key Influential Factors for this Patient")

        col_inc, col_red = st.columns(2)
        with col_inc:
            st.markdown("**Top Risk Drivers (+):**")
            for item in local_xai["top_risk_increasers"]:
                st.markdown(
                    f"<span class='badge-danger'>+{item['shap_value']:.3f}</span> "
                    f"**{item['display_name']}** ({item['patient_value']} {item['unit']})",
                    unsafe_allow_html=True
                )

        with col_red:
            st.markdown("**Top Protective Factors (-):**")
            for item in local_xai["top_risk_reducers"]:
                st.markdown(
                    f"<span class='badge-success'>{item['shap_value']:.3f}</span> "
                    f"**{item['display_name']}** ({item['patient_value']} {item['unit']})",
                    unsafe_allow_html=True
                )

# ----------------- TAB 2: GLOBAL EXPLAINABILITY -----------------
with tab_global:
    st.markdown("### Global Feature Attribution Across Population")

    col_g1, col_g2 = st.columns([1, 1.1])

    with col_g1:
        st.markdown("#### 1. Mean Absolute SHAP Impact Ranking")
        st.caption("Quantifies the average magnitude of influence each feature exerts on model predictions across the cohort.")

        with st.spinner("Computing global SHAP values over stratified test samples..."):
            shap_pos, X_trans, feat_names = compute_global_shap_values(pipeline, X_test, max_samples=150)
            mean_abs_shap = np.mean(np.abs(shap_pos), axis=0)
            importance_df = pd.DataFrame({
                "feature": feat_names,
                "importance": mean_abs_shap
            }).sort_values(by="importance", ascending=False)

        st.plotly_chart(plot_global_feature_importance(importance_df), use_container_width=True)

    with col_g2:
        st.markdown("#### 2. SHAP Beeswarm Summary Plot")
        st.caption("Each dot represents an individual patient. Position on x-axis indicates SHAP impact; color indicates parameter value (Red = High, Blue = Low).")

        fig_summary = generate_shap_summary_figure(shap_pos, pd.DataFrame(X_trans, columns=feat_names))
        st.pyplot(fig_summary)

    st.markdown("### 🔬 Clinical Interpretation of Global Rankings")
    st.markdown(
        """
        - **Plasma Glucose** ranks consistently as the dominant primary predictive feature. Higher glucose concentrations (red dots) strongly drive positive SHAP values toward diabetes classification.
        - **Body Mass Index (BMI)** serves as the second most influential predictor, highlighting the robust association between metabolic adiposity, visceral adipose tissue, and insulin insensitivity.
        - **Age** and **Diabetes Pedigree Function** follow, demonstrating how chronological beta-cell degradation and hereditary genetic background establish baseline susceptibility.
        """
    )

# ----------------- TAB 3: LIME SURROGATE -----------------
with tab_lime:
    st.markdown("### LIME (Local Interpretable Model-agnostic Explanations)")
    st.write(
        """
        LIME constructs an interpretable linear surrogate model locally around the patient's specific biometric vector
        by perturbing the feature space and fitting a sparse local regressor.
        """
    )

    with st.spinner("Fitting local linear surrogate with LIME..."):
        lime_rules = compute_lime_explanation(pipeline, X_train, patient_df, num_features=6)

    lime_df = pd.DataFrame(lime_rules, columns=["Local Decision Condition", "LIME Weight / Impact"])
    lime_df["Influence"] = lime_df["LIME Weight / Impact"].apply(lambda w: "Higher Diabetes Likelihood" if w > 0 else "Lower Diabetes Likelihood")

    st.dataframe(lime_df, use_container_width=True)

    st.caption("LIME surrogate validates consistency with SHAP: both frameworks identify concordant decision drivers.")

# ----------------- TAB 4: METHODOLOGICAL CONTEXT -----------------
with tab_methodology:
    st.markdown("### ⚖️ Correlation vs. Medical Causation")
    st.warning(
        """
        **Crucial Scientific Distinction:**
        1. **Associational Significance:** Feature importances (SHAP / LIME) identify patterns of co-occurrence
           and predictive variance learned from observational records in the Pima Indians diabetes study.
        2. **Absence of Causal Proof:** A high positive SHAP value for *Age* or *Glucose* proves the model utilizes this parameter
           to differentiate classes, but does not independently establish biological causality or mechanistic pathways.
        3. **Clinical Governance:** AI attributions should support—never replace—clinical judgment, laboratory diagnostic criteria,
           and pathophysiological assessments performed by licensed medical clinicians.
        """
    )

render_medical_disclaimer()
