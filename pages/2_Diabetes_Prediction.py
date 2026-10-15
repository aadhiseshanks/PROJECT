"""
Page 2: Diabetes Risk Prediction.
Dynamic input form with clinical parameter validation, calibrated probability estimation,
documented risk tier stratification, and patient record logging.
"""

import streamlit as st
import pandas as pd
from src.ui_components import apply_custom_theme, render_hero_banner, render_sidebar_summary, render_medical_disclaimer
from src.data_loader import FEATURE_METADATA, FEATURE_COLUMNS
from src.prediction import load_prediction_pipeline, predict_diabetes_risk, validate_patient_input
from src.health_insights import save_patient_record


render_hero_banner(
    title="Diabetes Risk Prediction & Probability Estimation",
    subtitle="Input verified patient clinical parameters to obtain supervised risk predictions, calibrated probabilities, and risk stratification.",
    badge_text="Inference Engine"
)

# Load pipeline
try:
    pipeline = load_prediction_pipeline()
except Exception as e:
    st.error(f"Failed to load trained pipeline: {e}. Please run 'python -m src.train_model'.")
    st.stop()

# Presets for demonstration
presets = {
    "Healthy": {
        "Pregnancies": 1, "Glucose": 88.0, "BloodPressure": 68.0,
        "SkinThickness": 22.0, "Insulin": 85.0, "BMI": 22.4,
        "DiabetesPedigreeFunction": 0.22, "Age": 26
    },
    "Borderline": {
        "Pregnancies": 3, "Glucose": 128.0, "BloodPressure": 78.0,
        "SkinThickness": 28.0, "Insulin": 135.0, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 38
    },
    "HighRisk": {
        "Pregnancies": 6, "Glucose": 172.0, "BloodPressure": 88.0,
        "SkinThickness": 38.0, "Insulin": 240.0, "BMI": 36.8,
        "DiabetesPedigreeFunction": 0.78, "Age": 52
    }
}

# Function to apply preset to session state keys
def apply_preset(preset_key):
    for col, val in presets[preset_key].items():
        st.session_state[f"slider_{col}"] = val

st.markdown("### ⚡ Quick-Load Clinical Profiles (Demonstration Presets)")
p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    if st.button("🟢 Load Normal / Low-Risk Patient", use_container_width=True, on_click=apply_preset, args=("Healthy",)):
        pass
with p_col2:
    if st.button("🟡 Load Borderline / Intermediate Patient", use_container_width=True, on_click=apply_preset, args=("Borderline",)):
        pass
with p_col3:
    if st.button("🔴 Load High-Risk / Diabetic Patient", use_container_width=True, on_click=apply_preset, args=("HighRisk",)):
        pass

st.markdown("<br>", unsafe_allow_html=True)

# Patient Metadata Inputs
st.markdown("### 📋 Patient Clinical Record Entry")
c_id1, c_id2 = st.columns([1, 2])
with c_id1:
    patient_id = st.text_input("Patient ID / MRN:", value="PATIENT-2026", max_chars=20)
with c_id2:
    patient_name = st.text_input("Patient Name:", value="John Doe")

st.markdown("---")

# Grouped Parameter Inputs
col_group1, col_group2 = st.columns(2)

input_data = {}

# Group 1: Glycemic & Metabolic Biomarkers
with col_group1:
    st.markdown(
        """
        <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">
            <h4 style="margin: 0; color: #38bdf8; font-size: 1.05rem;">🩸 Glycemic & Metabolic Biomarkers</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #94a3b8;">Primary observational predictors of insulin resistance</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    group1_cols = ["Glucose", "BMI", "Insulin", "SkinThickness"]
    for col_name in group1_cols:
        meta = FEATURE_METADATA[col_name]
        default_val = st.session_state.get(f"slider_{col_name}", meta["default"])
        
        st.markdown(f"**{meta['display_name']}** ({meta['unit']})")
        val = st.slider(
            label=f"{meta['description']}",
            min_value=float(meta["min_valid"]),
            max_value=float(meta["max_valid"]),
            value=float(default_val),
            step=float(meta["step"]),
            key=f"slider_{col_name}",
            help=meta["clinical_note"]
        )
        input_data[col_name] = val
        st.caption(f"💡 *Reference:* {meta['clinical_note']}")
        st.markdown("<br>", unsafe_allow_html=True)

# Group 2: Demographic & Cardiovascular Biomarkers
with col_group2:
    st.markdown(
        """
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(129, 140, 248, 0.2); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">
            <h4 style="margin: 0; color: #818cf8; font-size: 1.05rem;">🧬 Demographic & Cardiovascular Markers</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #94a3b8;">Vascular strain and hereditary susceptibility indicators</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    group2_cols = ["BloodPressure", "Age", "DiabetesPedigreeFunction", "Pregnancies"]
    for col_name in group2_cols:
        meta = FEATURE_METADATA[col_name]
        default_val = st.session_state.get(f"slider_{col_name}", meta["default"])
        
        st.markdown(f"**{meta['display_name']}** ({meta['unit']})")
        if meta["type"] == "integer":
            val = st.slider(
                label=f"{meta['description']}",
                min_value=int(meta["min_valid"]),
                max_value=int(meta["max_valid"]),
                value=int(default_val),
                step=int(meta["step"]),
                key=f"slider_{col_name}",
                help=meta["clinical_note"]
            )
        else:
            val = st.slider(
                label=f"{meta['description']}",
                min_value=float(meta["min_valid"]),
                max_value=float(meta["max_valid"]),
                value=float(default_val),
                step=float(meta["step"]),
                key=f"slider_{col_name}",
                help=meta["clinical_note"]
            )
        input_data[col_name] = val
        st.caption(f"💡 *Reference:* {meta['clinical_note']}")
        st.markdown("<br>", unsafe_allow_html=True)

# Submit button
submit_btn = st.button("🚀 Calculate Diabetes Risk Prediction", type="primary", use_container_width=True)

if submit_btn:
    is_valid, errors = validate_patient_input(input_data)
    if not is_valid:
        for err in errors:
            st.error(f"Validation Error: {err}")
    else:
        pred_result = predict_diabetes_risk(input_data, pipeline=pipeline)
        st.session_state["current_prediction"] = pred_result
        st.session_state["current_patient_id"] = patient_id
        st.session_state["current_patient_name"] = patient_name

# Display prediction result if available
if "current_prediction" in st.session_state:
    res = st.session_state["current_prediction"]
    st.markdown("---")
    st.markdown("## 🔍 Diagnostic Assessment Summary")

    is_pos = res["predicted_class"] == 1
    card_class = "result-card-positive" if is_pos else "result-card-negative"
    icon = "⚠️" if is_pos else "✅"

    res_col1, res_col2 = st.columns([1.2, 0.8])

    with res_col1:
        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <h2 style="margin: 0; font-size: 1.85rem;">{icon} {res['predicted_label']}</h2>
                    <span style="font-size: 1.7rem; font-weight: 800; font-family: monospace;">{res['percentage_diabetic']}% Risk</span>
                </div>
                <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1rem 1.2rem; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <span style="font-weight: 600; color: #e2e8f0;">Statistical Stratification:</span>
                        <span style="font-weight: 700; color: {res['tier_color']};">{res['risk_tier']}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;">
                        <strong>Actionable Clinical Guidance:</strong> {res['tier_guidance']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(res["probability_diabetic"], text=f"Statistical Model Risk Confidence: {res['percentage_diabetic']}%")

    with res_col2:
        st.markdown("### 📊 Probability Breakdown")
        prob_df = pd.DataFrame({
            "Classification": ["Non-Diabetic (Outcome = 0)", "Diabetic (Outcome = 1)"],
            "Probability": [f"{res['percentage_non_diabetic']}%", f"{res['percentage_diabetic']}%"],
            "Raw Score": [f"{res['probability_non_diabetic']:.4f}", f"{res['probability_diabetic']:.4f}"]
        })
        st.dataframe(prob_df, use_container_width=True)

        st.info(
            """
            **Statistical Probability vs. Clinical Diagnosis:**
            The percentage represents the classifier's statistical confidence relative to observational cohort distributions.
            It does not replace diagnostic blood tests (HbA1c / FPG) evaluated by licensed medical practitioners.
            """
        )

    # Action buttons: Save record and Navigate to Explainable AI
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        if st.button("💾 Save Record to Longitudinal Monitoring Log", use_container_width=True):
            saved = save_patient_record(
                patient_id=st.session_state.get("current_patient_id", "PATIENT-2026"),
                patient_name=st.session_state.get("current_patient_name", "Anonymous"),
                parameters=input_data,
                prediction=res,
                notes=f"Automated risk evaluation. Probability: {res['percentage_diabetic']}%"
            )
            st.success(f"Record successfully saved! Record ID: `{saved['record_id']}`. View in 'Health Monitoring' tab.")

    with act_col2:
        st.caption("👉 Open **3 Explainable AI** from the left sidebar to inspect SHAP waterfall and LIME attributions for this patient.")

render_medical_disclaimer()
