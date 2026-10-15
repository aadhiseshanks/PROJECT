"""
Page 4: Personalized Health Monitoring.
Clinical guidelines benchmarking (ADA, WHO, AHA), evidence-based preventive education,
and longitudinal patient record tracking with multi-trace trajectory visualization.
"""

import streamlit as st
import pandas as pd
from src.ui_components import apply_custom_theme, render_hero_banner, render_sidebar_summary, render_medical_disclaimer
from src.health_insights import (
    benchmark_health_parameter,
    generate_personalized_guidance,
    get_patient_timeline,
    save_patient_record,
    load_patient_history
)
from src.prediction import load_prediction_pipeline, predict_diabetes_risk
from src.visualizations import plot_patient_longitudinal_trajectory
from src.data_loader import FEATURE_COLUMNS, FEATURE_METADATA


render_hero_banner(
    title="Personalized Health Monitoring & Clinical Guidance",
    subtitle="Benchmark clinical parameters against ADA and WHO standards, review preventive lifestyle recommendations, and track health trajectories over time.",
    badge_text="Longitudinal Healthcare Surveillance"
)

# Load current patient or fallback sample
if "current_prediction" in st.session_state:
    current_inputs = {col: float(st.session_state["current_prediction"]["input_df"][col].iloc[0]) for col in FEATURE_COLUMNS}
    current_prediction = st.session_state["current_prediction"]
    current_id = st.session_state.get("current_patient_id", "ACTIVE-PATIENT")
    current_name = st.session_state.get("current_patient_name", "Evaluated Patient")
else:
    current_inputs = {
        "Pregnancies": 2, "Glucose": 138.0, "BloodPressure": 82.0,
        "SkinThickness": 30.0, "Insulin": 150.0, "BMI": 31.8,
        "DiabetesPedigreeFunction": 0.52, "Age": 41
    }
    pipe = load_prediction_pipeline()
    current_prediction = predict_diabetes_risk(current_inputs, pipe)
    current_id = "SAMPLE-PATIENT"
    current_name = "Demonstration Patient"

# Tabs
tab_benchmark, tab_guidance, tab_longitudinal, tab_log_entry = st.tabs([
    "🩺 Clinical Guidelines Benchmarking",
    "🥗 Evidence-Based Preventive Guidance",
    "📈 Longitudinal Health Trajectory",
    "📝 Add New Follow-up Assessment"
])

# ----------------- TAB 1: BENCHMARKING -----------------
with tab_benchmark:
    st.markdown(f"### Parameter Evaluation for **{current_name}** (`{current_id}`)")
    st.caption("Each parameter is benchmarked against published American Diabetes Association (ADA) and WHO standards.")

    bench_records = []
    for col in ["Glucose", "BMI", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"]:
        val = current_inputs.get(col, 0.0)
        res = benchmark_health_parameter(col, val)
        bench_records.append(res)

    col_b1, col_b2 = st.columns(2)
    for i, b in enumerate(bench_records):
        target_col = col_b1 if i % 2 == 0 else col_b2
        with target_col:
            badge_class = f"badge-{b['badge']}"
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="metric-label">{b['display_name']}</span>
                        <span class="{badge_class}">{b['status']}</span>
                    </div>
                    <div class="metric-val" style="font-size: 1.5rem; margin: 0.3rem 0;">
                        {b['value']} <span style="font-size: 0.9rem; color: #64748b;">{b['unit']}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.45;">{b['clinical_message']}</div>
                    <div style="font-size: 0.75rem; color: #38bdf8; margin-top: 0.4rem; font-weight: 500;">Source: {b['guideline_source']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------- TAB 2: PREVENTIVE GUIDANCE -----------------
with tab_guidance:
    st.markdown("### Actionable Lifestyle & Preventive Guidance")
    st.write(
        """
        Lifestyle recommendations synthesized based on the patient's individual glycemic and metabolic markers.
        These recommendations adhere to the clinical trial findings of the **Diabetes Prevention Program (DPP)**.
        """
    )

    guidance_list = generate_personalized_guidance(current_inputs, current_prediction)

    for item in guidance_list:
        st.markdown(
            f"""
            <div style="background: linear-gradient(145deg, rgba(17, 26, 46, 0.85) 0%, rgba(13, 20, 36, 0.95) 100%); border: 1px solid rgba(56, 189, 248, 0.18); border-radius: 14px; padding: 1.35rem; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.25);">
                <div style="display: flex; align-items: center; gap: 0.85rem;">
                    <span style="font-size: 2rem;">{item['icon']}</span>
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em;">{item['category']}</div>
                        <h4 style="margin: 0; color: #ffffff; font-size: 1.15rem;">{item['title']}</h4>
                    </div>
                </div>
                <p style="margin-top: 0.85rem; margin-bottom: 0; color: #cbd5e1; font-size: 0.93rem; line-height: 1.55;">
                    {item['recommendation']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------- TAB 3: LONGITUDINAL TRAJECTORY -----------------
with tab_longitudinal:
    st.markdown("### Patient Health Trajectory Over Time")
    st.caption("Track changes in blood glucose, BMI, and statistical diabetes probability across follow-up clinic visits.")

    df_history = get_patient_timeline()

    if df_history.empty:
        st.warning("No longitudinal health records logged yet.")
    else:
        # Patient selector
        all_patient_ids = list(df_history["patient_id"].unique())
        selected_pid = st.selectbox(
            "Select Patient ID to View Longitudinal Trajectory:",
            options=["All Patients"] + all_patient_ids
        )

        filter_pid = None if selected_pid == "All Patients" else selected_pid
        patient_timeline = get_patient_timeline(filter_pid)

        # Plot longitudinal chart
        st.plotly_chart(plot_patient_longitudinal_trajectory(patient_timeline), use_container_width=True)

        st.markdown("#### Logged Historical Records Table")
        cols_to_show = ["record_id", "timestamp", "patient_id", "patient_name", "Glucose", "BMI", "BloodPressure", "risk_percentage", "risk_tier", "notes"]
        valid_cols = [c for c in cols_to_show if c in patient_timeline.columns]
        st.dataframe(patient_timeline[valid_cols], use_container_width=True)

# ----------------- TAB 4: ADD FOLLOW-UP ENTRY -----------------
with tab_log_entry:
    st.markdown("### 📝 Log a Follow-Up Consultation Record")
    st.caption("Record new health parameters for a returning patient to monitor longitudinal progress.")

    with st.form("new_consultation_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            form_pid = st.text_input("Patient ID:", value="PATIENT-101")
            form_name = st.text_input("Patient Name:", value="Demo Patient A")
            form_glucose = st.number_input("Fasting / 2-Hr Glucose (mg/dL):", min_value=40.0, max_value=300.0, value=120.0, step=1.0)
            form_bmi = st.number_input("Body Mass Index (BMI):", min_value=10.0, max_value=70.0, value=30.5, step=0.1)

        with fc2:
            form_bp = st.number_input("Diastolic Blood Pressure (mm Hg):", min_value=30.0, max_value=180.0, value=78.0, step=1.0)
            form_insulin = st.number_input("2-Hr Serum Insulin (μU/mL):", min_value=10.0, max_value=900.0, value=120.0, step=1.0)
            form_age = st.number_input("Age (Years):", min_value=18, max_value=100, value=42, step=1)
            form_notes = st.text_area("Physician / Clinical Notes:", value="Follow-up visit. Assessing lifestyle adherence.")

        form_submit = st.form_submit_button("💾 Save Follow-up Consultation Record", type="primary")

        if form_submit:
            pipe = load_prediction_pipeline()
            new_params = {
                "Pregnancies": 2,
                "Glucose": float(form_glucose),
                "BloodPressure": float(form_bp),
                "SkinThickness": 28.0,
                "Insulin": float(form_insulin),
                "BMI": float(form_bmi),
                "DiabetesPedigreeFunction": 0.58,
                "Age": int(form_age)
            }
            new_pred = predict_diabetes_risk(new_params, pipe)
            saved_entry = save_patient_record(
                patient_id=form_pid,
                patient_name=form_name,
                parameters=new_params,
                prediction=new_pred,
                notes=form_notes
            )
            st.success(f"Assessment successfully recorded! (Record ID: `{saved_entry['record_id']}`, Predicted Risk: {new_pred['percentage_diabetic']}%)")
            st.rerun()

render_medical_disclaimer()
