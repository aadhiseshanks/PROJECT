"""
Page 1: Dataset Explorer.
Comprehensive exploratory data analysis (EDA), zero-value clinical audit,
summary statistics, and interactive Plotly visualizations.
"""

import streamlit as st
import pandas as pd
from src.ui_components import apply_custom_theme, render_hero_banner, render_sidebar_summary
from src.data_loader import load_dataset, validate_dataset, FEATURE_METADATA, FEATURE_COLUMNS
from src.preprocessing import detect_outliers_iqr
from src.visualizations import (
    plot_target_distribution,
    plot_feature_distribution,
    plot_correlation_heatmap,
    plot_missing_and_zeros
)


render_hero_banner(
    title="Dataset Explorer & Clinical Data Quality Audit",
    subtitle="Examine cohort characteristics, inspect physiological distributions, and analyze zero-value clinical missingness.",
    badge_text="Exploratory Data Analysis"
)

# Load dataset
df = load_dataset("diabetes.csv")
val_report = validate_dataset(df)

# Top Summary Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<p style='font-size:24px; font-weight:600; color:#e2e8f0; margin:0;'>Total Records</p><p style='font-size:20px; color:#e2e8f0;'>{len(df):,}</p>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<p style='font-size:24px; font-weight:600; color:#e2e8f0; margin:0;'>Feature Columns</p><p style='font-size:20px; color:#e2e8f0;'>{len(FEATURE_COLUMNS)}</p>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<p style='font-size:24px; font-weight:600; color:#e2e8f0; margin:0;'>Target Variable</p><p style='font-size:20px; color:#e2e8f0;'>Outcome (0 / 1)</p>", unsafe_allow_html=True)
    with c4:
        non_diab = val_report["target_distribution"]["0"]["count"]
        diab = val_report["target_distribution"]["1"]["count"]
        st.markdown(f"<p style='font-size:24px; font-weight:600; color:#e2e8f0; margin:0;'>Class Imbalance</p><p style='font-size:20px; color:#e2e8f0;'>{non_diab} Neg / {diab} Pos</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for structured exploration
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data Preview & Schema",
    "🔍 Zero-Value & Missingness Audit",
    "📈 Interactive Distributions",
    "🔗 Correlation & Outlier Analytics"
])

with tab1:
    st.markdown("### Clinical Dataset Preview")
    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        outcome_filter = st.selectbox(
            "Filter by Clinical Outcome:",
            options=["All Records", "Non-Diabetic (Outcome = 0)", "Diabetic (Outcome = 1)"]
        )
    with col_search:
        selected_cols = st.multiselect(
            "Select Columns to Display:",
            options=list(df.columns),
            default=list(df.columns)
        )

    filtered_df = df.copy()
    if outcome_filter == "Non-Diabetic (Outcome = 0)":
        filtered_df = filtered_df[filtered_df["Outcome"] == 0]
    elif outcome_filter == "Diabetic (Outcome = 1)":
        filtered_df = filtered_df[filtered_df["Outcome"] == 1]

    st.dataframe(filtered_df[selected_cols], use_container_width=True, height=350)

    st.markdown("### Descriptive Summary Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

with tab2:
    st.markdown("### ⚠️ Clinical Missingness: The 'Zero-Value' Phenomenon")
    st.info(
        """
        **Clinical Domain Insight:** In standard data collection systems, missing measurements are frequently recorded as `0`.
        However, in human physiology:
        - A **Glucose** or **Diastolic Blood Pressure** of `0` is incompatible with life in ambulatory patients.
        - **Triceps Skinfold Thickness**, **Serum Insulin**, and **BMI** cannot physically equal `0`.
        - Conversely, **Pregnancies = 0** is completely valid (nulliparous women).
        - Our framework explicitly flags these invalid zeroes as `NaN` and applies **median imputation** inside Scikit-learn pipelines to prevent data distortion.
        """
    )

    st.plotly_chart(plot_missing_and_zeros(val_report["zero_statistics"]), use_container_width=True)

    # Detailed Zero-Stats Table
    zero_df = pd.DataFrame([
        {
            "Feature": col,
            "Clinical Meaning": FEATURE_METADATA.get(col, {}).get("description", ""),
            "Zero Count": stats["count"],
            "Zero Percentage": f"{stats['percentage']}%",
            "Imputation Strategy": "Converted to NaN & Median Imputed" if stats["is_missing_flag"] else "Preserved as Legitimate Zero"
        }
        for col, stats in val_report["zero_statistics"].items()
    ])
    st.dataframe(zero_df, use_container_width=True)

with tab3:
    st.markdown("### Interactive Feature Distributions")
    col_dist1, col_dist2 = st.columns([1, 1.2])

    with col_dist1:
        st.plotly_chart(plot_target_distribution(df), use_container_width=True)

    with col_dist2:
        selected_param = st.selectbox(
            "Select Feature to Visualize Distribution:",
            options=FEATURE_COLUMNS,
            index=1  # Glucose
        )
        group_target = st.checkbox("Overlay Class Separation (Diabetic vs Non-Diabetic)", value=True)
        st.plotly_chart(plot_feature_distribution(df, selected_param, group_by_target=group_target), use_container_width=True)

    st.caption(f"Note: Zero values for '{selected_param}' have been omitted from this plot if flagged as physiologically implausible.")

with tab4:
    st.markdown("### Feature Intercorrelations and Outlier Analysis")
    col_corr, col_iqr = st.columns([1.1, 0.9])

    with col_corr:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

    with col_iqr:
        st.markdown("#### Outlier Detection (IQR Method)")
        st.caption("Outliers are identified using Tukey's Interquartile Range ($Q1 - 1.5 \\times IQR$, $Q3 + 1.5 \\times IQR$) without deletion, as extreme clinical measurements reflect genuine pathology.")
        outliers = detect_outliers_iqr(df)
        outlier_summary = pd.DataFrame([
            {
                "Feature": k,
                "Outlier Count": v["outlier_count"],
                "Prevalence (%)": f"{v['outlier_percentage']}%",
                "Normal Range": f"[{v['lower_bound']}, {v['upper_bound']}]"
            }
            for k, v in outliers.items()
        ])
        st.dataframe(outlier_summary, use_container_width=True)
