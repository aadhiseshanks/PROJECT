"""
Page 5: Model Performance & Cross-Algorithm Benchmarking.
Evaluates Random Forest, Gradient Boosting, Logistic Regression, SVM, Decision Tree, and AdaBoost
across Accuracy, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, and Confusion Matrices.
"""

import streamlit as st
import pandas as pd
from src.ui_components import apply_custom_theme, render_hero_banner, render_sidebar_summary, render_medical_disclaimer
from src.prediction import load_model_metadata
from src.visualizations import (
    plot_model_comparison,
    plot_roc_curves,
    plot_confusion_matrix
)


render_hero_banner(
    title="Machine Learning Model Evaluation & Cross-Algorithm Benchmark",
    subtitle="Comparative performance analysis on held-out stratified test data (N=154) across sensitivity, specificity, ROC-AUC, and clinical composite fitness.",
    badge_text="Empirical Model Validation"
)

# Load metadata
try:
    metadata = load_model_metadata()
    results = metadata["models_evaluation"]
    best_model_name = metadata["best_model_name"]
    criterion = metadata["selection_criterion"]
except Exception as e:
    st.error(f"Failed to load model evaluation metadata: {e}")
    st.stop()

# Best Model Callout Banner
st.success(
    f"""
    🏆 **Selected Production Model: {best_model_name}**  
    **Clinical Selection Rationale:** In biomedical screening, failing to detect a diabetic patient (False Negative)
    can lead to irreversible microvascular damage. Hence, our selection algorithm prioritizes **Sensitivity (Recall)**
    and **Discriminative Power (ROC-AUC)** alongside overall balance, rather than naive overall accuracy.  
    *Criterion:* `{criterion}`
    """
)

# Performance Comparison Table
st.markdown("### 📊 Cross-Algorithm Performance Metrics (Test Split: 20%)")
table_rows = []
for m_name, m_data in results.items():
    is_best = " ⭐ (Best)" if m_name == best_model_name else ""
    table_rows.append({
        "Model Architecture": f"{m_name}{is_best}",
        "Accuracy": f"{m_data['accuracy']:.4f}",
        "Recall (Sensitivity)": f"{m_data['recall']:.4f}",
        "Specificity": f"{m_data['specificity']:.4f}",
        "Precision": f"{m_data['precision']:.4f}",
        "F1-Score": f"{m_data['f1_score']:.4f}",
        "ROC-AUC": f"{m_data['roc_auc']:.4f}" if m_data['roc_auc'] else "N/A",
        "Clinical Score": f"{m_data['clinical_score']:.4f}"
    })

df_metrics = pd.DataFrame(table_rows)
st.dataframe(df_metrics, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Comparative Charts
col_chart1, col_chart2 = st.columns([1.1, 0.9])

with col_chart1:
    st.plotly_chart(plot_model_comparison(results), use_container_width=True)

with col_chart2:
    st.plotly_chart(plot_roc_curves(results), use_container_width=True)

st.markdown("---")

# Confusion Matrix and Classification Report Deep-Dive
st.markdown("### 🔍 Model Diagnostic Deep-Dive (Confusion Matrix & Classification Report)")

selected_model_inspect = st.selectbox(
    "Select Model to Inspect Confusion Matrix and Per-Class Report:",
    options=list(results.keys()),
    index=list(results.keys()).index(best_model_name)
)

inspect_data = results[selected_model_inspect]

col_cm, col_report = st.columns([1, 1.2])

with col_cm:
    st.plotly_chart(plot_confusion_matrix(inspect_data["confusion_matrix"], selected_model_inspect), use_container_width=True)
    cm = inspect_data["confusion_matrix"]
    st.caption(
        f"**True Negatives (TN):** {cm['tn']} | **False Positives (FP):** {cm['fp']} | "
        f"**False Negatives (FN):** {cm['fn']} | **True Positives (TP):** {cm['tp']}"
    )

with col_report:
    st.markdown(f"#### Classification Report: **{selected_model_inspect}**")
    clf_rep = inspect_data["classification_report"]
    rep_rows = []
    for cls_k in ["0", "1"]:
        if cls_k in clf_rep:
            rep_rows.append({
                "Target Class": "0: Non-Diabetic" if cls_k == "0" else "1: Diabetic",
                "Precision": f"{clf_rep[cls_k]['precision']:.4f}",
                "Recall (Sensitivity)": f"{clf_rep[cls_k]['recall']:.4f}",
                "F1-Score": f"{clf_rep[cls_k]['f1-score']:.4f}",
                "Support": int(clf_rep[cls_k]['support'])
            })
    st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)

    st.markdown(
        f"""
        - **Balanced Accuracy:** `{inspect_data['balanced_accuracy']:.4f}`
        - **Clinical Specificity:** `{inspect_data['specificity']:.1%}` (Capacity to correctly identify non-diabetic patients without false alarms)
        - **Clinical Sensitivity:** `{inspect_data['recall']:.1%}` (Capacity to capture patients with diabetes)
        """
    )

render_medical_disclaimer()
