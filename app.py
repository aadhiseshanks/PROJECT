"""
Main Application Entrypoint.
Initializes the unified multi-page clinical intelligence framework
using Streamlit's native st.navigation.
Directly loads the functional prediction and explainability engine without abstract essays.
"""

import streamlit as st
from src.ui_components import apply_custom_theme, render_sidebar_summary

# Global Page Configuration
st.set_page_config(
    page_title="Diabetes Risk & Explainable AI Framework",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global dark glassmorphism styling
apply_custom_theme()

# Render real-time system status card in sidebar
render_sidebar_summary()

# Define project navigation: pure practical functionality, no abstract
pages = [
    st.Page(
        "pages/2_Diabetes_Prediction.py",
        title="Diabetes Risk Prediction",
        icon="🔮",
        default=True
    ),
    st.Page(
        "pages/3_Explainable_AI.py",
        title="Explainable AI (SHAP & LIME)",
        icon="🧠"
    ),
    st.Page(
        "pages/4_Health_Monitoring.py",
        title="Personalized Health Monitoring",
        icon="📈"
    ),
    st.Page(
        "pages/1_Dataset_Explorer.py",
        title="Clinical Dataset Explorer",
        icon="📊"
    ),
    st.Page(
        "pages/5_Model_Performance.py",
        title="Model Benchmark & Validation",
        icon="⚡"
    )
]

# Run native navigation router
pg = st.navigation(pages)
pg.run()
