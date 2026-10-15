"""
Shared UI components, dark-mode healthcare glassmorphism styling, and layout helpers for Streamlit.
"""

import streamlit as st
import pandas as pd
from src.prediction import load_model_metadata


CUSTOM_CSS = """
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f1f5f9;
    }

    /* Main Container */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }

    /* Seamless Dark Sidebar Override */
    section[data-testid="stSidebar"] {
        background-color: #080d1a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] div {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Sidebar Navigation Links */
    div[data-testid="stSidebarNav"] ul {
        padding-top: 0.5rem;
    }
    div[data-testid="stSidebarNav"] li a {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
        margin-bottom: 0.25rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stSidebarNav"] li a:hover {
        background: rgba(14, 165, 233, 0.1) !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
    }
    div[data-testid="stSidebarNav"] li a[aria-current="page"] {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.22) 0%, rgba(14, 165, 233, 0.05) 100%) !important;
        border-left: 3px solid #0ea5e9 !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
    }
    div[data-testid="stSidebarNav"] li a span {
        font-weight: 500 !important;
        color: #e2e8f0 !important;
    }

    /* Hero Banner with Futuristic Glow */
    .hero-banner {
        background: linear-gradient(135deg, #091326 0%, #0f213d 50%, #024b75 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(2, 75, 117, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.2) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.15rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        letter-spacing: -0.02em;
        line-height: 1.25;
    }
    .hero-banner p {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        margin-bottom: 0;
        line-height: 1.55;
        max-width: 900px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.85rem;
    }
    .hero-badge::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        background: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38bdf8;
    }

    /* Metric Glass Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(17, 26, 46, 0.85) 0%, rgba(13, 20, 36, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.15);
    }
    .metric-label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .metric-val {
        color: #ffffff !important;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.01em;
    }
    .metric-sub {
        color: #38bdf8 !important;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.45rem;
    }

    /* Result Cards (Glassmorphism) */
    .result-card-positive {
        background: linear-gradient(135deg, rgba(69, 10, 10, 0.65) 0%, rgba(26, 10, 16, 0.85) 100%);
        border: 1px solid rgba(239, 68, 68, 0.6);
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem;
        color: #fee2e2;
        margin: 1.2rem 0;
    }
    .result-card-positive h2 {
        color: #fca5a5 !important;
    }
    .result-card-negative {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.65) 0%, rgba(8, 38, 28, 0.85) 100%);
        border: 1px solid rgba(16, 185, 129, 0.6);
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem;
        color: #d1fae5;
        margin: 1.2rem 0;
    }
    .result-card-negative h2 {
        color: #6ee7b7 !important;
    }

    /* Medical Disclaimer Banner */
    .medical-disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #f59e0b;
        color: #fde68a !important;
        padding: 1.1rem 1.35rem;
        border-radius: 10px;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 1.8rem 0;
    }
    .medical-disclaimer strong {
        color: #fbbf24 !important;
    }

    /* Badges */
    .badge-success {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24 !important;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Sidebar System Widget Card */
    .sidebar-widget {
        background: linear-gradient(145deg, rgba(19, 29, 51, 0.9) 0%, rgba(13, 20, 36, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0 1.2rem 0;
    }
    .sidebar-widget-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #38bdf8 !important;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.84rem;
    }
    .sidebar-stat-row:last-child {
        border-bottom: none;
    }
    .sidebar-stat-label {
        color: #94a3b8 !important;
    }
    .sidebar-stat-val {
        color: #f8fafc !important;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Enhanced Streamlit Inputs & Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    button[kind="secondary"] {
        background: rgba(19, 29, 51, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(30, 41, 59, 0.9) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #38bdf8 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.6);
        padding: 0.35rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #94a3b8 !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.2) !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }
</style>
"""


def apply_custom_theme():
    """Inject custom responsive dark healthcare glassmorphism styling."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero_banner(
    title: str,
    subtitle: str,
    badge_text: str = "Explainable Clinical Intelligence"
):
    """Render high-impact gradient hero header banner."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-badge">{badge_text}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_medical_disclaimer():
    """Render standard institutional medical disclaimer."""
    st.markdown(
        """
        <div class="medical-disclaimer">
            <strong>⚠️ Institutional Medical & Research Disclaimer:</strong> This application is engineered strictly for
            academic research, engineering demonstrations, and decision-support exploration.
            The predicted probabilities and feature attributions are mathematical approximations from observational cohort data
            and do <strong>NOT</strong> constitute a clinical diagnosis, medical prescription, or therapeutic directive.
            Always consult a qualified healthcare professional or licensed physician for formal clinical assessment.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar_summary():
    """Display modern glassmorphic system status in the sidebar without broken formatting."""
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.4rem;">🩺</span>
            <span style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">ClinAI Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        meta = load_model_metadata("models/model_metadata.joblib")
        best_model = meta.get("best_model_name", "Random Forest")
        best_score = meta.get("models_evaluation", {}).get(best_model, {}).get("clinical_score", 0.7553)
        recall = meta.get("models_evaluation", {}).get(best_model, {}).get("recall", 0.7593)
        roc_auc = meta.get("models_evaluation", {}).get(best_model, {}).get("roc_auc", 0.8180)
        cohort_len = meta.get("train_samples", 614) + meta.get("test_samples", 154)

        st.sidebar.markdown(
            f"""
            <div class="sidebar-widget">
                <div class="sidebar-widget-title">
                    <span>⚡</span> System Intelligence
                </div>
                <div class="sidebar-stat-row">
                    <span class="sidebar-stat-label">Production Model</span>
                    <span class="badge-success">{best_model}</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="sidebar-stat-label">Clinical Fitness</span>
                    <span class="sidebar-stat-val">{best_score:.4f}</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="sidebar-stat-label">Sensitivity (Recall)</span>
                    <span class="sidebar-stat-val" style="color: #34d399;">{recall:.1%}</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="sidebar-stat-label">Test ROC-AUC</span>
                    <span class="sidebar-stat-val" style="color: #38bdf8;">{roc_auc:.3f}</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="sidebar-stat-label">Pima Cohort</span>
                    <span class="sidebar-stat-val">{cohort_len} rows</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="sidebar-stat-label">Explainability</span>
                    <span class="sidebar-stat-val">SHAP + LIME</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.sidebar.caption("System status loading...")
