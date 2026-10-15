"""
Visualizations module.
Builds interactive, dark-mode healthcare Plotly charts for EDA, model benchmarking,
Explainable AI diagnostics, and patient health trajectory monitoring.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dark-mode vibrant healthcare color palette
COLOR_PRIMARY = "#0ea5e9"    # Electric Sky Blue
COLOR_SECONDARY = "#14b8a6"  # Teal
COLOR_POSITIVE = "#f43f5e"   # Neon Rose / Diabetic
COLOR_NEGATIVE = "#10b981"   # Emerald / Non-diabetic
COLOR_WARNING = "#f59e0b"    # Amber
COLOR_GRID = "rgba(255, 255, 255, 0.07)"
COLOR_TEXT = "#e2e8f0"
COLOR_MUTED = "#94a3b8"


def set_custom_chart_layout(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    """Apply consistent, sleek dark glassmorphism styling across all Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 15, "family": "'Plus Jakarta Sans', sans-serif", "color": "#ffffff"},
            "x": 0.02,
            "xanchor": "left"
        },
        height=height,
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": COLOR_TEXT}
        },
        xaxis={
            "showgrid": True,
            "gridcolor": COLOR_GRID,
            "linecolor": "rgba(255, 255, 255, 0.15)",
            "tickfont": {"size": 11, "color": COLOR_MUTED},
            "title_font": {"size": 12, "color": COLOR_TEXT}
        },
        yaxis={
            "showgrid": True,
            "gridcolor": COLOR_GRID,
            "linecolor": "rgba(255, 255, 255, 0.15)",
            "tickfont": {"size": 11, "color": COLOR_MUTED},
            "title_font": {"size": 12, "color": COLOR_TEXT}
        }
    )
    return fig


def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    """Create interactive Donut Chart for target class distribution."""
    counts = df["Outcome"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    counts["Label"] = counts["Outcome"].map({0: "Non-Diabetic (Negative)", 1: "Diabetic (Positive)"})
    counts["Percentage"] = (counts["Count"] / len(df)) * 100

    fig = go.Figure(data=[
        go.Pie(
            labels=counts["Label"],
            values=counts["Count"],
            hole=0.6,
            marker={"colors": [COLOR_NEGATIVE, COLOR_POSITIVE], "line": {"color": "#0b1120", "width": 2}},
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            textfont={"size": 12, "color": "#ffffff"},
            insidetextorientation="horizontal"
        )
    ])
    fig = set_custom_chart_layout(fig, "Target Class Distribution (Outcome)")
    fig.update_layout(showlegend=True)
    return fig


def plot_feature_distribution(
    df: pd.DataFrame,
    column: str,
    group_by_target: bool = True
) -> go.Figure:
    """Create interactive distribution plot for a specific health parameter."""
    df_plot = df.copy()

    # Filter invalid zeros for clinical features to show true distribution
    if column in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df_plot = df_plot[df_plot[column] > 0]

    if group_by_target and "Outcome" in df_plot.columns:
        df_plot["Outcome_Label"] = df_plot["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})
        fig = px.histogram(
            df_plot,
            x=column,
            color="Outcome_Label",
            barmode="overlay",
            marginal="box",
            opacity=0.75,
            color_discrete_map={"Non-Diabetic": COLOR_NEGATIVE, "Diabetic": COLOR_POSITIVE},
            labels={"Outcome_Label": "Condition"}
        )
    else:
        fig = px.histogram(
            df_plot,
            x=column,
            marginal="box",
            color_discrete_sequence=[COLOR_PRIMARY],
            opacity=0.85
        )

    fig = set_custom_chart_layout(fig, f"Distribution & Outlier Analysis: {column}")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create interactive correlation heatmap matrix."""
    corr = df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="Blues",
        zmin=-1,
        zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#ffffff"},
        hoverongaps=False
    ))
    fig = set_custom_chart_layout(fig, "Feature Correlation Matrix (Pearson)", height=450)
    return fig


def plot_missing_and_zeros(zero_stats: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Bar chart illustrating percentage of zero values and clinical validity."""
    features = list(zero_stats.keys())
    percentages = [zero_stats[f]["percentage"] for f in features]
    is_missing = [zero_stats[f]["is_missing_flag"] for f in features]
    colors = [COLOR_POSITIVE if m else COLOR_PRIMARY for m in is_missing]
    labels = ["Biologically Implausible (Missing)" if m else "Valid Zero (e.g. Parity)" for m in is_missing]

    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=percentages,
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.2)", "width": 1}},
            text=[f"{p:.1f}%" for p in percentages],
            textposition="outside",
            textfont={"color": "#e2e8f0"},
            customdata=labels,
            hovertemplate="<b>%{x}</b><br>Zero Count: %{y:.2f}%<br>Classification: %{customdata}<extra></extra>"
        )
    ])
    fig = set_custom_chart_layout(fig, "Zero-Value Prevalence & Clinical Imputation Flag", height=400)
    fig.update_layout(
        yaxis_title="Percentage of Dataset (%)",
        xaxis_title="Features"
    )
    return fig


def plot_model_comparison(evaluation_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing all candidate classification models."""
    models = list(evaluation_results.keys())
    metrics_to_plot = ["accuracy", "recall", "precision", "specificity", "f1_score", "roc_auc"]
    metric_labels = {
        "accuracy": "Accuracy",
        "recall": "Recall (Sens.)",
        "precision": "Precision",
        "specificity": "Specificity",
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC"
    }

    colors = ["#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6", "#f43f5e", "#06b6d4"]

    fig = go.Figure()
    for i, m_key in enumerate(metrics_to_plot):
        vals = [evaluation_results[m].get(m_key, 0.0) or 0.0 for m in models]
        fig.add_trace(go.Bar(
            name=metric_labels[m_key],
            x=models,
            y=vals,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont={"size": 10, "color": "#e2e8f0"}
        ))

    fig = set_custom_chart_layout(fig, "Cross-Algorithm Model Performance Benchmark", height=480)
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1.15], "title": "Score"},
        xaxis_title="Model Architecture"
    )
    return fig


def plot_roc_curves(evaluation_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Generate multi-model interactive ROC curves."""
    fig = go.Figure()

    palette = ["#0ea5e9", "#10b981", "#f59e0b", "#a855f7", "#ec4899", "#14b8a6"]

    for i, (name, metrics) in enumerate(evaluation_results.items()):
        roc_data = metrics.get("roc_curve")
        if roc_data:
            auc_val = metrics.get("roc_auc", 0.0)
            fig.add_trace(go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"{name} (AUC = {auc_val:.3f})",
                line={"width": 2.5, "color": palette[i % len(palette)]}
            ))

    # Diagonal chance line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.500)",
        line={"dash": "dash", "color": "#64748b", "width": 1.5}
    ))

    fig = set_custom_chart_layout(fig, "Receiver Operating Characteristic (ROC) Curves", height=450)
    fig.update_layout(
        xaxis={"title": "False Positive Rate (1 - Specificity)", "range": [-0.02, 1.02]},
        yaxis={"title": "True Positive Rate (Sensitivity / Recall)", "range": [-0.02, 1.02]}
    )
    return fig


def plot_confusion_matrix(cm_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Generate interactive annotated confusion matrix heatmap."""
    matrix = np.array(cm_data.get("matrix", [[0, 0], [0, 0]]))
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]

    text_labels = [
        [f"True Negative (TN)<br><b>{matrix[0, 0]}</b>", f"False Positive (FP)<br><b>{matrix[0, 1]}</b>"],
        [f"False Negative (FN)<br><b>{matrix[1, 0]}</b>", f"True Positive (TP)<br><b>{matrix[1, 1]}</b>"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        text=text_labels,
        texttemplate="%{text}",
        colorscale="Blues",
        reversescale=False,
        showscale=False
    ))

    fig = set_custom_chart_layout(fig, f"Confusion Matrix: {model_name}", height=380)
    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        yaxis={"autorange": "reversed"}
    )
    return fig


def plot_local_shap_waterfall(contributions: List[Dict[str, Any]]) -> go.Figure:
    """Create interactive diverging horizontal bar chart of local SHAP feature contributions."""
    df_c = pd.DataFrame(contributions)
    df_c = df_c.sort_values(by="shap_value", ascending=True)

    colors = [COLOR_POSITIVE if sv > 0 else COLOR_NEGATIVE for sv in df_c["shap_value"]]
    hover_texts = [
        f"<b>{row['display_name']}</b><br>"
        f"Patient Value: {row['patient_value']} {row['unit']}<br>"
        f"SHAP Impact: {row['shap_value']:+.4f}<br>"
        f"Direction: {row['direction']}"
        for _, row in df_c.iterrows()
    ]

    fig = go.Figure(data=[
        go.Bar(
            y=df_c["display_name"],
            x=df_c["shap_value"],
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.2)", "width": 1}},
            hoverinfo="text",
            hovertext=hover_texts,
            text=[f"{v:+.3f}" for v in df_c["shap_value"]],
            textposition="outside",
            textfont={"size": 11, "color": "#f8fafc"}
        )
    ])

    fig = set_custom_chart_layout(fig, "Individual Patient SHAP Contributions", height=420)
    fig.update_layout(
        xaxis={"title": "SHAP Impact Score (Positive = Higher Risk, Negative = Protective)"},
        yaxis_title="Patient Biomarkers"
    )
    return fig


def plot_global_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for global feature importance."""
    df_sorted = importance_df.sort_values(by="importance", ascending=True)

    fig = go.Figure(data=[
        go.Bar(
            y=df_sorted["feature"],
            x=df_sorted["importance"],
            orientation="h",
            marker={"color": COLOR_PRIMARY, "line": {"color": "#38bdf8", "width": 1}},
            text=[f"{v:.4f}" for v in df_sorted["importance"]],
            textposition="outside",
            textfont={"size": 11, "color": "#f8fafc"}
        )
    ])
    fig = set_custom_chart_layout(fig, "Global Feature Importance (Gini / Mean |SHAP|)", height=380)
    fig.update_layout(
        xaxis_title="Relative Feature Importance Score",
        yaxis_title="Feature Name"
    )
    return fig


def plot_patient_longitudinal_trajectory(df_timeline: pd.DataFrame) -> go.Figure:
    """Plot multi-trace timeline of Glucose, BMI, and Risk Probability over time."""
    if df_timeline.empty:
        fig = go.Figure()
        fig.add_annotation(text="No historical monitoring records available.", showarrow=False, font={"color": "#94a3b8"})
        return fig

    fig = go.Figure()

    # Probability Trace (Normalized to percentage)
    fig.add_trace(go.Scatter(
        x=df_timeline["timestamp"],
        y=df_timeline["risk_percentage"],
        mode="lines+markers",
        name="Predicted Risk Probability (%)",
        line={"color": COLOR_POSITIVE, "width": 3},
        marker={"size": 8, "symbol": "circle"}
    ))

    # Fasting Glucose Trace
    if "Glucose" in df_timeline.columns:
        fig.add_trace(go.Scatter(
            x=df_timeline["timestamp"],
            y=df_timeline["Glucose"],
            mode="lines+markers",
            name="Plasma Glucose (mg/dL)",
            line={"color": COLOR_PRIMARY, "width": 2, "dash": "dot"},
            marker={"size": 6}
        ))

    # BMI Trace
    if "BMI" in df_timeline.columns:
        fig.add_trace(go.Scatter(
            x=df_timeline["timestamp"],
            y=df_timeline["BMI"],
            mode="lines+markers",
            name="BMI (kg/m²)",
            line={"color": COLOR_WARNING, "width": 2, "dash": "dash"},
            marker={"size": 6}
        ))

    # Threshold zone for Prediabetes (100 mg/dL)
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(245, 158, 11, 0.7)", annotation_text="Glucose Normal Limit (100)", annotation_font_color="#fbbf24")
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(244, 63, 94, 0.7)", annotation_text="50% Statistical Risk Threshold", annotation_font_color="#f43f5e")

    fig = set_custom_chart_layout(fig, "Longitudinal Patient Health & Trajectory Tracking", height=450)
    fig.update_layout(
        xaxis_title="Assessment Date / Time",
        yaxis_title="Measurement Values / Risk Percentage (%)"
    )
    return fig
