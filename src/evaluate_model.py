"""
Model evaluation module.
Calculates comprehensive classification metrics including sensitivity/recall,
specificity, F1-score, ROC-AUC, PR-AUC, and confusion matrix statistics.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute full suite of classification metrics.

    Parameters:
    - y_true: True binary labels (0 or 1)
    - y_pred: Predicted binary labels (0 or 1)
    - y_prob: Predicted positive class probabilities [0, 1] (optional)

    Returns:
    - Dictionary of calculated metrics and curve points.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    # Binary confusion matrix unpack
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0
        sensitivity = 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recalls, precisions))

            metrics["roc_auc"] = round(roc_auc, 4)
            metrics["pr_auc"] = round(pr_auc, 4)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }
            metrics["pr_curve"] = {
                "precision": precisions.tolist(),
                "recall": recalls.tolist()
            }
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def calculate_composite_clinical_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate documented clinical composite score for model selection:
    Clinical priority:
    - Minimizing False Negatives (High Sensitivity/Recall): weight 0.35
    - High Discriminative Capacity (ROC-AUC): weight 0.35
    - Balanced Overall Precision-Recall (F1-score): weight 0.20
    - Avoiding excessive False Positives (Specificity): weight 0.10

    Score = 0.35*ROC_AUC + 0.35*Recall + 0.20*F1 + 0.10*Specificity
    """
    roc = metrics.get("roc_auc") or metrics.get("accuracy", 0.0)
    recall = metrics.get("recall", 0.0)
    f1 = metrics.get("f1_score", 0.0)
    spec = metrics.get("specificity", 0.0)

    composite = (0.35 * roc) + (0.35 * recall) + (0.20 * f1) + (0.10 * spec)
    return round(float(composite), 4)
