"""
Model training and pipeline persistence script.
Trains and compares multiple classification algorithms:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (RBF Kernel)
- Gradient Boosting Classifier
- AdaBoost Classifier

Evaluates models on held-out stratified test set, applies documented multi-metric
clinical selection criteria, and persists trained pipelines and metadata via Joblib.
"""

import os
import sys
import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

# Scikit-learn models & pipelines
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# Project internal modules
from src.data_loader import load_dataset, validate_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import create_preprocessor_pipeline, split_data
from src.evaluate_model import compute_classification_metrics, calculate_composite_clinical_score


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define dictionary of candidate classification algorithms with tuned hyperparameters."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state
        )
    }


def train_and_evaluate_all_models(
    dataset_path: str = "diabetes.csv",
    output_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute end-to-end model training, validation, multi-metric benchmarking,
    and artifact serialization.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading dataset from '{dataset_path}'...")
    df = load_dataset(dataset_path)
    val_report = validate_dataset(df)
    print(f"Dataset loaded successfully: {val_report['total_rows']} rows, {val_report['total_features']} features.")
    print(f"Class Distribution: {val_report['target_distribution']}")

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=TARGET_COLUMN,
        test_size=0.20,
        random_state=random_state
    )
    print(f"Training split: {X_train.shape[0]} samples; Test split: {X_test.shape[0]} samples.")

    candidate_models = get_candidate_models(random_state=random_state)
    trained_pipelines = {}
    evaluation_results = {}
    composite_scores = {}

    print("\n" + "=" * 80)
    print(f"{'Model Name':<25} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Clinical Score':<14}")
    print("=" * 80)

    for name, clf in candidate_models.items():
        # Build full pipeline: Preprocessing -> Estimator
        preprocessor = create_preprocessor_pipeline(scaler_type="robust", imputer_strategy="median")
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # Train pipeline strictly on training split
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Predictions on held-out test split
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            decision = pipe.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision))
        else:
            y_prob = None

        # Compute full classification metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        clinical_score = calculate_composite_clinical_score(metrics)
        metrics["clinical_score"] = clinical_score

        evaluation_results[name] = metrics
        composite_scores[name] = clinical_score

        roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        print(f"{name:<25} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<9.4f} | {metrics['recall']:<8.4f} | {metrics['f1_score']:<8.4f} | {roc_auc_str:<8} | {clinical_score:<14.4f}")

    print("=" * 80)

    # Multi-metric selection: Choose model with highest composite clinical score
    best_model_name = max(composite_scores, key=composite_scores.get)
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = evaluation_results[best_model_name]

    print(f"\n>>> Selected Best Model: '{best_model_name}'")
    print(f"    Clinical Composite Score: {best_metrics['clinical_score']:.4f}")
    print(f"    Recall (Sensitivity):    {best_metrics['recall']:.4f}")
    print(f"    Specificity:             {best_metrics['specificity']:.4f}")
    print(f"    F1-Score:                {best_metrics['f1_score']:.4f}")
    print(f"    ROC-AUC:                 {best_metrics['roc_auc']:.4f}")

    # Prepare metadata container
    metadata = {
        "best_model_name": best_model_name,
        "selection_criterion": "Composite Clinical Score: 0.35*ROC_AUC + 0.35*Recall + 0.20*F1 + 0.10*Specificity",
        "models_evaluation": evaluation_results,
        "feature_names": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_labels": {0: "Non-Diabetic", 1: "Diabetic"},
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_timestamp": datetime.datetime.now().isoformat(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    # Save to disk
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    all_models_path = os.path.join(output_dir, "all_models.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.joblib")

    joblib.dump(best_pipeline, best_model_path)
    joblib.dump(trained_pipelines, all_models_path)
    joblib.dump(metadata, metadata_path)

    print(f"\nArtifacts successfully persisted:")
    print(f"  - Best Pipeline:  {best_model_path}")
    print(f"  - All Pipelines:  {all_models_path}")
    print(f"  - Metadata:       {metadata_path}")

    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "metadata_path": metadata_path,
        "best_model_path": best_model_path
    }


if __name__ == "__main__":
    train_and_evaluate_all_models()
