# An Explainable Machine Learning Framework for Diabetes Risk Prediction and Personalized Health Monitoring

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-green.svg)](https://github.com/shap/shap)
[![LIME](https://img.shields.io/badge/LIME-0.2.0+-yellow.svg)](https://github.com/marcotcr/lime)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, academic-grade clinical decision-support web application developed for an **M.Tech Computer Science and Engineering** final-year project. The system couples robust biomedical data preprocessing with supervised ensemble learning, game-theoretic Explainable AI (SHAP & LIME), and longitudinal patient trajectory tracking.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [Dataset Inspection & Preprocessing](#-dataset-inspection--preprocessing)
5. [Model Architecture & Selection](#-model-architecture--selection)
6. [Explainable AI (XAI) Framework](#-explainable-ai-xai-framework)
7. [Directory Structure](#-directory-structure)
8. [Installation & Execution Guide](#-installation--execution-guide)
9. [Automated Testing & Validation](#-automated-testing--validation)
10. [Troubleshooting & FAQs](#-troubleshooting--faqs)
11. [Clinical & Ethical Disclaimer](#-clinical--ethical-disclaimer)

---

## 📖 Project Overview

Diabetes mellitus represents one of the most critical metabolic health emergencies globally. While modern supervised machine learning models demonstrate high accuracy, clinical deployment requires **interpretability** and **transparency**—physicians and patients must understand *why* a particular risk level was estimated.

This framework resolves the clinical "black box" dilemma through:
- **Biomedical Imputation:** Correcting biologically implausible zero values in clinical features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) without data leakage.
- **Cross-Algorithm Benchmarking:** Comparing Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVMs, and AdaBoost.
- **Composite Clinical Fitness:** Model selection driven by Sensitivity/Recall (0.35) and ROC-AUC (0.35) to minimize dangerous False Negatives.
- **Dual-Layer Explainability:** Population-level global feature hierarchies and patient-specific local SHAP waterfall / LIME surrogate interpretations.
- **Longitudinal Patient Monitoring:** Benchmarking against American Diabetes Association (ADA) and WHO guidelines with temporal multi-visit trajectory tracking.

---

## ✨ Key Features

- **Multi-Page Streamlit Interface:**
  - `Home`: Executive abstract, architecture overview, system KPIs, and medical disclaimer.
  - `1_📊_Dataset_Explorer`: Interactive Exploratory Data Analysis (EDA), zero-value audit, distributions, and correlation heatmaps.
  - `2_🔮_Diabetes_Prediction`: Dynamic input form with clinical boundaries, preset profiles, and calibrated probability estimation.
  - `3_🧠_Explainable_AI`: SHAP summary beeswarm, local waterfall plots, LIME rules, and plain-English clinical translations.
  - `4_📈_Health_Monitoring`: ADA/WHO reference benchmarking, lifestyle recommendations, and historical multi-visit trajectory tracking.
  - `5_⚡_Model_Performance`: Cross-model benchmarking, ROC curves, confusion matrices, and per-class metrics.
  - `6_ℹ️_About_Project`: Academic abstract, system design, citations, and evaluation checklist.
- **Strict Data Leakage Prevention:** All preprocessing steps (`ZeroToNanTransformer`, `SimpleImputer`, `RobustScaler`) are encapsulated in a single `Pipeline` fitted strictly on training data.
- **Persistent State & History:** Patient assessments can be saved directly to `data/patient_history.json` and tracked across longitudinal clinic visits.

---

## 🛠️ Technology Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.13 |
| **Web Interface** | Streamlit (Custom Responsive CSS & Components) |
| **Machine Learning** | Scikit-learn (Pipelines, Estimators, Metrics) |
| **Data Processing** | Pandas, NumPy, Scipy |
| **Explainable AI (XAI)** | SHAP (Shapley Additive exPlanations), LIME |
| **Interactive Visualizations** | Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Unit Testing** | Pytest |

---

## 🔬 Dataset Inspection & Preprocessing

The framework uses the benchmark **Pima Indians Diabetes Dataset** (`diabetes.csv`), comprising 768 patient records and 8 clinical features.

### Zero-Value Clinical Audit
In standard health data recording, missing measurements are frequently entered as `0`. In human physiology, however, several measurements cannot equal zero in living patients:
- **Glucose:** 5 records (0.65%) = Missing
- **Blood Pressure:** 35 records (4.56%) = Missing
- **Skinfold Thickness:** 227 records (29.56%) = Missing
- **Serum Insulin:** 374 records (48.70%) = Missing
- **BMI:** 11 records (1.43%) = Missing
- **Pregnancies:** 111 records (14.45%) = *Valid zero (nulliparous women)*

### Preprocessing Pipeline:
1. `ZeroToNanTransformer`: Replaces zeros with `NaN` strictly for the 5 physiologically implausible features.
2. `SimpleImputer(strategy='median')`: Uses median imputation to handle highly skewed features (Insulin, Skinfold).
3. `RobustScaler`: Centers and scales data using median and interquartile range (IQR), mitigating extreme outliers without deletion.
4. `StratifiedSplit`: 80% Training ($N=614$) / 20% Held-Out Testing ($N=154$) maintaining identical class proportions ($34.9\%$ positive).

---

## 🤖 Model Architecture & Selection

Six candidate classification algorithms were trained and benchmarked on the held-out test split:

| Model Architecture | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | Clinical Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** ⭐ | **0.7273** | **0.7593** | **0.7100** | **0.5857** | **0.6613** | **0.8180** | **0.7553** |
| Logistic Regression | 0.7338 | 0.7037 | 0.7500 | 0.6032 | 0.6496 | 0.8126 | 0.7356 |
| Decision Tree | 0.7273 | 0.6667 | 0.7600 | 0.6000 | 0.6316 | 0.8124 | 0.7200 |
| Gradient Boosting | 0.7532 | 0.5926 | 0.8400 | 0.6667 | 0.6275 | 0.8300 | 0.7074 |
| Support Vector Machine (RBF) | 0.6688 | 0.6852 | 0.6600 | 0.5211 | 0.5920 | 0.7765 | 0.6960 |
| AdaBoost | 0.7468 | 0.4815 | 0.8900 | 0.7027 | 0.5714 | 0.8194 | 0.6586 |

### Documented Multi-Metric Clinical Selection Criteria:
$$\text{Clinical Composite Score} = 0.35 \times \text{ROC-AUC} + 0.35 \times \text{Recall} + 0.20 \times \text{F1} + 0.10 \times \text{Specificity}$$

**Rationale:** Failing to identify a diabetic individual (**False Negative**) delays intervention and causes irreversible microvascular damage. Random Forest was selected because it maximizes sensitivity (**75.93%**) and discriminative ability (**ROC-AUC = 0.8180**) while maintaining strong balance.

---

## 🧠 Explainable AI (XAI) Framework

1. **Global Population Interpretability:**
   - **SHAP Summary Plot (Beeswarm):** Visualizes the direction and magnitude of feature effects across patients.
   - **Mean Absolute SHAP Ranking:** Identifies `Glucose` as the primary driver, followed by `BMI`, `Age`, and `DiabetesPedigreeFunction`.
2. **Local Patient Interpretability:**
   - **SHAP Waterfall & Diverging Bar Chart:** Deconstructs an individual patient's prediction into exact positive (+) risk-increasing and negative (-) protective contributions.
   - **LIME Tabular Explainer:** Generates local linear surrogate decision rules.
   - **Plain-English Translation:** Converts mathematical scores into clear summaries for non-technical patients.
   - **Correlation vs. Causation Caveat:** Clarifies that statistical weights reflect observational patterns in the dataset, not biological causality.

---

## 📁 Directory Structure

```plaintext
PROJECT/
│
├── app.py                         # Streamlit landing page (Home)
├── requirements.txt               # Complete Python package dependencies
├── README.md                      # Comprehensive project documentation
├── diabetes.csv                   # Benchmark clinical dataset
│
├── data/
│   └── patient_history.json       # Longitudinal multi-visit patient records log
│
├── models/
│   ├── best_model.joblib          # Persisted best trained Pipeline
│   ├── all_models.joblib          # Persisted candidate model pipelines
│   └── model_metadata.joblib      # Training metrics, test partitions, & metadata
│
├── pages/
│   ├── 1_📊_Dataset_Explorer.py   # Exploratory Data Analysis & Zero audit
│   ├── 2_🔮_Diabetes_Prediction.py# Dynamic patient risk prediction form
│   ├── 3_🧠_Explainable_AI.py     # SHAP & LIME global and local interpretability
│   ├── 4_📈_Health_Monitoring.py  # ADA/WHO benchmarking & trajectory tracking
│   ├── 5_⚡_Model_Performance.py  # Cross-algorithm benchmark & ROC curves
│   └── 6_ℹ️_About_Project.py       # Academic abstract, citations, & methodology
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Data ingestion, schema validation, & metadata
│   ├── preprocessing.py          # Custom ZeroToNanTransformer & scaling pipelines
│   ├── train_model.py             # Training & evaluation script
│   ├── evaluate_model.py          # Sensitivity, Specificity, ROC-AUC, F1 metrics
│   ├── prediction.py              # Inference pipeline & clinical bounds validation
│   ├── explainability.py          # SHAP TreeExplainer & LIME surrogate engines
│   ├── health_insights.py         # ADA/WHO reference ranges & record logger
│   ├── visualizations.py          # Interactive Plotly healthcare analytics charts
│   └── ui_components.py           # Shared responsive CSS, banners, & cards
│
└── tests/
    ├── __init__.py                # Test package initialization
    ├── test_preprocessing.py      # Unit tests for ZeroToNan & preprocessor pipeline
    ├── test_prediction.py         # Unit tests for inference, validation, & risk tiers
    └── test_models.py             # Unit tests for SHAP, LIME, & ADA benchmarking
```

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites
Ensure you have **Python 3.10+** (tested on Python 3.13) installed on Windows.

### 2. Open Windows PowerShell
Navigate to your project root folder:
```powershell
cd "c:\Users\AADHISESHAN K S\OneDrive\ドキュメント\PROJECT"
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Train and Benchmark Models
Execute the standalone training pipeline. This will validate the dataset, train all 6 algorithms on held-out stratified test data, compute metrics, and persist the artifacts into `models/`:
```powershell
python -m src.train_model
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 🧪 Automated Testing & Validation

The framework includes a comprehensive Pytest test suite covering data integrity, preprocessing transformers, model inference, probability bounds, Explainable AI engines, and clinical guideline benchmarking.

Run all tests from the project root:
```powershell
python -m pytest tests/
```

### Test Suite Summary:
- `tests/test_preprocessing.py`: Verifies `ZeroToNanTransformer`, data leakage absence, and stratified split ratios.
- `tests/test_prediction.py`: Validates input boundary enforcement, model loading, and probability calibration.
- `tests/test_models.py`: Confirms SHAP local waterfall outputs, LIME explanations, and ADA/WHO guideline classifications.

*Result:* **14 passed in 4.30s (100% test pass rate)**.

---

## ❓ Troubleshooting & FAQs

- **Q: Streamlit displays `FileNotFoundError: Model file not found`**  
  *Solution:* Run `python -m src.train_model` from the project root to generate `models/best_model.joblib`.
- **Q: Why does the model show different metrics than simple accuracy?**  
  *Solution:* In medical screening, accuracy is misleading due to class imbalance ($65\%$ negative). Our framework evaluates **Sensitivity/Recall** and **Specificity** to ensure diabetic cases are not missed.
- **Q: Why does the dataset contain zero values for insulin and skin thickness?**  
  *Solution:* In the original Pima study, uncollected lab assays were entered as `0`. Our custom pipeline replaces these with `NaN` and applies median imputation during training.

---

## ⚠️ Clinical & Ethical Disclaimer

This software application is engineered strictly for **academic research, engineering evaluation, and algorithmic demonstration**. 
- The risk probabilities and feature explanations generated are mathematical approximations derived from observational data.
- They **do not** constitute a medical diagnosis, clinical prognosis, or treatment prescription.
- Neither this application nor its underlying models replace formal diagnostic laboratory evaluations (Fasting Plasma Glucose, HbA1c, Oral Glucose Tolerance Test) or the judgment of a licensed healthcare practitioner.
=======
# PROJECT
AN EXPLAINABLE MACHINE LEARNING FRAMEWORK FOR DIABETES RISK PREDICTION AND PERSONALIZED HEALTH MONITORING
