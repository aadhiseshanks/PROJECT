"""
Data loading and dataset validation module.
Responsible for reading the clinical dataset, verifying schema integrity,
and providing comprehensive feature metadata and physiological reference bounds.
"""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Clinical features where a measurement of 0 is physiologically implausible
# and represents missing data in the Pima Indians Diabetes Dataset
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

TARGET_COLUMN = "Outcome"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_METADATA = {
    "Pregnancies": {
        "display_name": "Pregnancies",
        "unit": "count",
        "description": "Number of times pregnant",
        "type": "integer",
        "min_valid": 0,
        "max_valid": 20,
        "default": 3,
        "step": 1,
        "zero_as_missing": False,
        "clinical_note": "Parity has been linked to gestational metabolic changes and insulin resistance."
    },
    "Glucose": {
        "display_name": "Plasma Glucose",
        "unit": "mg/dL",
        "description": "Plasma glucose concentration at 2 hours in an oral glucose tolerance test",
        "type": "continuous",
        "min_valid": 40.0,
        "max_valid": 300.0,
        "default": 117.0,
        "step": 1.0,
        "zero_as_missing": True,
        "clinical_note": "Normal fasting: <100 mg/dL; Impaired (prediabetes): 100-125 mg/dL; Diabetic: >=126 mg/dL (ADA Guidelines)."
    },
    "BloodPressure": {
        "display_name": "Diastolic Blood Pressure",
        "unit": "mm Hg",
        "description": "Diastolic blood pressure measurement",
        "type": "continuous",
        "min_valid": 30.0,
        "max_valid": 180.0,
        "default": 72.0,
        "step": 1.0,
        "zero_as_missing": True,
        "clinical_note": "Normal: <80 mm Hg; Elevated/Prehypertension: 80-89 mm Hg; Hypertension: >=90 mm Hg (AHA/ACC Guidelines)."
    },
    "SkinThickness": {
        "display_name": "Triceps Skinfold Thickness",
        "unit": "mm",
        "description": "Triceps skin fold thickness measuring subcutaneous adipose tissue",
        "type": "continuous",
        "min_valid": 5.0,
        "max_valid": 100.0,
        "default": 29.0,
        "step": 1.0,
        "zero_as_missing": True,
        "clinical_note": "Surrogate measure of body fat distribution and peripheral subcutaneous adiposity."
    },
    "Insulin": {
        "display_name": "2-Hour Serum Insulin",
        "unit": "μU/mL",
        "description": "2-Hour post-load serum insulin concentration",
        "type": "continuous",
        "min_valid": 10.0,
        "max_valid": 900.0,
        "default": 125.0,
        "step": 1.0,
        "zero_as_missing": True,
        "clinical_note": "Elevated insulin indicates compensatory hyperinsulinemia secondary to peripheral insulin resistance."
    },
    "BMI": {
        "display_name": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "description": "Body mass index calculated as weight in kg divided by height in meters squared",
        "type": "continuous",
        "min_valid": 10.0,
        "max_valid": 70.0,
        "default": 32.0,
        "step": 0.1,
        "zero_as_missing": True,
        "clinical_note": "Underweight: <18.5; Normal: 18.5-24.9; Overweight: 25.0-29.9; Obese: >=30.0 kg/m² (WHO Criteria)."
    },
    "DiabetesPedigreeFunction": {
        "display_name": "Diabetes Pedigree Function",
        "unit": "score",
        "description": "Genetic score synthesizing diabetes history among first and second-degree relatives",
        "type": "continuous",
        "min_valid": 0.05,
        "max_valid": 2.50,
        "default": 0.37,
        "step": 0.01,
        "zero_as_missing": False,
        "clinical_note": "Quantifies hereditary genetic predisposition based on familial diabetes prevalence and degree of kinship."
    },
    "Age": {
        "display_name": "Age",
        "unit": "years",
        "description": "Chronological age of patient",
        "type": "integer",
        "min_valid": 18,
        "max_valid": 100,
        "default": 33,
        "step": 1,
        "zero_as_missing": False,
        "clinical_note": "Type 2 diabetes incidence increases progressively with age due to decreased beta-cell function."
    }
}


def load_dataset(filepath: str = "diabetes.csv") -> pd.DataFrame:
    """
    Load dataset from specified CSV path with existence verification.
    """
    if not os.path.exists(filepath):
        # Fallback to current working directory or parent directories
        potential_paths = [
            filepath,
            os.path.join(os.getcwd(), filepath),
            os.path.join(os.path.dirname(__file__), "..", filepath),
        ]
        for p in potential_paths:
            if os.path.exists(p):
                filepath = p
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. Please ensure 'diabetes.csv' is present."
            )

    df = pd.read_csv(filepath)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify schema integrity, identify columns, missing/zero counts, and class balance.
    """
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    total_records = len(df)
    duplicates = int(df.duplicated().sum())

    # Calculate zeroes for clinical attributes
    zero_stats = {}
    for col in FEATURE_COLUMNS:
        zero_cnt = int((df[col] == 0).sum())
        zero_stats[col] = {
            "count": zero_cnt,
            "percentage": round((zero_cnt / total_records) * 100, 2),
            "is_missing_flag": col in ZERO_AS_MISSING_COLUMNS
        }

    # Class balance
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    target_dist = {
        str(k): {
            "count": int(v),
            "percentage": round((v / total_records) * 100, 2),
            "label": "Diabetic (Positive)" if k == 1 else "Non-Diabetic (Negative)"
        }
        for k, v in target_counts.items()
    }

    return {
        "total_rows": total_records,
        "total_features": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_distribution": target_dist,
        "zero_statistics": zero_stats,
        "duplicates": duplicates,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns}
    }


def get_feature_metadata() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of feature definitions and clinical metadata."""
    return FEATURE_METADATA
