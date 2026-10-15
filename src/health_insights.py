"""
Personalized Health Monitoring and Clinical Guidance module.
Provides evidence-based benchmarking against established medical guidelines (ADA, WHO, AHA),
synthesizes non-prescriptive lifestyle guidance, and manages longitudinal patient health records.
"""

import os
import json
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from src.data_loader import FEATURE_METADATA

HISTORY_FILE_PATH = os.path.join("data", "patient_history.json")


def benchmark_health_parameter(param_name: str, value: float) -> Dict[str, Any]:
    """
    Evaluate an individual health parameter against published clinical guidelines.
    """
    val = float(value)

    if param_name == "Glucose":
        if val < 100:
            status = "Normal"
            badge = "success"
            message = "Within standard healthy glycemic parameters (<100 mg/dL)."
        elif 100 <= val < 126:
            status = "Impaired (Prediabetes Range)"
            badge = "warning"
            message = "Elevated glycemic levels consistent with prediabetes (100–125 mg/dL, ADA Standards)."
        else:
            status = "Elevated (Diabetic Range)"
            badge = "danger"
            message = "Markedly elevated plasma glucose (≥126 mg/dL). Diagnostic confirmation recommended."
        guideline = "ADA Standards of Medical Care in Diabetes"

    elif param_name == "BloodPressure":
        if val < 80:
            status = "Normal"
            badge = "success"
            message = "Optimal diastolic arterial pressure (<80 mm Hg)."
        elif 80 <= val < 90:
            status = "Prehypertension / Elevated"
            badge = "warning"
            message = "Elevated diastolic pressure (80–89 mm Hg). Lifestyle modifications recommended."
        else:
            status = "Stage 1 / 2 Hypertension"
            badge = "danger"
            message = "Diastolic pressure ≥90 mm Hg indicates hypertension. Physician consultation recommended."
        guideline = "AHA / ACC Blood Pressure Guidelines"

    elif param_name == "BMI":
        if val < 18.5:
            status = "Underweight"
            badge = "warning"
            message = "BMI below recommended range (<18.5 kg/m²)."
        elif 18.5 <= val < 25.0:
            status = "Normal Weight"
            badge = "success"
            message = "Healthy body mass index range (18.5–24.9 kg/m²)."
        elif 25.0 <= val < 30.0:
            status = "Overweight"
            badge = "warning"
            message = "Moderately elevated adiposity (25.0–29.9 kg/m²). Weight management beneficial."
        else:
            status = "Obese Range"
            badge = "danger"
            message = "Adiposity index ≥30.0 kg/m² significantly increases metabolic and cardiovascular strain."
        guideline = "WHO International BMI Classification"

    elif param_name == "Insulin":
        if val <= 166:
            status = "Within Typical Post-load Range"
            badge = "success"
            message = "Serum insulin levels within normal physiological response envelope."
        else:
            status = "Hyperinsulinemia"
            badge = "warning"
            message = "Elevated post-load insulin indicates compensatory pancreatic response to peripheral insulin resistance."
        guideline = "Clinical Endocrinology Reference Limits"

    elif param_name == "DiabetesPedigreeFunction":
        if val < 0.35:
            status = "Low Familial Burden"
            badge = "success"
            message = "Lower documented hereditary diabetes density in familial lineage."
        elif 0.35 <= val < 0.70:
            status = "Moderate Familial Burden"
            badge = "warning"
            message = "Moderate historical diabetes occurrences in relatives."
        else:
            status = "High Familial Burden"
            badge = "danger"
            message = "Strong familial genetic clustering. Proactive monitoring indicated."
        guideline = "Observational Pedigree Scoring"

    else:
        status = "Observed"
        badge = "info"
        message = f"Recorded measurement: {val}"
        guideline = "Clinical Reference"

    return {
        "parameter": param_name,
        "value": val,
        "unit": FEATURE_METADATA.get(param_name, {}).get("unit", ""),
        "display_name": FEATURE_METADATA.get(param_name, {}).get("display_name", param_name),
        "status": status,
        "badge": badge,
        "clinical_message": message,
        "guideline_source": guideline
    }


def generate_personalized_guidance(
    input_dict: Dict[str, Any],
    prediction_result: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Generate evidence-based preventive health lifestyle insights
    tailored to the specific patient's biometric markers.
    """
    insights = []
    glucose = float(input_dict.get("Glucose", 0))
    bmi = float(input_dict.get("BMI", 0))
    bp = float(input_dict.get("BloodPressure", 0))
    pred_prob = prediction_result.get("probability_diabetic", 0.0)

    # Glycemic Management
    if glucose >= 100 or pred_prob > 0.40:
        insights.append({
            "category": "Glycemic Nutrition & Diet",
            "icon": "🥗",
            "title": "Adopt Low-Glycemic Complex Carbohydrates",
            "recommendation": (
                "Prioritize whole grains, legumes, leafy vegetables, and dietary fiber (target 25–35 grams/day). "
                "Minimize refined sugars, sweetened beverages, and high-glycemic carbohydrates to prevent postprandial glucose surges."
            )
        })

    # Weight and Metabolic Fitness
    if bmi >= 25.0:
        target_loss = round(bmi * 0.07, 1)
        insights.append({
            "category": "Weight & Metabolic Fitness",
            "icon": "⚖️",
            "title": "Target 5–7% Moderate Weight Reduction",
            "recommendation": (
                "Clinical trials (Diabetes Prevention Program) demonstrate that achieving a 5% to 7% reduction "
                "in baseline body weight reduces type 2 diabetes incidence by over 58% in at-risk cohorts."
            )
        })

    # Physical Activity
    insights.append({
        "category": "Physical Exercise",
        "icon": "🏃",
        "title": "150 Minutes/Week Moderate Aerobic Exercise",
        "recommendation": (
            "Engage in at least 150 minutes per week of moderate-intensity aerobic physical activity "
            "(such as brisk walking, cycling, or swimming) combined with 2 to 3 weekly sessions of resistance training "
            "to enhance GLUT4 translocation and insulin sensitivity."
        )
    })

    # Cardiovascular & Blood Pressure
    if bp >= 80:
        insights.append({
            "category": "Cardiovascular Health",
            "icon": "❤️",
            "title": "Sodium Moderation & Arterial Health",
            "recommendation": (
                "Maintain dietary sodium intake below 2,000 mg/day (DASH dietary pattern). "
                "Hypertension synergistically accelerates microvascular and macrovascular complications when combined with dysglycemia."
            )
        })

    # Professional Follow-up
    if pred_prob >= 0.50 or glucose >= 126:
        urgency = "Diagnostic Consultation Strongly Recommended"
        urgency_text = (
            "Given elevated statistical indicators and glycemic markers, schedule a formal consultation with a physician. "
            "Request laboratory confirmation via venous fasting plasma glucose and glycated hemoglobin (HbA1c)."
        )
    else:
        urgency = "Routine Preventive Health Surveillance"
        urgency_text = (
            "Maintain annual wellness screenings. Re-evaluate fasting glucose and lipid panels periodically "
            "as part of comprehensive preventive healthcare."
        )

    insights.append({
        "category": "Clinical Follow-up",
        "icon": "🩺",
        "title": urgency,
        "recommendation": urgency_text
    })

    return insights


def save_patient_record(
    patient_id: str,
    patient_name: str,
    parameters: Dict[str, Any],
    prediction: Dict[str, Any],
    notes: str = ""
) -> Dict[str, Any]:
    """
    Persist patient health record to historical JSON log for longitudinal tracking.
    """
    os.makedirs(os.path.dirname(HISTORY_FILE_PATH), exist_ok=True)

    records = load_patient_history()

    new_entry = {
        "record_id": f"REC_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": patient_id.strip() if patient_id else "ANONYMOUS",
        "patient_name": patient_name.strip() if patient_name else "Anonymous Patient",
        "parameters": {k: float(v) for k, v in parameters.items()},
        "prediction_class": int(prediction.get("predicted_class", 0)),
        "prediction_label": prediction.get("predicted_label", ""),
        "probability_diabetic": float(prediction.get("probability_diabetic", 0.0)),
        "risk_tier": prediction.get("risk_tier", ""),
        "notes": notes.strip()
    }

    records.append(new_entry)

    with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return new_entry


def load_patient_history() -> List[Dict[str, Any]]:
    """Load all stored longitudinal patient records."""
    if not os.path.exists(HISTORY_FILE_PATH):
        # Initialize with sample data if file does not exist
        sample_records = _generate_initial_sample_history()
        os.makedirs(os.path.dirname(HISTORY_FILE_PATH), exist_ok=True)
        with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_records, f, indent=2)
        return sample_records

    try:
        with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def get_patient_timeline(patient_id: Optional[str] = None) -> pd.DataFrame:
    """Retrieve historical records as a sorted DataFrame for plotting."""
    records = load_patient_history()
    if not records:
        return pd.DataFrame()

    flat_records = []
    for r in records:
        if patient_id and r.get("patient_id") != patient_id:
            continue
        entry = {
            "record_id": r["record_id"],
            "timestamp": pd.to_datetime(r["timestamp"]),
            "patient_id": r["patient_id"],
            "patient_name": r.get("patient_name", "Anonymous"),
            "probability_diabetic": r["probability_diabetic"],
            "risk_percentage": round(r["probability_diabetic"] * 100, 1),
            "predicted_class": r["prediction_class"],
            "risk_tier": r.get("risk_tier", ""),
            "notes": r.get("notes", "")
        }
        entry.update(r["parameters"])
        flat_records.append(entry)

    if not flat_records:
        return pd.DataFrame()

    df_hist = pd.DataFrame(flat_records)
    df_hist = df_hist.sort_values(by="timestamp").reset_index(drop=True)
    return df_hist


def _generate_initial_sample_history() -> List[Dict[str, Any]]:
    """Seed sample longitudinal records demonstrating health tracking over time."""
    return [
        {
            "record_id": "REC_20260601_100000",
            "timestamp": "2026-06-01 10:00:00",
            "patient_id": "PATIENT-101",
            "patient_name": "Demo Patient A",
            "parameters": {
                "Pregnancies": 2, "Glucose": 158.0, "BloodPressure": 86.0,
                "SkinThickness": 32.0, "Insulin": 180.0, "BMI": 34.2,
                "DiabetesPedigreeFunction": 0.58, "Age": 42
            },
            "prediction_class": 1,
            "prediction_label": "Diabetic (Outcome = 1)",
            "probability_diabetic": 0.784,
            "risk_tier": "Elevated Statistical Probability",
            "notes": "Initial clinic intake. Elevated fasting glucose."
        },
        {
            "record_id": "REC_20260801_103000",
            "timestamp": "2026-08-01 10:30:00",
            "patient_id": "PATIENT-101",
            "patient_name": "Demo Patient A",
            "parameters": {
                "Pregnancies": 2, "Glucose": 134.0, "BloodPressure": 80.0,
                "SkinThickness": 30.0, "Insulin": 140.0, "BMI": 32.1,
                "DiabetesPedigreeFunction": 0.58, "Age": 42
            },
            "prediction_class": 1,
            "prediction_label": "Diabetic (Outcome = 1)",
            "probability_diabetic": 0.592,
            "risk_tier": "Moderate / Borderline Probability",
            "notes": "2-Month follow-up after starting dietary modification."
        },
        {
            "record_id": "REC_20261001_091500",
            "timestamp": "2026-10-01 09:15:00",
            "patient_id": "PATIENT-101",
            "patient_name": "Demo Patient A",
            "parameters": {
                "Pregnancies": 2, "Glucose": 112.0, "BloodPressure": 76.0,
                "SkinThickness": 27.0, "Insulin": 110.0, "BMI": 29.8,
                "DiabetesPedigreeFunction": 0.58, "Age": 42
            },
            "prediction_class": 0,
            "prediction_label": "Non-Diabetic (Outcome = 0)",
            "probability_diabetic": 0.328,
            "risk_tier": "Low Statistical Probability",
            "notes": "4-Month progress: significant improvement in glucose and BMI."
        }
    ]
