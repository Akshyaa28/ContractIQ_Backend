from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "quality_prediction"
    / "quality_score_pipeline.joblib"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# All 32 year-T feature columns expected by the pipeline.
# ACO_ID and Year_T1 (prediction target year) are NOT model features.

MODEL_FEATURES = [
    "Year_T",
    "Primary_State",
    "Revenue_Category",
    "Track",
    "Agreement_Period_Num",
    "Policy_Version",
    "COVID_Period",
    "Beneficiary_Count",
    "Hospital_Count",
    "PCP_Count",
    "Specialist_Count",
    "Risk_Score",
    "Chronic_Disease_Rate_Pct",
    "Current_Quality_Score",
    "Previous_Quality_Score",
    "Readmission_Rate_Pct",
    "Admission_Rate_Per_1000",
    "ED_Visit_Rate_Per_1000",
    "Preventable_Admission_Rate_Per_1000",
    "Patient_Experience_Score",
    "Diabetes_Control_Rate_Pct",
    "Blood_Pressure_Control_Rate_Pct",
    "Preventive_Screening_Rate_Pct",
    "Followup_Compliance_Rate_Pct",
    "Expenditure_Per_Beneficiary",
    "Benchmark_Per_Beneficiary",
    "Total_Expenditure",
    "Benchmark_Expenditure",
    "Savings_Amount",
    "Savings_Rate",
    "Final_Share_Rate",
    "Earned_Savings_Loss",
]

# Quality band thresholds (based on target distribution 55–100)
HIGH_QUALITY_THRESHOLD    = 90.0   # >= 90  → HIGH
MODERATE_QUALITY_THRESHOLD = 75.0  # 75–90  → MODERATE
                                    # < 75   → LOW


# ============================================================
# LOAD MODEL (lazy, cached)
# ============================================================

_model = None


def get_model():
    """Load the Ridge Pipeline once and reuse it."""

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Quality model was not found:\n{MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


# ============================================================
# QUALITY BAND
# ============================================================

def get_quality_band(score: float) -> str:
    """
    Classify a predicted quality score into a named band.

    HIGH     : score >= 90.0   (top performer)
    MODERATE : 75.0 <= score < 90.0
    LOW      : score < 75.0
    """
    if score >= HIGH_QUALITY_THRESHOLD:
        return "HIGH"
    if score >= MODERATE_QUALITY_THRESHOLD:
        return "MODERATE"
    return "LOW"


# ============================================================
# PREDICTION
# ============================================================

def predict_quality(input_data: dict) -> dict:
    """
    Predict next-year quality score for an ACO.

    The pipeline handles its own preprocessing (imputation,
    scaling, one-hot encoding), so raw feature values are fine.

    Parameters
    ----------
    input_data : dict
        Must contain all 32 MODEL_FEATURES. Missing numeric
        values may be None — the pipeline's SimpleImputer will
        fill them with the training-set median.

    Returns
    -------
    dict with:
        predicted_quality_score      float  (0–100, clipped)
        quality_band                 str    ("HIGH" | "MODERATE" | "LOW")
        model_r2_pct                 float  (83.38)
        model_mae                    float  (2.461)
    """

    model = get_model()

    # Build single-row DataFrame in exact feature order
    row = {feature: [input_data.get(feature)] for feature in MODEL_FEATURES}
    X = pd.DataFrame(row, columns=MODEL_FEATURES)

    # Predict and clip to valid quality score range
    raw = float(model.predict(X)[0])
    score = float(np.clip(raw, 0.0, 100.0))

    return {
        "predicted_quality_score": round(score, 4),
        "quality_band":            get_quality_band(score),
        "model_r2_pct":            83.38,
        "model_mae":               2.461,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> dict:
    """Return metadata about the production quality model."""

    model = get_model()

    return {
        "model_type":          type(model.named_steps["model"]).__name__,
        "pipeline_steps":      [s for s in model.named_steps],
        "model_file":          MODEL_PATH.name,
        "target":              "target_Quality_Score_T1",
        "features":            MODEL_FEATURES,
        "total_features":      len(MODEL_FEATURES),
        "test_r2_pct":         83.38,
        "test_mae":            2.461,
        "test_rmse":           3.112,
        "train_years":         "T+1 years 2017–2023",
        "test_year":           2024,
        "quality_bands": {
            "HIGH":     "predicted score >= 90.0",
            "MODERATE": "75.0 <= score < 90.0",
            "LOW":      "score < 75.0",
        },
        "target_range":        "55 – 100",
        "disclosure": (
            "Trained on a SYNTHETIC ACO dataset. "
            "Not real CMS outcome data."
        ),
    }
