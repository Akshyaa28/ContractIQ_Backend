"""
Quality Score Prediction Service
================================
Loads the production Ridge Pipeline and predicts next-year
quality scores for ACOs based on 32 input features.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "quality_prediction" / "quality_score_pipeline.joblib"

MODEL_FEATURES = [
    "Year_T", "Primary_State", "Revenue_Category", "Track",
    "Agreement_Period_Num", "Policy_Version", "COVID_Period",
    "Beneficiary_Count", "Hospital_Count", "PCP_Count", "Specialist_Count",
    "Risk_Score", "Chronic_Disease_Rate_Pct", "Current_Quality_Score",
    "Previous_Quality_Score", "Readmission_Rate_Pct",
    "Admission_Rate_Per_1000", "ED_Visit_Rate_Per_1000",
    "Preventable_Admission_Rate_Per_1000", "Patient_Experience_Score",
    "Diabetes_Control_Rate_Pct", "Blood_Pressure_Control_Rate_Pct",
    "Preventive_Screening_Rate_Pct", "Followup_Compliance_Rate_Pct",
    "Expenditure_Per_Beneficiary", "Benchmark_Per_Beneficiary",
    "Total_Expenditure", "Benchmark_Expenditure",
    "Savings_Amount", "Savings_Rate", "Final_Share_Rate", "Earned_Savings_Loss",
]

HIGH_QUALITY_THRESHOLD = 90.0
MODERATE_QUALITY_THRESHOLD = 75.0

_model = None


def get_model():
    """Load the Ridge Pipeline (cached after first call)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Quality model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_quality_band(score: float) -> str:
    """Classify score into HIGH (>=90) / MODERATE (75-90) / LOW (<75)."""
    if score >= HIGH_QUALITY_THRESHOLD:
        return "HIGH"
    if score >= MODERATE_QUALITY_THRESHOLD:
        return "MODERATE"
    return "LOW"


def predict_quality(input_data: dict) -> dict:
    """Predict next-year quality score. Pipeline handles preprocessing internally."""
    model = get_model()

    X = pd.DataFrame(
        {feat: [input_data.get(feat)] for feat in MODEL_FEATURES},
        columns=MODEL_FEATURES,
    )

    raw = float(model.predict(X)[0])
    score = float(np.clip(raw, 0.0, 100.0))

    return {
        "predicted_quality_score": round(score, 4),
        "quality_band": get_quality_band(score),
        "model_r2_pct": 83.38,
        "model_mae": 2.461,
    }


def get_model_info() -> dict:
    """Return production model metadata."""
    model = get_model()
    return {
        "model_type": type(model.named_steps["model"]).__name__,
        "pipeline_steps": list(model.named_steps.keys()),
        "model_file": MODEL_PATH.name,
        "target": "target_Quality_Score_T1",
        "features": MODEL_FEATURES,
        "total_features": len(MODEL_FEATURES),
        "test_r2_pct": 83.38,
        "test_mae": 2.461,
        "test_rmse": 3.112,
        "train_years": "2017-2023",
        "test_year": 2024,
        "quality_bands": {
            "HIGH": "score >= 90.0",
            "MODERATE": "75.0 <= score < 90.0",
            "LOW": "score < 75.0",
        },
    }
