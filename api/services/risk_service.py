"""
Risk Prediction Service
=======================
Loads the production Random Forest classifier and provides
risk predictions for ACOs based on 8 input features.
"""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "risk_prediction" / "risk_random_forest_85_87.joblib"

MODEL_FEATURES = [
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
]

RISK_THRESHOLD = 0.23

_model = None


def get_model():
    """Load the Random Forest model (cached after first call)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Risk model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        if hasattr(_model, "feature_names_in_"):
            if list(_model.feature_names_in_) != MODEL_FEATURES:
                raise ValueError("Model feature mismatch with API configuration.")
    return _model


def get_risk_level(probability: float) -> str:
    """Classify probability into LOW / MEDIUM / HIGH."""
    if probability < RISK_THRESHOLD:
        return "LOW"
    if probability < 0.35:
        return "MEDIUM"
    return "HIGH"


def predict_risk(input_data: dict) -> dict:
    """Generate a risk prediction from the trained Random Forest."""
    model = get_model()

    X = pd.DataFrame(
        {feat: [input_data[feat]] for feat in MODEL_FEATURES},
        columns=MODEL_FEATURES,
    )

    risk_probability = float(model.predict_proba(X)[0][1])
    prediction = int(risk_probability >= RISK_THRESHOLD)

    return {
        "risk_probability": round(risk_probability, 6),
        "risk_probability_pct": round(risk_probability * 100, 2),
        "prediction": prediction,
        "risk_label": "RISK" if prediction == 1 else "NON-RISK",
        "risk_level": get_risk_level(risk_probability),
        "threshold": RISK_THRESHOLD,
    }


def get_model_info() -> dict:
    """Return production model metadata."""
    model = get_model()
    return {
        "model_type": type(model).__name__,
        "model_file": MODEL_PATH.name,
        "threshold": RISK_THRESHOLD,
        "features": MODEL_FEATURES,
        "risk_levels": {
            "LOW": "probability < 0.23",
            "MEDIUM": "0.23 <= probability < 0.35",
            "HIGH": "probability >= 0.35",
        },
    }
