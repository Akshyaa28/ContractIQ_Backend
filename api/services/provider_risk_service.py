"""
Provider Risk Prediction Service
================================
Loads the production XGBoost classifier and predicts provider
risk tier (HIGH / MEDIUM / LOW) based on 41 input features.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "deployment_package" / "models"
MODEL_PATH = MODEL_DIR / "xgboost_provider_risk.joblib"
ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
TARGET_ENCODER_PATH = MODEL_DIR / "target_encoder.pkl"
IMPUTATION_PATH = MODEL_DIR / "imputation_values.pkl"

MODEL_FEATURES = [
    "performance_year", "provider_type", "specialty", "beneficiary_count",
    "hcc_risk_aged_dual", "admissions_per_1000", "inpatient_expenditure",
    "outpatient_expenditure", "professional_expenditure", "snf_expenditure",
    "home_health_expenditure", "ed_visits_per_1000", "snf_admission_rate",
    "pcp_visits_per_1000", "specialist_visits_per_1000", "hcc_risk_aged_nondual",
    "hcc_risk_disabled", "hcc_risk_esrd", "pcp_visits_per_1000_missing",
    "specialist_visits_per_1000_missing", "aco_avg_total_expenditure",
    "aco_avg_per_capita_expenditure", "aco_avg_quality_score",
    "aco_avg_admissions_per_1000", "aco_avg_ed_visits_per_1000",
    "aco_avg_snf_admission_rate", "aco_avg_hcc_risk_aged_dual",
    "aco_std_total_expenditure", "aco_std_per_capita_expenditure",
    "aco_std_quality_score", "aco_std_admissions_per_1000",
    "aco_std_ed_visits_per_1000", "aco_std_snf_admission_rate",
    "aco_std_hcc_risk_aged_dual", "cost_vs_aco", "per_capita_vs_aco",
    "quality_vs_aco", "admissions_vs_aco", "ed_visits_vs_aco",
    "snf_admission_vs_aco", "risk_vs_aco",
]

CATEGORICAL_FEATURES = ["provider_type", "specialty"]

_model = None
_label_encoders = None
_target_encoder = None
_imputation_values = None


def _load_all():
    """Load model + preprocessing artifacts (cached)."""
    global _model, _label_encoders, _target_encoder, _imputation_values

    if _model is not None:
        return

    for label, path in [("Model", MODEL_PATH), ("Encoders", ENCODERS_PATH),
                        ("Target encoder", TARGET_ENCODER_PATH), ("Imputation", IMPUTATION_PATH)]:
        if not path.exists():
            raise FileNotFoundError(f"Provider risk {label} not found: {path}")

    _model = joblib.load(MODEL_PATH)
    _label_encoders = joblib.load(ENCODERS_PATH)
    _target_encoder = joblib.load(TARGET_ENCODER_PATH)
    _imputation_values = joblib.load(IMPUTATION_PATH)


def get_model():
    """Public accessor."""
    _load_all()
    return _model


def predict_provider_risk(input_data: dict) -> dict:
    """
    Predict risk tier for a provider.
    input_data must contain all 41 MODEL_FEATURES.
    Returns: { predicted_risk_tier, confidence, class_probabilities }
    """
    _load_all()

    # Build DataFrame
    row = {feat: [input_data.get(feat)] for feat in MODEL_FEATURES}
    X = pd.DataFrame(row, columns=MODEL_FEATURES)

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        le = _label_encoders[col]
        val = str(X[col].iloc[0])
        X[col] = le.transform([val])[0] if val in le.classes_ else -1

    # Impute missing numericals
    for col, median_val in _imputation_values.items():
        if col in X.columns and pd.isna(X[col].iloc[0]):
            X[col] = median_val

    # Predict
    probs = _model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    predicted_class = _target_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    # Build probability dict
    class_probs = {}
    for i, cls in enumerate(_target_encoder.classes_):
        class_probs[cls] = round(float(probs[i]), 4)

    return {
        "predicted_risk_tier": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs,
    }


def get_model_info() -> dict:
    """Return model metadata."""
    _load_all()
    return {
        "model_type": type(_model).__name__,
        "model_file": MODEL_PATH.name,
        "target": "risk_tier",
        "classes": list(_target_encoder.classes_),
        "features": MODEL_FEATURES,
        "total_features": len(MODEL_FEATURES),
        "accuracy": 0.9139,
        "macro_f1": 0.9142,
    }
