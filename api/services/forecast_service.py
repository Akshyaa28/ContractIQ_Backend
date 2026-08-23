"""
Forecast Service
================
Loads the production Random Forest regressor and provides
next-year savings rate forecasts for ACOs.
"""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "forecasting" / "forecast_random_forest.joblib"

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

HIGH_SAVINGS_THRESHOLD = 0.03
MODERATE_SAVINGS_THRESHOLD = 0.00

_model = None


def get_model():
    """Load the Random Forest forecast model (cached after first call)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Forecast model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        if hasattr(_model, "feature_names_in_"):
            if list(_model.feature_names_in_) != MODEL_FEATURES:
                raise ValueError("Forecast model feature mismatch with API configuration.")
    return _model


def get_savings_category(forecast: float) -> str:
    """Classify forecast into HIGH SAVINGS / MODERATE SAVINGS / LOW SAVINGS."""
    if forecast >= HIGH_SAVINGS_THRESHOLD:
        return "HIGH SAVINGS"
    if forecast >= MODERATE_SAVINGS_THRESHOLD:
        return "MODERATE SAVINGS"
    return "LOW SAVINGS"


def predict_forecast(input_data: dict) -> dict:
    """Generate a next-year savings rate forecast."""
    model = get_model()

    X = pd.DataFrame(
        {feat: [input_data[feat]] for feat in MODEL_FEATURES},
        columns=MODEL_FEATURES,
    )

    raw = float(model.predict(X)[0])

    return {
        "forecasted_savings_rate": round(raw, 6),
        "forecasted_savings_rate_pct": round(raw * 100, 2),
        "savings_category": get_savings_category(raw),
        "savings_direction": "POSITIVE" if raw >= 0 else "NEGATIVE",
        "model_r2_pct": 88.82,
    }


def get_model_info() -> dict:
    """Return production model metadata."""
    model = get_model()
    return {
        "model_type": type(model).__name__,
        "model_file": MODEL_PATH.name,
        "target": "NEXT_YEAR_SAVINGS_RATE",
        "features": MODEL_FEATURES,
        "validation_r2_pct": 88.82,
        "validation_mae": 0.008054,
        "validation_rmse": 0.010574,
        "train_years": "2018-2022",
        "validation_year": 2023,
        "savings_categories": {
            "HIGH SAVINGS": "forecast >= 3.0%",
            "MODERATE SAVINGS": "0.0% <= forecast < 3.0%",
            "LOW SAVINGS": "forecast < 0.0%",
        },
    }
