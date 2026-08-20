from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# forecast_service.py
#   I:\ContractIQ(!)\api\services\forecast_service.py
#
# parents[0] = services
# parents[1] = api
# parents[2] = ContractIQ(!)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "forecast_random_forest.joblib"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

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

# Savings category thresholds
# HIGH SAVINGS     : forecast >= 0.03   (3%)
# MODERATE SAVINGS : 0.00 <= forecast < 0.03
# LOW SAVINGS      : forecast < 0.00   (spending above benchmark)

HIGH_SAVINGS_THRESHOLD     = 0.03
MODERATE_SAVINGS_THRESHOLD = 0.00


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def get_model():
    """
    Load the Random Forest forecast model once and reuse it.
    Raises FileNotFoundError at startup if the .joblib file is missing.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Forecast model was not found:\n{MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

        # Validate feature names if the model stores them
        if hasattr(_model, "feature_names_in_"):

            trained_features = list(_model.feature_names_in_)

            if trained_features != MODEL_FEATURES:
                raise ValueError(
                    "Forecast model feature configuration does not match "
                    "API configuration.\n"
                    f"Expected : {MODEL_FEATURES}\n"
                    f"Model has: {trained_features}"
                )

    return _model


# ============================================================
# SAVINGS CATEGORY
# ============================================================

def get_savings_category(forecast: float) -> str:
    """
    Classify a forecasted savings rate into a named category.

    HIGH SAVINGS
        forecast >= 0.03  (≥ 3 % above benchmark — strong performer)

    MODERATE SAVINGS
        0.00 <= forecast < 0.03  (saving, but under 3 %)

    LOW SAVINGS
        forecast < 0.00  (spending above benchmark — at-risk)
    """

    if forecast >= HIGH_SAVINGS_THRESHOLD:
        return "HIGH SAVINGS"

    if forecast >= MODERATE_SAVINGS_THRESHOLD:
        return "MODERATE SAVINGS"

    return "LOW SAVINGS"


# ============================================================
# FORECAST
# ============================================================

def predict_forecast(input_data: dict) -> dict:
    """
    Generate a next-year savings rate forecast from the trained
    Random Forest regressor.

    Parameters
    ----------
    input_data : dict
        Must contain all 8 MODEL_FEATURES as numeric values.

    Returns
    -------
    dict with keys:
        forecasted_savings_rate      float  (e.g. 0.032)
        forecasted_savings_rate_pct  float  (e.g. 3.20)
        savings_category             str    ("HIGH SAVINGS" | "MODERATE SAVINGS" | "LOW SAVINGS")
        savings_direction            str    ("POSITIVE" | "NEGATIVE")
        model_r2_pct                 float  (72.63 — reported for transparency)
    """

    model = get_model()

    # --------------------------------------------------------
    # Build single-row DataFrame in exact feature order
    # --------------------------------------------------------

    feature_data = {
        feature: [input_data[feature]]
        for feature in MODEL_FEATURES
    }

    X = pd.DataFrame(feature_data, columns=MODEL_FEATURES)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    raw_prediction = float(model.predict(X)[0])

    # --------------------------------------------------------
    # Derive outputs
    # --------------------------------------------------------

    forecasted_pct      = round(raw_prediction * 100, 4)
    savings_category    = get_savings_category(raw_prediction)
    savings_direction   = "POSITIVE" if raw_prediction >= 0 else "NEGATIVE"

    return {
        "forecasted_savings_rate":     round(raw_prediction, 6),
        "forecasted_savings_rate_pct": round(forecasted_pct, 2),
        "savings_category":            savings_category,
        "savings_direction":           savings_direction,
        "model_r2_pct":                88.82,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> dict:
    """
    Return metadata about the production forecast model.
    """

    model = get_model()

    return {
        "model_type":          type(model).__name__,
        "model_file":          MODEL_PATH.name,
        "target":              "NEXT_YEAR_SAVINGS_RATE",
        "features":            MODEL_FEATURES,
        "validation_r2_pct":   88.82,
        "validation_mae":      0.008054,
        "validation_rmse":     0.010574,
        "train_years":         "2018–2022",
        "validation_year":     2023,
        "savings_categories": {
            "HIGH SAVINGS":     "forecast >= 3.0%",
            "MODERATE SAVINGS": "0.0% <= forecast < 3.0%",
            "LOW SAVINGS":      "forecast < 0.0%",
        },
    }
