from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# risk_service.py
#   I:\ContractIQ(!)\api\services\risk_service.py
#
# parents[0] = services
# parents[1] = api
# parents[2] = ContractIQ(!)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "risk_prediction"
    / "risk_random_forest_85_87.joblib"
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

# This is the validated production threshold
RISK_THRESHOLD = 0.23


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def get_model():
    """
    Load the Random Forest model once and reuse it.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Risk model was not found:\n{MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

        # Validate model feature configuration
        if hasattr(_model, "feature_names_in_"):

            trained_features = list(_model.feature_names_in_)

            if trained_features != MODEL_FEATURES:
                raise ValueError(
                    "Model feature configuration does not match API configuration.\n"
                    f"Expected: {MODEL_FEATURES}\n"
                    f"Model has: {trained_features}"
                )

    return _model


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability: float) -> str:
    """
    Convert probability into LOW / MEDIUM / HIGH.

    Thresholds are calibrated against the actual probability
    distribution of the production model (max ~0.865, p99 ~0.559).
    Using 0.50 as the HIGH cutoff would classify only 1.7% of ACOs
    as HIGH — too narrow to be actionable. 0.35 splits the risk
    population into meaningful thirds.

    LOW:
        probability < 0.23

    MEDIUM:
        0.23 <= probability < 0.35

    HIGH:
        probability >= 0.35
    """

    if probability < RISK_THRESHOLD:
        return "LOW"

    if probability < 0.35:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# PREDICTION
# ============================================================

def predict_risk(input_data: dict) -> dict:
    """
    Generate a risk prediction from the trained Random Forest.
    """

    model = get_model()

    # --------------------------------------------------------
    # Create DataFrame in EXACT model feature order
    # --------------------------------------------------------

    feature_data = {
        feature: [input_data[feature]]
        for feature in MODEL_FEATURES
    }

    X = pd.DataFrame(
        feature_data,
        columns=MODEL_FEATURES
    )

    # --------------------------------------------------------
    # Generate probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)

    # Probability of class 1 = Risk
    risk_probability = float(probabilities[0][1])

    # --------------------------------------------------------
    # Apply production threshold
    # --------------------------------------------------------

    prediction = int(
        risk_probability >= RISK_THRESHOLD
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    risk_level = get_risk_level(
        risk_probability
    )

    # --------------------------------------------------------
    # Risk label
    # --------------------------------------------------------

    if prediction == 1:
        risk_label = "RISK"
    else:
        risk_label = "NON-RISK"

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return {
        "risk_probability": round(
            risk_probability,
            6
        ),

        "risk_probability_pct": round(
            risk_probability * 100,
            2
        ),

        "prediction": prediction,

        "risk_label": risk_label,

        "risk_level": risk_level,

        "threshold": RISK_THRESHOLD,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> dict:
    """
    Return information about the production model.
    """

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