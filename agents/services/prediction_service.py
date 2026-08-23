"""
Prediction Retrieval Service
============================
Fetches model inputs + outputs from PostgreSQL for agent consumption.
Validates ownership, existence, and consistency.
"""

from typing import Optional
from sqlalchemy.orm import Session

from api.models.prediction import PredictionInput, PredictionResult
from api.models.quality_data import QualityPredictionInput, QualityPredictionResult


def get_prediction_context(db: Session, input_id: str, user_id: str) -> dict:
    """
    Retrieve prediction input + result from PostgreSQL.
    Returns: { analysis_type, aco_id, model_input, model_output }
    Raises ValueError on any validation failure.
    """
    pred_input = db.query(PredictionInput).filter(PredictionInput.id == input_id).first()
    if not pred_input:
        raise ValueError(f"Prediction input not found: {input_id}")

    if pred_input.user_id and str(pred_input.user_id) != user_id:
        raise ValueError("Access denied: this prediction belongs to another user.")

    pred_result = db.query(PredictionResult).filter(PredictionResult.input_id == input_id).first()
    if not pred_result:
        raise ValueError(f"Prediction result not found for input_id: {input_id}")

    if pred_input.analysis_type != pred_result.analysis_type:
        raise ValueError(f"Analysis type mismatch: input={pred_input.analysis_type}, result={pred_result.analysis_type}")

    if not pred_result.result_json:
        raise ValueError("Prediction result_json is empty or invalid.")

    return {
        "analysis_type": pred_input.analysis_type,
        "aco_id": pred_input.aco_id,
        "model_input": {
            "n_ab": pred_input.n_ab,
            "previous_savings_rate": pred_input.previous_savings_rate,
            "previous_quality_score": pred_input.previous_quality_score,
            "previous_performance_gap_pct": pred_input.previous_performance_gap_pct,
            "expenditure_growth_pct": pred_input.expenditure_growth_pct,
            "benchmark_growth_pct": pred_input.benchmark_growth_pct,
            "beneficiary_growth_pct": pred_input.beneficiary_growth_pct,
            "quality_change": pred_input.quality_change,
        },
        "model_output": pred_result.result_json,
    }


def get_quality_prediction_context(db: Session, input_id: str, user_id: str) -> dict:
    """Retrieve quality prediction input + result from PostgreSQL."""
    pred_input = db.query(QualityPredictionInput).filter(QualityPredictionInput.id == input_id).first()
    if not pred_input:
        raise ValueError(f"Quality prediction input not found: {input_id}")

    if pred_input.user_id and str(pred_input.user_id) != user_id:
        raise ValueError("Access denied: this prediction belongs to another user.")

    pred_result = db.query(QualityPredictionResult).filter(QualityPredictionResult.input_id == input_id).first()
    if not pred_result:
        raise ValueError(f"Quality prediction result not found for input_id: {input_id}")

    return {
        "analysis_type": "quality",
        "aco_id": pred_input.aco_id,
        "year_t": pred_input.year_t,
        "model_input": pred_result.features_json,
        "model_output": pred_result.result_json,
    }
