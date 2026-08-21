"""
Prediction Orchestrator
-----------------------
Routes prediction requests to the correct model service,
stores inputs and outputs in PostgreSQL, and returns results.
"""

from typing import Optional

from sqlalchemy.orm import Session

from api.models.prediction import PredictionInput, PredictionResult
from api.services.risk_service import predict_risk
from api.services.forecast_service import predict_forecast
from api.services.twin_service import find_similar_twins


def run_prediction(
    db: Session,
    aco_id: str,
    analysis_type: str,
    n_ab: float,
    previous_savings_rate: float,
    previous_quality_score: float,
    previous_performance_gap: float,
    expenditure_growth: float,
    benchmark_growth: float,
    beneficiary_growth: float,
    quality_change: float,
    user_id: Optional[str] = None,
) -> dict:
    """
    1. Store the input in prediction_inputs table
    2. Route to the correct model
    3. Store the result in prediction_results table
    4. Return input_id, result_id, and prediction
    """

    # ──────────────────────────────────────────────────────────
    # STEP 1: Store input
    # ──────────────────────────────────────────────────────────

    input_record = PredictionInput(
        user_id=user_id,
        aco_id=aco_id,
        analysis_type=analysis_type,
        n_ab=n_ab,
        previous_savings_rate=previous_savings_rate,
        previous_quality_score=previous_quality_score,
        previous_performance_gap_pct=previous_performance_gap,
        expenditure_growth_pct=expenditure_growth,
        benchmark_growth_pct=benchmark_growth,
        beneficiary_growth_pct=beneficiary_growth,
        quality_change=quality_change,
    )

    db.add(input_record)
    db.flush()  # get input_record.id without committing yet

    # ──────────────────────────────────────────────────────────
    # STEP 2: Build model payload and route to correct model
    # ──────────────────────────────────────────────────────────

    # Convert frontend percentage values to decimal for models
    # Frontend sends: 3.5 (meaning 3.5%)
    # Models expect:  0.035 (decimal)
    model_payload = {
        "N_AB": n_ab,
        "PREVIOUS_SAVINGS_RATE": previous_savings_rate / 100.0,
        "PREVIOUS_QUALITY_SCORE": previous_quality_score,
        "PREVIOUS_PERFORMANCE_GAP_PCT": previous_performance_gap / 100.0,
        "EXPENDITURE_GROWTH_PCT": expenditure_growth / 100.0,
        "BENCHMARK_GROWTH_PCT": benchmark_growth / 100.0,
        "BENEFICIARY_GROWTH_PCT": beneficiary_growth / 100.0,
        "QUALITY_CHANGE": quality_change,
    }

    if analysis_type == "risk":
        prediction = predict_risk(model_payload)

    elif analysis_type == "forecast":
        prediction = predict_forecast(model_payload)

    elif analysis_type == "twin":
        # Twin service also accepts ACO_ID for self-exclusion
        twin_payload = {**model_payload, "ACO_ID": aco_id}
        prediction = find_similar_twins(twin_payload)

    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    # ──────────────────────────────────────────────────────────
    # STEP 3: Store result
    # ──────────────────────────────────────────────────────────

    result_record = PredictionResult(
        input_id=input_record.id,
        user_id=user_id,
        aco_id=aco_id,
        analysis_type=analysis_type,
        result_json=prediction,
    )

    db.add(result_record)
    db.commit()
    db.refresh(input_record)
    db.refresh(result_record)

    # ──────────────────────────────────────────────────────────
    # STEP 4: Return
    # ──────────────────────────────────────────────────────────

    return {
        "input_id": str(input_record.id),
        "result_id": str(result_record.id),
        "aco_id": aco_id,
        "analysis_type": analysis_type,
        "prediction": prediction,
    }
