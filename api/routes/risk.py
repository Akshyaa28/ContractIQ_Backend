from typing import List

from fastapi import APIRouter, HTTPException, status

from api.schemas.risk import (
    RiskPredictionRequest,
    RiskPredictionResponse,
    RiskBatchRequest,
    RiskBatchResponse,
    ModelInfoResponse,
)
from api.services.risk_service import predict_risk, get_model_info


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/risk",
    tags=["Risk Prediction"],
)


# ============================================================
# SINGLE PREDICTION
# ============================================================

@router.post(
    "/predict",
    response_model=RiskPredictionResponse,
    summary="Predict ACO Risk",
    description=(
        "Submit one ACO's 8 feature values and receive a risk "
        "probability, binary prediction, and risk level classification "
        "using the production Random Forest model (threshold = 0.23)."
    ),
    status_code=status.HTTP_200_OK,
)
def predict(request: RiskPredictionRequest) -> RiskPredictionResponse:
    """
    Run the production risk model on a single ACO record.

    Returns:
    - **risk_probability**: raw probability (0–1) from the model
    - **risk_probability_pct**: probability as a percentage (0–100)
    - **prediction**: 1 = RISK, 0 = NON-RISK
    - **risk_label**: "RISK" or "NON-RISK"
    - **risk_level**: "LOW" | "MEDIUM" | "HIGH"
    - **threshold**: the production classification threshold (0.23)
    """

    try:
        result = predict_risk(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model file not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during prediction: {exc}",
        ) from exc

    return RiskPredictionResponse(**result)


# ============================================================
# BATCH PREDICTION
# ============================================================

@router.post(
    "/predict/batch",
    response_model=RiskBatchResponse,
    summary="Batch Predict ACO Risk",
    description=(
        "Submit up to 500 ACO records in a single request. "
        "Each record is scored independently and returned in the "
        "same order as submitted."
    ),
    status_code=status.HTTP_200_OK,
)
def predict_batch(request: RiskBatchRequest) -> RiskBatchResponse:
    """
    Run the production risk model across multiple ACO records at once.

    - Maximum 500 records per request.
    - Results are returned in the same order as the input.
    - **total**: total records submitted
    - **risk_count**: number predicted as RISK (prediction = 1)
    - **non_risk_count**: number predicted as NON-RISK (prediction = 0)
    - **predictions**: list of individual `RiskPredictionResponse` objects
    """

    predictions: List[RiskPredictionResponse] = []

    try:
        for record in request.records:
            result = predict_risk(record.model_dump())
            predictions.append(RiskPredictionResponse(**result))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model file not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during batch prediction: {exc}",
        ) from exc

    risk_count = sum(1 for p in predictions if p.prediction == 1)

    return RiskBatchResponse(
        total=len(predictions),
        risk_count=risk_count,
        non_risk_count=len(predictions) - risk_count,
        predictions=predictions,
    )


# ============================================================
# MODEL INFO
# ============================================================

@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description=(
        "Returns metadata about the loaded production model: "
        "type, filename, threshold, feature list, and risk level definitions."
    ),
    status_code=status.HTTP_200_OK,
)
def model_info() -> ModelInfoResponse:
    """
    Retrieve production model metadata.

    Useful for confirming which model version is active and
    verifying the feature/threshold configuration.
    """

    try:
        info = get_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model file not found: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve model info: {exc}",
        ) from exc

    return ModelInfoResponse(**info)
