from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, status

from api.schemas.quality import (
    QualityPredictionRequest,
    QualityPredictionResponse,
    QualityBatchRequest,
    QualityBatchResponse,
    QualityModelInfoResponse,
)
from api.services.quality_service import predict_quality, get_model_info


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/quality",
    tags=["Quality Score Prediction"],
)


# ============================================================
# SINGLE PREDICTION
# ============================================================

@router.post(
    "/predict",
    response_model=QualityPredictionResponse,
    summary="Predict ACO Next-Year Quality Score",
    description=(
        "Submit an ACO's 32 year-T feature values and receive a predicted "
        "next-year quality score (0–100) plus a quality band classification "
        "using the production Ridge Pipeline "
        "(test R² = 83.38% on 2024 hold-out).\n\n"
        "**Disclosure:** trained on a SYNTHETIC dataset — not real CMS data."
    ),
    status_code=status.HTTP_200_OK,
)
def predict(request: QualityPredictionRequest) -> QualityPredictionResponse:
    """
    Run the production quality model on a single ACO record.

    Returns:
    - **predicted_quality_score**: predicted next-year score (0–100)
    - **quality_band**: "HIGH" (≥90) | "MODERATE" (75–90) | "LOW" (<75)
    - **model_r2_pct**: model test R² (83.38%)
    - **model_mae**: mean absolute error (2.461 quality points)
    """

    try:
        result = predict_quality(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Quality model file not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during quality prediction: {exc}",
        ) from exc

    return QualityPredictionResponse(**result)


# ============================================================
# BATCH PREDICTION
# ============================================================

@router.post(
    "/predict/batch",
    response_model=QualityBatchResponse,
    summary="Batch Predict ACO Quality Scores",
    description=(
        "Submit up to 500 ACO records in a single request. "
        "Each record is scored independently and returned in submission order, "
        "with aggregate band counts and the mean predicted score."
    ),
    status_code=status.HTTP_200_OK,
)
def predict_batch(request: QualityBatchRequest) -> QualityBatchResponse:
    """
    Run quality predictions across multiple ACO records.

    - Maximum 500 records per request.
    - **total**: total records submitted
    - **high_count**: records predicted HIGH (≥90)
    - **moderate_count**: records predicted MODERATE (75–90)
    - **low_count**: records predicted LOW (<75)
    - **mean_predicted_score**: average predicted quality score
    - **predictions**: individual QualityPredictionResponse objects
    """

    predictions: List[QualityPredictionResponse] = []

    try:
        for record in request.records:
            result = predict_quality(record.model_dump())
            predictions.append(QualityPredictionResponse(**result))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Quality model file not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during batch quality prediction: {exc}",
        ) from exc

    high_count     = sum(1 for p in predictions if p.quality_band == "HIGH")
    moderate_count = sum(1 for p in predictions if p.quality_band == "MODERATE")
    low_count      = sum(1 for p in predictions if p.quality_band == "LOW")
    mean_score     = round(
        float(np.mean([p.predicted_quality_score for p in predictions])), 2
    )

    return QualityBatchResponse(
        total=len(predictions),
        high_count=high_count,
        moderate_count=moderate_count,
        low_count=low_count,
        mean_predicted_score=mean_score,
        predictions=predictions,
    )


# ============================================================
# MODEL INFO
# ============================================================

@router.get(
    "/model/info",
    response_model=QualityModelInfoResponse,
    summary="Quality Model Information",
    description=(
        "Returns metadata about the loaded quality model: "
        "pipeline steps, features, validation metrics, "
        "quality band definitions, and data disclosure."
    ),
    status_code=status.HTTP_200_OK,
)
def quality_model_info() -> QualityModelInfoResponse:
    """Retrieve production quality model metadata."""

    try:
        info = get_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Quality model file not found: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve quality model info: {exc}",
        ) from exc

    return QualityModelInfoResponse(**info)
