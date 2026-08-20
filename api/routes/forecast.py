from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, status

from api.schemas.forecast import (
    ForecastRequest,
    ForecastResponse,
    ForecastBatchRequest,
    ForecastBatchResponse,
    ForecastModelInfoResponse,
)
from api.services.forecast_service import predict_forecast, get_model_info


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/forecast",
    tags=["Savings Forecast"],
)


# ============================================================
# SINGLE FORECAST
# ============================================================

@router.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Forecast ACO Savings Rate",
    description=(
        "Submit one ACO's 8 feature values and receive a predicted "
        "next-year savings rate, a savings category classification, "
        "and savings direction using the production Random Forest model "
        "(validation R² = 88.82% on 2023 hold-out)."
    ),
    status_code=status.HTTP_200_OK,
)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Run the production savings forecast model on a single ACO record.

    Returns:
    - **forecasted_savings_rate**: predicted rate as a decimal (e.g. 0.032)
    - **forecasted_savings_rate_pct**: predicted rate as a percentage (e.g. 3.20)
    - **savings_category**: "HIGH SAVINGS" | "MODERATE SAVINGS" | "LOW SAVINGS"
    - **savings_direction**: "POSITIVE" | "NEGATIVE"
    - **model_r2_pct**: model validation R² (88.82%)
    """

    try:
        result = predict_forecast(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Forecast model file not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Forecast error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during forecast: {exc}",
        ) from exc

    return ForecastResponse(**result)


# ============================================================
# BATCH FORECAST
# ============================================================

@router.post(
    "/predict/batch",
    response_model=ForecastBatchResponse,
    summary="Batch Forecast ACO Savings Rates",
    description=(
        "Submit up to 500 ACO records in a single request. "
        "Each record is forecast independently and returned in the "
        "same order as submitted, with aggregate category counts "
        "and the mean forecasted savings rate."
    ),
    status_code=status.HTTP_200_OK,
)
def forecast_batch(request: ForecastBatchRequest) -> ForecastBatchResponse:
    """
    Run the production savings forecast model across multiple ACO records.

    - Maximum 500 records per request.
    - Results are returned in submission order.
    - **total**: total records submitted
    - **high_savings_count**: records forecast as HIGH SAVINGS (≥3%)
    - **moderate_savings_count**: records forecast as MODERATE SAVINGS (0–3%)
    - **low_savings_count**: records forecast as LOW SAVINGS (<0%)
    - **mean_forecasted_savings_rate_pct**: average forecast across all records
    - **forecasts**: individual ForecastResponse objects
    """

    forecasts: List[ForecastResponse] = []

    try:
        for record in request.records:
            result = predict_forecast(record.model_dump())
            forecasts.append(ForecastResponse(**result))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Forecast model file not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Forecast error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during batch forecast: {exc}",
        ) from exc

    high_count     = sum(1 for f in forecasts if f.savings_category == "HIGH SAVINGS")
    moderate_count = sum(1 for f in forecasts if f.savings_category == "MODERATE SAVINGS")
    low_count      = sum(1 for f in forecasts if f.savings_category == "LOW SAVINGS")

    mean_pct = round(
        float(np.mean([f.forecasted_savings_rate_pct for f in forecasts])),
        2,
    )

    return ForecastBatchResponse(
        total=len(forecasts),
        high_savings_count=high_count,
        moderate_savings_count=moderate_count,
        low_savings_count=low_count,
        mean_forecasted_savings_rate_pct=mean_pct,
        forecasts=forecasts,
    )


# ============================================================
# MODEL INFO
# ============================================================

@router.get(
    "/model/info",
    response_model=ForecastModelInfoResponse,
    summary="Forecast Model Information",
    description=(
        "Returns metadata about the loaded production forecast model: "
        "type, filename, validation metrics, feature list, "
        "and savings category definitions."
    ),
    status_code=status.HTTP_200_OK,
)
def forecast_model_info() -> ForecastModelInfoResponse:
    """
    Retrieve production forecast model metadata.

    Useful for confirming which model version is active and
    understanding its validation performance.
    """

    try:
        info = get_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Forecast model file not found: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve forecast model info: {exc}",
        ) from exc

    return ForecastModelInfoResponse(**info)
