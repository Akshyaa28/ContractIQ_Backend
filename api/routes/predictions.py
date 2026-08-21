from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.models.prediction import PredictionInput, PredictionResult
from api.models.user import User
from api.schemas.prediction import (
    PredictionRunRequest,
    PredictionRunResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
)
from api.services.prediction_orchestrator import run_prediction
from api.middleware.auth_middleware import get_current_user


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


# ============================================================
# RUN PREDICTION
# ============================================================

@router.post(
    "/run",
    response_model=PredictionRunResponse,
    summary="Run a prediction (risk / forecast / twin)",
    description=(
        "Submit an ACO ID + 8 performance inputs + analysis type. "
        "The backend stores the inputs, runs the selected model, "
        "stores the result, and returns the prediction. "
        "Both input and output are saved in PostgreSQL for audit."
    ),
    status_code=status.HTTP_200_OK,
)
def run_prediction_endpoint(
    request: PredictionRunRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PredictionRunResponse:
    """
    Unified prediction endpoint for the CMS portal.

    Flow:
    1. Stores inputs in `prediction_inputs` table
    2. Routes to the selected model (risk/forecast/twin)
    3. Stores result in `prediction_results` table
    4. Returns the full prediction to the frontend
    """

    try:
        result = run_prediction(
            db=db,
            aco_id=request.aco_id,
            analysis_type=request.analysis_type,
            n_ab=request.n_ab,
            previous_savings_rate=request.previous_savings_rate,
            previous_quality_score=request.previous_quality_score,
            previous_performance_gap=request.previous_performance_gap,
            expenditure_growth=request.expenditure_growth,
            benchmark_growth=request.benchmark_growth,
            beneficiary_growth=request.beneficiary_growth,
            quality_change=request.quality_change,
            user_id=str(user.id),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model file not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionRunResponse(**result)


# ============================================================
# PREDICTION HISTORY
# ============================================================

@router.get(
    "/history",
    response_model=PredictionHistoryResponse,
    summary="Get prediction history",
    description=(
        "Retrieve past predictions for the authenticated user. "
        "Optionally filter by aco_id or analysis_type."
    ),
    status_code=status.HTTP_200_OK,
)
def get_prediction_history(
    aco_id: Optional[str] = Query(None, description="Filter by ACO ID"),
    analysis_type: Optional[str] = Query(None, description="Filter by type: risk/forecast/twin"),
    limit: int = Query(50, ge=1, le=200, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PredictionHistoryResponse:
    """
    Returns the prediction history for the current user.
    Results are ordered by most recent first.
    """

    query = (
        db.query(PredictionResult)
        .filter(PredictionResult.user_id == user.id)
    )

    if aco_id:
        query = query.filter(PredictionResult.aco_id == aco_id)

    if analysis_type:
        query = query.filter(PredictionResult.analysis_type == analysis_type)

    total = query.count()

    results = (
        query
        .order_by(PredictionResult.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = [
        PredictionHistoryItem(
            input_id=str(r.input_id),
            result_id=str(r.id),
            aco_id=r.aco_id,
            analysis_type=r.analysis_type,
            prediction=r.result_json,
            created_at=r.created_at.isoformat(),
        )
        for r in results
    ]

    return PredictionHistoryResponse(total=total, items=items)
