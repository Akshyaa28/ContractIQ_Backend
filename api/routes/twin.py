from fastapi import APIRouter, HTTPException, status

from api.schemas.twin import (
    TwinRequest,
    TwinResponse,
    TwinModelInfoResponse,
)
from api.services.twin_service import find_similar_twins, get_model_info


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/twin",
    tags=["Similar Twin Matching"],
)


# ============================================================
# FIND TWINS
# ============================================================

@router.post(
    "/find",
    response_model=TwinResponse,
    summary="Find Top-5 Similar Twins for an ACO",
    description=(
        "Submit an ACO's 8 feature values and receive its top-5 most "
        "similar historical ACO-year peers, ranked by weighted Euclidean "
        "similarity. Includes savings rate outcomes and outperformance flags."
    ),
    status_code=status.HTTP_200_OK,
)
def find_twins(request: TwinRequest) -> TwinResponse:
    """
    Find the 5 most similar historical ACO-year observations.

    - Self-match excluded (if ACO_ID provided)
    - Weighted Euclidean distance on 8 standardised features
    - Outperforming = twin savings rate above Top-5 median
    """

    try:
        result = find_similar_twins(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Twin model file not found: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during twin matching: {exc}",
        ) from exc

    return TwinResponse(**result)


# ============================================================
# MODEL INFO
# ============================================================

@router.get(
    "/model/info",
    response_model=TwinModelInfoResponse,
    summary="Twin Model Information",
    description=(
        "Returns metadata about the twin matching model: "
        "algorithm, features, reference set, and configuration."
    ),
    status_code=status.HTTP_200_OK,
)
def twin_model_info() -> TwinModelInfoResponse:
    """Retrieve twin model metadata."""

    try:
        info = get_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Twin model file not found: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve twin model info: {exc}",
        ) from exc

    return TwinModelInfoResponse(**info)
