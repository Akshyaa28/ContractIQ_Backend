"""
Provider Risk — Lookup + What-If Simulation
============================================
ACO Portal endpoints for provider risk assessment and scenario simulation.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.models.provider_data import ProviderData
from api.services.provider_risk_service import predict_provider_risk, get_model_info, MODEL_FEATURES

router = APIRouter(
    prefix="/provider",
    tags=["Provider Risk (ACO Portal)"],
)

# 5 key simulation metrics
SIMULATION_METRICS = [
    {"field": "ed_visits_vs_aco", "label": "ER Visits vs ACO Average", "min": -749, "max": 1695, "impact_pct": 12.9},
    {"field": "quality_vs_aco", "label": "Quality Score vs ACO Average", "min": -18.2, "max": 19.3, "impact_pct": 9.3},
    {"field": "per_capita_vs_aco", "label": "Per Capita Cost vs ACO Average", "min": -34534, "max": 59729, "impact_pct": 8.3},
    {"field": "admissions_vs_aco", "label": "Admissions vs ACO Average", "min": -357, "max": 888, "impact_pct": 7.9},
    {"field": "snf_admission_vs_aco", "label": "SNF Admissions vs ACO Average", "min": -327, "max": 534, "impact_pct": 7.9},
]

SIMULATION_FIELDS = [m["field"] for m in SIMULATION_METRICS]


# ============================================================
# SCHEMAS
# ============================================================

class ProviderLookupRequest(BaseModel):
    provider_id: str = Field(..., description="Provider ID (e.g. PRV_SYN_ACO_000002_001383)")
    performance_year: int = Field(..., description="Year (2021-2024)")


class ProviderLookupResponse(BaseModel):
    provider_id: str
    aco_id: str
    performance_year: int
    provider_type: str
    specialty: str
    beneficiary_count: int
    predicted_risk_tier: str
    confidence: float
    class_probabilities: Dict[str, float]
    show_simulator: bool
    current_metrics: Dict[str, float]


class SimulationRequest(BaseModel):
    provider_id: str = Field(..., description="Provider ID")
    performance_year: int = Field(..., description="Year")
    modifications: Dict[str, float] = Field(..., description="Modified metric values")


class SimulationResponse(BaseModel):
    provider_id: str
    original: Dict
    modified: Dict
    risk_change: str
    modifications_applied: Dict


class ProviderModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_type: str
    model_file: str
    target: str
    classes: List[str]
    features: List[str]
    total_features: int
    accuracy: float
    macro_f1: float


# ============================================================
# HELPERS
# ============================================================

def row_to_features(row: ProviderData) -> dict:
    """Convert a DB row to a 41-feature dict for the model."""
    return {
        "performance_year": row.performance_year,
        "provider_type": row.provider_type,
        "specialty": row.specialty,
        "beneficiary_count": row.beneficiary_count,
        "hcc_risk_aged_dual": row.hcc_risk_aged_dual,
        "admissions_per_1000": row.admissions_per_1000,
        "inpatient_expenditure": row.inpatient_expenditure,
        "outpatient_expenditure": row.outpatient_expenditure,
        "professional_expenditure": row.professional_expenditure,
        "snf_expenditure": row.snf_expenditure,
        "home_health_expenditure": row.home_health_expenditure,
        "ed_visits_per_1000": row.ed_visits_per_1000,
        "snf_admission_rate": row.snf_admission_rate,
        "pcp_visits_per_1000": row.pcp_visits_per_1000,
        "specialist_visits_per_1000": row.specialist_visits_per_1000,
        "hcc_risk_aged_nondual": row.hcc_risk_aged_nondual,
        "hcc_risk_disabled": row.hcc_risk_disabled,
        "hcc_risk_esrd": row.hcc_risk_esrd,
        "pcp_visits_per_1000_missing": row.pcp_visits_per_1000_missing,
        "specialist_visits_per_1000_missing": row.specialist_visits_per_1000_missing,
        "aco_avg_total_expenditure": row.aco_avg_total_expenditure,
        "aco_avg_per_capita_expenditure": row.aco_avg_per_capita_expenditure,
        "aco_avg_quality_score": row.aco_avg_quality_score,
        "aco_avg_admissions_per_1000": row.aco_avg_admissions_per_1000,
        "aco_avg_ed_visits_per_1000": row.aco_avg_ed_visits_per_1000,
        "aco_avg_snf_admission_rate": row.aco_avg_snf_admission_rate,
        "aco_avg_hcc_risk_aged_dual": row.aco_avg_hcc_risk_aged_dual,
        "aco_std_total_expenditure": row.aco_std_total_expenditure,
        "aco_std_per_capita_expenditure": row.aco_std_per_capita_expenditure,
        "aco_std_quality_score": row.aco_std_quality_score,
        "aco_std_admissions_per_1000": row.aco_std_admissions_per_1000,
        "aco_std_ed_visits_per_1000": row.aco_std_ed_visits_per_1000,
        "aco_std_snf_admission_rate": row.aco_std_snf_admission_rate,
        "aco_std_hcc_risk_aged_dual": row.aco_std_hcc_risk_aged_dual,
        "cost_vs_aco": row.cost_vs_aco,
        "per_capita_vs_aco": row.per_capita_vs_aco,
        "quality_vs_aco": row.quality_vs_aco,
        "admissions_vs_aco": row.admissions_vs_aco,
        "ed_visits_vs_aco": row.ed_visits_vs_aco,
        "snf_admission_vs_aco": row.snf_admission_vs_aco,
        "risk_vs_aco": row.risk_vs_aco,
    }


def determine_risk_change(original: str, modified: str) -> str:
    """Determine if risk improved, worsened, or stayed the same."""
    levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    if levels.get(modified, 2) < levels.get(original, 2):
        return "improved"
    elif levels.get(modified, 2) > levels.get(original, 2):
        return "worsened"
    return "unchanged"


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/lookup", response_model=ProviderLookupResponse, summary="Lookup provider and predict risk")
def lookup_provider(request: ProviderLookupRequest, db: Session = Depends(get_db)):
    """Fetch provider from DB by ID + year, run model, return risk + current metrics."""

    row = db.query(ProviderData).filter(
        ProviderData.provider_id == request.provider_id,
        ProviderData.performance_year == request.performance_year,
    ).first()

    if not row:
        # Check if provider exists in other years
        other = db.query(ProviderData.performance_year).filter(
            ProviderData.provider_id == request.provider_id
        ).distinct().all()

        if other:
            years = sorted([y[0] for y in other])
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {request.provider_id} not found for year {request.performance_year}. Available years: {years}",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {request.provider_id} not found in any year.",
        )

    features = row_to_features(row)
    prediction = predict_provider_risk(features)

    current_metrics = {field: features[field] for field in SIMULATION_FIELDS}

    return ProviderLookupResponse(
        provider_id=row.provider_id,
        aco_id=row.aco_id,
        performance_year=row.performance_year,
        provider_type=row.provider_type,
        specialty=row.specialty,
        beneficiary_count=row.beneficiary_count,
        predicted_risk_tier=prediction["predicted_risk_tier"],
        confidence=prediction["confidence"],
        class_probabilities=prediction["class_probabilities"],
        show_simulator=prediction["predicted_risk_tier"] in ("MEDIUM", "HIGH"),
        current_metrics=current_metrics,
    )


@router.post("/simulate", response_model=SimulationResponse, summary="What-If simulation")
def simulate_what_if(request: SimulationRequest, db: Session = Depends(get_db)):
    """Modify up to 5 metrics, re-predict, compare original vs modified."""

    row = db.query(ProviderData).filter(
        ProviderData.provider_id == request.provider_id,
        ProviderData.performance_year == request.performance_year,
    ).first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {request.provider_id} not found for year {request.performance_year}.",
        )

    # Validate modifications — only allow the 5 simulation fields
    invalid_fields = [f for f in request.modifications if f not in SIMULATION_FIELDS]
    if invalid_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid modification fields: {invalid_fields}. Allowed: {SIMULATION_FIELDS}",
        )

    # Original prediction
    original_features = row_to_features(row)
    original_prediction = predict_provider_risk(original_features)

    # Modified prediction
    modified_features = original_features.copy()
    modifications_applied = {}

    for field, new_value in request.modifications.items():
        original_value = original_features[field]
        modified_features[field] = new_value
        modifications_applied[field] = {
            "original": round(original_value, 4) if original_value else 0,
            "modified": round(new_value, 4),
            "change": round(new_value - (original_value or 0), 4),
        }

    modified_prediction = predict_provider_risk(modified_features)

    return SimulationResponse(
        provider_id=request.provider_id,
        original={
            "risk_tier": original_prediction["predicted_risk_tier"],
            "confidence": original_prediction["confidence"],
            "class_probabilities": original_prediction["class_probabilities"],
        },
        modified={
            "risk_tier": modified_prediction["predicted_risk_tier"],
            "confidence": modified_prediction["confidence"],
            "class_probabilities": modified_prediction["class_probabilities"],
        },
        risk_change=determine_risk_change(
            original_prediction["predicted_risk_tier"],
            modified_prediction["predicted_risk_tier"],
        ),
        modifications_applied=modifications_applied,
    )


@router.get("/simulation-config", summary="Get What-If slider configuration")
def get_simulation_config():
    """Return the 5 key metrics configuration for the frontend sliders."""
    return {"metrics": SIMULATION_METRICS}


@router.get("/model/info", response_model=ProviderModelInfoResponse, summary="Provider risk model info")
def model_info():
    """Return provider risk model metadata."""
    try:
        info = get_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return ProviderModelInfoResponse(**info)
