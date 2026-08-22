"""
Quality prediction endpoint that takes ACO_ID + Year_T,
looks up the full feature set from the database,
runs the quality model, and stores both input and output.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Any, Dict, Optional

from api.models.database import get_db
from api.models.user import User
from api.models.quality_data import QualityData, QualityPredictionInput, QualityPredictionResult
from api.services.quality_service import predict_quality
from api.middleware.auth_middleware import get_current_user


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/quality",
    tags=["Quality Score Prediction"],
)


# ============================================================
# SCHEMAS
# ============================================================

class QualityByAcoRequest(BaseModel):
    """Frontend sends just ACO_ID + Year."""
    aco_id: str = Field(..., description="ACO identifier (e.g. 'A00001')", examples=["A00001"])
    year_t: int = Field(..., description="Performance year (e.g. 2021)", examples=[2021])

    model_config = {
        "json_schema_extra": {
            "example": {"aco_id": "A00001", "year_t": 2021}
        }
    }


class QualityByAcoResponse(BaseModel):
    """Response with prediction result."""
    input_id: str
    aco_id: str
    year_t: int
    predicted_quality_score: float
    quality_band: str


# ============================================================
# ENDPOINT: Predict by ACO ID + Year
# ============================================================

@router.post(
    "/predict-by-aco",
    response_model=QualityByAcoResponse,
    summary="Predict quality score by ACO ID + Year",
    description=(
        "Send an ACO_ID and Year_T. The backend looks up the full "
        "feature set from the quality_data table, runs the quality model, "
        "and returns the predicted next-year quality score. "
        "Both inputs and outputs are stored in PostgreSQL."
    ),
    status_code=status.HTTP_200_OK,
)
def predict_quality_by_aco(
    request: QualityByAcoRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QualityByAcoResponse:

    # ──────────────────────────────────────────────────────────
    # STEP 1: Look up the row in quality_data table
    # ──────────────────────────────────────────────────────────

    row = (
        db.query(QualityData)
        .filter(
            QualityData.aco_id == request.aco_id,
            QualityData.year_t == request.year_t,
        )
        .first()
    )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for ACO_ID='{request.aco_id}' and Year_T={request.year_t}. "
                   f"Check that this combination exists in the quality dataset.",
        )

    # ──────────────────────────────────────────────────────────
    # STEP 2: Build the 32-feature payload for the model
    # ──────────────────────────────────────────────────────────

    features = {
        "Year_T": row.year_t,
        "Primary_State": row.primary_state,
        "Revenue_Category": row.revenue_category,
        "Track": row.track,
        "Agreement_Period_Num": row.agreement_period_num,
        "Policy_Version": row.policy_version,
        "COVID_Period": row.covid_period,
        "Beneficiary_Count": row.beneficiary_count,
        "Hospital_Count": row.hospital_count,
        "PCP_Count": row.pcp_count,
        "Specialist_Count": row.specialist_count,
        "Risk_Score": row.risk_score,
        "Chronic_Disease_Rate_Pct": row.chronic_disease_rate_pct,
        "Current_Quality_Score": row.current_quality_score,
        "Previous_Quality_Score": row.previous_quality_score,
        "Readmission_Rate_Pct": row.readmission_rate_pct,
        "Admission_Rate_Per_1000": row.admission_rate_per_1000,
        "ED_Visit_Rate_Per_1000": row.ed_visit_rate_per_1000,
        "Preventable_Admission_Rate_Per_1000": row.preventable_admission_rate_per_1000,
        "Patient_Experience_Score": row.patient_experience_score,
        "Diabetes_Control_Rate_Pct": row.diabetes_control_rate_pct,
        "Blood_Pressure_Control_Rate_Pct": row.blood_pressure_control_rate_pct,
        "Preventive_Screening_Rate_Pct": row.preventive_screening_rate_pct,
        "Followup_Compliance_Rate_Pct": row.followup_compliance_rate_pct,
        "Expenditure_Per_Beneficiary": row.expenditure_per_beneficiary,
        "Benchmark_Per_Beneficiary": row.benchmark_per_beneficiary,
        "Total_Expenditure": row.total_expenditure,
        "Benchmark_Expenditure": row.benchmark_expenditure,
        "Savings_Amount": row.savings_amount,
        "Savings_Rate": row.savings_rate,
        "Final_Share_Rate": row.final_share_rate,
        "Earned_Savings_Loss": row.earned_savings_loss,
    }

    # ──────────────────────────────────────────────────────────
    # STEP 3: Run the quality model
    # ──────────────────────────────────────────────────────────

    try:
        prediction = predict_quality(features)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality model prediction failed: {exc}",
        ) from exc

    # ──────────────────────────────────────────────────────────
    # STEP 4: Store input record
    # ──────────────────────────────────────────────────────────

    input_record = QualityPredictionInput(
        user_id=user.id,
        aco_id=request.aco_id,
        year_t=request.year_t,
    )
    db.add(input_record)
    db.flush()

    # ──────────────────────────────────────────────────────────
    # STEP 5: Store result record
    # ──────────────────────────────────────────────────────────

    result_record = QualityPredictionResult(
        input_id=input_record.id,
        user_id=user.id,
        aco_id=request.aco_id,
        year_t=request.year_t,
        features_json=features,
        result_json=prediction,
    )
    db.add(result_record)
    db.commit()
    db.refresh(input_record)
    db.refresh(result_record)

    # ──────────────────────────────────────────────────────────
    # STEP 6: Return to frontend
    # ──────────────────────────────────────────────────────────

    return QualityByAcoResponse(
        input_id=str(input_record.id),
        aco_id=request.aco_id,
        year_t=request.year_t,
        predicted_quality_score=prediction["predicted_quality_score"],
        quality_band=prediction["quality_band"],
    )
