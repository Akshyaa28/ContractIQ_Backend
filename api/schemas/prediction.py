from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# REQUEST
# ============================================================

class PredictionRunRequest(BaseModel):
    """
    Unified prediction request from the CMS portal.
    Includes ACO selection + 8 model inputs + analysis type.
    """

    aco_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Selected ACO identifier (e.g. 'A00001')",
        examples=["A00001"],
    )
    analysis_type: str = Field(
        ...,
        pattern="^(risk|forecast|twin)$",
        description="Type of analysis: 'risk', 'forecast', or 'twin'",
        examples=["risk"],
    )

    # 8 model inputs
    n_ab: float = Field(..., description="Number of attributed beneficiaries", examples=[12500.0])
    previous_savings_rate: float = Field(..., description="Previous savings rate (%)", examples=[3.5])
    previous_quality_score: float = Field(..., description="Previous quality score (0–100)", examples=[84.0])
    previous_performance_gap: float = Field(..., description="Previous performance gap (%)", examples=[-1.2])
    expenditure_growth: float = Field(..., description="Expenditure growth (%)", examples=[2.1])
    benchmark_growth: float = Field(..., description="Benchmark growth (%)", examples=[3.0])
    beneficiary_growth: float = Field(..., description="Beneficiary growth (%)", examples=[1.5])
    quality_change: float = Field(..., description="Quality score change (points)", examples=[2.0])

    model_config = {
        "json_schema_extra": {
            "example": {
                "aco_id": "A00001",
                "analysis_type": "risk",
                "n_ab": 12500,
                "previous_savings_rate": 3.5,
                "previous_quality_score": 84,
                "previous_performance_gap": -1.2,
                "expenditure_growth": 2.1,
                "benchmark_growth": 3.0,
                "beneficiary_growth": 1.5,
                "quality_change": 2.0,
            }
        }
    }


# ============================================================
# RESPONSE
# ============================================================

class PredictionRunResponse(BaseModel):
    """
    Response after running a prediction.
    Contains IDs for audit trail + the full model result.
    """

    input_id: str = Field(..., description="UUID of the stored input record")
    result_id: str = Field(..., description="UUID of the stored result record")
    aco_id: str = Field(..., description="ACO that was analyzed")
    analysis_type: str = Field(..., description="Type of analysis run")
    prediction: Dict[str, Any] = Field(
        ...,
        description="Full model prediction output (shape varies by analysis_type)",
    )


class PredictionHistoryItem(BaseModel):
    """One prediction history entry."""

    input_id: str
    result_id: str
    aco_id: str
    analysis_type: str
    prediction: Dict[str, Any]
    created_at: str

    model_config = {"from_attributes": True}


class PredictionHistoryResponse(BaseModel):
    """Paginated prediction history."""

    total: int
    items: List[PredictionHistoryItem]
