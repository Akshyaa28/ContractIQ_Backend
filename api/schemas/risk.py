from typing import Dict, List

from pydantic import BaseModel, Field


# ============================================================
# INPUT SCHEMAS
# ============================================================

class RiskPredictionRequest(BaseModel):
    """
    Input features required by the trained Dataset 1 Risk Model.

    All 8 features must be present and numeric. Values must not
    be null, infinite, or missing — the production dataset has
    been validated to contain no missing feature values.
    """

    N_AB: float = Field(
        ...,
        description="Number of attributed beneficiaries",
        examples=[5200.0],
    )
    PREVIOUS_SAVINGS_RATE: float = Field(
        ...,
        description=(
            "Prior-year savings rate. "
            "Negative values indicate the ACO spent above benchmark."
        ),
        examples=[0.012],
    )
    PREVIOUS_QUALITY_SCORE: float = Field(
        ...,
        description="Prior-year quality score (typical range 60–100)",
        examples=[87.5],
    )
    PREVIOUS_PERFORMANCE_GAP_PCT: float = Field(
        ...,
        description=(
            "Prior-year performance gap as a percentage. "
            "Top predictor in the production model."
        ),
        examples=[-0.034],
    )
    EXPENDITURE_GROWTH_PCT: float = Field(
        ...,
        description="Year-over-year expenditure growth percentage",
        examples=[0.021],
    )
    BENCHMARK_GROWTH_PCT: float = Field(
        ...,
        description="Year-over-year benchmark growth percentage",
        examples=[0.018],
    )
    BENEFICIARY_GROWTH_PCT: float = Field(
        ...,
        description="Year-over-year beneficiary count growth percentage",
        examples=[0.005],
    )
    QUALITY_CHANGE: float = Field(
        ...,
        description="Change in quality score versus prior year",
        examples=[-1.5],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "N_AB": 5200.0,
                "PREVIOUS_SAVINGS_RATE": 0.012,
                "PREVIOUS_QUALITY_SCORE": 87.5,
                "PREVIOUS_PERFORMANCE_GAP_PCT": -0.034,
                "EXPENDITURE_GROWTH_PCT": 0.021,
                "BENCHMARK_GROWTH_PCT": 0.018,
                "BENEFICIARY_GROWTH_PCT": 0.005,
                "QUALITY_CHANGE": -1.5,
            }
        }
    }


class RiskBatchRequest(BaseModel):
    """
    A batch of ACO records to score in a single request.
    Maximum 500 records per call.
    """

    records: List[RiskPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of ACO records to score (1–500 records)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
                        "N_AB": 5200.0,
                        "PREVIOUS_SAVINGS_RATE": 0.012,
                        "PREVIOUS_QUALITY_SCORE": 87.5,
                        "PREVIOUS_PERFORMANCE_GAP_PCT": -0.034,
                        "EXPENDITURE_GROWTH_PCT": 0.021,
                        "BENCHMARK_GROWTH_PCT": 0.018,
                        "BENEFICIARY_GROWTH_PCT": 0.005,
                        "QUALITY_CHANGE": -1.5,
                    },
                    {
                        "N_AB": 12000.0,
                        "PREVIOUS_SAVINGS_RATE": -0.008,
                        "PREVIOUS_QUALITY_SCORE": 72.0,
                        "PREVIOUS_PERFORMANCE_GAP_PCT": 0.091,
                        "EXPENDITURE_GROWTH_PCT": 0.053,
                        "BENCHMARK_GROWTH_PCT": 0.027,
                        "BENEFICIARY_GROWTH_PCT": 0.038,
                        "QUALITY_CHANGE": -4.0,
                    },
                ]
            }
        }
    }


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class RiskPredictionResponse(BaseModel):
    """
    Risk prediction result for a single ACO record.
    """

    risk_probability: float = Field(
        ...,
        description="Raw model probability (0.0 – 1.0)",
    )
    risk_probability_pct: float = Field(
        ...,
        description="Probability as a percentage (0.0 – 100.0)",
    )
    prediction: int = Field(
        ...,
        description="Binary prediction: 1 = RISK, 0 = NON-RISK",
    )
    risk_label: str = Field(
        ...,
        description='"RISK" or "NON-RISK"',
    )
    risk_level: str = Field(
        ...,
        description='"LOW" (p < 0.23) | "MEDIUM" (0.23 ≤ p < 0.35) | "HIGH" (p ≥ 0.35)',
    )
    threshold: float = Field(
        ...,
        description="Classification threshold applied (production = 0.23)",
    )


class RiskBatchResponse(BaseModel):
    """
    Aggregated results for a batch prediction request.
    """

    total: int = Field(..., description="Total records scored")
    risk_count: int = Field(..., description="Records classified as RISK")
    non_risk_count: int = Field(..., description="Records classified as NON-RISK")
    predictions: List[RiskPredictionResponse] = Field(
        ...,
        description="Individual prediction results in submission order",
    )


class ModelInfoResponse(BaseModel):
    """
    Metadata about the currently loaded production model.
    """

    model_config = {"protected_namespaces": ()}

    model_type: str = Field(
        ...,
        description="Scikit-learn estimator class name",
    )
    model_file: str = Field(
        ...,
        description="Filename of the loaded .joblib model",
    )
    threshold: float = Field(
        ...,
        description="Production classification threshold",
    )
    features: List[str] = Field(
        ...,
        description="Ordered list of features the model expects",
    )
    risk_levels: Dict[str, str] = Field(
        ...,
        description="Mapping of risk level labels to their probability ranges",
    )
