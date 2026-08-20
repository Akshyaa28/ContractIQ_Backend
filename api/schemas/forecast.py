from typing import Dict, List

from pydantic import BaseModel, Field


# ============================================================
# INPUT SCHEMAS
# ============================================================

class ForecastRequest(BaseModel):
    """
    Input features required by the Dataset 1 Savings Forecast Model.

    All 8 features must be present and numeric.
    The model predicts NEXT_YEAR_SAVINGS_RATE — the ACO's savings rate
    relative to the CMS benchmark in the following performance year.
    """

    N_AB: float = Field(
        ...,
        description="Number of attributed beneficiaries",
        examples=[5200.0],
    )
    PREVIOUS_SAVINGS_RATE: float = Field(
        ...,
        description=(
            "Prior-year savings rate (strongest predictor, corr=0.68). "
            "Positive = saved vs benchmark. Negative = spent above benchmark."
        ),
        examples=[0.032],
    )
    PREVIOUS_QUALITY_SCORE: float = Field(
        ...,
        description="Prior-year quality score (typical range 60–100)",
        examples=[88.0],
    )
    PREVIOUS_PERFORMANCE_GAP_PCT: float = Field(
        ...,
        description="Prior-year performance gap as a percentage",
        examples=[-0.05],
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
        examples=[1.5],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "N_AB": 5200.0,
                "PREVIOUS_SAVINGS_RATE": 0.032,
                "PREVIOUS_QUALITY_SCORE": 88.0,
                "PREVIOUS_PERFORMANCE_GAP_PCT": -0.05,
                "EXPENDITURE_GROWTH_PCT": 0.021,
                "BENCHMARK_GROWTH_PCT": 0.018,
                "BENEFICIARY_GROWTH_PCT": 0.005,
                "QUALITY_CHANGE": 1.5,
            }
        }
    }


class ForecastBatchRequest(BaseModel):
    """
    A batch of ACO records to forecast in a single request.
    Maximum 500 records per call.
    """

    records: List[ForecastRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of ACO records to forecast (1–500 records)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
                        "N_AB": 5200.0,
                        "PREVIOUS_SAVINGS_RATE": 0.032,
                        "PREVIOUS_QUALITY_SCORE": 88.0,
                        "PREVIOUS_PERFORMANCE_GAP_PCT": -0.05,
                        "EXPENDITURE_GROWTH_PCT": 0.021,
                        "BENCHMARK_GROWTH_PCT": 0.018,
                        "BENEFICIARY_GROWTH_PCT": 0.005,
                        "QUALITY_CHANGE": 1.5,
                    },
                    {
                        "N_AB": 1800.0,
                        "PREVIOUS_SAVINGS_RATE": -0.045,
                        "PREVIOUS_QUALITY_SCORE": 68.0,
                        "PREVIOUS_PERFORMANCE_GAP_PCT": 0.12,
                        "EXPENDITURE_GROWTH_PCT": 0.078,
                        "BENCHMARK_GROWTH_PCT": 0.021,
                        "BENEFICIARY_GROWTH_PCT": -0.12,
                        "QUALITY_CHANGE": -6.0,
                    },
                ]
            }
        }
    }


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class ForecastResponse(BaseModel):
    """
    Savings rate forecast result for a single ACO record.
    """

    forecasted_savings_rate: float = Field(
        ...,
        description="Predicted next-year savings rate (decimal, e.g. 0.032 = 3.2%)",
    )
    forecasted_savings_rate_pct: float = Field(
        ...,
        description="Predicted next-year savings rate as a percentage (e.g. 3.20)",
    )
    savings_category: str = Field(
        ...,
        description=(
            '"HIGH SAVINGS" (≥3%) | '
            '"MODERATE SAVINGS" (0%–3%) | '
            '"LOW SAVINGS" (<0%, spending above benchmark)'
        ),
    )
    savings_direction: str = Field(
        ...,
        description='"POSITIVE" (saving vs benchmark) | "NEGATIVE" (spending above benchmark)',
    )
    model_r2_pct: float = Field(
        ...,
        description="Validation R² of the model (88.82% on 2023 hold-out year)",
    )


class ForecastBatchResponse(BaseModel):
    """
    Aggregated results for a batch forecast request.
    """

    total: int = Field(
        ...,
        description="Total records forecasted",
    )
    high_savings_count: int = Field(
        ...,
        description="Records forecast as HIGH SAVINGS (≥3%)",
    )
    moderate_savings_count: int = Field(
        ...,
        description="Records forecast as MODERATE SAVINGS (0%–3%)",
    )
    low_savings_count: int = Field(
        ...,
        description="Records forecast as LOW SAVINGS (<0%)",
    )
    mean_forecasted_savings_rate_pct: float = Field(
        ...,
        description="Mean forecasted savings rate across all submitted records (%)",
    )
    forecasts: List[ForecastResponse] = Field(
        ...,
        description="Individual forecast results in submission order",
    )


class ForecastModelInfoResponse(BaseModel):
    """
    Metadata about the currently loaded production forecast model.
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
    target: str = Field(
        ...,
        description="The regression target variable",
    )
    features: List[str] = Field(
        ...,
        description="Ordered list of features the model expects",
    )
    validation_r2_pct: float = Field(
        ...,
        description="Out-of-time validation R² percentage (2023 hold-out)",
    )
    validation_mae: float = Field(
        ...,
        description="Mean absolute error on 2023 validation set",
    )
    validation_rmse: float = Field(
        ...,
        description="Root mean squared error on 2023 validation set",
    )
    train_years: str = Field(
        ...,
        description="Years used for training",
    )
    validation_year: int = Field(
        ...,
        description="Hold-out year used for validation",
    )
    savings_categories: Dict[str, str] = Field(
        ...,
        description="Mapping of category labels to their forecast rate ranges",
    )
