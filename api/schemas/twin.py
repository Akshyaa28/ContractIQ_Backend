from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# INPUT SCHEMAS
# ============================================================

class TwinRequest(BaseModel):
    """
    8 features for a query ACO to find its top-5 similar twins.
    ACO_ID is optional — used only to exclude self-match.
    """

    ACO_ID: Optional[str] = Field(
        None,
        description="Query ACO identifier (used to exclude self-match)",
        examples=["A00001"],
    )
    N_AB: float = Field(..., description="Number of attributed beneficiaries", examples=[5200.0])
    PREVIOUS_SAVINGS_RATE: float = Field(..., description="Prior-year savings rate", examples=[0.032])
    PREVIOUS_QUALITY_SCORE: float = Field(..., description="Prior-year quality score", examples=[88.0])
    PREVIOUS_PERFORMANCE_GAP_PCT: float = Field(..., description="Prior-year performance gap %", examples=[-0.05])
    EXPENDITURE_GROWTH_PCT: float = Field(..., description="Expenditure growth %", examples=[0.021])
    BENCHMARK_GROWTH_PCT: float = Field(..., description="Benchmark growth %", examples=[0.018])
    BENEFICIARY_GROWTH_PCT: float = Field(..., description="Beneficiary growth %", examples=[0.005])
    QUALITY_CHANGE: float = Field(..., description="Change in quality score", examples=[1.5])

    model_config = {
        "json_schema_extra": {
            "example": {
                "ACO_ID": "A00001",
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


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class TwinDetail(BaseModel):
    """One matched twin."""
    rank: int = Field(..., description="Rank 1–5 (1 = most similar)")
    twin_aco_id: str = Field(..., description="Twin ACO identifier")
    twin_aco_name: str = Field(..., description="Twin ACO name")
    twin_state: str = Field(..., description="Twin ACO state")
    twin_year: int = Field(..., description="Historical year of this twin observation")
    similarity_score: float = Field(..., description="Similarity score (higher = more similar)")
    twin_savings_rate_pct: float = Field(..., description="Twin's next-year savings rate (%)")
    outperforming: bool = Field(..., description="True if twin savings > Top-5 median")


class TwinResponse(BaseModel):
    """Top-5 similar twin results for a query ACO."""
    twins: List[TwinDetail] = Field(..., description="Top 5 most similar historical ACO-years")
    top5_avg_savings_rate_pct: float = Field(..., description="Average savings rate of top 5 (%)")
    top5_median_savings_rate_pct: float = Field(..., description="Median savings rate of top 5 (%)")
    top5_best_savings_rate_pct: float = Field(..., description="Best savings rate among top 5 (%)")
    top5_worst_savings_rate_pct: float = Field(..., description="Worst savings rate among top 5 (%)")
    outperformer_count: int = Field(..., description="Number of outperforming twins (above median)")
    outperformer_rate: float = Field(..., description="Outperformer rate (0–1)")


class TwinModelInfoResponse(BaseModel):
    """Metadata about the twin matching model."""

    model_config = {"protected_namespaces": ()}

    model_type: str = Field(..., description="Algorithm description")
    model_file: str = Field(..., description="NearestNeighbors .joblib filename")
    scaler_file: str = Field(..., description="StandardScaler .joblib filename")
    features: List[str] = Field(..., description="8 matching features")
    top_k: int = Field(..., description="Number of twins returned per query")
    candidate_k: int = Field(..., description="Initial candidate pool size")
    reference_rows: int = Field(..., description="Number of reference observations")
    reference_years: str = Field(..., description="Historical years used")
    target: str = Field(..., description="Outcome variable (not used for matching)")
    target_used_for_similarity: bool = Field(..., description="False — no leakage")
    self_match_excluded: bool = Field(..., description="True — ACO cannot match itself")
