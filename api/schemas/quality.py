from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# INPUT SCHEMAS
# ============================================================

class QualityPredictionRequest(BaseModel):
    """
    All 32 year-T features required by the quality score pipeline.

    Categorical fields (Primary_State, Revenue_Category, Track,
    Policy_Version) should be provided as strings.  Unknown
    categories are handled gracefully by the pipeline's
    OneHotEncoder (handle_unknown='ignore').

    Numeric fields may be omitted (None) — the pipeline's
    SimpleImputer will substitute the training-set median.
    """

    # ── Temporal / administrative ──────────────────────────
    Year_T: int = Field(
        ...,
        description="Current performance year (e.g. 2023 to predict 2024)",
        examples=[2023],
    )
    Primary_State: str = Field(
        ...,
        description="Two-letter state abbreviation (e.g. 'CA', 'TX')",
        examples=["CA"],
    )
    Revenue_Category: str = Field(
        ...,
        description="'High Revenue' or 'Low Revenue'",
        examples=["High Revenue"],
    )
    Track: str = Field(
        ...,
        description="'One-Sided' or 'Two-Sided'",
        examples=["Two-Sided"],
    )
    Agreement_Period_Num: int = Field(
        ...,
        description="Agreement period number (1–4)",
        examples=[4],
    )
    Policy_Version: str = Field(
        ...,
        description="Policy version string (e.g. 'QPV-3')",
        examples=["QPV-3"],
    )
    COVID_Period: int = Field(
        ...,
        description="1 if the performance year was affected by COVID, else 0",
        examples=[0],
    )

    # ── Provider / beneficiary structure ───────────────────
    Beneficiary_Count: int = Field(
        ...,
        description="Number of attributed beneficiaries",
        examples=[47360],
    )
    Hospital_Count: int = Field(
        ...,
        description="Number of hospitals in the ACO",
        examples=[4],
    )
    PCP_Count: int = Field(
        ...,
        description="Number of primary care physicians",
        examples=[243],
    )
    Specialist_Count: int = Field(
        ...,
        description="Number of specialist physicians",
        examples=[387],
    )

    # ── Risk / disease burden ───────────────────────────────
    Risk_Score: float = Field(
        ...,
        description="HCC risk score (typical range 0.5–1.5)",
        examples=[1.10],
    )
    Chronic_Disease_Rate_Pct: float = Field(
        ...,
        description="Percentage of beneficiaries with chronic disease",
        examples=[44.5],
    )

    # ── Quality measures ────────────────────────────────────
    Current_Quality_Score: float = Field(
        ...,
        description="Current-year composite quality score (0–100)",
        examples=[67.5],
    )
    Previous_Quality_Score: Optional[float] = Field(
        None,
        description="Prior-year quality score (None for first year of ACO)",
        examples=[71.5],
    )
    Readmission_Rate_Pct: float = Field(
        ...,
        description="30-day all-cause readmission rate (%)",
        examples=[13.0],
    )
    Admission_Rate_Per_1000: float = Field(
        ...,
        description="All-cause hospital admissions per 1,000 beneficiaries",
        examples=[171.0],
    )
    ED_Visit_Rate_Per_1000: float = Field(
        ...,
        description="Emergency department visits per 1,000 beneficiaries",
        examples=[485.0],
    )
    Preventable_Admission_Rate_Per_1000: float = Field(
        ...,
        description="Preventable admissions per 1,000 beneficiaries",
        examples=[36.2],
    )
    Patient_Experience_Score: float = Field(
        ...,
        description="Patient experience composite score (0–100)",
        examples=[71.7],
    )
    Diabetes_Control_Rate_Pct: float = Field(
        ...,
        description="Percentage of diabetic beneficiaries with controlled HbA1c",
        examples=[73.9],
    )
    Blood_Pressure_Control_Rate_Pct: float = Field(
        ...,
        description="Percentage of hypertensive beneficiaries with controlled BP",
        examples=[75.7],
    )
    Preventive_Screening_Rate_Pct: float = Field(
        ...,
        description="Preventive screening completion rate (%)",
        examples=[70.0],
    )
    Followup_Compliance_Rate_Pct: float = Field(
        ...,
        description="Post-discharge follow-up compliance rate (%)",
        examples=[68.2],
    )

    # ── Financial ───────────────────────────────────────────
    Expenditure_Per_Beneficiary: float = Field(
        ...,
        description="Per-beneficiary expenditure (USD)",
        examples=[13775.0],
    )
    Benchmark_Per_Beneficiary: float = Field(
        ...,
        description="Per-beneficiary CMS benchmark (USD)",
        examples=[13584.0],
    )
    Total_Expenditure: int = Field(
        ...,
        description="Total ACO expenditure (USD)",
        examples=[652382299],
    )
    Benchmark_Expenditure: int = Field(
        ...,
        description="Total CMS benchmark expenditure (USD)",
        examples=[643337666],
    )
    Savings_Amount: int = Field(
        ...,
        description="Savings vs benchmark (USD, negative = spending above benchmark)",
        examples=[-9044633],
    )
    Savings_Rate: float = Field(
        ...,
        description="Savings rate (negative = spending above benchmark)",
        examples=[-0.014],
    )
    Final_Share_Rate: float = Field(
        ...,
        description="Shared savings participation rate (0.50 or 0.75)",
        examples=[0.75],
    )
    Earned_Savings_Loss: int = Field(
        ...,
        description="Earned shared savings or loss (USD)",
        examples=[-6783474],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Year_T": 2023,
                "Primary_State": "CA",
                "Revenue_Category": "High Revenue",
                "Track": "Two-Sided",
                "Agreement_Period_Num": 4,
                "Policy_Version": "QPV-3",
                "COVID_Period": 0,
                "Beneficiary_Count": 47360,
                "Hospital_Count": 4,
                "PCP_Count": 243,
                "Specialist_Count": 387,
                "Risk_Score": 1.1001,
                "Chronic_Disease_Rate_Pct": 44.475,
                "Current_Quality_Score": 67.485,
                "Previous_Quality_Score": 71.452,
                "Readmission_Rate_Pct": 12.964,
                "Admission_Rate_Per_1000": 170.985,
                "ED_Visit_Rate_Per_1000": 484.518,
                "Preventable_Admission_Rate_Per_1000": 36.181,
                "Patient_Experience_Score": 71.706,
                "Diabetes_Control_Rate_Pct": 73.912,
                "Blood_Pressure_Control_Rate_Pct": 75.66,
                "Preventive_Screening_Rate_Pct": 69.979,
                "Followup_Compliance_Rate_Pct": 68.169,
                "Expenditure_Per_Beneficiary": 13774.96,
                "Benchmark_Per_Beneficiary": 13583.99,
                "Total_Expenditure": 652382299,
                "Benchmark_Expenditure": 643337666,
                "Savings_Amount": -9044633,
                "Savings_Rate": -0.014059,
                "Final_Share_Rate": 0.75,
                "Earned_Savings_Loss": -6783474,
            }
        }
    }


class QualityBatchRequest(BaseModel):
    """Batch of ACO records for quality score prediction (max 500)."""

    records: List[QualityPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of ACO records (1–500)",
    )


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class QualityPredictionResponse(BaseModel):
    """Quality score prediction result for a single ACO record."""

    predicted_quality_score: float = Field(
        ...,
        description="Predicted next-year quality score (0–100)",
    )
    quality_band: str = Field(
        ...,
        description='"HIGH" (≥90) | "MODERATE" (75–90) | "LOW" (<75)',
    )
    model_r2_pct: float = Field(
        ...,
        description="Test R² of the model on the 2024 hold-out year (83.38%)",
    )
    model_mae: float = Field(
        ...,
        description="Mean absolute error on 2024 hold-out (2.461 quality points)",
    )


class QualityBatchResponse(BaseModel):
    """Aggregated results for a batch quality prediction request."""

    total: int = Field(..., description="Total records scored")
    high_count:     int = Field(..., description="Records predicted as HIGH quality (≥90)")
    moderate_count: int = Field(..., description="Records predicted as MODERATE quality (75–90)")
    low_count:      int = Field(..., description="Records predicted as LOW quality (<75)")
    mean_predicted_score: float = Field(
        ...,
        description="Mean predicted quality score across all submitted records",
    )
    predictions: List[QualityPredictionResponse] = Field(
        ...,
        description="Individual prediction results in submission order",
    )


class QualityModelInfoResponse(BaseModel):
    """Metadata about the loaded quality score model."""

    model_config = {"protected_namespaces": ()}

    model_type:      str              = Field(..., description="Estimator class (e.g. Ridge)")
    pipeline_steps:  List[str]        = Field(..., description="Steps in the sklearn Pipeline")
    model_file:      str              = Field(..., description="Filename of the .joblib model")
    target:          str              = Field(..., description="Regression target variable")
    features:        List[str]        = Field(..., description="All 32 input features")
    total_features:  int              = Field(..., description="Number of input features")
    test_r2_pct:     float            = Field(..., description="Hold-out R² percentage")
    test_mae:        float            = Field(..., description="Hold-out MAE")
    test_rmse:       float            = Field(..., description="Hold-out RMSE")
    train_years:     str              = Field(..., description="Years used for training")
    test_year:       int              = Field(..., description="Hold-out year")
    quality_bands:   Dict[str, str]   = Field(..., description="Band definitions")
    target_range:    str              = Field(..., description="Observed target range")
    disclosure:      str              = Field(..., description="Data provenance disclosure")
