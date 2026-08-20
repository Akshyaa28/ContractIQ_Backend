from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONTRACTIQ — DATASET 1 SIMILAR TWIN MODEL DATASET BUILDER
# ============================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 SIMILAR TWIN MODEL DATASET BUILDER")
print("=" * 78)


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# preprocessing/dataset1/similar_twin/build_twin_model_data.py
#
# parents[0] = similar_twin
# parents[1] = dataset1
# parents[2] = preprocessing
# parents[3] = ContractIQ(!)

PROJECT_ROOT = SCRIPT_PATH.parents[3]


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Clean.csv"
)


OUTPUT_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Model_Twin.csv"
)


OUTPUT_SCORING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Scoring_2024_Twin.csv"
)


# ============================================================
# TWIN MODEL FEATURES
# ============================================================
#
# These variables describe the ACO's current/historical
# characteristics and are appropriate for similarity matching.
#
# IMPORTANT:
#
# NEXT_YEAR_SAVINGS_RATE is intentionally NOT included.
#
# The twin model is NOT a forecasting model.
# It identifies structurally similar ACOs.
# ============================================================

TWIN_FEATURES = [

    "N_AB",

    "PREVIOUS_SAVINGS_RATE",

    "PREVIOUS_QUALITY_SCORE",

    "PREVIOUS_PERFORMANCE_GAP_PCT",

    "EXPENDITURE_GROWTH_PCT",

    "BENCHMARK_GROWTH_PCT",

    "BENEFICIARY_GROWTH_PCT",

    "QUALITY_CHANGE",

]


# ============================================================
# IDENTIFIERS
# ============================================================

IDENTIFIERS = [

    "ACO_ID",

    "ACO_NAME",

    "STATE",

    "YEAR",

]


# ============================================================
# FUTURE / TARGET COLUMNS
# ============================================================

TARGET = "NEXT_YEAR_SAVINGS_RATE"


# ============================================================
# DISPLAY PATHS
# ============================================================

print()
print("=" * 78)
print("PROJECT PATHS")
print("=" * 78)

print()
print("SCRIPT LOCATION:")
print(SCRIPT_PATH)

print()
print("PROJECT ROOT:")
print(PROJECT_ROOT)

print()
print("INPUT:")
print(INPUT_PATH)

print()
print("TWIN MODEL OUTPUT:")
print(OUTPUT_MODEL_PATH)

print()
print("2024 TWIN SCORING OUTPUT:")
print(OUTPUT_SCORING_PATH)


# ============================================================
# CHECK INPUT
# ============================================================

print()
print("=" * 78)
print("CHECKING INPUT DATASET")
print("=" * 78)


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"""
Clean Dataset 1 was not found.

Expected:
{INPUT_PATH}

Run the Dataset 1 preprocessing pipeline first.
"""
    )


print()
print("Input dataset found.")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 78)
print("LOADING CLEAN DATASET")
print("=" * 78)


df = pd.read_csv(
    INPUT_PATH
)


print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

print()
print("=" * 78)
print("STANDARDIZING COLUMN NAMES")
print("=" * 78)


df.columns = (
    df.columns
    .str.strip()
    .str.upper()
)


print(
    "Column names standardized."
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

print()
print("=" * 78)
print("CHECKING REQUIRED COLUMNS")
print("=" * 78)


required_columns = (
    IDENTIFIERS
    + TWIN_FEATURES
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print()
    print("Missing columns:")

    for column in missing_columns:

        print(
            f" - {column}"
        )

    raise ValueError(
        "Required Similar Twin columns are missing."
    )


print(
    "All required Similar Twin columns are present."
)


# ============================================================
# REMOVE MISSING INDICATORS
# ============================================================

print()
print("=" * 78)
print("REMOVING MISSING-VALUE INDICATOR COLUMNS")
print("=" * 78)


missing_indicator_columns = [

    column
    for column in df.columns
    if column.endswith("_MISSING")

]


if missing_indicator_columns:

    print()
    print("Removing:")

    for column in missing_indicator_columns:

        print(
            f" - {column}"
        )

    df = df.drop(
        columns=missing_indicator_columns
    )

else:

    print(
        "No missing-value indicator columns found."
    )


# ============================================================
# IDENTIFIER VALIDATION
# ============================================================

print()
print("=" * 78)
print("VALIDATING IDENTIFIERS")
print("=" * 78)


for column in IDENTIFIERS[:3]:

    missing_count = (
        df[column]
        .isna()
        .sum()
    )

    print(
        f"{column:<12}: "
        f"missing = {missing_count}"
    )

    if missing_count > 0:

        raise ValueError(
            f"{column} contains missing values."
        )


# ============================================================
# YEAR VALIDATION
# ============================================================

print()
print("=" * 78)
print("VALIDATING YEAR")
print("=" * 78)


df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce",
)


if df["YEAR"].isna().any():

    raise ValueError(
        "YEAR contains invalid values."
    )


df["YEAR"] = (
    df["YEAR"]
    .astype(int)
)


print(
    f"Year range: "
    f"{df['YEAR'].min()}–{df['YEAR'].max()}"
)


print()
print("Rows by year:")


print(
    df["YEAR"]
    .value_counts()
    .sort_index()
)


# ============================================================
# ACO-YEAR UNIQUENESS
# ============================================================

print()
print("=" * 78)
print("CHECKING ACO-YEAR UNIQUENESS")
print("=" * 78)


duplicate_aco_years = (
    df
    .duplicated(
        subset=[
            "ACO_ID",
            "YEAR",
        ]
    )
    .sum()
)


print(
    f"Duplicate ACO-year rows: "
    f"{duplicate_aco_years}"
)


if duplicate_aco_years > 0:

    raise ValueError(
        "ACO-year uniqueness failed."
    )


print(
    "ACO-year uniqueness PASSED."
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print()
print("=" * 78)
print("VALIDATING TWIN MODEL FEATURES")
print("=" * 78)


for feature in TWIN_FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )


    missing_count = (
        df[feature]
        .isna()
        .sum()
    )


    infinite_count = (
        np.isinf(
            df[feature]
        )
        .sum()
    )


    print(
        f"{feature:<38}"
        f"missing={missing_count:6d} "
        f"infinite={infinite_count:6d}"
    )


    if missing_count > 0:

        raise ValueError(
            f"Missing values found in Twin feature: "
            f"{feature}"
        )


    if infinite_count > 0:

        raise ValueError(
            f"Infinite values found in Twin feature: "
            f"{feature}"
        )


print()
print(
    "All Twin features passed validation."
)


# ============================================================
# TARGET / FUTURE INFORMATION CHECK
# ============================================================

print()
print("=" * 78)
print("TARGET / FUTURE INFORMATION CHECK")
print("=" * 78)


future_columns = [

    column
    for column in df.columns
    if (
        "NEXT" in column
        or "FUTURE" in column
        or "LEAD" in column
    )

]


print()
print("Future-related columns detected:")


if future_columns:

    for column in future_columns:

        print(
            f" - {column}"
        )

else:

    print(
        " - None"
    )


print()
print(
    f"Forecast target detected: "
    f"{TARGET}"
)


# ------------------------------------------------------------
# Confirm target is NOT a Twin feature.
# ------------------------------------------------------------

if TARGET in TWIN_FEATURES:

    raise ValueError(
        """
TARGET LEAKAGE DETECTED.

NEXT_YEAR_SAVINGS_RATE cannot be used
as a Similar Twin matching feature.
"""
    )


print()
print(
    "NEXT_YEAR_SAVINGS_RATE is NOT used "
    "for Twin similarity."
)


print(
    "Leakage protection PASSED."
)


# ============================================================
# DATASET STRUCTURE
# ============================================================

print()
print("=" * 78)
print("ANALYZING DATASET STRUCTURE")
print("=" * 78)


print()
print(
    f"Total historical records: "
    f"{len(df):,}"
)


print(
    f"Unique ACOs: "
    f"{df['ACO_ID'].nunique():,}"
)


print(
    f"Years: "
    f"{df['YEAR'].min()}–{df['YEAR'].max()}"
)


# ============================================================
# SPLIT HISTORICAL AND 2024 SCORING DATA
# ============================================================

print()
print("=" * 78)
print("CREATING HISTORICAL TWIN DATA + 2024 SCORING DATA")
print("=" * 78)


scoring_year = 2024


historical_df = (
    df[
        df["YEAR"] < scoring_year
    ]
    .copy()
)


scoring_df = (
    df[
        df["YEAR"] == scoring_year
    ]
    .copy()
)


print()
print(
    f"Historical Twin rows : "
    f"{len(historical_df):,}"
)


print(
    f"2024 scoring rows    : "
    f"{len(scoring_df):,}"
)


# ============================================================
# CHECK 2024 DATA
# ============================================================

if len(scoring_df) == 0:

    raise ValueError(
        """
No 2024 scoring records were found.

Expected Dataset 1 to contain 2024 records
for Similar Twin scoring.
"""
    )


# ============================================================
# HISTORICAL TARGET INFORMATION
# ============================================================
#
# We keep NEXT_YEAR_SAVINGS_RATE in the historical dataset
# only as an OUTCOME/PERFORMANCE variable.
#
# It is NOT a matching feature.
#
# This allows the later Twin model to answer:
#
# "Among structurally similar ACOs, which ones performed better?"
#
# For the 2024 scoring dataset, this target should not exist
# or should not be required.
# ============================================================

print()
print("=" * 78)
print("VALIDATING HISTORICAL PERFORMANCE INFORMATION")
print("=" * 78)


if TARGET in historical_df.columns:

    historical_df[TARGET] = pd.to_numeric(
        historical_df[TARGET],
        errors="coerce",
    )


    target_missing = (
        historical_df[TARGET]
        .isna()
        .sum()
    )


    target_infinite = (
        np.isinf(
            historical_df[TARGET]
        )
        .sum()
    )


    print(
        f"{TARGET} missing   : "
        f"{target_missing}"
    )


    print(
        f"{TARGET} infinite  : "
        f"{target_infinite}"
    )


    if target_missing > 0:

        raise ValueError(
            """
Historical Twin performance data contains
missing NEXT_YEAR_SAVINGS_RATE values.
"""
        )


    if target_infinite > 0:

        raise ValueError(
            """
Historical Twin performance data contains
infinite NEXT_YEAR_SAVINGS_RATE values.
"""
        )


else:

    raise ValueError(
        f"Historical dataset does not contain {TARGET}."
    )


# ============================================================
# CREATE HISTORICAL TWIN DATASET
# ============================================================

print()
print("=" * 78)
print("CREATING HISTORICAL TWIN MODEL DATASET")
print("=" * 78)


MODEL_COLUMNS = (

    IDENTIFIERS

    + TWIN_FEATURES

    + [TARGET]

)


model_df = (
    historical_df[
        MODEL_COLUMNS
    ]
    .copy()
)


print()
print(
    f"Historical Twin rows    : "
    f"{len(model_df):,}"
)


print(
    f"Historical Twin columns : "
    f"{len(model_df.columns)}"
)


# ============================================================
# CREATE 2024 SCORING DATASET
# ============================================================
#
# 2024 is the target population.
#
# We intentionally do NOT include
# NEXT_YEAR_SAVINGS_RATE.
# ============================================================

print()
print("=" * 78)
print("CREATING 2024 TWIN SCORING DATASET")
print("=" * 78)


SCORING_COLUMNS = (

    IDENTIFIERS

    + TWIN_FEATURES

)


scoring_output_df = (
    scoring_df[
        SCORING_COLUMNS
    ]
    .copy()
)


print()
print(
    f"2024 Twin scoring rows    : "
    f"{len(scoring_output_df):,}"
)


print(
    f"2024 Twin scoring columns : "
    f"{len(scoring_output_df.columns)}"
)


# ============================================================
# FINAL MODEL DATA VALIDATION
# ============================================================

print()
print("=" * 78)
print("FINAL HISTORICAL TWIN DATA VALIDATION")
print("=" * 78)


missing_model = (
    model_df
    .isna()
    .sum()
    .sum()
)


duplicate_model = (
    model_df
    .duplicated()
    .sum()
)


duplicate_aco_year_model = (
    model_df
    .duplicated(
        subset=[
            "ACO_ID",
            "YEAR",
        ]
    )
    .sum()
)


print(
    f"Missing values          : "
    f"{missing_model}"
)


print(
    f"Exact duplicate rows    : "
    f"{duplicate_model}"
)


print(
    f"Duplicate ACO-year rows : "
    f"{duplicate_aco_year_model}"
)


if missing_model > 0:

    raise ValueError(
        "Missing values detected in final Twin model dataset."
    )


if duplicate_model > 0:

    raise ValueError(
        "Exact duplicate rows detected."
    )


if duplicate_aco_year_model > 0:

    raise ValueError(
        "Duplicate ACO-year rows detected."
    )


print()
print(
    "Historical Twin dataset validation PASSED."
)


# ============================================================
# FINAL 2024 SCORING VALIDATION
# ============================================================

print()
print("=" * 78)
print("FINAL 2024 TWIN SCORING VALIDATION")
print("=" * 78)


missing_scoring_values = (
    scoring_output_df
    .isna()
    .sum()
    .sum()
)


duplicate_scoring = (
    scoring_output_df
    .duplicated()
    .sum()
)


duplicate_scoring_aco_year = (
    scoring_output_df
    .duplicated(
        subset=[
            "ACO_ID",
            "YEAR",
        ]
    )
    .sum()
)


print(
    f"Missing values          : "
    f"{missing_scoring_values}"
)


print(
    f"Exact duplicate rows    : "
    f"{duplicate_scoring}"
)


print(
    f"Duplicate ACO-year rows : "
    f"{duplicate_scoring_aco_year}"
)


if missing_scoring_values > 0:

    raise ValueError(
        "Missing values detected in 2024 Twin scoring dataset."
    )


if duplicate_scoring > 0:

    raise ValueError(
        "Exact duplicate rows detected in 2024 scoring dataset."
    )


if duplicate_scoring_aco_year > 0:

    raise ValueError(
        "Duplicate ACO-year rows detected in 2024 scoring dataset."
    )


print()
print(
    "2024 Twin scoring validation PASSED."
)


# ============================================================
# VERIFY TARGET IS NOT IN SCORING DATASET
# ============================================================

print()
print("=" * 78)
print("VERIFYING 2024 TARGET EXCLUSION")
print("=" * 78)


if TARGET in scoring_output_df.columns:

    raise ValueError(
        """
NEXT_YEAR_SAVINGS_RATE was incorrectly included
in the 2024 Twin scoring dataset.
"""
    )


print(
    f"{TARGET} is NOT present in 2024 scoring data."
)


print(
    "Target exclusion PASSED."
)


# ============================================================
# SAVE HISTORICAL TWIN MODEL DATASET
# ============================================================

print()
print("=" * 78)
print("SAVING HISTORICAL TWIN MODEL DATASET")
print("=" * 78)


OUTPUT_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


model_df.to_csv(
    OUTPUT_MODEL_PATH,
    index=False,
)


print()
print(
    f"Historical Twin model dataset saved:"
)

print(
    OUTPUT_MODEL_PATH
)


# ============================================================
# SAVE 2024 SCORING DATASET
# ============================================================

print()
print("=" * 78)
print("SAVING 2024 TWIN SCORING DATASET")
print("=" * 78)


scoring_output_df.to_csv(
    OUTPUT_SCORING_PATH,
    index=False,
)


print()
print(
    f"2024 Twin scoring dataset saved:"
)

print(
    OUTPUT_SCORING_PATH
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print()
print("=" * 78)
print("TWIN MATCHING FEATURES")
print("=" * 78)


for index, feature in enumerate(
    TWIN_FEATURES,
    start=1,
):

    print(
        f"{index:2d}. {feature}"
    )


# ============================================================
# PERFORMANCE VARIABLE
# ============================================================

print()
print("=" * 78)
print("TWIN PERFORMANCE VARIABLE")
print("=" * 78)


print(
    f"Performance variable:"
)

print(
    f" - {TARGET}"
)


print()
print(
    "IMPORTANT:"
)

print(
    f"{TARGET} is retained ONLY as a historical "
    "performance/outcome variable."
)

print(
    "It is NOT used when calculating similarity."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("FINAL SUMMARY")
print("=" * 78)


print()
print(
    f"Original clean dataset : "
    f"{len(df):,} rows"
)


print(
    f"Historical Twin data   : "
    f"{len(model_df):,} rows"
)


print(
    f"2024 Twin scoring data : "
    f"{len(scoring_output_df):,} rows"
)


print(
    f"Unique historical ACOs : "
    f"{model_df['ACO_ID'].nunique():,}"
)


print(
    f"2024 scoring ACOs     : "
    f"{scoring_output_df['ACO_ID'].nunique():,}"
)


print()
print(
    "Twin matching features:"
)


for feature in TWIN_FEATURES:

    print(
        f" - {feature}"
    )


print()
print(
    "Historical performance variable:"
)

print(
    f" - {TARGET}"
)


print()
print(
    "Leakage protection:"
)

print(
    "PASSED"
)


print()
print(
    "2024 is reserved for Similar Twin scoring."
)


print()
print("=" * 78)
print("SIMILAR TWIN MODEL DATASET BUILD COMPLETE")
print("=" * 78)