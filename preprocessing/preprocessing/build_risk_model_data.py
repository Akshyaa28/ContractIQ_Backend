"""
===============================================================================
CONTRACTIQ — DATASET 1 RISK MODEL DATASET BUILDER
===============================================================================

Purpose
-------
Create the final modeling dataset for the Dataset 1 risk prediction model.

Input
-----
data/processed/Dataset1_Clean.csv

Output
------
data/processed/Dataset1_Model_Risk.csv
data/processed/Dataset1_Scoring_2024.csv

Risk model predictors
---------------------
1. N_AB
2. PREVIOUS_SAVINGS_RATE
3. PREVIOUS_QUALITY_SCORE
4. PREVIOUS_PERFORMANCE_GAP_PCT
5. EXPENDITURE_GROWTH_PCT
6. BENCHMARK_GROWTH_PCT
7. BENEFICIARY_GROWTH_PCT
8. QUALITY_CHANGE

Target
------
NEXT_YEAR_RISK

Important
---------
NEXT_YEAR_SAVINGS_RATE is NOT used as a predictor because it is a
future-year outcome and could cause target leakage.

The *_MISSING indicator columns are removed because the current
validated dataset contains no missing values in the source features.

2024 is kept separately as a scoring dataset because in a real
deployment setting, a 2024 row cannot know its genuine 2025 outcome
at prediction time.
===============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

# This script is located at:
# I:\ContractIQ(!)\preprocessing\dataset1\preprocessing
#
# Project root is therefore four levels above this file.

# =============================================================================
# PROJECT PATH
# =============================================================================

# Current script location:
# I:\ContractIQ(!)\preprocessing\preprocessing\build_risk_model_data.py
#
# Therefore:
# parents[0] = preprocessing
# parents[1] = preprocessing
# parents[2] = ContractIQ(!)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "Dataset1_Clean.csv"

MODEL_OUTPUT = PROCESSED_DIR / "Dataset1_Model_Risk.csv"

SCORING_OUTPUT = PROCESSED_DIR / "Dataset1_Scoring_2024.csv"

INPUT_FILE = PROCESSED_DIR / "Dataset1_Clean.csv"

MODEL_OUTPUT = PROCESSED_DIR / "Dataset1_Model_Risk.csv"

SCORING_OUTPUT = PROCESSED_DIR / "Dataset1_Scoring_2024.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

ID_COLUMNS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]

FEATURE_COLUMNS = [
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
]

TARGET_COLUMN = "NEXT_YEAR_RISK"

FUTURE_OUTCOME_COLUMN = "NEXT_YEAR_SAVINGS_RATE"

MISSING_INDICATOR_SUFFIX = "_MISSING"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    print()
    print("ERROR:")
    print(message)
    print()
    sys.exit(1)


# =============================================================================
# START
# =============================================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 RISK MODEL DATASET BUILDER")
print("=" * 78)

print()
print("PROJECT ROOT:")
print(PROJECT_ROOT)

print()
print("INPUT:")
print(INPUT_FILE)

print()
print("MODEL OUTPUT:")
print(MODEL_OUTPUT)

print()
print("2024 SCORING OUTPUT:")
print(SCORING_OUTPUT)


# =============================================================================
# CHECK INPUT
# =============================================================================

print_section("CHECKING INPUT DATASET")

if not INPUT_FILE.exists():
    fail(
        f"""
Clean dataset was not found.

Expected:
{INPUT_FILE}

Run preprocessing first:
python preprocess_dataset1.py
"""
    )

print("Input dataset found.")


# =============================================================================
# LOAD DATA
# =============================================================================

print_section("LOADING CLEAN DATASET")

try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
except Exception as exc:
    fail(f"Could not read input dataset.\n\n{exc}")

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# =============================================================================
# STANDARDIZE COLUMN NAMES
# =============================================================================

print_section("STANDARDIZING COLUMN NAMES")

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.upper()
)

print("Column names standardized.")


# =============================================================================
# REQUIRED COLUMN CHECK
# =============================================================================

print_section("CHECKING REQUIRED COLUMNS")

required_columns = (
    ID_COLUMNS
    + FEATURE_COLUMNS
    + [TARGET_COLUMN]
    + [FUTURE_OUTCOME_COLUMN]
)

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    fail(
        "The following required columns are missing:\n\n"
        + "\n".join(f" - {col}" for col in missing_required)
    )

print("All required columns are present.")


# =============================================================================
# REMOVE MISSING INDICATOR COLUMNS
# =============================================================================

print_section("REMOVING UNNECESSARY MISSING-VALUE INDICATORS")

missing_indicator_columns = [
    col
    for col in df.columns
    if col.endswith(MISSING_INDICATOR_SUFFIX)
]

if missing_indicator_columns:

    print("Removing:")

    for col in missing_indicator_columns:
        print(f" - {col}")

    df = df.drop(columns=missing_indicator_columns)

else:
    print("No *_MISSING columns found.")


# =============================================================================
# IDENTIFIER VALIDATION
# =============================================================================

print_section("VALIDATING IDENTIFIERS")

for col in ["ACO_ID", "ACO_NAME", "STATE"]:
    missing_count = df[col].isna().sum()

    print(
        f"{col:<12}: "
        f"missing = {missing_count:,}"
    )

    if missing_count > 0:
        fail(
            f"{col} contains missing values. "
            "Fix the clean dataset before modeling."
        )


# =============================================================================
# YEAR VALIDATION
# =============================================================================

print_section("VALIDATING YEAR")

df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce"
)

if df["YEAR"].isna().any():
    fail("YEAR contains invalid or missing values.")

df["YEAR"] = df["YEAR"].astype(int)

print(
    f"Year range: "
    f"{df['YEAR'].min()}–{df['YEAR'].max()}"
)

print()
print("Rows by year:")
print(df["YEAR"].value_counts().sort_index())


# =============================================================================
# ACO-YEAR UNIQUENESS
# =============================================================================

print_section("CHECKING ACO-YEAR UNIQUENESS")

duplicate_mask = df.duplicated(
    subset=["ACO_ID", "YEAR"],
    keep=False
)

duplicate_count = int(duplicate_mask.sum())

print(
    f"Duplicate ACO-year rows: "
    f"{duplicate_count:,}"
)

if duplicate_count > 0:

    duplicate_keys = (
        df.loc[
            duplicate_mask,
            ["ACO_ID", "YEAR"]
        ]
        .drop_duplicates()
    )

    print()
    print("Duplicate keys:")
    print(duplicate_keys.head(20).to_string(index=False))

    fail(
        "Duplicate ACO-year observations detected. "
        "Do not train until this is fixed."
    )

print("ACO-year uniqueness PASSED.")


# =============================================================================
# TARGET VALIDATION
# =============================================================================

print_section("VALIDATING RISK TARGET")

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

if df[TARGET_COLUMN].isna().any():

    missing_target = int(
        df[TARGET_COLUMN].isna().sum()
    )

    fail(
        f"{TARGET_COLUMN} contains "
        f"{missing_target:,} missing values."
    )

unique_target_values = sorted(
    df[TARGET_COLUMN].unique().tolist()
)

print("Target values:")
print(unique_target_values)

invalid_target_values = [
    value
    for value in unique_target_values
    if value not in [0, 1]
]

if invalid_target_values:

    fail(
        "Invalid NEXT_YEAR_RISK values detected:\n"
        + str(invalid_target_values)
    )

df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

print()
print("Target validation PASSED.")


# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

print_section("TARGET DISTRIBUTION")

target_counts = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

print(target_counts)

risk_count = int(
    (df[TARGET_COLUMN] == 1).sum()
)

nonrisk_count = int(
    (df[TARGET_COLUMN] == 0).sum()
)

total_count = len(df)

risk_rate = risk_count / total_count

print()
print(f"Risk cases     : {risk_count:,}")
print(f"Non-risk cases : {nonrisk_count:,}")
print(f"Risk rate      : {risk_rate:.4%}")


# =============================================================================
# TARGET BY YEAR
# =============================================================================

print_section("NEXT_YEAR_RISK BY YEAR")

year_target_summary = (
    df.groupby("YEAR")
    .agg(
        rows=(TARGET_COLUMN, "size"),
        risk_cases=(TARGET_COLUMN, "sum"),
        risk_rate=(TARGET_COLUMN, "mean"),
    )
    .reset_index()
)

print(
    year_target_summary.to_string(
        index=False
    )
)


# =============================================================================
# NUMERIC FEATURE VALIDATION
# =============================================================================

print_section("VALIDATING MODEL FEATURES")

for col in FEATURE_COLUMNS:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    missing = int(df[col].isna().sum())

    infinite = int(
        np.isinf(
            df[col].dropna()
        ).sum()
    )

    print(
        f"{col:<38} "
        f"missing={missing:>6,} "
        f"infinite={infinite:>6,}"
    )

    if missing > 0:
        fail(
            f"Feature {col} contains missing values."
        )

    if infinite > 0:
        fail(
            f"Feature {col} contains infinite values."
        )


# =============================================================================
# FUTURE OUTCOME LEAKAGE CHECK
# =============================================================================

print_section("TARGET LEAKAGE CHECK")

print(
    "Future outcome column detected:"
)
print(
    f"  {FUTURE_OUTCOME_COLUMN}"
)

print()
print(
    "This column will NOT be used as a risk-model feature."
)

print()
print("Risk model features:")

for i, col in enumerate(
    FEATURE_COLUMNS,
    start=1
):
    print(
        f"{i:>2}. {col}"
    )

print()
print(
    f"Target: {TARGET_COLUMN}"
)

if FUTURE_OUTCOME_COLUMN in FEATURE_COLUMNS:

    fail(
        f"LEAKAGE ERROR: "
        f"{FUTURE_OUTCOME_COLUMN} is incorrectly included "
        "as a predictor."
    )

print()
print("Leakage check PASSED.")


# =============================================================================
# CHECK THAT TARGET IS NOT INCLUDED IN FEATURES
# =============================================================================

if TARGET_COLUMN in FEATURE_COLUMNS:
    fail(
        "NEXT_YEAR_RISK cannot be included as an input feature."
    )

if TARGET_COLUMN in ID_COLUMNS:
    fail(
        "NEXT_YEAR_RISK cannot be an identifier column."
    )


# =============================================================================
# HISTORICAL MODEL DATA
# =============================================================================

print_section("CREATING HISTORICAL MODELING DATA")

# 2024 is excluded from training because a real 2024 observation
# cannot know its genuine 2025 outcome at prediction time.
#
# Therefore:
#
# 2018–2023 -> model training/evaluation dataset
# 2024       -> future scoring dataset

model_df = df[
    df["YEAR"] < 2024
].copy()

scoring_df = df[
    df["YEAR"] == 2024
].copy()

print(
    f"Historical modeling rows : "
    f"{len(model_df):,}"
)

print(
    f"2024 scoring rows        : "
    f"{len(scoring_df):,}"
)

if len(model_df) == 0:
    fail(
        "No historical modeling observations remain."
    )

if len(scoring_df) == 0:
    print(
        "WARNING: No 2024 scoring observations found."
    )


# =============================================================================
# CREATE MODEL DATASET
# =============================================================================

print_section("CREATING FINAL RISK MODEL DATASET")

model_columns = (
    ID_COLUMNS
    + FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

model_data = model_df[
    model_columns
].copy()

# Sort for reproducibility
model_data = model_data.sort_values(
    by=["YEAR", "ACO_ID"]
).reset_index(drop=True)


# =============================================================================
# CREATE 2024 SCORING DATASET
# =============================================================================

print_section("CREATING 2024 SCORING DATASET")

scoring_columns = (
    ID_COLUMNS
    + FEATURE_COLUMNS
)

scoring_data = scoring_df[
    scoring_columns
].copy()

scoring_data = scoring_data.sort_values(
    by=["YEAR", "ACO_ID"]
).reset_index(drop=True)


# =============================================================================
# FINAL MODEL DATA VALIDATION
# =============================================================================

print_section("FINAL MODEL DATA VALIDATION")

print(
    f"Model rows    : {len(model_data):,}"
)

print(
    f"Model columns : {len(model_data.columns):,}"
)

print()
print("Model columns:")

for i, col in enumerate(
    model_data.columns,
    start=1
):
    print(
        f"{i:>2}. {col}"
    )


# Check missing values
model_missing = (
    model_data.isna()
    .sum()
)

model_missing = (
    model_missing[
        model_missing > 0
    ]
)

if len(model_missing) > 0:

    print()
    print(
        "Missing values detected:"
    )
    print(model_missing)

    fail(
        "Final modeling dataset contains missing values."
    )

print()
print(
    "No missing values in final modeling dataset."
)


# Check duplicate rows
duplicate_model_rows = int(
    model_data.duplicated().sum()
)

print(
    f"Exact duplicate rows: "
    f"{duplicate_model_rows:,}"
)

if duplicate_model_rows > 0:
    fail(
        "Duplicate rows detected in final model dataset."
    )


# Check ACO-year uniqueness
duplicate_model_keys = int(
    model_data.duplicated(
        subset=["ACO_ID", "YEAR"]
    ).sum()
)

print(
    f"Duplicate ACO-year rows: "
    f"{duplicate_model_keys:,}"
)

if duplicate_model_keys > 0:
    fail(
        "Duplicate ACO-year rows detected."
    )


# =============================================================================
# FINAL SCORING DATA VALIDATION
# =============================================================================

if len(scoring_data) > 0:

    print_section("2024 SCORING DATA VALIDATION")

    scoring_missing = (
        scoring_data.isna()
        .sum()
    )

    scoring_missing = (
        scoring_missing[
            scoring_missing > 0
        ]
    )

    if len(scoring_missing) > 0:

        print(
            "Missing values detected:"
        )
        print(scoring_missing)

        fail(
            "2024 scoring data contains missing values."
        )

    print(
        "2024 scoring data contains no missing values."
    )

    duplicate_scoring_keys = int(
        scoring_data.duplicated(
            subset=["ACO_ID", "YEAR"]
        ).sum()
    )

    print(
        f"Duplicate ACO-year rows: "
        f"{duplicate_scoring_keys:,}"
    )

    if duplicate_scoring_keys > 0:
        fail(
            "Duplicate ACO-year rows detected "
            "in 2024 scoring data."
        )


# =============================================================================
# SAVE MODEL DATASET
# =============================================================================

print_section("SAVING MODEL DATASET")

MODEL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_data.to_csv(
    MODEL_OUTPUT,
    index=False
)

print(
    "Risk model dataset saved:"
)

print(
    MODEL_OUTPUT
)


# =============================================================================
# SAVE SCORING DATASET
# =============================================================================

if len(scoring_data) > 0:

    print_section("SAVING 2024 SCORING DATASET")

    scoring_data.to_csv(
        SCORING_OUTPUT,
        index=False
    )

    print(
        "2024 scoring dataset saved:"
    )

    print(
        SCORING_OUTPUT
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_section("FINAL SUMMARY")

print(
    f"Original clean dataset : "
    f"{len(df):,} rows"
)

print(
    f"Risk model dataset     : "
    f"{len(model_data):,} rows"
)

print(
    f"2024 scoring dataset   : "
    f"{len(scoring_data):,} rows"
)

print()
print(
    f"Model years: "
    f"{model_data['YEAR'].min()}–"
    f"{model_data['YEAR'].max()}"
)

print()
print(
    "Risk model predictors:"
)

for col in FEATURE_COLUMNS:
    print(
        f" - {col}"
    )

print()
print(
    f"Risk target:"
)
print(
    f" - {TARGET_COLUMN}"
)

print()
print(
    "Excluded from predictors:"
)

print(
    f" - {FUTURE_OUTCOME_COLUMN}"
)

print(
    " - All *_MISSING columns"
)

print()
print(
    "2024 is reserved for future scoring."
)

print()
print("=" * 78)
print("RISK MODEL DATASET BUILD COMPLETE")
print("=" * 78)