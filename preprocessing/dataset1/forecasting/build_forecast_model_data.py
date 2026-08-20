from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONTRACTIQ — DATASET 1 FORECAST MODEL DATASET BUILDER
# ============================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 SAVINGS FORECAST DATASET BUILDER")
print("=" * 78)


# ============================================================
# PROJECT ROOT
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# build_forecast_model_data.py
# forecasting
# dataset1
# preprocessing
# ContractIQ(!)

PROJECT_ROOT = SCRIPT_PATH.parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "Dataset1_Clean.csv"

MODEL_OUTPUT_PATH = (
    DATA_DIR / "Dataset1_Model_Forecast.csv"
)

SCORING_OUTPUT_PATH = (
    DATA_DIR / "Dataset1_Scoring_2024_Forecast.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
]

TARGET = "NEXT_YEAR_SAVINGS_RATE"

IDENTIFIERS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]

REQUIRED_COLUMNS = (
    IDENTIFIERS
    + FEATURES
    + [TARGET]
)


# ============================================================
# PATH CHECK
# ============================================================

print("\n" + "=" * 78)
print("CHECKING INPUT DATASET")
print("=" * 78)

print(f"Project root:\n{PROJECT_ROOT}")
print(f"\nInput:\n{INPUT_PATH}")
print(f"\nForecast output:\n{MODEL_OUTPUT_PATH}")
print(f"\n2024 scoring output:\n{SCORING_OUTPUT_PATH}")


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"""
Clean dataset was not found.

Expected:
{INPUT_PATH}

Run:
python preprocess_dataset1.py
"""
    )

print("\nInput dataset found.")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 78)
print("LOADING CLEAN DATASET")
print("=" * 78)

df = pd.read_csv(INPUT_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

print("\n" + "=" * 78)
print("STANDARDIZING COLUMN NAMES")
print("=" * 78)

df.columns = (
    df.columns
    .str.strip()
    .str.upper()
)

print("Column names standardized.")


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

print("\n" + "=" * 78)
print("CHECKING REQUIRED COLUMNS")
print("=" * 78)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )

print("All required columns are present.")


# ============================================================
# REMOVE MISSING INDICATOR COLUMNS
# ============================================================

missing_indicator_columns = [
    column
    for column in df.columns
    if column.endswith("_MISSING")
]

if missing_indicator_columns:

    print("\n" + "=" * 78)
    print("REMOVING MISSING-VALUE INDICATORS")
    print("=" * 78)

    for column in missing_indicator_columns:
        print(f" - {column}")

    df = df.drop(
        columns=missing_indicator_columns
    )


# ============================================================
# IDENTIFIER VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("VALIDATING IDENTIFIERS")
print("=" * 78)

for column in IDENTIFIERS[:3]:

    missing = df[column].isna().sum()

    print(
        f"{column:<12}: missing = {missing:,}"
    )

    if missing > 0:

        raise ValueError(
            f"Identifier {column} contains missing values."
        )


# ============================================================
# YEAR VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("VALIDATING YEAR")
print("=" * 78)

df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce"
)

if df["YEAR"].isna().any():

    raise ValueError(
        "YEAR contains invalid values."
    )

df["YEAR"] = df["YEAR"].astype(int)

print(
    f"Year range: "
    f"{df['YEAR'].min()}–{df['YEAR'].max()}"
)

print("\nRows by year:")
print(
    df["YEAR"]
    .value_counts()
    .sort_index()
)


# ============================================================
# ACO-YEAR UNIQUENESS
# ============================================================

print("\n" + "=" * 78)
print("CHECKING ACO-YEAR UNIQUENESS")
print("=" * 78)

duplicate_aco_year = df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print(
    f"Duplicate ACO-year rows: "
    f"{duplicate_aco_year:,}"
)

if duplicate_aco_year > 0:

    raise ValueError(
        "Duplicate ACO-year rows detected."
    )

print("ACO-year uniqueness PASSED.")


# ============================================================
# NUMERIC VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("VALIDATING FORECAST FEATURES")
print("=" * 78)

for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    missing = df[column].isna().sum()

    infinite = np.isinf(
        df[column].to_numpy()
    ).sum()

    print(
        f"{column:<38}"
        f"missing={missing:>7,} "
        f"infinite={infinite:>7,}"
    )

    if missing > 0:
        raise ValueError(
            f"{column} contains missing values."
        )

    if infinite > 0:
        raise ValueError(
            f"{column} contains infinite values."
        )


# ============================================================
# TARGET VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("VALIDATING FORECAST TARGET")
print("=" * 78)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

target_missing = df[TARGET].isna().sum()

target_infinite = np.isinf(
    df[TARGET].to_numpy()
).sum()

print(
    f"Target missing   : {target_missing:,}"
)

print(
    f"Target infinite  : {target_infinite:,}"
)

if target_missing > 0:

    raise ValueError(
        "NEXT_YEAR_SAVINGS_RATE contains missing values."
    )

if target_infinite > 0:

    raise ValueError(
        "NEXT_YEAR_SAVINGS_RATE contains infinite values."
    )


print("\nTarget statistics:")
print(
    df[TARGET].describe()
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 78)
print("TARGET LEAKAGE CHECK")
print("=" * 78)

print(
    f"""
Forecast target:
 - {TARGET}

The target is NOT included in the model features.

Forecast model features:
"""
)

for index, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{index}. {feature}"
    )


if TARGET in FEATURES:

    raise ValueError(
        "TARGET LEAKAGE DETECTED: "
        "NEXT_YEAR_SAVINGS_RATE is included as a feature."
    )

print("\nLeakage check PASSED.")


# ============================================================
# HISTORICAL MODEL DATA
# ============================================================

print("\n" + "=" * 78)
print("CREATING HISTORICAL FORECAST DATA")
print("=" * 78)

historical_df = df[
    df["YEAR"] < 2024
].copy()

scoring_df = df[
    df["YEAR"] == 2024
].copy()

print(
    f"Historical modeling rows : "
    f"{len(historical_df):,}"
)

print(
    f"2024 scoring rows        : "
    f"{len(scoring_df):,}"
)


if len(historical_df) == 0:

    raise ValueError(
        "No historical modeling rows found."
    )

if len(scoring_df) == 0:

    raise ValueError(
        "No 2024 scoring rows found."
    )


# ============================================================
# MODEL DATASET
# ============================================================

model_columns = (
    IDENTIFIERS
    + FEATURES
    + [TARGET]
)

model_df = historical_df[
    model_columns
].copy()


# ============================================================
# SCORING DATASET
# ============================================================

scoring_columns = (
    IDENTIFIERS
    + FEATURES
)

scoring_df = scoring_df[
    scoring_columns
].copy()


# ============================================================
# FINAL MODEL VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("FINAL FORECAST MODEL DATA VALIDATION")
print("=" * 78)

print(
    f"Model rows    : {len(model_df):,}"
)

print(
    f"Model columns : {len(model_df.columns):,}"
)

missing_model_values = (
    model_df[
        FEATURES + [TARGET]
    ]
    .isna()
    .sum()
    .sum()
)

if missing_model_values > 0:

    raise ValueError(
        "Missing values found in model dataset."
    )

duplicate_rows = model_df.duplicated().sum()

duplicate_aco_year = model_df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print(
    f"Missing feature/target values : "
    f"{missing_model_values:,}"
)

print(
    f"Exact duplicate rows          : "
    f"{duplicate_rows:,}"
)

print(
    f"Duplicate ACO-year rows       : "
    f"{duplicate_aco_year:,}"
)


# ============================================================
# 2024 SCORING VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("2024 SCORING DATA VALIDATION")
print("=" * 78)

scoring_missing = (
    scoring_df[FEATURES]
    .isna()
    .sum()
    .sum()
)

print(
    f"Missing scoring values: "
    f"{scoring_missing:,}"
)

if scoring_missing > 0:

    raise ValueError(
        "2024 scoring dataset contains missing values."
    )

print(
    "2024 scoring data contains no missing values."
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 78)
print("SAVING FORECAST MODEL DATASET")
print("=" * 78)

model_df.to_csv(
    MODEL_OUTPUT_PATH,
    index=False
)

print(
    f"Saved:\n{MODEL_OUTPUT_PATH}"
)


print("\n" + "=" * 78)
print("SAVING 2024 FORECAST SCORING DATASET")
print("=" * 78)

scoring_df.to_csv(
    SCORING_OUTPUT_PATH,
    index=False
)

print(
    f"Saved:\n{SCORING_OUTPUT_PATH}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("FINAL SUMMARY")
print("=" * 78)

print(
    f"""
Original clean dataset : {len(df):,} rows
Forecast model dataset : {len(model_df):,} rows
2024 scoring dataset    : {len(scoring_df):,} rows

Model years:
{model_df["YEAR"].min()}–{model_df["YEAR"].max()}

Forecast target:
 - {TARGET}

Forecast predictors:
"""
)

for feature in FEATURES:

    print(
        f" - {feature}"
    )

print(
    """
2024 is reserved for future forecasting.

No future target is used as a predictor.
"""
)

print("=" * 78)
print("FORECAST DATASET BUILD COMPLETE")
print("=" * 78)