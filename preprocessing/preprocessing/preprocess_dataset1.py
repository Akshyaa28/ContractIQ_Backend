# ================================================================
# CONTRACTIQ — DATASET 1 PREPROCESSING
# ================================================================
# Purpose:
#   Clean and validate the 50,000-row Dataset 1 before ML modeling.
#
# Input:
#   Datasets\Dataset1\Raw\Dataset1_ACO_Model_Master_50000(1).csv
#
# Output:
#   data\processed\Dataset1_Clean.csv
#
# Reports:
#   preprocessing\dataset1\reports\preprocessing_summary.csv
#   preprocessing\dataset1\reports\preprocessing_missing_values.csv
#   preprocessing\dataset1\reports\preprocessing_duplicates.csv
# ================================================================

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"I:\ContractIQ(!)"

RAW_FILE = os.path.join(
    BASE_DIR,
    "Datasets",
    "Dataset1",
    "Raw",
    "Dataset1_ACO_Model_Master_50000(1).csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "preprocessing",
    "dataset1",
    "reports"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "Dataset1_Clean.csv"
)


# ================================================================
# CREATE DIRECTORIES
# ================================================================

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def clean_column_name(column):
    """
    Convert column names into consistent uppercase snake_case.
    """

    column = str(column).strip()

    column = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        column
    )

    column = re.sub(
        r"_+",
        "_",
        column
    )

    column = column.strip("_").upper()

    return column


def find_column(df, candidates):
    """
    Find the first matching column from a list of candidates.
    """

    lookup = {
        str(c).upper(): c
        for c in df.columns
    }

    for candidate in candidates:

        candidate_upper = candidate.upper()

        if candidate_upper in lookup:
            return lookup[candidate_upper]

    return None


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 78)
print("CONTRACTIQ — DATASET 1 PREPROCESSING")
print("=" * 78)

print()
print("RAW DATASET")
print("-" * 78)
print(RAW_FILE)


# ================================================================
# CHECK INPUT
# ================================================================

if not os.path.exists(RAW_FILE):

    raise FileNotFoundError(
        f"""
Raw dataset was not found.

Expected:

{RAW_FILE}

Please check the file name and location.
"""
    )


# ================================================================
# LOAD DATA
# ================================================================

print()
print("Loading dataset...")

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

original_rows = len(df)
original_columns = len(df.columns)

print()
print("Dataset loaded successfully.")
print(f"Rows    : {original_rows:,}")
print(f"Columns : {original_columns:,}")


# ================================================================
# STANDARDIZE COLUMN NAMES
# ================================================================

print()
print("=" * 78)
print("STANDARDIZING COLUMN NAMES")
print("=" * 78)

df.columns = [
    clean_column_name(c)
    for c in df.columns
]

print("Column names standardized.")


# ================================================================
# HANDLE DUPLICATE COLUMN NAMES
# ================================================================

print()
print("Checking duplicate column names...")

duplicate_columns = df.columns[
    df.columns.duplicated()
].tolist()

if duplicate_columns:

    print(
        "Duplicate column names detected:"
    )

    print(duplicate_columns)

    new_columns = []
    counts = {}

    for col in df.columns:

        if col not in counts:

            counts[col] = 0
            new_columns.append(col)

        else:

            counts[col] += 1

            new_columns.append(
                f"{col}_{counts[col]}"
            )

    df.columns = new_columns

else:

    print("No duplicate column names.")


# ================================================================
# IDENTIFY IMPORTANT COLUMNS
# ================================================================

ACO_ID_COL = find_column(
    df,
    [
        "ACO_ID",
        "ACO_NUM",
        "ACO_NUMBER",
        "ACO"
    ]
)

ACO_NAME_COL = find_column(
    df,
    [
        "ACO_NAME",
        "ACO_NAME_1",
        "ACO_NAME_2"
    ]
)

STATE_COL = find_column(
    df,
    [
        "STATE",
        "ACO_STATE"
    ]
)

YEAR_COL = find_column(
    df,
    [
        "YEAR",
        "REPORTING_YEAR"
    ]
)

N_AB_COL = find_column(
    df,
    [
        "N_AB",
        "BENEFICIARIES",
        "NUMBER_OF_BENEFICIARIES"
    ]
)

RISK_TARGET_COL = find_column(
    df,
    [
        "NEXT_YEAR_RISK"
    ]
)

FORECAST_TARGET_COL = find_column(
    df,
    [
        "NEXT_YEAR_SAVINGS_RATE"
    ]
)


print()
print("=" * 78)
print("IMPORTANT COLUMNS")
print("=" * 78)

print(f"ACO ID             : {ACO_ID_COL}")
print(f"ACO Name           : {ACO_NAME_COL}")
print(f"State              : {STATE_COL}")
print(f"Year               : {YEAR_COL}")
print(f"Beneficiaries      : {N_AB_COL}")
print(f"Risk target        : {RISK_TARGET_COL}")
print(f"Forecast target    : {FORECAST_TARGET_COL}")


# ================================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# ================================================================

print()
print("=" * 78)
print("REMOVING COMPLETELY EMPTY COLUMNS")
print("=" * 78)

empty_columns = []

for col in df.columns:

    if df[col].isna().all():

        empty_columns.append(col)

if empty_columns:

    print(
        f"Completely empty columns removed: "
        f"{len(empty_columns)}"
    )

    for col in empty_columns:
        print(f"  - {col}")

    df = df.drop(
        columns=empty_columns
    )

else:

    print("No completely empty columns.")


# ================================================================
# REMOVE EXACT DUPLICATE ROWS
# ================================================================

print()
print("=" * 78)
print("DUPLICATE ROW CHECK")
print("=" * 78)

exact_duplicates = int(
    df.duplicated().sum()
)

print(
    f"Exact duplicate rows: "
    f"{exact_duplicates:,}"
)

if exact_duplicates > 0:

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    print(
        f"Removed {exact_duplicates:,} "
        "exact duplicate rows."
    )

else:

    print("No exact duplicate rows.")


# ================================================================
# CLEAN IDENTIFIERS
# ================================================================

print()
print("=" * 78)
print("CLEANING IDENTIFIERS")
print("=" * 78)

if ACO_ID_COL:

    df[ACO_ID_COL] = (
        df[ACO_ID_COL]
        .astype("string")
        .str.strip()
    )

    df[ACO_ID_COL] = (
        df[ACO_ID_COL]
        .replace(
            {
                "": pd.NA,
                "NAN": pd.NA,
                "NONE": pd.NA,
                "NULL": pd.NA
            }
        )
    )

if ACO_NAME_COL:

    df[ACO_NAME_COL] = (
        df[ACO_NAME_COL]
        .astype("string")
        .str.strip()
    )

    df[ACO_NAME_COL] = (
        df[ACO_NAME_COL]
        .replace(
            {
                "": pd.NA,
                "NAN": pd.NA,
                "NONE": pd.NA,
                "NULL": pd.NA
            }
        )
    )

if STATE_COL:

    df[STATE_COL] = (
        df[STATE_COL]
        .astype("string")
        .str.strip()
    )

    df[STATE_COL] = (
        df[STATE_COL]
        .replace(
            {
                "": pd.NA,
                "NAN": pd.NA,
                "NONE": pd.NA,
                "NULL": pd.NA
            }
        )
    )


# ================================================================
# YEAR
# ================================================================

if YEAR_COL:

    print()
    print("Cleaning YEAR...")

    df[YEAR_COL] = pd.to_numeric(
        df[YEAR_COL],
        errors="coerce"
    )

    df[YEAR_COL] = (
        df[YEAR_COL]
        .round()
        .astype("Int64")
    )

    print(
        "Year range:",
        df[YEAR_COL].min(),
        "to",
        df[YEAR_COL].max()
    )


# ================================================================
# NUMERIC CONVERSION
# ================================================================

print()
print("=" * 78)
print("CONVERTING NUMERIC COLUMNS")
print("=" * 78)

identifier_columns = {
    c for c in [
        ACO_ID_COL,
        ACO_NAME_COL,
        STATE_COL,
        YEAR_COL
    ]
    if c is not None
}

numeric_candidates = []

for col in df.columns:

    if col in identifier_columns:
        continue

    numeric_candidates.append(col)


converted_numeric = 0

for col in numeric_candidates:

    if df[col].dtype == "object" or str(
        df[col].dtype
    ).startswith("string"):

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        original_non_null = (
            df[col].notna().sum()
        )

        converted_non_null = (
            converted.notna().sum()
        )

        # Convert only when the column is
        # predominantly numeric.
        if (
            original_non_null > 0
            and
            converted_non_null
            / original_non_null
            >= 0.90
        ):

            df[col] = converted
            converted_numeric += 1

print(
    f"Numeric columns converted: "
    f"{converted_numeric:,}"
)


# ================================================================
# INFINITE VALUE CHECK
# ================================================================

print()
print("=" * 78)
print("INFINITE VALUE CHECK")
print("=" * 78)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

infinite_summary = []

for col in numeric_columns:

    count_inf = np.isinf(
        df[col].to_numpy(
            dtype=float,
            na_value=np.nan
        )
    ).sum()

    if count_inf > 0:

        infinite_summary.append(
            {
                "COLUMN": col,
                "INFINITE_VALUES": int(count_inf)
            }
        )

        df[col] = df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

if infinite_summary:

    infinite_df = pd.DataFrame(
        infinite_summary
    )

    print(infinite_df.to_string(
        index=False
    ))

else:

    print("No infinite values found.")


# ================================================================
# NEGATIVE VALUE CHECK
# ================================================================

print()
print("=" * 78)
print("NEGATIVE VALUE CHECK")
print("=" * 78)

negative_summary = []

for col in numeric_columns:

    negative_count = int(
        (df[col] < 0).sum()
    )

    if negative_count > 0:

        negative_summary.append(
            {
                "COLUMN": col,
                "NEGATIVE_VALUES": negative_count
            }
        )

if negative_summary:

    negative_df = pd.DataFrame(
        negative_summary
    )

    print(
        "Negative values detected."
    )

    print(
        negative_df
        .sort_values(
            "NEGATIVE_VALUES",
            ascending=False
        )
        .head(30)
        .to_string(index=False)
    )

    negative_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "negative_values_after_preprocessing.csv"
        ),
        index=False
    )

else:

    print("No negative values detected.")


# ================================================================
# ZERO VALUE CHECK
# ================================================================

print()
print("=" * 78)
print("ZERO VALUE CHECK")
print("=" * 78)

zero_summary = []

for col in numeric_columns:

    zero_count = int(
        (df[col] == 0).sum()
    )

    if zero_count > 0:

        zero_summary.append(
            {
                "COLUMN": col,
                "ZERO_VALUES": zero_count,
                "ZERO_PCT":
                    round(
                        zero_count
                        / len(df)
                        * 100,
                        2
                    )
            }
        )

zero_df = pd.DataFrame(
    zero_summary
)

if not zero_df.empty:

    zero_df = zero_df.sort_values(
        "ZERO_PCT",
        ascending=False
    )

    print(
        zero_df.head(30).to_string(
            index=False
        )
    )

    zero_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "zero_value_analysis_after_preprocessing.csv"
        ),
        index=False
    )

else:

    print("No zero values detected.")


# ================================================================
# TARGET PROTECTION
# ================================================================

print()
print("=" * 78)
print("TARGET VALIDATION")
print("=" * 78)

if RISK_TARGET_COL:

    print(
        f"Risk target: {RISK_TARGET_COL}"
    )

    df[RISK_TARGET_COL] = pd.to_numeric(
        df[RISK_TARGET_COL],
        errors="coerce"
    )

    available_risk = df[
        RISK_TARGET_COL
    ].dropna()

    if len(available_risk) > 0:

        print(
            "Risk target values:"
        )

        print(
            sorted(
                available_risk.unique()
            )
        )

        invalid_risk = (
            ~available_risk.isin([0, 1])
        ).sum()

        print(
            f"Invalid risk target values: "
            f"{invalid_risk}"
        )

        if invalid_risk > 0:

            df.loc[
                ~df[RISK_TARGET_COL].isin(
                    [0, 1]
                ),
                RISK_TARGET_COL
            ] = np.nan


if FORECAST_TARGET_COL:

    print(
        f"Forecast target: "
        f"{FORECAST_TARGET_COL}"
    )

    df[FORECAST_TARGET_COL] = pd.to_numeric(
        df[FORECAST_TARGET_COL],
        errors="coerce"
    )


# ================================================================
# MODEL FEATURE MISSINGNESS FLAGS
# ================================================================

print()
print("=" * 78)
print("CREATING MISSING-VALUE INDICATORS")
print("=" * 78)

important_model_features = [
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE"
]

created_flags = []

for feature in important_model_features:

    if feature in df.columns:

        flag_name = (
            feature + "_MISSING"
        )

        df[flag_name] = (
            df[feature]
            .isna()
            .astype("int8")
        )

        created_flags.append(
            flag_name
        )

print(
    f"Missing-value indicators created: "
    f"{len(created_flags)}"
)


# ================================================================
# MISSING VALUE REPORT
# ================================================================

print()
print("=" * 78)
print("MISSING VALUE ANALYSIS")
print("=" * 78)

missing_rows = []

for col in df.columns:

    missing_count = int(
        df[col].isna().sum()
    )

    missing_pct = (
        missing_count
        / len(df)
        * 100
    )

    missing_rows.append(
        {
            "COLUMN": col,
            "MISSING": missing_count,
            "MISSING_PCT": round(
                missing_pct,
                4
            )
        }
    )

missing_df = pd.DataFrame(
    missing_rows
)

missing_df = missing_df.sort_values(
    "MISSING_PCT",
    ascending=False
)

missing_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "preprocessing_missing_values.csv"
    ),
    index=False
)

print(
    missing_df.head(20).to_string(
        index=False
    )
)


# ================================================================
# IMPORTANT:
# DO NOT BLINDLY IMPUTE MISSING VALUES HERE
# ================================================================
#
# Why?
#
# Some missing values are structurally meaningful.
#
# Example:
#
# ACO first observed in 2019
# -> no 2018 historical value
# -> PREVIOUS_SAVINGS_RATE is legitimately missing
#
# Replacing that with 0 would falsely tell the model:
#
# "The ACO had a 0% savings rate."
#
# Instead:
#
# - retain NaN in the clean dataset
# - later use a model Pipeline
# - fit the imputer only on the training data
#
# This prevents data leakage.
# ================================================================


# ================================================================
# SORT DATA
# ================================================================

print()
print("=" * 78)
print("SORTING DATA")
print("=" * 78)

sort_columns = []

if ACO_ID_COL:
    sort_columns.append(ACO_ID_COL)

if YEAR_COL:
    sort_columns.append(YEAR_COL)

if sort_columns:

    df = df.sort_values(
        sort_columns,
        kind="stable"
    ).reset_index(
        drop=True
    )


# ================================================================
# FINAL DATA TYPE CHECK
# ================================================================

print()
print("=" * 78)
print("FINAL DATA TYPES")
print("=" * 78)

dtype_report = pd.DataFrame(
    {
        "COLUMN": df.columns,
        "DATA_TYPE": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "MISSING": [
            int(df[c].isna().sum())
            for c in df.columns
        ]
    }
)

dtype_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "preprocessing_data_types.csv"
    ),
    index=False
)


# ================================================================
# FINAL ACO-YEAR CHECK
# ================================================================

print()
print("=" * 78)
print("ACO-YEAR VALIDATION")
print("=" * 78)

duplicate_aco_years = 0

if ACO_ID_COL and YEAR_COL:

    valid_keys = df[
        ACO_ID_COL
    ].notna() & df[
        YEAR_COL
    ].notna()

    key_duplicates = df.loc[
        valid_keys
    ].duplicated(
        subset=[
            ACO_ID_COL,
            YEAR_COL
        ]
    )

    duplicate_aco_years = int(
        key_duplicates.sum()
    )

    print(
        f"Duplicate ACO-year rows: "
        f"{duplicate_aco_years:,}"
    )

    if duplicate_aco_years > 0:

        print(
            "WARNING: duplicate ACO-year "
            "records remain."
        )

    else:

        print(
            "ACO-year uniqueness check passed."
        )


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 78)
print("FINAL DATASET SUMMARY")
print("=" * 78)

print(
    f"Original rows       : "
    f"{original_rows:,}"
)

print(
    f"Final rows          : "
    f"{len(df):,}"
)

print(
    f"Original columns    : "
    f"{original_columns:,}"
)

print(
    f"Final columns       : "
    f"{len(df.columns):,}"
)

if ACO_ID_COL:

    print(
        f"Unique ACOs        : "
        f"{df[ACO_ID_COL].nunique(dropna=True):,}"
    )

if YEAR_COL:

    print(
        f"Year range          : "
        f"{df[YEAR_COL].min()}–"
        f"{df[YEAR_COL].max()}"
    )

print(
    f"Exact duplicate rows: "
    f"{df.duplicated().sum():,}"
)

if ACO_ID_COL:

    print(
        f"Missing ACO IDs     : "
        f"{df[ACO_ID_COL].isna().sum():,}"
    )

if YEAR_COL:

    print(
        f"Missing YEAR        : "
        f"{df[YEAR_COL].isna().sum():,}"
    )


# ================================================================
# SAVE CLEAN DATASET
# ================================================================

print()
print("=" * 78)
print("SAVING CLEAN DATASET")
print("=" * 78)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    "Clean dataset saved successfully:"
)

print(
    OUTPUT_FILE
)


# ================================================================
# SAVE PREPROCESSING SUMMARY
# ================================================================

summary = pd.DataFrame(
    [
        {
            "METRIC": "ORIGINAL_ROWS",
            "VALUE": original_rows
        },
        {
            "METRIC": "FINAL_ROWS",
            "VALUE": len(df)
        },
        {
            "METRIC": "ORIGINAL_COLUMNS",
            "VALUE": original_columns
        },
        {
            "METRIC": "FINAL_COLUMNS",
            "VALUE": len(df.columns)
        },
        {
            "METRIC": "EXACT_DUPLICATES_REMOVED",
            "VALUE": exact_duplicates
        },
        {
            "METRIC": "EMPTY_COLUMNS_REMOVED",
            "VALUE": len(empty_columns)
        },
        {
            "METRIC": "DUPLICATE_ACO_YEAR_ROWS",
            "VALUE": duplicate_aco_years
        },
        {
            "METRIC": "MISSING_ACO_IDS",
            "VALUE":
                (
                    int(df[ACO_ID_COL].isna().sum())
                    if ACO_ID_COL
                    else -1
                )
        },
        {
            "METRIC": "MISSING_YEAR",
            "VALUE":
                (
                    int(df[YEAR_COL].isna().sum())
                    if YEAR_COL
                    else -1
                )
        }
    ]
)

summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "preprocessing_summary.csv"
    ),
    index=False
)


# ================================================================
# FINAL STATUS
# ================================================================

print()
print("=" * 78)
print("PREPROCESSING COMPLETE")
print("=" * 78)

print()
print("OUTPUT:")
print(
    OUTPUT_FILE
)

print()
print("REPORTS:")
print(
    REPORT_DIR
)

print()
print("IMPORTANT:")
print(
    "Missing historical values were NOT blindly "
    "converted to zero."
)

print(
    "Legitimate zero values were preserved."
)

print(
    "Target columns were preserved for model training."
)

print()
print("=" * 78)