from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# CONTRACTIQ — DATASET 1 VALIDATION
# =============================================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 CLEAN DATA VALIDATION")
print("=" * 78)


# =============================================================================
# PROJECT PATHS
# =============================================================================

# validate_risk_data.py is located at:
# ContractIQ(!)\preprocessing\preprocessing\validate_risk_data.py
#
# Therefore:
# parents[0] = preprocessing
# parents[1] = preprocessing
# parents[2] = ContractIQ(!)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DATASET = PROJECT_ROOT / "data" / "processed" / "Dataset1_Clean.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "preprocessing"
    / "dataset1"
    / "reports"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


print("\nPROJECT ROOT")
print("-" * 78)
print(PROJECT_ROOT)

print("\nCLEAN DATASET")
print("-" * 78)
print(CLEAN_DATASET)


# =============================================================================
# CHECK FILE
# =============================================================================

if not CLEAN_DATASET.exists():
    raise FileNotFoundError(
        f"""
Clean dataset was not found.

Expected:
{CLEAN_DATASET}

Please run preprocessing first.
"""
    )

print("\nClean dataset found.")


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 78)
print("LOADING CLEAN DATASET")
print("=" * 78)

df = pd.read_csv(CLEAN_DATASET, low_memory=False)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# =============================================================================
# BASIC STRUCTURE
# =============================================================================

print("\n" + "=" * 78)
print("BASIC DATASET VALIDATION")
print("=" * 78)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")

print("\nColumn names:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

required_columns = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
    "NEXT_YEAR_SAVINGS_RATE",
    "NEXT_YEAR_RISK",
]

print("\n" + "=" * 78)
print("REQUIRED COLUMN CHECK")
print("=" * 78)

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    print("MISSING REQUIRED COLUMNS:")
    for col in missing_required:
        print(f"  - {col}")

    raise ValueError(
        "Required columns are missing from the clean dataset."
    )

print("All required columns are present.")


# =============================================================================
# IDENTIFIER VALIDATION
# =============================================================================

print("\n" + "=" * 78)
print("IDENTIFIER VALIDATION")
print("=" * 78)

missing_aco_id = df["ACO_ID"].isna().sum()
missing_name = df["ACO_NAME"].isna().sum()
missing_state = df["STATE"].isna().sum()
missing_year = df["YEAR"].isna().sum()

print(f"Missing ACO_ID   : {missing_aco_id:,}")
print(f"Missing ACO_NAME : {missing_name:,}")
print(f"Missing STATE    : {missing_state:,}")
print(f"Missing YEAR     : {missing_year:,}")

duplicate_rows = df.duplicated().sum()

print(f"Exact duplicate rows : {duplicate_rows:,}")


# =============================================================================
# ACO-YEAR UNIQUENESS
# =============================================================================

print("\n" + "=" * 78)
print("ACO-YEAR UNIQUENESS")
print("=" * 78)

aco_year_duplicates = df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print(f"Duplicate ACO-year rows : {aco_year_duplicates:,}")

if aco_year_duplicates == 0:
    print("ACO-year uniqueness check PASSED.")
else:
    print("WARNING: Duplicate ACO-year records detected.")


# =============================================================================
# YEAR VALIDATION
# =============================================================================

print("\n" + "=" * 78)
print("YEAR VALIDATION")
print("=" * 78)

df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce"
)

print(df["YEAR"].value_counts().sort_index())

print("\nYear range:")

if df["YEAR"].notna().any():
    print(
        f"Minimum year : {int(df['YEAR'].min())}"
    )
    print(
        f"Maximum year : {int(df['YEAR'].max())}"
    )


# =============================================================================
# TARGET VALIDATION
# =============================================================================

print("\n" + "=" * 78)
print("RISK TARGET VALIDATION")
print("=" * 78)

risk = pd.to_numeric(
    df["NEXT_YEAR_RISK"],
    errors="coerce"
)

print("NEXT_YEAR_RISK values:")

print(
    risk.value_counts(
        dropna=False
    ).sort_index()
)

invalid_risk = risk[
    risk.notna() &
    ~risk.isin([0, 1])
]

print(
    f"\nInvalid risk values : {len(invalid_risk):,}"
)

if len(invalid_risk) == 0:
    print("Risk target values are valid: 0 / 1.")
else:
    print("WARNING: Invalid risk target values found.")


# =============================================================================
# TARGET BY YEAR
# =============================================================================

print("\n" + "=" * 78)
print("NEXT_YEAR_RISK BY YEAR")
print("=" * 78)

year_target = (
    df.assign(
        NEXT_YEAR_RISK_NUM=risk
    )
    .groupby("YEAR")
    .agg(
        rows=("YEAR", "size"),
        target_rows=("NEXT_YEAR_RISK_NUM", "count"),
        risk_cases=("NEXT_YEAR_RISK_NUM", "sum")
    )
    .reset_index()
)

year_target["risk_rate"] = np.where(
    year_target["target_rows"] > 0,
    year_target["risk_cases"] /
    year_target["target_rows"],
    np.nan
)

print(
    year_target.to_string(index=False)
)


# =============================================================================
# MISSING VALUE ANALYSIS
# =============================================================================

print("\n" + "=" * 78)
print("MISSING VALUE ANALYSIS")
print("=" * 78)

missing_report = pd.DataFrame({
    "COLUMN": df.columns,
    "MISSING": [
        df[col].isna().sum()
        for col in df.columns
    ]
})

missing_report["MISSING_PCT"] = (
    missing_report["MISSING"] /
    len(df) *
    100
)

missing_report = missing_report.sort_values(
    "MISSING_PCT",
    ascending=False
)

print(
    missing_report.head(30).to_string(index=False)
)


# =============================================================================
# ZERO VALUE ANALYSIS
# =============================================================================

print("\n" + "=" * 78)
print("ZERO VALUE ANALYSIS")
print("=" * 78)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

zero_rows = []

for col in numeric_columns:

    zero_count = (
        df[col] == 0
    ).sum()

    zero_pct = (
        zero_count /
        len(df) *
        100
    )

    zero_rows.append({
        "COLUMN": col,
        "ZERO_VALUES": zero_count,
        "ZERO_PCT": zero_pct
    })

zero_report = pd.DataFrame(zero_rows)

zero_report = zero_report.sort_values(
    "ZERO_PCT",
    ascending=False
)

print(
    zero_report.head(30).to_string(index=False)
)


# =============================================================================
# NEGATIVE VALUE ANALYSIS
# =============================================================================

print("\n" + "=" * 78)
print("NEGATIVE VALUE ANALYSIS")
print("=" * 78)

negative_rows = []

for col in numeric_columns:

    negative_count = (
        df[col] < 0
    ).sum()

    if negative_count > 0:

        negative_rows.append({
            "COLUMN": col,
            "NEGATIVE_VALUES": negative_count,
            "NEGATIVE_PCT":
                negative_count / len(df) * 100
        })

negative_report = pd.DataFrame(
    negative_rows
)

if len(negative_report) > 0:

    negative_report = negative_report.sort_values(
        "NEGATIVE_VALUES",
        ascending=False
    )

    print(
        negative_report.to_string(index=False)
    )

else:

    print("No negative numeric values found.")


# =============================================================================
# DATA TYPES
# =============================================================================

print("\n" + "=" * 78)
print("DATA TYPE VALIDATION")
print("=" * 78)

dtype_report = pd.DataFrame({
    "COLUMN": df.columns,
    "DATA_TYPE": [
        str(df[col].dtype)
        for col in df.columns
    ],
    "UNIQUE_VALUES": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

print(
    dtype_report.to_string(index=False)
)


# =============================================================================
# NUMERIC SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("NUMERIC STATISTICS")
print("=" * 78)

numeric_summary = df[
    numeric_columns
].describe().T

numeric_summary.to_csv(
    REPORT_DIR / "validation_numeric_statistics.csv"
)

print(
    numeric_summary.head(20).to_string()
)


# =============================================================================
# ACO SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("ACO SUMMARY")
print("=" * 78)

unique_acos = df["ACO_ID"].nunique()

print(
    f"Unique ACOs : {unique_acos:,}"
)

print(
    f"Rows        : {len(df):,}"
)

print(
    f"Average rows per ACO : "
    f"{len(df) / unique_acos:.2f}"
)


# =============================================================================
# YEAR SUMMARY
# =============================================================================

year_summary = (
    df.groupby("YEAR")
    .agg(
        ROWS=("YEAR", "size"),
        UNIQUE_ACOS=("ACO_ID", "nunique"),
        AVG_BENEFICIARIES=("N_AB", "mean"),
        MEDIAN_BENEFICIARIES=("N_AB", "median"),
    )
    .reset_index()
)

print("\nYear summary:")
print(
    year_summary.to_string(index=False)
)


# =============================================================================
# SAVE REPORTS
# =============================================================================

print("\n" + "=" * 78)
print("SAVING VALIDATION REPORTS")
print("=" * 78)

missing_report.to_csv(
    REPORT_DIR / "validation_missing_values.csv",
    index=False
)

zero_report.to_csv(
    REPORT_DIR / "validation_zero_values.csv",
    index=False
)

negative_report.to_csv(
    REPORT_DIR / "validation_negative_values.csv",
    index=False
)

dtype_report.to_csv(
    REPORT_DIR / "validation_data_types.csv",
    index=False
)

year_target.to_csv(
    REPORT_DIR / "validation_target_by_year.csv",
    index=False
)

year_summary.to_csv(
    REPORT_DIR / "validation_year_summary.csv",
    index=False
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print("\n" + "=" * 78)
print("FINAL VALIDATION STATUS")
print("=" * 78)

problems = []

if missing_aco_id > 0:
    problems.append(
        "Missing ACO_ID values"
    )

if missing_year > 0:
    problems.append(
        "Missing YEAR values"
    )

if duplicate_rows > 0:
    problems.append(
        "Exact duplicate rows"
    )

if aco_year_duplicates > 0:
    problems.append(
        "Duplicate ACO-year records"
    )

if len(invalid_risk) > 0:
    problems.append(
        "Invalid NEXT_YEAR_RISK values"
    )


if problems:

    print("VALIDATION STATUS: WARNING")

    print("\nIssues detected:")

    for problem in problems:
        print(f" - {problem}")

else:

    print("VALIDATION STATUS: PASSED")
    print("No structural validation errors detected.")


print("\nReports saved to:")
print(REPORT_DIR)

print("\n" + "=" * 78)
print("DATASET VALIDATION COMPLETE")
print("=" * 78)