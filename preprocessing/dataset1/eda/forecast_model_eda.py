import os
import sys
import json
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt


# =============================================================================
# CONTRACTIQ — DATASET 1 FORECAST MODEL EDA
# =============================================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 FORECAST MODEL EDA")
print("=" * 78)


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "Dataset1_Model_Forecast.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "preprocessing"
    / "dataset1"
    / "reports"
    / "forecast_model_eda"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "preprocessing"
    / "dataset1"
    / "plots"
    / "forecast_model_eda"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

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


# =============================================================================
# INPUT CHECK
# =============================================================================

print()
print("=" * 78)
print("PROJECT ROOT")
print("=" * 78)
print(PROJECT_ROOT)

print()
print("=" * 78)
print("INPUT DATASET")
print("=" * 78)
print(INPUT_PATH)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"""
Forecast model dataset was not found.

Expected:
{INPUT_PATH}

Run:

python build_forecast_model_data.py
"""
    )

print()
print("Input dataset found.")


# =============================================================================
# LOAD DATA
# =============================================================================

print()
print("=" * 78)
print("LOADING FORECAST MODEL DATASET")
print("=" * 78)

df = pd.read_csv(INPUT_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# =============================================================================
# STANDARDIZE COLUMN NAMES
# =============================================================================

print()
print("=" * 78)
print("STANDARDIZING COLUMN NAMES")
print("=" * 78)

df.columns = (
    df.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_")
)

print("Column names standardized.")


# =============================================================================
# REQUIRED COLUMN CHECK
# =============================================================================

print()
print("=" * 78)
print("CHECKING REQUIRED COLUMNS")
print("=" * 78)

required_columns = IDENTIFIERS + FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(f" - {x}" for x in missing_columns)
    )

print("All required columns are present.")


# =============================================================================
# BASIC VALIDATION
# =============================================================================

print()
print("=" * 78)
print("BASIC VALIDATION")
print("=" * 78)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

exact_duplicates = df.duplicated().sum()

aco_year_duplicates = df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print()
print(f"Exact duplicate rows    : {exact_duplicates}")
print(f"Duplicate ACO-year rows : {aco_year_duplicates}")

if exact_duplicates > 0:
    print("WARNING: Exact duplicate rows detected.")

if aco_year_duplicates > 0:
    raise ValueError("Duplicate ACO-year rows detected.")


# =============================================================================
# MISSING VALUE ANALYSIS
# =============================================================================

print()
print("=" * 78)
print("MISSING VALUE ANALYSIS")
print("=" * 78)

missing_report = []

for col in required_columns:

    missing = int(df[col].isna().sum())
    pct = missing / len(df) * 100

    missing_report.append({
        "COLUMN": col,
        "MISSING": missing,
        "MISSING_PCT": pct
    })

missing_df = pd.DataFrame(missing_report)

print(
    missing_df.to_string(
        index=False,
        formatters={
            "MISSING_PCT": "{:.4f}".format
        }
    )
)

missing_df.to_csv(
    REPORT_DIR / "missing_value_analysis.csv",
    index=False
)


# =============================================================================
# YEAR ANALYSIS
# =============================================================================

print()
print("=" * 78)
print("YEAR ANALYSIS")
print("=" * 78)

print(
    f"Year range: "
    f"{int(df['YEAR'].min())}–{int(df['YEAR'].max())}"
)

year_counts = (
    df.groupby("YEAR")
    .size()
    .reset_index(name="ROWS")
)

print()
print(year_counts.to_string(index=False))

year_counts.to_csv(
    REPORT_DIR / "rows_by_year.csv",
    index=False
)


# =============================================================================
# TARGET ANALYSIS
# =============================================================================

print()
print("=" * 78)
print("FORECAST TARGET ANALYSIS")
print("=" * 78)

target = df[TARGET]

target_stats = target.describe()

print(target_stats.to_string())

print()
print(f"Target mean   : {target.mean():.8f}")
print(f"Target median : {target.median():.8f}")
print(f"Target std    : {target.std():.8f}")
print(f"Target min    : {target.min():.8f}")
print(f"Target max    : {target.max():.8f}")

target_stats.to_csv(
    REPORT_DIR / "target_statistics.csv"
)


# =============================================================================
# TARGET BY YEAR
# =============================================================================

print()
print("=" * 78)
print("TARGET STATISTICS BY YEAR")
print("=" * 78)

target_by_year = (
    df.groupby("YEAR")[TARGET]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .reset_index()
)

print(target_by_year.to_string(index=False))

target_by_year.to_csv(
    REPORT_DIR / "target_by_year.csv",
    index=False
)


# =============================================================================
# FEATURE STATISTICS
# =============================================================================

print()
print("=" * 78)
print("FEATURE STATISTICS")
print("=" * 78)

feature_stats = df[FEATURES].describe().T

feature_stats["MISSING"] = df[FEATURES].isna().sum()
feature_stats["MISSING_PCT"] = (
    feature_stats["MISSING"] / len(df) * 100
)

print(feature_stats.to_string())

feature_stats.to_csv(
    REPORT_DIR / "feature_statistics.csv"
)


# =============================================================================
# NEGATIVE VALUE ANALYSIS
# =============================================================================

print()
print("=" * 78)
print("NEGATIVE VALUE ANALYSIS")
print("=" * 78)

negative_rows = []

for feature in FEATURES + [TARGET]:

    count = int((df[feature] < 0).sum())
    pct = count / len(df) * 100

    negative_rows.append({
        "FEATURE": feature,
        "NEGATIVE_VALUES": count,
        "NEGATIVE_PCT": pct
    })

negative_df = pd.DataFrame(negative_rows)

print(
    negative_df.to_string(
        index=False,
        formatters={
            "NEGATIVE_PCT": "{:.4f}".format
        }
    )
)

negative_df.to_csv(
    REPORT_DIR / "negative_value_analysis.csv",
    index=False
)


# =============================================================================
# ZERO VALUE ANALYSIS
# =============================================================================

print()
print("=" * 78)
print("ZERO VALUE ANALYSIS")
print("=" * 78)

zero_rows = []

for feature in FEATURES + [TARGET]:

    count = int((df[feature] == 0).sum())
    pct = count / len(df) * 100

    zero_rows.append({
        "FEATURE": feature,
        "ZERO_VALUES": count,
        "ZERO_PCT": pct
    })

zero_df = pd.DataFrame(zero_rows)

print(
    zero_df.to_string(
        index=False,
        formatters={
            "ZERO_PCT": "{:.4f}".format
        }
    )
)

zero_df.to_csv(
    REPORT_DIR / "zero_value_analysis.csv",
    index=False
)


# =============================================================================
# OUTLIER ANALYSIS
# =============================================================================

print()
print("=" * 78)
print("OUTLIER ANALYSIS")
print("=" * 78)

outlier_rows = []

for feature in FEATURES + [TARGET]:

    series = df[feature]

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (series < lower) |
        (series > upper)
    ).sum()

    outlier_pct = outliers / len(df) * 100

    outlier_rows.append({
        "FEATURE": feature,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LOWER_BOUND": lower,
        "UPPER_BOUND": upper,
        "OUTLIERS": int(outliers),
        "OUTLIER_PCT": outlier_pct,
    })

outlier_df = pd.DataFrame(outlier_rows)

print(
    outlier_df.to_string(
        index=False,
        formatters={
            "OUTLIER_PCT": "{:.4f}".format
        }
    )
)

outlier_df.to_csv(
    REPORT_DIR / "outlier_analysis.csv",
    index=False
)


# =============================================================================
# FEATURE-TARGET CORRELATION
# =============================================================================

print()
print("=" * 78)
print("FEATURE — TARGET CORRELATION")
print("=" * 78)

correlations = []

for feature in FEATURES:

    corr = df[feature].corr(df[TARGET])

    correlations.append({
        "FEATURE": feature,
        "TARGET_CORRELATION": corr
    })

correlation_df = (
    pd.DataFrame(correlations)
    .sort_values(
        "TARGET_CORRELATION",
        key=lambda x: x.abs(),
        ascending=False
    )
)

print(correlation_df.to_string(index=False))

correlation_df.to_csv(
    REPORT_DIR / "feature_target_correlation.csv",
    index=False
)


# =============================================================================
# FEATURE VS TARGET REGRESSION VISUALIZATIONS
# =============================================================================

print()
print("Creating forecast feature distribution plots...")

for feature in FEATURES:

    plt.figure(figsize=(9, 6))

    plt.scatter(
        df[feature],
        df[TARGET],
        alpha=0.15,
        s=8
    )

    plt.xlabel(feature)
    plt.ylabel(TARGET)
    plt.title(f"{feature} vs {TARGET}")
    plt.tight_layout()

    filename = (
        feature.lower()
        + "_vs_target.png"
    )

    plt.savefig(
        PLOT_DIR / filename,
        dpi=150
    )

    plt.close()


# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

print("Creating target distribution plot...")

plt.figure(figsize=(9, 6))

plt.hist(
    df[TARGET],
    bins=50
)

plt.xlabel(TARGET)
plt.ylabel("Frequency")
plt.title("Distribution of NEXT_YEAR_SAVINGS_RATE")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "target_distribution.png",
    dpi=150
)

plt.close()


# =============================================================================
# TARGET BY YEAR
# =============================================================================

print("Creating target-by-year plot...")

year_means = (
    df.groupby("YEAR")[TARGET]
    .mean()
)

plt.figure(figsize=(9, 6))

plt.plot(
    year_means.index,
    year_means.values,
    marker="o"
)

plt.xlabel("YEAR")
plt.ylabel("Mean NEXT_YEAR_SAVINGS_RATE")
plt.title("Forecast Target Mean by Year")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "target_mean_by_year.png",
    dpi=150
)

plt.close()


# =============================================================================
# MODEL READINESS
# =============================================================================

print()
print("=" * 78)
print("MODEL READINESS CHECK")
print("=" * 78)

readiness_rows = []

for feature in FEATURES:

    missing = int(df[feature].isna().sum())

    infinite = int(
        np.isinf(
            pd.to_numeric(
                df[feature],
                errors="coerce"
            )
        ).sum()
    )

    unique_values = int(
        df[feature].nunique()
    )

    ready = (
        missing == 0
        and infinite == 0
        and unique_values > 1
    )

    readiness_rows.append({
        "FEATURE": feature,
        "MISSING": missing,
        "INFINITE": infinite,
        "UNIQUE_VALUES": unique_values,
        "READY": ready
    })

readiness_df = pd.DataFrame(readiness_rows)

print(readiness_df.to_string(index=False))

readiness_df.to_csv(
    REPORT_DIR / "model_readiness.csv",
    index=False
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 78)
print("FINAL FORECAST EDA SUMMARY")
print("=" * 78)

print()
print("Dataset:")
print("    Dataset1_Model_Forecast.csv")

print()
print("Rows:")
print(f"    {len(df):,}")

print()
print("Columns:")
print(f"    {len(df.columns)}")

print()
print("Unique ACOs:")
print(f"    {df['ACO_ID'].nunique():,}")

print()
print("Years:")
print(
    f"    {int(df['YEAR'].min())}–"
    f"{int(df['YEAR'].max())}"
)

print()
print("Forecast features:")
for feature in FEATURES:
    print(f"    - {feature}")

print()
print("Forecast target:")
print(f"    - {TARGET}")

print()
print("Target mean:")
print(f"    {target.mean():.8f}")

print()
print("Target range:")
print(
    f"    {target.min():.8f} "
    f"to "
    f"{target.max():.8f}"
)

print()
print("Exact duplicate rows:")
print(f"    {exact_duplicates}")

print()
print("Duplicate ACO-year rows:")
print(f"    {aco_year_duplicates}")

print()
print("=" * 78)
print("FORECAST MODEL EDA COMPLETE")
print("=" * 78)

print()
print("Reports:")
print(REPORT_DIR)

print()
print("Plots:")
print(PLOT_DIR)

print()
print("Important:")
print(
    "Do NOT modify the forecast model dataset based only on EDA."
)

print(
    "Review the EDA results before training the forecast model."
)

print("=" * 78)