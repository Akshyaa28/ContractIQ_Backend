import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

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


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 SAVINGS FORECAST MODEL EDA")
print("=" * 78)

print("\nPROJECT ROOT")
print("-" * 78)
print(PROJECT_ROOT)

print("\nINPUT DATASET")
print("-" * 78)
print(INPUT_PATH)


# ============================================================
# CHECK INPUT
# ============================================================

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

print("\nInput dataset found.")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 78)
print("LOADING FORECAST MODEL DATASET")
print("=" * 78)

df = pd.read_csv(INPUT_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


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
    .str.replace(" ", "_", regex=False)
)

print("Column names standardized.")


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

print("\n" + "=" * 78)
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
        + "\n".join(f" - {c}" for c in missing_columns)
    )

print("All required columns are present.")


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("DUPLICATE ANALYSIS")
print("=" * 78)

exact_duplicates = int(df.duplicated().sum())

aco_year_duplicates = int(
    df.duplicated(subset=["ACO_ID", "YEAR"]).sum()
)

print(f"Exact duplicate rows       : {exact_duplicates}")
print(f"Duplicate ACO-year rows    : {aco_year_duplicates}")


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("MISSING VALUE ANALYSIS")
print("=" * 78)

missing_rows = []

for col in required_columns:

    missing = int(df[col].isna().sum())
    missing_pct = missing / len(df) * 100

    missing_rows.append({
        "COLUMN": col,
        "MISSING": missing,
        "MISSING_PCT": round(missing_pct, 4),
    })

missing_df = pd.DataFrame(missing_rows)

print(missing_df.to_string(index=False))

missing_df.to_csv(
    REPORT_DIR / "missing_value_analysis.csv",
    index=False
)


# ============================================================
# TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("TARGET ANALYSIS")
print("=" * 78)

target = pd.to_numeric(df[TARGET], errors="coerce")

print("\nTarget statistics:")
print(target.describe())

print(f"\nTarget missing   : {target.isna().sum()}")
print(f"Target infinite  : {np.isinf(target).sum()}")


# ============================================================
# TARGET BY YEAR
# ============================================================

print("\n" + "=" * 78)
print("TARGET BY YEAR")
print("=" * 78)

target_by_year = (
    df.groupby("YEAR")[TARGET]
    .agg(
        ROWS="count",
        MEAN="mean",
        MEDIAN="median",
        STD="std",
        MIN="min",
        MAX="max",
    )
    .reset_index()
)

print(target_by_year.to_string(index=False))

target_by_year.to_csv(
    REPORT_DIR / "target_by_year.csv",
    index=False
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 78)
print("FEATURE STATISTICS")
print("=" * 78)

feature_stats = []

for feature in FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    feature_stats.append({
        "FEATURE": feature,
        "COUNT": values.count(),
        "MEAN": values.mean(),
        "STD": values.std(),
        "MIN": values.min(),
        "Q1": values.quantile(0.25),
        "MEDIAN": values.median(),
        "Q3": values.quantile(0.75),
        "MAX": values.max(),
        "MISSING": values.isna().sum(),
        "INFINITE": np.isinf(values).sum(),
    })

feature_stats_df = pd.DataFrame(feature_stats)

print(
    feature_stats_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

feature_stats_df.to_csv(
    REPORT_DIR / "feature_statistics.csv",
    index=False
)


# ============================================================
# NEGATIVE VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("NEGATIVE VALUE ANALYSIS")
print("=" * 78)

negative_rows = []

for feature in FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    negative = int((values < 0).sum())

    negative_rows.append({
        "FEATURE": feature,
        "NEGATIVE_VALUES": negative,
        "NEGATIVE_PCT": round(
            negative / len(df) * 100,
            4
        ),
    })

negative_df = pd.DataFrame(negative_rows)

print(negative_df.to_string(index=False))

negative_df.to_csv(
    REPORT_DIR / "negative_value_analysis.csv",
    index=False
)


# ============================================================
# ZERO VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("ZERO VALUE ANALYSIS")
print("=" * 78)

zero_rows = []

for feature in FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    zero = int((values == 0).sum())

    zero_rows.append({
        "FEATURE": feature,
        "ZERO_VALUES": zero,
        "ZERO_PCT": round(
            zero / len(df) * 100,
            4
        ),
    })

zero_df = pd.DataFrame(zero_rows)

print(zero_df.to_string(index=False))

zero_df.to_csv(
    REPORT_DIR / "zero_value_analysis.csv",
    index=False
)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("OUTLIER ANALYSIS")
print("=" * 78)

outlier_rows = []

for feature in FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = int(
        ((values < lower) | (values > upper)).sum()
    )

    outlier_rows.append({
        "FEATURE": feature,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LOWER_BOUND": lower,
        "UPPER_BOUND": upper,
        "OUTLIERS": outliers,
        "OUTLIER_PCT": round(
            outliers / len(values) * 100,
            4
        ),
    })

outlier_df = pd.DataFrame(outlier_rows)

print(
    outlier_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

outlier_df.to_csv(
    REPORT_DIR / "outlier_analysis.csv",
    index=False
)


# ============================================================
# FEATURE-TARGET CORRELATION
# ============================================================

print("\n" + "=" * 78)
print("FEATURE — TARGET CORRELATION")
print("=" * 78)

correlation_rows = []

for feature in FEATURES:

    correlation = df[feature].corr(df[TARGET])

    correlation_rows.append({
        "FEATURE": feature,
        "TARGET_CORRELATION": correlation,
        "ABS_CORRELATION": abs(correlation),
    })

correlation_df = (
    pd.DataFrame(correlation_rows)
    .sort_values(
        "ABS_CORRELATION",
        ascending=False
    )
)

print(
    correlation_df[
        ["FEATURE", "TARGET_CORRELATION"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

correlation_df.to_csv(
    REPORT_DIR / "feature_target_correlation.csv",
    index=False
)


# ============================================================
# FEATURE CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 78)
print("FEATURE CORRELATION MATRIX")
print("=" * 78)

feature_corr = df[FEATURES].corr()

feature_corr.to_csv(
    REPORT_DIR / "feature_correlation_matrix.csv"
)

print(
    feature_corr.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# YEAR-BASED TARGET TREND
# ============================================================

print("\nCreating target trend plot...")

year_target = (
    df.groupby("YEAR")[TARGET]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    year_target.index,
    year_target.values,
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Mean Next-Year Savings Rate")
plt.title("Mean NEXT_YEAR_SAVINGS_RATE by Year")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "target_by_year.png",
    dpi=150
)

plt.close()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("Creating target distribution plot...")

plt.figure(figsize=(10, 6))

plt.hist(
    df[TARGET],
    bins=50
)

plt.xlabel("NEXT_YEAR_SAVINGS_RATE")
plt.ylabel("Frequency")
plt.title("Distribution of NEXT_YEAR_SAVINGS_RATE")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "target_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# FEATURE DISTRIBUTIONS
# ============================================================

print("Creating feature distribution plots...")

for feature in FEATURES:

    plt.figure(figsize=(10, 6))

    plt.hist(
        df[feature],
        bins=50
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature}")
    plt.grid(True)

    plt.tight_layout()

    safe_name = feature.lower()

    plt.savefig(
        PLOT_DIR / f"{safe_name}_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# FEATURE VS TARGET
# ============================================================

print("Creating feature-vs-target plots...")

for feature in FEATURES:

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df[feature],
        df[TARGET],
        alpha=0.15,
        s=8
    )

    plt.xlabel(feature)
    plt.ylabel(TARGET)
    plt.title(
        f"{feature} vs {TARGET}"
    )
    plt.grid(True)

    plt.tight_layout()

    safe_name = feature.lower()

    plt.savefig(
        PLOT_DIR / f"{safe_name}_vs_target.png",
        dpi=150
    )

    plt.close()


# ============================================================
# MODEL READINESS CHECK
# ============================================================

print("\n" + "=" * 78)
print("MODEL READINESS CHECK")
print("=" * 78)

readiness_rows = []

for feature in FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    missing = int(values.isna().sum())
    infinite = int(np.isinf(values).sum())
    unique = int(values.nunique())

    ready = (
        missing == 0
        and infinite == 0
        and unique > 1
    )

    readiness_rows.append({
        "FEATURE": feature,
        "MISSING": missing,
        "INFINITE": infinite,
        "UNIQUE_VALUES": unique,
        "READY": ready,
    })

readiness_df = pd.DataFrame(readiness_rows)

print(readiness_df.to_string(index=False))

readiness_df.to_csv(
    REPORT_DIR / "model_readiness.csv",
    index=False
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 78)
print("TARGET LEAKAGE CHECK")
print("=" * 78)

future_columns = [
    col
    for col in df.columns
    if "NEXT_YEAR" in col
]

print("Future-related columns detected:")

for col in future_columns:
    print(f" - {col}")

unexpected_future_features = [
    col
    for col in FEATURES
    if "NEXT_YEAR" in col
]

if unexpected_future_features:

    raise ValueError(
        "Potential target leakage detected:\n"
        + "\n".join(
            f" - {c}"
            for c in unexpected_future_features
        )
    )

print("\nForecast target:")
print(f" - {TARGET}")

print("\nForecast features:")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i}. {feature}")

print("\nLeakage check PASSED.")


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "dataset": INPUT_PATH.name,
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "unique_acos": int(df["ACO_ID"].nunique()),
    "min_year": int(df["YEAR"].min()),
    "max_year": int(df["YEAR"].max()),
    "features": FEATURES,
    "target": TARGET,
    "target_mean": float(df[TARGET].mean()),
    "target_std": float(df[TARGET].std()),
    "target_min": float(df[TARGET].min()),
    "target_max": float(df[TARGET].max()),
    "exact_duplicates": exact_duplicates,
    "duplicate_aco_year": aco_year_duplicates,
}

with open(
    REPORT_DIR / "forecast_eda_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 78)
print("FINAL FORECAST EDA SUMMARY")
print("=" * 78)

print(f"""
Dataset:
    {INPUT_PATH.name}

Rows:
    {len(df):,}

Columns:
    {len(df.columns)}

Unique ACOs:
    {df["ACO_ID"].nunique():,}

Years:
    {df["YEAR"].min()}–{df["YEAR"].max()}

Forecast target:
    {TARGET}

Target mean:
    {df[TARGET].mean():.6f}

Target standard deviation:
    {df[TARGET].std():.6f}

Target minimum:
    {df[TARGET].min():.6f}

Target maximum:
    {df[TARGET].max():.6f}

Exact duplicate rows:
    {exact_duplicates}

Duplicate ACO-year rows:
    {aco_year_duplicates}

Model features:
    {len(FEATURES)}
""")

print("Reports:")
print(REPORT_DIR)

print("\nPlots:")
print(PLOT_DIR)

print("\nIMPORTANT:")
print(
    "Do NOT modify the forecast model dataset based only on EDA."
)

print(
    "Review the EDA before training the forecasting model."
)

print("\n" + "=" * 78)
print("FORECAST MODEL EDA COMPLETE")
print("=" * 78)