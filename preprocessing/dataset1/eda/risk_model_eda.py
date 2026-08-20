import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONTRACTIQ — DATASET 1 RISK MODEL EDA
# ============================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 RISK MODEL EDA")
print("=" * 78)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "Dataset1_Model_Risk.csv"
)

EDA_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "dataset1",
    "eda"
)

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "dataset1",
    "reports",
    "risk_model_eda"
)

PLOT_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "dataset1",
    "plots",
    "risk_model_eda"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


print("\nPROJECT ROOT")
print("-" * 78)
print(PROJECT_ROOT)

print("\nINPUT DATASET")
print("-" * 78)
print(INPUT_FILE)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"""
Risk model dataset was not found.

Expected:
{INPUT_FILE}

Please run:

python build_risk_model_data.py
"""
    )

print("\nInput dataset found.")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 78)
print("LOADING RISK MODEL DATASET")
print("=" * 78)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# EXPECTED MODEL STRUCTURE
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

TARGET = "NEXT_YEAR_RISK"

IDENTIFIERS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("BASIC VALIDATION")
print("=" * 78)

required_columns = IDENTIFIERS + FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 78)
print("DUPLICATE ANALYSIS")
print("=" * 78)

exact_duplicates = df.duplicated().sum()

aco_year_duplicates = df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print(f"Exact duplicate rows       : {exact_duplicates:,}")
print(f"Duplicate ACO-year rows    : {aco_year_duplicates:,}")


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("MISSING VALUE ANALYSIS")
print("=" * 78)

missing_report = pd.DataFrame({
    "COLUMN": df.columns,
    "MISSING": df.isna().sum().values,
})

missing_report["MISSING_PCT"] = (
    missing_report["MISSING"] / len(df) * 100
)

missing_report = missing_report.sort_values(
    "MISSING",
    ascending=False
)

print(missing_report.to_string(index=False))

missing_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_model_missingness.csv"
    ),
    index=False
)


# ============================================================
# TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("TARGET ANALYSIS")
print("=" * 78)

print("\nNEXT_YEAR_RISK distribution:")
print(df[TARGET].value_counts().sort_index())

risk_cases = int((df[TARGET] == 1).sum())
nonrisk_cases = int((df[TARGET] == 0).sum())

risk_rate = risk_cases / len(df)

print(f"\nRisk cases     : {risk_cases:,}")
print(f"Non-risk cases : {nonrisk_cases:,}")
print(f"Risk rate      : {risk_rate:.4%}")


# ============================================================
# TARGET BY YEAR
# ============================================================

print("\n" + "=" * 78)
print("RISK RATE BY YEAR")
print("=" * 78)

year_risk = (
    df.groupby("YEAR")
    .agg(
        ROWS=(TARGET, "size"),
        RISK_CASES=(TARGET, "sum"),
        RISK_RATE=(TARGET, "mean"),
        UNIQUE_ACOS=("ACO_ID", "nunique")
    )
    .reset_index()
)

print(year_risk.to_string(index=False))

year_risk.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_rate_by_year.csv"
    ),
    index=False
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 78)
print("FEATURE STATISTICS")
print("=" * 78)

feature_stats = df[FEATURES].describe().T

feature_stats["MISSING"] = df[FEATURES].isna().sum()
feature_stats["MISSING_PCT"] = (
    feature_stats["MISSING"] / len(df) * 100
)

print(feature_stats.to_string())

feature_stats.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_feature_statistics.csv"
    )
)


# ============================================================
# NEGATIVE VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("NEGATIVE VALUE ANALYSIS")
print("=" * 78)

negative_rows = []

for col in FEATURES:
    negative_count = int((df[col] < 0).sum())

    negative_rows.append({
        "FEATURE": col,
        "NEGATIVE_VALUES": negative_count,
        "NEGATIVE_PCT": negative_count / len(df) * 100
    })

negative_report = pd.DataFrame(negative_rows)

print(negative_report.to_string(index=False))

negative_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_feature_negative_values.csv"
    ),
    index=False
)


# ============================================================
# ZERO VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("ZERO VALUE ANALYSIS")
print("=" * 78)

zero_rows = []

for col in FEATURES:
    zero_count = int((df[col] == 0).sum())

    zero_rows.append({
        "FEATURE": col,
        "ZERO_VALUES": zero_count,
        "ZERO_PCT": zero_count / len(df) * 100
    })

zero_report = pd.DataFrame(zero_rows)

print(zero_report.to_string(index=False))

zero_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_feature_zero_values.csv"
    ),
    index=False
)


# ============================================================
# OUTLIER ANALYSIS — IQR
# ============================================================

print("\n" + "=" * 78)
print("OUTLIER ANALYSIS")
print("=" * 78)

outlier_rows = []

for col in FEATURES:

    series = df[col].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (df[col] < lower) |
        (df[col] > upper)
    ).sum()

    outlier_rows.append({
        "FEATURE": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LOWER_BOUND": lower,
        "UPPER_BOUND": upper,
        "OUTLIERS": int(outliers),
        "OUTLIER_PCT": outliers / len(df) * 100
    })

outlier_report = pd.DataFrame(outlier_rows)

print(outlier_report.to_string(index=False))

outlier_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_feature_outliers.csv"
    ),
    index=False
)


# ============================================================
# CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 78)
print("FEATURE — TARGET CORRELATION")
print("=" * 78)

correlation_rows = []

for col in FEATURES:

    correlation = df[col].corr(df[TARGET])

    correlation_rows.append({
        "FEATURE": col,
        "TARGET_CORRELATION": correlation
    })

correlation_report = pd.DataFrame(
    correlation_rows
).sort_values(
    "TARGET_CORRELATION",
    key=lambda x: x.abs(),
    ascending=False
)

print(correlation_report.to_string(index=False))

correlation_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_feature_correlation.csv"
    ),
    index=False
)


# ============================================================
# RISK VS NON-RISK FEATURE COMPARISON
# ============================================================

print("\n" + "=" * 78)
print("RISK VS NON-RISK COMPARISON")
print("=" * 78)

comparison = (
    df.groupby(TARGET)[FEATURES]
    .agg(["mean", "median", "std"])
)

print(comparison.to_string())

comparison.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_vs_nonrisk_comparison.csv"
    )
)


# ============================================================
# FEATURE CORRELATION MATRIX
# ============================================================

feature_correlation_matrix = df[FEATURES].corr()

feature_correlation_matrix.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_feature_correlation_matrix.csv"
    )
)


# ============================================================
# PLOT 1 — RISK RATE BY YEAR
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    year_risk["YEAR"],
    year_risk["RISK_RATE"],
    marker="o"
)

plt.title("Risk Rate by Year")
plt.xlabel("Year")
plt.ylabel("Risk Rate")

plt.xticks(year_risk["YEAR"])

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "risk_rate_by_year.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PLOT 2 — TARGET DISTRIBUTION
# ============================================================

target_counts = df[TARGET].value_counts().sort_index()

plt.figure(figsize=(8, 6))

plt.bar(
    ["Non-Risk (0)", "Risk (1)"],
    [
        target_counts.get(0, 0),
        target_counts.get(1, 0)
    ]
)

plt.title("Risk Target Distribution")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "risk_target_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PLOT 3+ — FEATURE DISTRIBUTIONS
# ============================================================

print("\nCreating feature distribution plots...")

for col in FEATURES:

    plt.figure(figsize=(9, 6))

    plt.hist(
        df[col].dropna(),
        bins=50
    )

    plt.title(f"Distribution — {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_name = col.lower()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{safe_name}_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT — RISK VS FEATURE
# ============================================================

print("Creating risk-vs-feature plots...")

for col in FEATURES:

    plt.figure(figsize=(9, 6))

    risk_data = df[df[TARGET] == 1][col].dropna()
    nonrisk_data = df[df[TARGET] == 0][col].dropna()

    plt.hist(
        nonrisk_data,
        bins=40,
        alpha=0.6,
        label="Non-Risk"
    )

    plt.hist(
        risk_data,
        bins=40,
        alpha=0.6,
        label="Risk"
    )

    plt.title(f"{col} — Risk vs Non-Risk")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_name = col.lower()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{safe_name}_risk_vs_nonrisk.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# MODEL READINESS CHECK
# ============================================================

print("\n" + "=" * 78)
print("MODEL READINESS CHECK")
print("=" * 78)

readiness = []

for col in FEATURES:

    missing = int(df[col].isna().sum())
    infinite = int(
        np.isinf(
            pd.to_numeric(
                df[col],
                errors="coerce"
            )
        ).sum()
    )

    unique = df[col].nunique()

    readiness.append({
        "FEATURE": col,
        "MISSING": missing,
        "INFINITE": infinite,
        "UNIQUE_VALUES": unique,
        "READY": (
            missing == 0 and
            infinite == 0 and
            unique > 1
        )
    })

readiness_df = pd.DataFrame(readiness)

print(readiness_df.to_string(index=False))

readiness_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_model_readiness.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("FINAL EDA SUMMARY")
print("=" * 78)

print(f"""
Dataset:
    Dataset1_Model_Risk.csv

Rows:
    {len(df):,}

Columns:
    {len(df.columns):,}

Unique ACOs:
    {df["ACO_ID"].nunique():,}

Years:
    {df["YEAR"].min()}–{df["YEAR"].max()}

Risk cases:
    {risk_cases:,}

Non-risk cases:
    {nonrisk_cases:,}

Risk rate:
    {risk_rate:.4%}

Exact duplicate rows:
    {exact_duplicates:,}

Duplicate ACO-year rows:
    {aco_year_duplicates:,}

Model features:
    {len(FEATURES)}

Target:
    {TARGET}
""")


# ============================================================
# SAVE COMPLETE EDA SUMMARY
# ============================================================

summary = pd.DataFrame({
    "METRIC": [
        "ROWS",
        "COLUMNS",
        "UNIQUE_ACOS",
        "MIN_YEAR",
        "MAX_YEAR",
        "RISK_CASES",
        "NON_RISK_CASES",
        "RISK_RATE",
        "EXACT_DUPLICATES",
        "DUPLICATE_ACO_YEAR_ROWS"
    ],
    "VALUE": [
        len(df),
        len(df.columns),
        df["ACO_ID"].nunique(),
        df["YEAR"].min(),
        df["YEAR"].max(),
        risk_cases,
        nonrisk_cases,
        risk_rate,
        exact_duplicates,
        aco_year_duplicates
    ]
})

summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "risk_model_eda_summary.csv"
    ),
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 78)
print("MODEL EDA COMPLETE")
print("=" * 78)

print("\nReports:")
print(REPORT_DIR)

print("\nPlots:")
print(PLOT_DIR)

print("\nImportant:")
print("Do NOT modify the model dataset based only on EDA.")
print("We will review the EDA results first and then decide")
print("whether preprocessing changes are actually necessary.")

print("\nNext step:")
print("Review the generated EDA reports before training the model.")
print("=" * 78)