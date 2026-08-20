# =============================================================================
# CONTRACTIQ — DATASET 1 EDA
# CMS ACO DATA — 50,000 ROWS
# =============================================================================
#
# PURPOSE:
#   1. Load the raw Dataset 1
#   2. Verify dataset structure
#   3. Analyze missing values
#   4. Analyze duplicates
#   5. Analyze zero values
#   6. Analyze negative values
#   7. Analyze numeric statistics
#   8. Analyze categorical variables
#   9. Analyze ACO IDs / names
#  10. Analyze year distribution
#  11. Detect suspicious columns
#  12. Create correlation matrix
#  13. Create EDA reports and plots
#
# IMPORTANT:
#   RAW DATASET IS NEVER MODIFIED.
# =============================================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = r"I:\ContractIQ(!)"

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "dataset1",
    "reports"
)

PLOT_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "dataset1",
    "plots"
)

DISTRIBUTION_DIR = os.path.join(
    PLOT_DIR,
    "distributions"
)

MISSINGNESS_DIR = os.path.join(
    PLOT_DIR,
    "missingness"
)

CORRELATION_DIR = os.path.join(
    PLOT_DIR,
    "correlations"
)

# Expected filename
EXPECTED_FILENAME = "ContractIQ_Dataset1_Synthetic_CMS_Similar_50000.csv"

RAW_FILE = os.path.join(
    RAW_DIR,
    EXPECTED_FILENAME
)


# =============================================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================================

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DISTRIBUTION_DIR, exist_ok=True)
os.makedirs(MISSINGNESS_DIR, exist_ok=True)
os.makedirs(CORRELATION_DIR, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def find_dataset():
    """
    Find the dataset robustly.

    First checks the expected location.
    Then searches the project if the expected file is missing.
    """

    if os.path.exists(RAW_FILE):
        return RAW_FILE

    print()
    print("Expected dataset was not found.")
    print()
    print("Expected:")
    print(RAW_FILE)
    print()
    print("Searching the ContractIQ project for a matching CSV...")

    candidates = []

    for root, dirs, files in os.walk(PROJECT_ROOT):

        # Ignore Python cache and model output folders
        dirs[:] = [
            d for d in dirs
            if d not in {
                "__pycache__",
                ".git",
                ".venv",
                "venv"
            }
        ]

        for file in files:

            if not file.lower().endswith(".csv"):
                continue

            full_path = os.path.join(root, file)

            if (
                "50000" in file.lower()
                or "synthetic" in file.lower()
                or "contractiq_dataset1" in file.lower()
            ):
                candidates.append(full_path)

    if len(candidates) == 0:

        raise FileNotFoundError(
            "\nDataset was not found.\n\n"
            "Please place your dataset here:\n"
            f"{RAW_FILE}\n\n"
            "Expected filename:\n"
            f"{EXPECTED_FILENAME}\n"
        )

    print()
    print("Candidate datasets found:")

    for i, path in enumerate(candidates, start=1):
        print(f"{i}. {path}")

    # Prefer exact filename
    for path in candidates:
        if os.path.basename(path).lower() == EXPECTED_FILENAME.lower():
            return path

    # Prefer 50k synthetic dataset
    for path in candidates:
        name = os.path.basename(path).lower()

        if "50000" in name and "synthetic" in name:
            return path

    # Otherwise use first candidate
    return candidates[0]


def safe_filename(text):
    """
    Make a safe filename.
    """

    text = str(text)

    invalid = '<>:"/\\|?*'

    for char in invalid:
        text = text.replace(char, "_")

    return text[:150]


# =============================================================================
# START
# =============================================================================

print_header("CONTRACTIQ — DATASET 1 EDA")

print("Project root:")
print(PROJECT_ROOT)

print()
print("Searching for raw dataset...")

dataset_path = find_dataset()

print()
print("Dataset selected:")
print(dataset_path)


# =============================================================================
# LOAD DATASET
# =============================================================================

print_header("LOADING DATASET")

try:

    df = pd.read_csv(
        dataset_path,
        low_memory=False
    )

except Exception as e:

    print()
    print("ERROR WHILE READING DATASET")
    print(e)
    sys.exit(1)


print()
print("Dataset loaded successfully.")

print()
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")


# =============================================================================
# BASIC DATASET INFORMATION
# =============================================================================

print_header("BASIC DATASET INFORMATION")

print("Memory usage:")
memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"{memory_mb:.2f} MB")

print()
print("Column names:")

for i, col in enumerate(df.columns, start=1):
    print(f"{i:3d}. {col}")


# =============================================================================
# DATA TYPES
# =============================================================================

print_header("DATA TYPES")

dtype_report = pd.DataFrame({
    "COLUMN": df.columns,
    "DATA_TYPE": df.dtypes.astype(str).values,
    "NON_NULL": df.notna().sum().values,
    "NULL_COUNT": df.isna().sum().values,
})

dtype_report["NULL_PCT"] = (
    dtype_report["NULL_COUNT"]
    / len(df)
    * 100
)

dtype_report = dtype_report.sort_values(
    "NULL_PCT",
    ascending=False
)

dtype_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "dataset1_column_data_types.csv"
    ),
    index=False
)

print(
    dtype_report.to_string(index=False)
)


# =============================================================================
# DUPLICATE ANALYSIS
# =============================================================================

print_header("DUPLICATE ANALYSIS")

duplicate_rows = df.duplicated().sum()

duplicate_pct = (
    duplicate_rows / len(df) * 100
)

print(f"Duplicate rows : {duplicate_rows:,}")
print(f"Duplicate %    : {duplicate_pct:.4f}%")


duplicate_report = pd.DataFrame({
    "METRIC": [
        "TOTAL_ROWS",
        "DUPLICATE_ROWS",
        "DUPLICATE_PERCENT"
    ],
    "VALUE": [
        len(df),
        duplicate_rows,
        duplicate_pct
    ]
})

duplicate_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "duplicate_report.csv"
    ),
    index=False
)


# =============================================================================
# MISSING VALUE ANALYSIS
# =============================================================================

print_header("MISSING VALUE ANALYSIS")

missing = df.isna().sum()

missing_pct = (
    missing / len(df) * 100
)

missing_report = pd.DataFrame({
    "COLUMN": df.columns,
    "MISSING": missing.values,
    "MISSING_PCT": missing_pct.values
})

missing_report = missing_report.sort_values(
    "MISSING",
    ascending=False
)

missing_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "missing_values.csv"
    ),
    index=False
)

print()
print(
    missing_report[
        missing_report["MISSING"] > 0
    ].to_string(index=False)
)

print()

if missing.sum() == 0:
    print("RESULT: No missing values detected.")
else:
    print(
        f"Total missing cells: {missing.sum():,}"
    )


# =============================================================================
# ZERO VALUE ANALYSIS
# =============================================================================

print_header("ZERO VALUE ANALYSIS")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

zero_records = []

for col in numeric_columns:

    zero_count = (
        df[col] == 0
    ).sum()

    zero_pct = (
        zero_count / len(df) * 100
    )

    zero_records.append({
        "COLUMN": col,
        "ZERO_COUNT": int(zero_count),
        "ZERO_PCT": zero_pct
    })


zero_report = pd.DataFrame(zero_records)

zero_report = zero_report.sort_values(
    "ZERO_COUNT",
    ascending=False
)

zero_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "zero_value_analysis.csv"
    ),
    index=False
)

print(
    zero_report.head(50).to_string(index=False)
)


# =============================================================================
# NEGATIVE VALUE ANALYSIS
# =============================================================================

print_header("NEGATIVE VALUE ANALYSIS")

negative_records = []

for col in numeric_columns:

    negative_count = (
        df[col] < 0
    ).sum()

    negative_pct = (
        negative_count / len(df) * 100
    )

    negative_records.append({
        "COLUMN": col,
        "NEGATIVE_COUNT": int(negative_count),
        "NEGATIVE_PCT": negative_pct
    })


negative_report = pd.DataFrame(
    negative_records
)

negative_report = negative_report.sort_values(
    "NEGATIVE_COUNT",
    ascending=False
)

negative_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "negative_value_analysis.csv"
    ),
    index=False
)

print(
    negative_report.head(50).to_string(index=False)
)


# =============================================================================
# NUMERIC STATISTICS
# =============================================================================

print_header("NUMERIC STATISTICS")

if len(numeric_columns) > 0:

    numeric_statistics = (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "COLUMN"
            }
        )
    )

    numeric_statistics.to_csv(
        os.path.join(
            REPORT_DIR,
            "numeric_statistics.csv"
        ),
        index=False
    )

    print(
        numeric_statistics.head(30).to_string(
            index=False
        )
    )

else:

    print("No numeric columns detected.")


# =============================================================================
# CATEGORICAL STATISTICS
# =============================================================================

print_header("CATEGORICAL STATISTICS")

categorical_columns = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

categorical_records = []

for col in categorical_columns:

    series = df[col]

    categorical_records.append({
        "COLUMN": col,
        "UNIQUE_VALUES": series.nunique(
            dropna=True
        ),
        "MISSING": series.isna().sum(),
        "TOP_VALUE": (
            series.mode(dropna=True).iloc[0]
            if not series.mode(dropna=True).empty
            else np.nan
        ),
        "TOP_VALUE_COUNT": (
            series.value_counts(
                dropna=True
            ).iloc[0]
            if not series.value_counts(
                dropna=True
            ).empty
            else 0
        )
    })


categorical_statistics = pd.DataFrame(
    categorical_records
)

categorical_statistics.to_csv(
    os.path.join(
        REPORT_DIR,
        "categorical_statistics.csv"
    ),
    index=False
)

if len(categorical_statistics) > 0:

    print(
        categorical_statistics.to_string(
            index=False
        )
    )

else:

    print("No categorical columns detected.")


# =============================================================================
# IDENTIFY IMPORTANT ACO COLUMNS
# =============================================================================

print_header("ACO IDENTIFIER CHECK")

columns_upper = {
    str(col).upper(): col
    for col in df.columns
}

possible_id_columns = [
    col for col in df.columns
    if any(
        term in str(col).upper()
        for term in [
            "ACO_ID",
            "ACOID",
            "ACO_NUM",
            "ACO_NUMBER",
            "ACO"
        ]
    )
]

possible_name_columns = [
    col for col in df.columns
    if any(
        term in str(col).upper()
        for term in [
            "ACO_NAME",
            "ACONAME",
            "ACO_NAME"
        ]
    )
]

print()
print("Possible ACO ID columns:")

for col in possible_id_columns:
    print(f"  - {col}")

print()
print("Possible ACO Name columns:")

for col in possible_name_columns:
    print(f"  - {col}")


# =============================================================================
# ACO ID / NAME QUALITY
# =============================================================================

identifier_records = []

for col in possible_id_columns + possible_name_columns:

    if col not in df.columns:
        continue

    identifier_records.append({
        "COLUMN": col,
        "ROWS": len(df),
        "UNIQUE_VALUES": df[col].nunique(
            dropna=True
        ),
        "MISSING": df[col].isna().sum(),
        "MISSING_PCT": (
            df[col].isna().mean() * 100
        )
    })


identifier_report = pd.DataFrame(
    identifier_records
)

identifier_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "aco_identifier_quality.csv"
    ),
    index=False
)

print()

if len(identifier_report) > 0:
    print(
        identifier_report.to_string(
            index=False
        )
    )
else:
    print(
        "WARNING: No obvious ACO identifier "
        "columns were detected."
    )


# =============================================================================
# YEAR ANALYSIS
# =============================================================================

print_header("YEAR ANALYSIS")

year_columns = [
    col for col in df.columns
    if str(col).upper() in {
        "YEAR",
        "PERFORMANCE_YEAR",
        "REPORTING_YEAR",
        "PY"
    }
]

if len(year_columns) > 0:

    year_col = year_columns[0]

    print(f"Year column detected: {year_col}")

    year_summary = (
        df.groupby(year_col)
        .size()
        .reset_index(
            name="ROWS"
        )
        .sort_values(
            year_col
        )
    )

    print()
    print(
        year_summary.to_string(
            index=False
        )
    )

    year_summary.to_csv(
        os.path.join(
            REPORT_DIR,
            "year_distribution.csv"
        ),
        index=False
    )

else:

    print(
        "WARNING: No YEAR column detected."
    )


# =============================================================================
# POTENTIAL TARGET COLUMNS
# =============================================================================

print_header("POTENTIAL MODEL TARGET COLUMNS")

target_keywords = [
    "RISK",
    "SAVINGS",
    "QUALITY",
    "EXPENDITURE",
    "PERFORMANCE",
    "FORECAST"
]

possible_targets = []

for col in df.columns:

    upper = str(col).upper()

    if any(
        keyword in upper
        for keyword in target_keywords
    ):

        possible_targets.append(col)


target_report = pd.DataFrame({
    "POTENTIAL_TARGET_COLUMN": possible_targets
})

target_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "potential_model_targets.csv"
    ),
    index=False
)

for col in possible_targets:
    print(f"  - {col}")


# =============================================================================
# CONSTANT / LOW-VARIANCE COLUMNS
# =============================================================================

print_header("CONSTANT / LOW-VARIANCE COLUMNS")

low_variance_records = []

for col in df.columns:

    unique_count = df[col].nunique(
        dropna=True
    )

    if unique_count <= 1:

        low_variance_records.append({
            "COLUMN": col,
            "UNIQUE_VALUES": unique_count
        })


low_variance_report = pd.DataFrame(
    low_variance_records
)

low_variance_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "constant_columns.csv"
    ),
    index=False
)

if len(low_variance_report) > 0:

    print(
        low_variance_report.to_string(
            index=False
        )
    )

else:

    print("No constant columns detected.")


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

print_header("CORRELATION ANALYSIS")

if len(numeric_columns) >= 2:

    correlation = df[
        numeric_columns
    ].corr(
        method="pearson"
    )

    correlation.to_csv(
        os.path.join(
            REPORT_DIR,
            "correlation_matrix.csv"
        )
    )

    # Plot only if manageable
    if len(numeric_columns) <= 60:

        plt.figure(
            figsize=(16, 13)
        )

        plt.imshow(
            correlation,
            aspect="auto",
            interpolation="nearest"
        )

        plt.colorbar()

        plt.xticks(
            range(len(numeric_columns)),
            numeric_columns,
            rotation=90,
            fontsize=6
        )

        plt.yticks(
            range(len(numeric_columns)),
            numeric_columns,
            fontsize=6
        )

        plt.title(
            "Dataset 1 Numeric Feature Correlation"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                CORRELATION_DIR,
                "correlation_matrix.png"
            ),
            dpi=200
        )

        plt.close()

        print(
            "Correlation matrix plot created."
        )

    else:

        print(
            f"Correlation plot skipped because "
            f"there are {len(numeric_columns)} "
            f"numeric columns."
        )

else:

    print(
        "Not enough numeric columns for correlation."
    )


# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================

print_header("CREATING DISTRIBUTION PLOTS")

plot_columns = numeric_columns[:]

# Limit the number of plots to avoid creating thousands
# of unnecessary files.
MAX_DISTRIBUTION_PLOTS = 40

if len(plot_columns) > MAX_DISTRIBUTION_PLOTS:

    print(
        f"{len(plot_columns)} numeric columns detected."
    )

    print(
        f"Creating plots for first "
        f"{MAX_DISTRIBUTION_PLOTS} numeric columns."
    )

    plot_columns = plot_columns[
        :MAX_DISTRIBUTION_PLOTS
    ]


for col in plot_columns:

    try:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) == 0:
            continue

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            series,
            bins=40
        )

        plt.title(
            f"Distribution — {col}"
        )

        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.tight_layout()

        filename = (
            safe_filename(col)
            + "_distribution.png"
        )

        plt.savefig(
            os.path.join(
                DISTRIBUTION_DIR,
                filename
            ),
            dpi=150
        )

        plt.close()

    except Exception as e:

        print(
            f"Could not plot {col}: {e}"
        )


print("Distribution plots created.")


# =============================================================================
# MISSINGNESS PLOT
# =============================================================================

print_header("CREATING MISSINGNESS PLOT")

missing_plot = missing_report[
    missing_report["MISSING"] > 0
].copy()

if len(missing_plot) > 0:

    missing_plot = missing_plot.head(40)

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        missing_plot["COLUMN"].astype(str),
        missing_plot["MISSING_PCT"]
    )

    plt.xlabel(
        "Missing Values (%)"
    )

    plt.ylabel(
        "Column"
    )

    plt.title(
        "Top Dataset 1 Columns by Missingness"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MISSINGNESS_DIR,
            "missing_values_top40.png"
        ),
        dpi=180
    )

    plt.close()

    print(
        "Missingness plot created."
    )

else:

    print(
        "No missing values — missingness plot not required."
    )


# =============================================================================
# DATASET SUMMARY
# =============================================================================

print_header("CREATING DATASET SUMMARY")

summary = pd.DataFrame({
    "METRIC": [
        "ROWS",
        "COLUMNS",
        "DUPLICATE_ROWS",
        "TOTAL_MISSING_CELLS",
        "TOTAL_NUMERIC_COLUMNS",
        "TOTAL_CATEGORICAL_COLUMNS",
        "TOTAL_ZERO_VALUES",
        "TOTAL_NEGATIVE_VALUES",
        "MEMORY_MB"
    ],
    "VALUE": [
        len(df),
        len(df.columns),
        int(duplicate_rows),
        int(df.isna().sum().sum()),
        len(numeric_columns),
        len(categorical_columns),
        int(
            zero_report["ZERO_COUNT"].sum()
        ),
        int(
            negative_report["NEGATIVE_COUNT"].sum()
        ),
        round(memory_mb, 2)
    ]
})

summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "dataset1_summary.csv"
    ),
    index=False
)

print(
    summary.to_string(index=False)
)


# =============================================================================
# FINAL EDA STATUS
# =============================================================================

print_header("EDA COMPLETE")

print("Dataset:")
print(dataset_path)

print()
print("Rows:")
print(f"{len(df):,}")

print()
print("Columns:")
print(f"{len(df.columns):,}")

print()
print("Reports saved to:")
print(REPORT_DIR)

print()
print("Plots saved to:")
print(PLOT_DIR)

print()
print("RAW DATASET WAS NOT MODIFIED.")

print()
print("=" * 78)
print("NEXT STEP: DATA CLEANING / PREPROCESSING")
print("=" * 78)