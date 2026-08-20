from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")


# ============================================================
# CONTRACTIQ — DATASET 1 RISK MODEL VALIDATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# train_risk_model.py / validate_risk_model.py
# are located at:
# I:\ContractIQ(!)\preprocessing\modeling
#
# Therefore:
# parent      = modeling
# parent[1]   = preprocessing
# parent[2]   = ContractIQ(!)

PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Dataset1_Model_Risk.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "risk_prediction"
    / "risk_random_forest_85_87.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "preprocessing"
    / "dataset1"
    / "reports"
    / "risk_model_validation"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

ID_COLUMNS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def print_metrics(metrics):
    print(f"Accuracy       : {metrics['accuracy']:.6f}")
    print(f"Precision      : {metrics['precision']:.6f}")
    print(f"Recall         : {metrics['recall']:.6f}")
    print(f"F1             : {metrics['f1']:.6f}")
    print(f"ROC-AUC        : {metrics['roc_auc']:.6f}")
    print(f"PR-AUC         : {metrics['pr_auc']:.6f}")
    print(f"True Negative  : {metrics['true_negative']}")
    print(f"False Positive : {metrics['false_positive']}")
    print(f"False Negative : {metrics['false_negative']}")
    print(f"True Positive  : {metrics['true_positive']}")


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 RISK MODEL VALIDATION")
print("=" * 78)

print()
print("SCRIPT LOCATION")
print("-" * 78)
print(SCRIPT_DIR)

print()
print("PROJECT ROOT")
print("-" * 78)
print(PROJECT_ROOT)

print()
print("INPUT DATASET")
print("-" * 78)
print(DATA_PATH)

print()
print("MODEL")
print("-" * 78)
print(MODEL_PATH)

print()
print("REPORT DIRECTORY")
print("-" * 78)
print(REPORT_DIR)


# ============================================================
# CHECK FILES
# ============================================================

print_section("CHECKING REQUIRED FILES")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"""
Risk model dataset was not found.

Expected:
{DATA_PATH}

Run:
python build_risk_model_data.py
"""
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"""
Random Forest model was not found.

Expected:
{MODEL_PATH}

Run:
python train_risk_model.py
"""
    )

print("Input dataset found.")
print("Random Forest model found.")


# ============================================================
# LOAD DATA
# ============================================================

print_section("LOADING MODEL DATASET")

df = pd.read_csv(DATA_PATH)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.upper()
)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

print_section("REQUIRED COLUMN CHECK")

required_columns = ID_COLUMNS + FEATURES + [TARGET]

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
# BASIC VALIDATION
# ============================================================

print_section("BASIC DATA VALIDATION")

print(f"Exact duplicate rows: {df.duplicated().sum():,}")

duplicate_aco_year = df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print(
    f"Duplicate ACO-year rows: {duplicate_aco_year:,}"
)

if duplicate_aco_year > 0:
    raise ValueError(
        "Duplicate ACO-year rows detected."
    )

missing_values = df[FEATURES + [TARGET]].isna().sum()

if missing_values.sum() > 0:
    print()
    print("Missing values detected:")
    print(
        missing_values[
            missing_values > 0
        ]
    )

    raise ValueError(
        "Missing values exist in model features/target."
    )

print("No missing values in model features or target.")


# ============================================================
# TARGET CHECK
# ============================================================

print_section("TARGET VALIDATION")

target_values = sorted(
    df[TARGET].dropna().unique().tolist()
)

print(f"Target values: {target_values}")

if target_values != [0, 1]:
    raise ValueError(
        f"Unexpected target values: {target_values}"
    )

print("Target validation PASSED.")

target_distribution = df[TARGET].value_counts().sort_index()

print()
print(target_distribution)

risk_rate = df[TARGET].mean()

print()
print(f"Overall risk rate: {risk_rate:.4%}")


# ============================================================
# LOAD MODEL
# ============================================================

print_section("LOADING RANDOM FOREST MODEL")

model = joblib.load(MODEL_PATH)

print("Random Forest loaded successfully.")
print(f"Model type: {type(model).__name__}")


# ============================================================
# FEATURE CHECK
# ============================================================

print_section("MODEL FEATURE CHECK")

if hasattr(model, "feature_names_in_"):

    model_features = list(
        model.feature_names_in_
    )

    print("Model features:")
    for i, feature in enumerate(
        model_features,
        start=1,
    ):
        print(f"{i}. {feature}")

    if model_features != FEATURES:
        print()
        print("WARNING:")
        print("Model feature order differs from expected.")

else:
    print(
        "Model does not expose feature_names_in_."
    )


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()


# ============================================================
# ACO ANALYSIS
# ============================================================

print_section("ACO-LEVEL VALIDATION")

unique_acos = df["ACO_ID"].nunique()

print(f"Unique ACOs : {unique_acos:,}")
print(f"Rows        : {len(df):,}")

rows_per_aco = df.groupby("ACO_ID").size()

print(
    f"Average rows per ACO : "
    f"{rows_per_aco.mean():.2f}"
)

print(
    f"Maximum rows per ACO : "
    f"{rows_per_aco.max()}"
)

print(
    f"Minimum rows per ACO : "
    f"{rows_per_aco.min()}"
)

aco_multi_year = (
    rows_per_aco > 1
).sum()

print(
    f"ACOs appearing in multiple years : "
    f"{aco_multi_year:,}"
)

print()
print(
    "IMPORTANT:"
)
print(
    "The dataset is longitudinal. "
    "Therefore random splitting can place the same ACO "
    "in both training and testing data."
)

print(
    "Time-based validation below is therefore the "
    "primary robustness check."
)


# ============================================================
# TIME-BASED VALIDATION
# ============================================================

print_section(
    "TIME-BASED VALIDATION — TRAIN 2018–2022 / TEST 2023"
)

train_time = df[
    df["YEAR"] <= 2022
].copy()

test_time = df[
    df["YEAR"] == 2023
].copy()

print(
    f"Training years : "
    f"{train_time['YEAR'].min()}–"
    f"{train_time['YEAR'].max()}"
)

print(
    f"Testing year   : "
    f"{test_time['YEAR'].min()}"
)

print(
    f"Training rows  : {len(train_time):,}"
)

print(
    f"Testing rows   : {len(test_time):,}"
)

print(
    f"Training risk rate : "
    f"{train_time[TARGET].mean():.4%}"
)

print(
    f"Testing risk rate  : "
    f"{test_time[TARGET].mean():.4%}"
)


time_model = joblib.load(MODEL_PATH)

time_model.fit(
    train_time[FEATURES],
    train_time[TARGET],
)

time_probabilities = time_model.predict_proba(
    test_time[FEATURES]
)[:, 1]


# Evaluate at the production threshold
production_threshold = 0.23

time_metrics = calculate_metrics(
    test_time[TARGET],
    time_probabilities,
    production_threshold,
)

print()
print(
    f"Production threshold: "
    f"{production_threshold:.2f}"
)

print()
print_metrics(time_metrics)


# Save time predictions

time_predictions = test_time[
    ID_COLUMNS
].copy()

time_predictions[
    "RISK_PROBABILITY"
] = time_probabilities

time_predictions[
    "PREDICTED_RISK"
] = (
    time_probabilities
    >= production_threshold
).astype(int)

time_predictions[
    "ACTUAL_RISK"
] = test_time[TARGET].values

time_predictions.to_csv(
    REPORT_DIR
    / "time_based_2023_predictions.csv",
    index=False,
)


pd.DataFrame(
    [time_metrics]
).to_csv(
    REPORT_DIR
    / "time_based_validation_metrics.csv",
    index=False,
)


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print_section(
    "5-FOLD STRATIFIED CROSS-VALIDATION"
)

cv_model = joblib.load(MODEL_PATH)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("Generating out-of-fold probabilities...")

cv_probabilities = cross_val_predict(
    cv_model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


cv_rows = []

for threshold in [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.30]:

    metrics = calculate_metrics(
        y,
        cv_probabilities,
        threshold,
    )

    cv_rows.append(metrics)


cv_results = pd.DataFrame(cv_rows)

print()
print(
    cv_results[
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(index=False)
)

cv_results.to_csv(
    REPORT_DIR
    / "cross_validation_threshold_results.csv",
    index=False,
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print_section(
    "THRESHOLD ANALYSIS"
)

thresholds = np.round(
    np.arange(
        0.10,
        0.51,
        0.01,
    ),
    2,
)

threshold_rows = []

for threshold in thresholds:

    metrics = calculate_metrics(
        y,
        cv_probabilities,
        float(threshold),
    )

    threshold_rows.append(metrics)


threshold_results = pd.DataFrame(
    threshold_rows
)

print(
    threshold_results[
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    ].to_string(index=False)
)

threshold_results.to_csv(
    REPORT_DIR
    / "threshold_analysis.csv",
    index=False,
)


# ============================================================
# FIND VALID 85–87% ACCURACY THRESHOLDS
# ============================================================

print_section(
    "85–87% ACCURACY THRESHOLD SEARCH"
)

valid_thresholds = threshold_results[
    (
        threshold_results["accuracy"] >= 0.85
    )
    &
    (
        threshold_results["accuracy"] <= 0.87
    )
].copy()

if valid_thresholds.empty:

    print(
        "No threshold produced "
        "85–87% accuracy in cross-validation."
    )

else:

    valid_thresholds = (
        valid_thresholds
        .sort_values(
            [
                "f1",
                "pr_auc",
                "recall",
            ],
            ascending=False,
        )
    )

    print(
        valid_thresholds[
            [
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    valid_thresholds.to_csv(
        REPORT_DIR
        / "valid_85_87_thresholds.csv",
        index=False,
    )


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

print_section(
    "CROSS-VALIDATION SUMMARY"
)

cv_default = calculate_metrics(
    y,
    cv_probabilities,
    production_threshold,
)

print(
    f"Threshold : {production_threshold:.2f}"
)

print_metrics(cv_default)

cv_summary = pd.DataFrame(
    [
        {
            "validation": "5-fold stratified CV",
            "threshold": production_threshold,
            "accuracy": cv_default["accuracy"],
            "precision": cv_default["precision"],
            "recall": cv_default["recall"],
            "f1": cv_default["f1"],
            "roc_auc": cv_default["roc_auc"],
            "pr_auc": cv_default["pr_auc"],
        }
    ]
)

cv_summary.to_csv(
    REPORT_DIR
    / "cross_validation_summary.csv",
    index=False,
)


# ============================================================
# COMPARE RANDOM SPLIT VS TIME SPLIT
# ============================================================

print_section(
    "RANDOM-SPLIT VS TIME-BASED VALIDATION"
)

comparison = pd.DataFrame(
    [
        {
            "validation": "Time-based 2023",
            **time_metrics,
        },
        {
            "validation": "5-fold CV",
            **cv_default,
        },
    ]
)

print(
    comparison[
        [
            "validation",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(index=False)
)

comparison.to_csv(
    REPORT_DIR
    / "validation_comparison.csv",
    index=False,
)


# ============================================================
# ROBUSTNESS CHECK
# ============================================================

print_section(
    "MODEL ROBUSTNESS CHECK"
)

time_accuracy = time_metrics["accuracy"]
cv_accuracy = cv_default["accuracy"]

time_roc_auc = time_metrics["roc_auc"]
cv_roc_auc = cv_default["roc_auc"]

print(
    f"Time-based accuracy : {time_accuracy:.4%}"
)

print(
    f"CV accuracy         : {cv_accuracy:.4%}"
)

print(
    f"Time-based ROC-AUC  : {time_roc_auc:.4f}"
)

print(
    f"CV ROC-AUC          : {cv_roc_auc:.4f}"
)

accuracy_difference = abs(
    time_accuracy - cv_accuracy
)

roc_difference = abs(
    time_roc_auc - cv_roc_auc
)

print()
print(
    f"Accuracy difference : "
    f"{accuracy_difference:.4%}"
)

print(
    f"ROC-AUC difference  : "
    f"{roc_difference:.4f}"
)

if accuracy_difference <= 0.05:
    accuracy_robust = True
else:
    accuracy_robust = False

if roc_difference <= 0.10:
    auc_robust = True
else:
    auc_robust = False

print()

print(
    "Accuracy robustness : "
    + ("PASS" if accuracy_robust else "REVIEW")
)

print(
    "ROC-AUC robustness  : "
    + ("PASS" if auc_robust else "REVIEW")
)


# ============================================================
# FINAL VALIDATION DECISION
# ============================================================

print_section(
    "FINAL VALIDATION DECISION"
)

production_accuracy = cv_default["accuracy"]
production_f1 = cv_default["f1"]
production_recall = cv_default["recall"]
production_pr_auc = cv_default["pr_auc"]

accuracy_requirement = (
    0.85
    <= production_accuracy
    <= 0.87
)

print(
    f"Cross-validation accuracy : "
    f"{production_accuracy:.4%}"
)

print(
    f"Cross-validation F1       : "
    f"{production_f1:.4f}"
)

print(
    f"Cross-validation recall   : "
    f"{production_recall:.4f}"
)

print(
    f"Cross-validation PR-AUC   : "
    f"{production_pr_auc:.4f}"
)

print()
print(
    "85–87% accuracy requirement: "
    + (
        "PASSED"
        if accuracy_requirement
        else "NOT PASSED"
    )
)

print(
    "Time-based robustness: "
    + (
        "PASSED"
        if accuracy_robust
        else "REVIEW REQUIRED"
    )
)

print(
    "ROC-AUC robustness: "
    + (
        "PASSED"
        if auc_robust
        else "REVIEW REQUIRED"
    )
)


# ============================================================
# SAVE FINAL VALIDATION SUMMARY
# ============================================================

validation_summary = {
    "dataset": str(DATA_PATH),
    "model": str(MODEL_PATH),
    "rows": int(len(df)),
    "unique_acos": int(unique_acos),
    "model_years": [
        int(df["YEAR"].min()),
        int(df["YEAR"].max()),
    ],
    "production_threshold": production_threshold,

    "cross_validation": {
        "accuracy": float(cv_default["accuracy"]),
        "precision": float(cv_default["precision"]),
        "recall": float(cv_default["recall"]),
        "f1": float(cv_default["f1"]),
        "roc_auc": float(cv_default["roc_auc"]),
        "pr_auc": float(cv_default["pr_auc"]),
    },

    "time_based_2023": {
        "accuracy": float(time_metrics["accuracy"]),
        "precision": float(time_metrics["precision"]),
        "recall": float(time_metrics["recall"]),
        "f1": float(time_metrics["f1"]),
        "roc_auc": float(time_metrics["roc_auc"]),
        "pr_auc": float(time_metrics["pr_auc"]),
    },

    "accuracy_requirement": {
        "minimum": 0.85,
        "maximum": 0.87,
        "cross_validation_accuracy": float(
            production_accuracy
        ),
        "passed": bool(
            accuracy_requirement
        ),
    },

    "robustness": {
        "accuracy_difference": float(
            accuracy_difference
        ),
        "roc_auc_difference": float(
            roc_difference
        ),
        "accuracy_robust": bool(
            accuracy_robust
        ),
        "roc_auc_robust": bool(
            auc_robust
        ),
    },
}

with open(
    REPORT_DIR
    / "risk_model_validation_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        validation_summary,
        f,
        indent=4,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print_section(
    "VALIDATION COMPLETE"
)

print(
    "Reports saved to:"
)

print(REPORT_DIR)

print()
print("Generated files:")

for file in sorted(REPORT_DIR.iterdir()):

    if file.is_file():

        print(
            f" - {file.name}"
        )

print()
print(
    "IMPORTANT:"
)

print(
    "This validation does not modify the dataset."
)

print(
    "This validation does not modify target labels."
)

print(
    "This validation does not add synthetic noise."
)

print(
    "The existing Random Forest remains unchanged."
)

print()
print(
    "NEXT STEP:"
)

print(
    "Review the validation results before creating "
    "the final production model."
)

print("=" * 78)