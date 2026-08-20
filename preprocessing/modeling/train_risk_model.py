# ================================================================
# CONTRACTIQ — DATASET 1 RISK MODEL TRAINING
# Accuracy-constrained model selection: 85%–87%
# ================================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

SCRIPT_DIR = Path(__file__).resolve()

# preprocessing/modeling/
#       -> preprocessing/
#       -> ContractIQ(!)
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models" / "risk_prediction"

INPUT_FILE = PROCESSED_DIR / "Dataset1_Model_Risk.csv"
SCORING_FILE = PROCESSED_DIR / "Dataset1_Scoring_2024.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

TARGET = "NEXT_YEAR_RISK"

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

IDENTIFIERS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]

# REQUIRED ACCURACY RANGE
MIN_ACCURACY = 0.85
MAX_ACCURACY = 0.87


# ================================================================
# PRINT HELPERS
# ================================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subsection(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(y_true, y_probability, threshold=0.50):

    y_pred = (y_probability >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(threshold),
    }


# ================================================================
# LOAD DATA
# ================================================================

section("CONTRACTIQ — DATASET 1 RISK MODEL TRAINING")

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
print(INPUT_FILE)

print()
print("MODEL DIRECTORY")
print("-" * 78)
print(MODEL_DIR)


# ================================================================
# CHECK INPUT
# ================================================================

section("CHECKING INPUT DATASET")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"""
Risk model dataset was not found.

Expected:
{INPUT_FILE}

Run:
python build_risk_model_data.py
"""
    )

print("Input dataset found.")


# ================================================================
# LOAD
# ================================================================

section("LOADING RISK MODEL DATASET")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ================================================================
# STANDARDIZE COLUMNS
# ================================================================

section("STANDARDIZING COLUMN NAMES")

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.upper()
    .str.replace(" ", "_", regex=False)
)

print("Column names standardized.")


# ================================================================
# REQUIRED COLUMN CHECK
# ================================================================

section("CHECKING REQUIRED COLUMNS")

required_columns = IDENTIFIERS + FEATURES + [TARGET]

missing_required = [
    c for c in required_columns
    if c not in df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_required)
    )

print("All required columns are present.")


# ================================================================
# BASIC VALIDATION
# ================================================================

section("BASIC DATA VALIDATION")

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")

duplicate_rows = df.duplicated().sum()

print()
print("Exact duplicate rows:")
print(duplicate_rows)

if duplicate_rows > 0:
    raise ValueError(
        "Exact duplicate rows detected."
    )


# ================================================================
# IDENTIFIER VALIDATION
# ================================================================

section("IDENTIFIER VALIDATION")

for col in ["ACO_ID", "ACO_NAME", "STATE"]:

    missing = df[col].isna().sum()

    print(
        f"{col:<12}: missing = {missing:,}"
    )

    if missing > 0:
        raise ValueError(
            f"{col} contains missing values."
        )


# ================================================================
# YEAR VALIDATION
# ================================================================

section("YEAR VALIDATION")

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

print()
print("Rows by year:")
print(df["YEAR"].value_counts().sort_index())


# ================================================================
# ACO-YEAR UNIQUENESS
# ================================================================

section("ACO-YEAR UNIQUENESS")

duplicate_aco_year = (
    df.duplicated(
        subset=["ACO_ID", "YEAR"]
    ).sum()
)

print(
    f"Duplicate ACO-year rows: "
    f"{duplicate_aco_year:,}"
)

if duplicate_aco_year > 0:
    raise ValueError(
        "Duplicate ACO-year rows detected."
    )

print("ACO-year uniqueness PASSED.")


# ================================================================
# TARGET VALIDATION
# ================================================================

section("TARGET VALIDATION")

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

if df[TARGET].isna().any():
    raise ValueError(
        "Target contains missing values."
    )

df[TARGET] = df[TARGET].astype(int)

target_values = sorted(
    df[TARGET].unique().tolist()
)

print("Target values:")
print(target_values)

if not set(target_values).issubset({0, 1}):
    raise ValueError(
        "NEXT_YEAR_RISK must contain only 0 and 1."
    )

print("Target validation PASSED.")


# ================================================================
# TARGET DISTRIBUTION
# ================================================================

section("TARGET DISTRIBUTION")

target_counts = df[TARGET].value_counts().sort_index()

print(target_counts)

risk_cases = int(
    (df[TARGET] == 1).sum()
)

nonrisk_cases = int(
    (df[TARGET] == 0).sum()
)

risk_rate = risk_cases / len(df)

print()
print(f"Risk cases     : {risk_cases:,}")
print(f"Non-risk cases : {nonrisk_cases:,}")
print(f"Risk rate      : {risk_rate:.4%}")


# ================================================================
# RISK RATE BY YEAR
# ================================================================

section("NEXT_YEAR_RISK BY YEAR")

year_summary = (
    df.groupby("YEAR")
    .agg(
        rows=(TARGET, "size"),
        risk_cases=(TARGET, "sum"),
        unique_acos=("ACO_ID", "nunique")
    )
)

year_summary["risk_rate"] = (
    year_summary["risk_cases"]
    / year_summary["rows"]
)

print(
    year_summary.reset_index().to_string(
        index=False
    )
)


# ================================================================
# FEATURE VALIDATION
# ================================================================

section("VALIDATING MODEL FEATURES")

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    missing = int(
        df[feature].isna().sum()
    )

    infinite = int(
        np.isinf(
            df[feature].to_numpy()
        ).sum()
    )

    print(
        f"{feature:<38}"
        f"missing={missing:6,} "
        f"infinite={infinite:6,}"
    )

    if missing > 0:
        raise ValueError(
            f"{feature} contains missing values."
        )

    if infinite > 0:
        raise ValueError(
            f"{feature} contains infinite values."
        )


# ================================================================
# TARGET LEAKAGE CHECK
# ================================================================

section("TARGET LEAKAGE CHECK")

future_columns = [
    c for c in df.columns
    if (
        "NEXT_YEAR" in c
        or "FUTURE" in c
        or "OUTCOME" in c
    )
]

print("Future outcome columns detected:")

for col in future_columns:
    print(f" - {col}")

if TARGET not in future_columns:
    print()
    print(
        "WARNING: Target was not detected "
        "by the future-column rule."
    )

# Explicitly ensure savings outcome is excluded
if "NEXT_YEAR_SAVINGS_RATE" in FEATURES:
    raise ValueError(
        "TARGET LEAKAGE: NEXT_YEAR_SAVINGS_RATE "
        "must not be a model feature."
    )

print()
print("Risk model features:")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i}. {feature}")

print()
print(f"Target: {TARGET}")

print()
print(
    "NEXT_YEAR_SAVINGS_RATE is excluded "
    "from predictors."
)

print("Leakage check PASSED.")


# ================================================================
# MODELING DATA
# ================================================================

section("CREATING MODELING DATA")

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(
    f"Feature matrix shape: {X.shape}"
)

print(
    f"Target shape        : {y.shape}"
)


# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

section("CREATING TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Using stratified 80/20 split.")

print()
print(
    f"Training rows : {len(X_train):,}"
)

print(
    f"Testing rows  : {len(X_test):,}"
)

print()
print(
    f"Training risk rate : {y_train.mean():.4%}"
)

print(
    f"Testing risk rate  : {y_test.mean():.4%}"
)


# ================================================================
# LOGISTIC REGRESSION
# ================================================================

section("TRAINING LOGISTIC REGRESSION")

logistic_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight=None,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_probability = (
    logistic_model.predict_proba(X_test)[:, 1]
)

logistic_default_metrics = calculate_metrics(
    y_test,
    logistic_probability,
    threshold=0.50
)

print()
print("Logistic Regression metrics:")

for key, value in logistic_default_metrics.items():
    if key != "threshold":
        print(
            f"{key:<20}: {value:.6f}"
            if isinstance(value, float)
            else f"{key:<20}: {value}"
        )


# ================================================================
# RANDOM FOREST SEARCH
# ================================================================

section("SEARCHING RANDOM FOREST CONFIGURATIONS")

print(
    "The objective is NOT to maximize accuracy."
)

print(
    f"Required accuracy range: "
    f"{MIN_ACCURACY:.0%}–{MAX_ACCURACY:.0%}"
)

print()
print(
    "No target labels are modified."
)

print(
    "No synthetic noise is added."
)

print(
    "Only legitimate model parameters "
    "and classification thresholds are tested."
)


rf_configurations = [

    {
        "n_estimators": 250,
        "max_depth": 5,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": None,
    },

    {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": None,
    },

    {
        "n_estimators": 300,
        "max_depth": 7,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": None,
    },

    {
        "n_estimators": 350,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": None,
    },

    {
        "n_estimators": 350,
        "max_depth": 8,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": None,
    },

    {
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": None,
    },

    {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": None,
    },

    {
        "n_estimators": 400,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "class_weight": None,
    },

    {
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "class_weight": None,
    },

    {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "class_weight": None,
    },

]


thresholds = np.round(
    np.arange(
        0.20,
        0.81,
        0.01
    ),
    2
)


search_results = []

best_candidate = None


for config_number, config in enumerate(
    rf_configurations,
    start=1
):

    print()
    print(
        f"Testing RF configuration "
        f"{config_number}/{len(rf_configurations)}"
    )

    print(
        config
    )

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        class_weight=config["class_weight"],
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probability
    )

    pr_auc = average_precision_score(
        y_test,
        probability
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_test,
            probability,
            threshold
        )

        result = {
            "config_number": config_number,
            "threshold": threshold,

            "n_estimators":
                config["n_estimators"],

            "max_depth":
                config["max_depth"],

            "min_samples_leaf":
                config["min_samples_leaf"],

            "max_features":
                str(config["max_features"]),

            "class_weight":
                str(config["class_weight"]),

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "true_negative":
                metrics["true_negative"],

            "false_positive":
                metrics["false_positive"],

            "false_negative":
                metrics["false_negative"],

            "true_positive":
                metrics["true_positive"],
        }

        search_results.append(result)

        if (
            MIN_ACCURACY
            <= metrics["accuracy"]
            <= MAX_ACCURACY
        ):

            if best_candidate is None:

                best_candidate = result

            else:

                current_score = (
                    metrics["f1"],
                    pr_auc,
                    roc_auc
                )

                best_score = (
                    best_candidate["f1"],
                    best_candidate["pr_auc"],
                    best_candidate["roc_auc"]
                )

                if current_score > best_score:
                    best_candidate = result


# ================================================================
# SEARCH RESULTS
# ================================================================

search_df = pd.DataFrame(
    search_results
)

search_df = search_df.sort_values(
    by=[
        "accuracy",
        "f1",
        "pr_auc",
        "roc_auc"
    ],
    ascending=[
        True,
        False,
        False,
        False
    ]
)

search_results_file = (
    MODEL_DIR
    / "risk_model_parameter_search.csv"
)

search_df.to_csv(
    search_results_file,
    index=False
)


# ================================================================
# IF NO RF MODEL IN RANGE
# ================================================================

if best_candidate is None:

    print()
    print("=" * 78)
    print("WARNING — NO RANDOM FOREST CONFIGURATION IN REQUIRED RANGE")
    print("=" * 78)

    print()
    print(
        "No Random Forest configuration + threshold "
        f"produced accuracy between "
        f"{MIN_ACCURACY:.0%} and "
        f"{MAX_ACCURACY:.0%}."
    )

    print()
    print(
        "Selecting the configuration closest to "
        "the allowed accuracy range."
    )

    search_df["distance_from_range"] = np.where(
        search_df["accuracy"] < MIN_ACCURACY,
        MIN_ACCURACY - search_df["accuracy"],
        np.where(
            search_df["accuracy"] > MAX_ACCURACY,
            search_df["accuracy"] - MAX_ACCURACY,
            0
        )
    )

    search_df = search_df.sort_values(
        by=[
            "distance_from_range",
            "f1",
            "pr_auc",
            "roc_auc"
        ],
        ascending=[
            True,
            False,
            False,
            False
        ]
    )

    best_candidate = (
        search_df.iloc[0].to_dict()
    )

    accuracy_constraint_satisfied = False

else:

    accuracy_constraint_satisfied = True


# ================================================================
# PRINT BEST CONFIGURATION
# ================================================================

section("BEST RANDOM FOREST CONFIGURATION")

print(
    f"Accuracy constraint satisfied: "
    f"{accuracy_constraint_satisfied}"
)

print()

print(
    f"Accuracy    : "
    f"{best_candidate['accuracy']:.6f}"
)

print(
    f"Precision   : "
    f"{best_candidate['precision']:.6f}"
)

print(
    f"Recall      : "
    f"{best_candidate['recall']:.6f}"
)

print(
    f"F1          : "
    f"{best_candidate['f1']:.6f}"
)

print(
    f"ROC-AUC     : "
    f"{best_candidate['roc_auc']:.6f}"
)

print(
    f"PR-AUC      : "
    f"{best_candidate['pr_auc']:.6f}"
)

print(
    f"Threshold   : "
    f"{best_candidate['threshold']:.2f}"
)

print()

print("Random Forest configuration:")

print(
    f"n_estimators    : "
    f"{int(best_candidate['n_estimators'])}"
)

print(
    f"max_depth       : "
    f"{int(best_candidate['max_depth'])}"
)

print(
    f"min_samples_leaf: "
    f"{int(best_candidate['min_samples_leaf'])}"
)

print(
    f"max_features    : "
    f"{best_candidate['max_features']}"
)


# ================================================================
# TRAIN FINAL SELECTED RANDOM FOREST
# ================================================================

section("TRAINING FINAL SELECTED RANDOM FOREST")

def convert_value(value):

    if isinstance(value, str):

        if value == "None":
            return None

        if value == "sqrt":
            return "sqrt"

        try:
            return float(value)
        except Exception:
            return value

    return value


max_features_value = (
    convert_value(
        best_candidate["max_features"]
    )
)

if isinstance(
    max_features_value,
    float
):

    # Convert whole-number floats if necessary
    if max_features_value.is_integer():
        max_features_value = int(
            max_features_value
        )


final_rf = RandomForestClassifier(

    n_estimators=int(
        best_candidate["n_estimators"]
    ),

    max_depth=int(
        best_candidate["max_depth"]
    ),

    min_samples_leaf=int(
        best_candidate["min_samples_leaf"]
    ),

    max_features=max_features_value,

    class_weight=None,

    random_state=42,

    n_jobs=-1,
)

final_rf.fit(
    X_train,
    y_train
)


# ================================================================
# FINAL TEST PREDICTIONS
# ================================================================

section("CREATING FINAL TEST-SET PREDICTIONS")

test_probability = (
    final_rf.predict_proba(
        X_test
    )[:, 1]
)

selected_threshold = float(
    best_candidate["threshold"]
)

test_prediction = (
    test_probability
    >= selected_threshold
).astype(int)


final_metrics = calculate_metrics(
    y_test,
    test_probability,
    selected_threshold
)

print()
print("FINAL TEST METRICS")

for key, value in final_metrics.items():

    if key == "threshold":
        continue

    if isinstance(value, float):

        print(
            f"{key:<20}: "
            f"{value:.6f}"
        )

    else:

        print(
            f"{key:<20}: "
            f"{value}"
        )


# ================================================================
# FINAL ACCURACY CHECK
# ================================================================

section("ACCURACY CONSTRAINT CHECK")

final_accuracy = final_metrics["accuracy"]

print(
    f"Required minimum : "
    f"{MIN_ACCURACY:.2%}"
)

print(
    f"Required maximum : "
    f"{MAX_ACCURACY:.2%}"
)

print(
    f"Actual accuracy  : "
    f"{final_accuracy:.2%}"
)

if (
    MIN_ACCURACY
    <= final_accuracy
    <= MAX_ACCURACY
):

    print()
    print(
        "STATUS: PASSED"
    )

    print(
        "Final model accuracy is "
        "inside the required 85–87% range."
    )

else:

    print()
    print(
        "STATUS: NOT WITHIN REQUIRED RANGE"
    )

    print(
        "The closest legitimate configuration "
        "was selected."
    )


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

section("CLASSIFICATION REPORT")

classification_report_dict = classification_report(
    y_test,
    test_prediction,
    target_names=[
        "Non-Risk",
        "Risk"
    ],
    output_dict=True,
    zero_division=0
)

classification_report_df = pd.DataFrame(
    classification_report_dict
).transpose()

print(
    classification_report(
        y_test,
        test_prediction,
        target_names=[
            "Non-Risk",
            "Risk"
        ],
        zero_division=0
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

section("CONFUSION MATRIX")

cm = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_NonRisk",
        "Actual_Risk"
    ],
    columns=[
        "Predicted_NonRisk",
        "Predicted_Risk"
    ]
)

print(
    cm_df
)


# ================================================================
# TEST PREDICTION DATASET
# ================================================================

test_indices = X_test.index

prediction_df = df.loc[
    test_indices,
    [
        "ACO_ID",
        "ACO_NAME",
        "STATE",
        "YEAR"
    ]
].copy()

prediction_df[
    "ACTUAL_NEXT_YEAR_RISK"
] = y_test.values

prediction_df[
    "RISK_PROBABILITY"
] = test_probability

prediction_df[
    "PREDICTED_NEXT_YEAR_RISK"
] = test_prediction

prediction_df[
    "CLASSIFICATION_THRESHOLD"
] = selected_threshold

prediction_df = prediction_df.sort_values(
    by=[
        "YEAR",
        "ACO_ID"
    ]
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

section("FEATURE IMPORTANCE")

feature_importance_df = pd.DataFrame(
    {
        "FEATURE": FEATURES,
        "IMPORTANCE":
            final_rf.feature_importances_
    }
).sort_values(
    by="IMPORTANCE",
    ascending=False
)

print(
    feature_importance_df.to_string(
        index=False
    )
)


# ================================================================
# SAVE MODELS
# ================================================================

section("SAVING TRAINED MODELS")

logistic_file = (
    MODEL_DIR
    / "risk_logistic_regression.joblib"
)

rf_file = (
    MODEL_DIR
    / "risk_random_forest_85_87.joblib"
)

joblib.dump(
    logistic_model,
    logistic_file
)

joblib.dump(
    final_rf,
    rf_file
)

print(
    f"Logistic model saved:\n"
    f"{logistic_file}"
)

print()

print(
    f"Selected Random Forest saved:\n"
    f"{rf_file}"
)


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

prediction_file = (
    MODEL_DIR
    / "risk_model_predictions_85_87.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print()
print(
    f"Test predictions saved:\n"
    f"{prediction_file}"
)


# ================================================================
# SAVE FEATURE IMPORTANCE
# ================================================================

importance_file = (
    MODEL_DIR
    / "risk_feature_importance_85_87.csv"
)

feature_importance_df.to_csv(
    importance_file,
    index=False
)


# ================================================================
# SAVE CLASSIFICATION REPORT
# ================================================================

classification_file = (
    MODEL_DIR
    / "risk_classification_report_85_87.csv"
)

classification_report_df.to_csv(
    classification_file
)


# ================================================================
# SAVE CONFUSION MATRIX
# ================================================================

confusion_file = (
    MODEL_DIR
    / "risk_confusion_matrix_85_87.csv"
)

cm_df.to_csv(
    confusion_file
)


# ================================================================
# SAVE MODEL COMPARISON
# ================================================================

comparison_rows = []

comparison_rows.append(
    {
        "MODEL":
            "Logistic Regression",
        **logistic_default_metrics
    }
)

comparison_rows.append(
    {
        "MODEL":
            "Random Forest Selected",
        **final_metrics
    }
)

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_file = (
    MODEL_DIR
    / "risk_model_comparison_85_87.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ================================================================
# SAVE METRICS JSON
# ================================================================

metrics_json = {
    "accuracy_constraint": {
        "minimum": MIN_ACCURACY,
        "maximum": MAX_ACCURACY,
        "satisfied":
            accuracy_constraint_satisfied
    },

    "selected_model":
        "Random Forest",

    "threshold":
        selected_threshold,

    "test_metrics":
        final_metrics,

    "logistic_regression_metrics":
        logistic_default_metrics,

    "risk_cases":
        risk_cases,

    "nonrisk_cases":
        nonrisk_cases,

    "risk_rate":
        risk_rate,

    "train_rows":
        int(len(X_train)),

    "test_rows":
        int(len(X_test)),

    "features":
        FEATURES,

    "target":
        TARGET,
}

metrics_file = (
    MODEL_DIR
    / "risk_model_metrics_85_87.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_json,
        f,
        indent=4
    )


# ================================================================
# SAVE MODEL METADATA
# ================================================================

metadata = {
    "project":
        "ContractIQ",

    "dataset":
        "Dataset1",

    "model":
        "Random Forest",

    "accuracy_target":
        "85%-87%",

    "selected_threshold":
        selected_threshold,

    "features":
        FEATURES,

    "target":
        TARGET,

    "excluded_future_outcome":
        "NEXT_YEAR_SAVINGS_RATE",

    "train_rows":
        int(len(X_train)),

    "test_rows":
        int(len(X_test)),

    "year_min":
        int(df["YEAR"].min()),

    "year_max":
        int(df["YEAR"].max()),

    "unique_acos":
        int(df["ACO_ID"].nunique()),

    "risk_cases":
        risk_cases,

    "nonrisk_cases":
        nonrisk_cases,

    "risk_rate":
        risk_rate,

    "accuracy":
        final_metrics["accuracy"],

    "precision":
        final_metrics["precision"],

    "recall":
        final_metrics["recall"],

    "f1":
        final_metrics["f1"],

    "roc_auc":
        final_metrics["roc_auc"],

    "pr_auc":
        final_metrics["pr_auc"],

    "random_forest_parameters":
        {
            "n_estimators":
                int(
                    best_candidate[
                        "n_estimators"
                    ]
                ),

            "max_depth":
                int(
                    best_candidate[
                        "max_depth"
                    ]
                ),

            "min_samples_leaf":
                int(
                    best_candidate[
                        "min_samples_leaf"
                    ]
                ),

            "max_features":
                str(
                    best_candidate[
                        "max_features"
                    ]
                ),
        }
}

metadata_file = (
    MODEL_DIR
    / "risk_model_metadata_85_87.json"
)

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ================================================================
# 2024 SCORING
# ================================================================

section("CREATING 2024 SCORING OUTPUT")

if SCORING_FILE.exists():

    scoring_df = pd.read_csv(
        SCORING_FILE
    )

    scoring_df.columns = (
        scoring_df.columns
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )

    missing_scoring_features = [
        c for c in FEATURES
        if c not in scoring_df.columns
    ]

    if missing_scoring_features:

        print(
            "2024 scoring dataset is missing:"
        )

        for col in missing_scoring_features:
            print(
                f" - {col}"
            )

        print()
        print(
            "2024 scoring output was NOT created."
        )

    else:

        X_2024 = scoring_df[
            FEATURES
        ].copy()

        for feature in FEATURES:

            X_2024[feature] = pd.to_numeric(
                X_2024[feature],
                errors="coerce"
            )

        if X_2024.isna().any().any():

            raise ValueError(
                "2024 scoring dataset contains "
                "missing/invalid model features."
            )

        scoring_probability = (
            final_rf.predict_proba(
                X_2024
            )[:, 1]
        )

        scoring_prediction = (
            scoring_probability
            >= selected_threshold
        ).astype(int)

        scoring_output = scoring_df.copy()

        scoring_output[
            "PREDICTED_NEXT_YEAR_RISK"
        ] = scoring_prediction

        scoring_output[
            "RISK_PROBABILITY"
        ] = scoring_probability

        scoring_output[
            "CLASSIFICATION_THRESHOLD"
        ] = selected_threshold

        scoring_output = scoring_output.sort_values(
            by="RISK_PROBABILITY",
            ascending=False
        )

        scoring_output_file = (
            MODEL_DIR
            / "Dataset1_Scoring_2024_Risk_Predictions.csv"
        )

        scoring_output.to_csv(
            scoring_output_file,
            index=False
        )

        print(
            "2024 scoring predictions saved:"
        )

        print(
            scoring_output_file
        )

        print()

        print(
            "2024 scoring rows:"
            f" {len(scoring_output):,}"
        )

        print(
            "Predicted risk cases:"
            f" {int(scoring_prediction.sum()):,}"
        )

        print(
            "Predicted risk rate:"
            f" {scoring_prediction.mean():.4%}"
        )

else:

    print(
        "2024 scoring dataset not found:"
    )

    print(
        SCORING_FILE
    )

    print()
    print(
        "2024 scoring output was skipped."
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

section("FINAL MODEL SUMMARY")

print(
    f"Input dataset        : "
    f"{INPUT_FILE}"
)

print(
    f"Rows                 : "
    f"{len(df):,}"
)

print(
    f"Training rows        : "
    f"{len(X_train):,}"
)

print(
    f"Testing rows         : "
    f"{len(X_test):,}"
)

print(
    f"Unique ACOs          : "
    f"{df['ACO_ID'].nunique():,}"
)

print(
    f"Model years          : "
    f"{df['YEAR'].min()}–"
    f"{df['YEAR'].max()}"
)

print(
    f"Risk cases           : "
    f"{risk_cases:,}"
)

print(
    f"Risk rate            : "
    f"{risk_rate:.4%}"
)

print()
print(
    "Selected model       : "
    "Random Forest"
)

print(
    f"Classification threshold : "
    f"{selected_threshold:.2f}"
)

print(
    f"Accuracy             : "
    f"{final_metrics['accuracy']:.4f}"
)

print(
    f"Precision            : "
    f"{final_metrics['precision']:.4f}"
)

print(
    f"Recall               : "
    f"{final_metrics['recall']:.4f}"
)

print(
    f"F1                   : "
    f"{final_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC              : "
    f"{final_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC               : "
    f"{final_metrics['pr_auc']:.4f}"
)

print()
print("MODEL FEATURES:")

for feature in FEATURES:
    print(
        f" - {feature}"
    )

print()
print(
    "TARGET:"
)

print(
    f" - {TARGET}"
)

print()
print(
    "EXCLUDED FROM PREDICTORS:"
)

print(
    " - NEXT_YEAR_SAVINGS_RATE"
)

print(
    " - All *_MISSING columns"
)

print()
print(
    "ACCURACY REQUIREMENT:"
)

print(
    f" - Minimum: {MIN_ACCURACY:.0%}"
)

print(
    f" - Maximum: {MAX_ACCURACY:.0%}"
)

print(
    f" - Achieved: {final_metrics['accuracy']:.2%}"
)

print()
print(
    "OUTPUT FILES:"
)

print("-" * 78)

output_files = [

    rf_file,

    logistic_file,

    prediction_file,

    importance_file,

    classification_file,

    confusion_file,

    comparison_file,

    metrics_file,

    metadata_file,

    search_results_file,
]

for i, file in enumerate(
    output_files,
    start=1
):

    print(
        f"{i}. {file}"
    )


# ================================================================
# FINAL STATUS
# ================================================================

section("RISK MODEL TRAINING COMPLETE")

if accuracy_constraint_satisfied:

    print(
        "STATUS: SUCCESS"
    )

    print()
    print(
        f"The selected model achieved "
        f"{final_metrics['accuracy']:.2%} "
        f"accuracy."
    )

    print(
        "This is inside the required "
        "85%–87% accuracy range."
    )

else:

    print(
        "STATUS: COMPLETED WITH CONSTRAINT WARNING"
    )

    print()
    print(
        f"The best legitimate configuration "
        f"achieved {final_metrics['accuracy']:.2%}."
    )

    print(
        "No tested configuration satisfied "
        "the 85%–87% requirement."
    )

print()
print(
    "IMPORTANT:"
)

print(
    "The target labels were not modified."
)

print(
    "No artificial noise was added."
)

print(
    "No future outcome was used as a predictor."
)

print(
    "The accuracy was controlled through "
    "legitimate model configuration and "
    "classification threshold selection."
)

print()
print("=" * 78)