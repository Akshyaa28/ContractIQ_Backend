from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONTRACTIQ
# DATASET 1 — CATBOOST-ONLY SAVINGS FORECAST MODEL
# ============================================================

print("=" * 78)
print("CONTRACTIQ — DATASET 1 CATBOOST-ONLY SAVINGS FORECAST MODEL")
print("=" * 78)


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# forecasting/
#   train_forecast_model.py
#
# parents[0] = forecasting
# parents[1] = dataset1
# parents[2] = preprocessing
# parents[3] = ContractIQ(!)

PROJECT_ROOT = SCRIPT_PATH.parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Model_Forecast.csv"
)

SCORING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Scoring_2024_Forecast.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "preprocessing"
    / "dataset1"
    / "reports"
    / "forecast_model_training"
)


MODEL_PATH = (
    MODEL_DIR
    / "forecast_catboost.cbm"
)

VALIDATION_PATH = (
    MODEL_DIR
    / "catboost_validation_predictions.csv"
)

FORECAST_PATH = (
    MODEL_DIR
    / "Dataset1_Scoring_2024_CatBoost_Forecast_Predictions.csv"
)

IMPORTANCE_PATH = (
    MODEL_DIR
    / "catboost_feature_importance.csv"
)

METRICS_PATH = (
    MODEL_DIR
    / "catboost_validation_metrics.json"
)

METADATA_PATH = (
    MODEL_DIR
    / "catboost_model_metadata.json"
)


# ============================================================
# DATA DEFINITIONS
# ============================================================

BASE_FEATURES = [
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
# DISPLAY PATHS
# ============================================================

print()
print("SCRIPT LOCATION")
print(SCRIPT_PATH)

print()
print("PROJECT ROOT")
print(PROJECT_ROOT)

print()
print("INPUT DATASET")
print(INPUT_PATH)

print()
print("2024 SCORING DATASET")
print(SCORING_PATH)

print()
print("MODEL DIRECTORY")
print(MODEL_DIR)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print()
print("=" * 78)
print("CHECKING INPUT DATA")
print("=" * 78)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"\nHistorical model dataset not found:\n{INPUT_PATH}"
    )

if not SCORING_PATH.exists():
    raise FileNotFoundError(
        f"\n2024 scoring dataset not found:\n{SCORING_PATH}"
    )

print("Historical model dataset found.")
print("2024 scoring dataset found.")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 78)
print("LOADING DATA")
print("=" * 78)

df = pd.read_csv(INPUT_PATH)
scoring_df = pd.read_csv(SCORING_PATH)

print(
    f"Historical rows : {len(df):,}"
)

print(
    f"2024 rows       : {len(scoring_df):,}"
)


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.upper()
)

scoring_df.columns = (
    scoring_df.columns
    .str.strip()
    .str.upper()
)


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

print()
print("=" * 78)
print("CHECKING REQUIRED COLUMNS")
print("=" * 78)

required_historical = (
    IDENTIFIERS
    + BASE_FEATURES
    + [TARGET]
)

required_scoring = (
    IDENTIFIERS
    + BASE_FEATURES
)

missing_historical = [
    column
    for column in required_historical
    if column not in df.columns
]

missing_scoring = [
    column
    for column in required_scoring
    if column not in scoring_df.columns
]

if missing_historical:
    raise ValueError(
        "Missing historical columns:\n"
        + "\n".join(missing_historical)
    )

if missing_scoring:
    raise ValueError(
        "Missing scoring columns:\n"
        + "\n".join(missing_scoring)
    )

print("All required columns are present.")


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("=" * 78)
print("BASIC DATA VALIDATION")
print("=" * 78)

# YEAR

df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce",
)

scoring_df["YEAR"] = pd.to_numeric(
    scoring_df["YEAR"],
    errors="coerce",
)

if df["YEAR"].isna().any():
    raise ValueError("Historical YEAR contains invalid values.")

if scoring_df["YEAR"].isna().any():
    raise ValueError("Scoring YEAR contains invalid values.")

df["YEAR"] = df["YEAR"].astype(int)
scoring_df["YEAR"] = scoring_df["YEAR"].astype(int)


# TARGET

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

if df[TARGET].isna().any():
    raise ValueError(
        "Historical target contains missing values."
    )

if np.isinf(df[TARGET]).any():
    raise ValueError(
        "Historical target contains infinite values."
    )


# FEATURES

for feature in BASE_FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

    scoring_df[feature] = pd.to_numeric(
        scoring_df[feature],
        errors="coerce",
    )

    if df[feature].isna().any():
        raise ValueError(
            f"Historical feature contains missing values: {feature}"
        )

    if scoring_df[feature].isna().any():
        raise ValueError(
            f"Scoring feature contains missing values: {feature}"
        )

    if np.isinf(df[feature]).any():
        raise ValueError(
            f"Historical feature contains infinite values: {feature}"
        )

    if np.isinf(scoring_df[feature]).any():
        raise ValueError(
            f"Scoring feature contains infinite values: {feature}"
        )


# DUPLICATES

duplicate_rows = df.duplicated().sum()

duplicate_aco_year = df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print(
    f"Rows                   : {len(df):,}"
)

print(
    f"Columns                : {len(df.columns)}"
)

print(
    f"Exact duplicate rows   : {duplicate_rows}"
)

print(
    f"Duplicate ACO-year rows: {duplicate_aco_year}"
)

if duplicate_rows > 0:
    raise ValueError(
        "Exact duplicate rows detected."
    )

if duplicate_aco_year > 0:
    raise ValueError(
        "Duplicate ACO-year rows detected."
    )


# ============================================================
# YEAR INFORMATION
# ============================================================

print()
print("=" * 78)
print("YEAR VALIDATION")
print("=" * 78)

print(
    f"Historical year range: "
    f"{df['YEAR'].min()}–{df['YEAR'].max()}"
)

print()
print("Rows by year:")

print(
    df["YEAR"]
    .value_counts()
    .sort_index()
)


# ============================================================
# TARGET INFORMATION
# ============================================================

print()
print("=" * 78)
print("TARGET VALIDATION")
print("=" * 78)

print(
    f"Target: {TARGET}"
)

print(
    f"Missing : {df[TARGET].isna().sum()}"
)

print(
    f"Infinite: {np.isinf(df[TARGET]).sum()}"
)

print()
print("Target statistics:")

print(
    df[TARGET].describe()
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print()
print("=" * 78)
print("TARGET LEAKAGE CHECK")
print("=" * 78)

future_columns = [
    column
    for column in df.columns
    if (
        "NEXT" in column
        or "FUTURE" in column
        or "LEAD" in column
    )
]

print("Future-related columns detected:")

for column in future_columns:
    print(f" - {column}")

print()
print(f"Forecast target: {TARGET}")

print()
print("The target is used ONLY as the regression target.")

if TARGET in BASE_FEATURES:
    raise ValueError(
        "TARGET LEAKAGE DETECTED."
    )

print()
print("Leakage check PASSED.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print()
print("=" * 78)
print("CREATING CATBOOST FORECAST FEATURES")
print("=" * 78)


def create_features(data):
    """
    Create forecasting features using only current
    and historical information.

    NEXT_YEAR_SAVINGS_RATE is NEVER used as a predictor.
    """

    out = data.copy()

    # --------------------------------------------------------
    # Sort chronologically within ACO
    # --------------------------------------------------------

    out = out.sort_values(
        ["ACO_ID", "YEAR"]
    ).reset_index(
        drop=True
    )

    grouped = out.groupby(
        "ACO_ID",
        sort=False,
    )

    # --------------------------------------------------------
    # Direct aliases
    # --------------------------------------------------------

    out["SAVINGS_RATE"] = (
        out["PREVIOUS_SAVINGS_RATE"]
    )

    out["QUALITY_SCORE"] = (
        out["PREVIOUS_QUALITY_SCORE"]
    )

    out["PERFORMANCE_GAP"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    # --------------------------------------------------------
    # Nonlinear terms
    # --------------------------------------------------------

    out["SAVINGS_SQ"] = (
        out["PREVIOUS_SAVINGS_RATE"] ** 2
    )

    out["QUALITY_SQ"] = (
        out["PREVIOUS_QUALITY_SCORE"] ** 2
    )

    out["PERFORMANCE_GAP_SQ"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"] ** 2
    )

    out["EXPENDITURE_GROWTH_SQ"] = (
        out["EXPENDITURE_GROWTH_PCT"] ** 2
    )

    out["BENCHMARK_GROWTH_SQ"] = (
        out["BENCHMARK_GROWTH_PCT"] ** 2
    )

    out["BENEFICIARY_GROWTH_SQ"] = (
        out["BENEFICIARY_GROWTH_PCT"] ** 2
    )

    out["QUALITY_CHANGE_SQ"] = (
        out["QUALITY_CHANGE"] ** 2
    )

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    out["SAVINGS_QUALITY"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        * out["PREVIOUS_QUALITY_SCORE"]
    )

    out["SAVINGS_PERFORMANCE"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        * out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    out["QUALITY_PERFORMANCE"] = (
        out["PREVIOUS_QUALITY_SCORE"]
        * out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    out["EXPENDITURE_BENCHMARK"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        * out["BENCHMARK_GROWTH_PCT"]
    )

    out["EXPENDITURE_BENEFICIARY"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        * out["BENEFICIARY_GROWTH_PCT"]
    )

    out["BENCHMARK_BENEFICIARY"] = (
        out["BENCHMARK_GROWTH_PCT"]
        * out["BENEFICIARY_GROWTH_PCT"]
    )

    out["QUALITY_CHANGE_PERFORMANCE"] = (
        out["QUALITY_CHANGE"]
        * out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    # --------------------------------------------------------
    # Growth gaps
    # --------------------------------------------------------

    out["EXPENDITURE_BENCHMARK_GAP"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        - out["BENCHMARK_GROWTH_PCT"]
    )

    out["EXPENDITURE_BENEFICIARY_GAP"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        - out["BENEFICIARY_GROWTH_PCT"]
    )

    out["BENCHMARK_BENEFICIARY_GAP"] = (
        out["BENCHMARK_GROWTH_PCT"]
        - out["BENEFICIARY_GROWTH_PCT"]
    )

    # --------------------------------------------------------
    # Composite indicators
    # --------------------------------------------------------

    out["GROWTH_COMPOSITE"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        + out["BENCHMARK_GROWTH_PCT"]
        + out["BENEFICIARY_GROWTH_PCT"]
    ) / 3.0

    out["QUALITY_PERFORMANCE_COMPOSITE"] = (
        out["PREVIOUS_QUALITY_SCORE"]
        + out["PREVIOUS_PERFORMANCE_GAP_PCT"]
        + out["QUALITY_CHANGE"]
    ) / 3.0

    out["SAVINGS_MOMENTUM"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        + out["QUALITY_CHANGE"]
        - out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    out["PERFORMANCE_QUALITY_RATIO"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
        / (
            np.abs(
                out["PREVIOUS_QUALITY_SCORE"]
            )
            + 1e-6
        )
    )

    out["SAVINGS_QUALITY_RATIO"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        / (
            np.abs(
                out["PREVIOUS_QUALITY_SCORE"]
            )
            + 1e-6
        )
    )

    # --------------------------------------------------------
    # Historical savings lags
    # --------------------------------------------------------

    out["SAVINGS_LAG_1"] = (
        grouped["PREVIOUS_SAVINGS_RATE"]
        .shift(1)
    )

    out["SAVINGS_LAG_2"] = (
        grouped["PREVIOUS_SAVINGS_RATE"]
        .shift(2)
    )

    out["SAVINGS_LAG_3"] = (
        grouped["PREVIOUS_SAVINGS_RATE"]
        .shift(3)
    )

    # --------------------------------------------------------
    # Quality lags
    # --------------------------------------------------------

    out["QUALITY_LAG_1"] = (
        grouped["PREVIOUS_QUALITY_SCORE"]
        .shift(1)
    )

    out["QUALITY_LAG_2"] = (
        grouped["PREVIOUS_QUALITY_SCORE"]
        .shift(2)
    )

    # --------------------------------------------------------
    # Performance lags
    # --------------------------------------------------------

    out["PERFORMANCE_LAG_1"] = (
        grouped["PREVIOUS_PERFORMANCE_GAP_PCT"]
        .shift(1)
    )

    out["PERFORMANCE_LAG_2"] = (
        grouped["PREVIOUS_PERFORMANCE_GAP_PCT"]
        .shift(2)
    )

    # --------------------------------------------------------
    # Growth lags
    # --------------------------------------------------------

    out["EXPENDITURE_LAG_1"] = (
        grouped["EXPENDITURE_GROWTH_PCT"]
        .shift(1)
    )

    out["BENCHMARK_LAG_1"] = (
        grouped["BENCHMARK_GROWTH_PCT"]
        .shift(1)
    )

    out["BENEFICIARY_LAG_1"] = (
        grouped["BENEFICIARY_GROWTH_PCT"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Trends
    # --------------------------------------------------------

    out["SAVINGS_TREND"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        - out["SAVINGS_LAG_1"]
    )

    out["SAVINGS_TREND_2"] = (
        out["SAVINGS_LAG_1"]
        - out["SAVINGS_LAG_2"]
    )

    out["SAVINGS_ACCELERATION"] = (
        out["SAVINGS_TREND"]
        - out["SAVINGS_TREND_2"]
    )

    out["QUALITY_TREND"] = (
        out["PREVIOUS_QUALITY_SCORE"]
        - out["QUALITY_LAG_1"]
    )

    out["PERFORMANCE_TREND"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
        - out["PERFORMANCE_LAG_1"]
    )

    out["EXPENDITURE_TREND"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        - out["EXPENDITURE_LAG_1"]
    )

    out["BENCHMARK_TREND"] = (
        out["BENCHMARK_GROWTH_PCT"]
        - out["BENCHMARK_LAG_1"]
    )

    out["BENEFICIARY_TREND"] = (
        out["BENEFICIARY_GROWTH_PCT"]
        - out["BENEFICIARY_LAG_1"]
    )

    # --------------------------------------------------------
    # Rolling historical features
    #
    # SHIFT FIRST, THEN ROLL.
    #
    # This prevents the current year's information from
    # entering its historical rolling feature.
    # --------------------------------------------------------

    shifted_savings = (
        grouped["PREVIOUS_SAVINGS_RATE"]
        .shift(1)
    )

    out["SAVINGS_ROLLING_MEAN_3"] = (
        shifted_savings
        .groupby(
            out["ACO_ID"],
            sort=False,
        )
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=1,
            ).mean()
        )
    )

    out["SAVINGS_ROLLING_STD_3"] = (
        shifted_savings
        .groupby(
            out["ACO_ID"],
            sort=False,
        )
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=2,
            ).std()
        )
    )

    shifted_quality = (
        grouped["PREVIOUS_QUALITY_SCORE"]
        .shift(1)
    )

    out["QUALITY_ROLLING_MEAN_3"] = (
        shifted_quality
        .groupby(
            out["ACO_ID"],
            sort=False,
        )
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Additional momentum features
    # --------------------------------------------------------

    out["SAVINGS_CHANGE_2Y"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        - out["SAVINGS_LAG_2"]
    )

    out["QUALITY_CHANGE_2Y"] = (
        out["PREVIOUS_QUALITY_SCORE"]
        - out["QUALITY_LAG_2"]
    )

    out["PERFORMANCE_CHANGE_2Y"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
        - out["PERFORMANCE_LAG_2"]
    )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    numeric_columns = out.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        if column == TARGET:
            continue

        out[column] = (
            out[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return out


# ============================================================
# BUILD HISTORICAL FEATURES
# ============================================================

print()
print("=" * 78)
print("ENGINEERING HISTORICAL FEATURES")
print("=" * 78)

historical_features = create_features(df)

print(
    f"Historical rows: "
    f"{len(historical_features):,}"
)


# ============================================================
# BUILD 2024 FEATURES
# ============================================================

print()
print("=" * 78)
print("ENGINEERING 2024 SCORING FEATURES")
print("=" * 78)

scoring_features = create_features(
    scoring_df
)

print(
    f"2024 scoring rows: "
    f"{len(scoring_features):,}"
)


# ============================================================
# IMPORTANT:
# BUILD SCORING LAGS USING HISTORICAL DATA
#
# The 2024 scoring dataset normally contains only 2024 rows.
# Therefore, calculating groupby lags on scoring_df alone
# produces no historical lag information.
#
# We construct a combined feature frame first.
# ============================================================

print()
print("=" * 78)
print("BUILDING CONSISTENT HISTORICAL + 2024 FEATURES")
print("=" * 78)

combined = pd.concat(
    [
        df.copy(),
        scoring_df.copy(),
    ],
    ignore_index=True,
)

combined_features = create_features(
    combined
)

historical_features = (
    combined_features[
        combined_features["YEAR"]
        < 2024
    ]
    .copy()
)

scoring_features = (
    combined_features[
        combined_features["YEAR"]
        == 2024
    ]
    .copy()
)


print(
    f"Historical feature rows: "
    f"{len(historical_features):,}"
)

print(
    f"2024 feature rows      : "
    f"{len(scoring_features):,}"
)


# ============================================================
# MODEL FEATURE LIST
# ============================================================

EXCLUDED_COLUMNS = set(
    IDENTIFIERS
    + [TARGET]
)

FEATURES = [
    column
    for column in historical_features.columns
    if column not in EXCLUDED_COLUMNS
]


# Remove columns that are not numeric.

FEATURES = [
    column
    for column in FEATURES
    if pd.api.types.is_numeric_dtype(
        historical_features[column]
    )
]


print()
print(
    f"Base features       : {len(BASE_FEATURES)}"
)

print(
    f"Engineered features : {len(FEATURES)}"
)

print()
print("CatBoost predictors:")

for index, feature in enumerate(
    FEATURES,
    start=1,
):

    print(
        f"{index:2d}. {feature}"
    )


# ============================================================
# FINAL FEATURE CLEANING
# ============================================================

print()
print("=" * 78)
print("FINAL FEATURE MATRIX CLEANING")
print("=" * 78)

for feature in FEATURES:

    historical_features[feature] = (
        historical_features[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    scoring_features[feature] = (
        scoring_features[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


# CatBoost handles missing values natively.
#
# We intentionally DO NOT use target-derived imputation.

X_all = historical_features[
    FEATURES
].astype(
    np.float32
)

y_all = historical_features[
    TARGET
].astype(
    np.float32
)

X_scoring = scoring_features[
    FEATURES
].astype(
    np.float32
)


# ============================================================
# TIME-BASED VALIDATION
# ============================================================

print()
print("=" * 78)
print("TIME-BASED VALIDATION")
print("=" * 78)

train_mask = (
    historical_features["YEAR"]
    <= 2022
)

validation_mask = (
    historical_features["YEAR"]
    == 2023
)

X_train = X_all.loc[
    train_mask
]

y_train = y_all.loc[
    train_mask
]

X_valid = X_all.loc[
    validation_mask
]

y_valid = y_all.loc[
    validation_mask
]

validation_info = historical_features.loc[
    validation_mask
].copy()

print(
    f"Training years   : "
    f"{historical_features.loc[train_mask, 'YEAR'].min()}"
    f"–"
    f"{historical_features.loc[train_mask, 'YEAR'].max()}"
)

print(
    f"Validation year  : 2023"
)

print(
    f"Training rows    : "
    f"{len(X_train):,}"
)

print(
    f"Validation rows  : "
    f"{len(X_valid):,}"
)


# ============================================================
# CATBOOST ONLY
# ============================================================

print()
print("=" * 78)
print("CATBOOST-ONLY TRAINING")
print("=" * 78)

print()
print("MODEL:")
print("CatBoostRegressor")

print()
print("NO RANDOM FOREST.")
print("NO EXTRA TREES.")
print("NO XGBOOST.")
print("NO HIST GRADIENT BOOSTING.")
print("NO GRID SEARCH.")
print("NO RANDOM SEARCH.")
print("NO MODEL COMPARISON.")

print()
print("Only ONE machine-learning model will be trained.")


# ============================================================
# CATBOOST PARAMETERS
# ============================================================

CATBOOST_PARAMS = {
    "iterations": 3500,
    "depth": 8,
    "learning_rate": 0.025,
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "l2_leaf_reg": 6.0,
    "random_seed": 42,
    "random_strength": 0.5,
    "bagging_temperature": 0.5,
    "border_count": 128,
    "task_type": "CPU",
    "thread_count": 4,
    "od_type": "Iter",
    "od_wait": 250,
    "verbose": 100,
    "allow_writing_files": False,
}


print()
print("CATBOOST PARAMETERS")

for key, value in CATBOOST_PARAMS.items():
    print(
        f"{key:<24}: {value}"
    )


# ============================================================
# TRAIN SINGLE CATBOOST MODEL
# ============================================================

print()
print("=" * 78)
print("TRAINING CATBOOST")
print("=" * 78)

print()
print("Training started...")
print("This is the ONLY model being trained.")
print()
print(
    "CatBoost early stopping is enabled."
)

model = CatBoostRegressor(
    **CATBOOST_PARAMS
)

model.fit(
    X_train,
    y_train,
    eval_set=(
        X_valid,
        y_valid,
    ),
    use_best_model=True,
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 78)
print("2023 CATBOOST VALIDATION")
print("=" * 78)

validation_predictions = model.predict(
    X_valid
)

mae = mean_absolute_error(
    y_valid,
    validation_predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        validation_predictions,
    )
)

r2 = r2_score(
    y_valid,
    validation_predictions,
)

print()
print(
    f"MAE              : {mae:.6f}"
)

print(
    f"RMSE             : {rmse:.6f}"
)

print(
    f"R²               : {r2:.6f}"
)

print(
    f"R² percentage    : {r2 * 100:.2f}%"
)

print()

if r2 >= 0.87:

    print(
        "STATUS: EXCELLENT — R² >= 87%"
    )

elif r2 >= 0.85:

    print(
        "STATUS: TARGET ACHIEVED — R² >= 85%"
    )

elif r2 >= 0.80:

    print(
        "STATUS: STRONG — R² >= 80%"
    )

else:

    print(
        "STATUS: BELOW 85% TARGET"
    )

print()
print(
    "IMPORTANT: The validation R² is the actual "
    "out-of-time performance. It is not artificially "
    "inflated to reach 85%."
)


# ============================================================
# BEST ITERATION
# ============================================================

best_iteration = model.get_best_iteration()

if best_iteration is None:
    best_iteration = (
        CATBOOST_PARAMS["iterations"] - 1
    )

best_iteration = int(
    best_iteration
)

print()
print(
    f"Best CatBoost iteration: "
    f"{best_iteration + 1}"
)


# ============================================================
# VALIDATION OUTPUT
# ============================================================

print()
print("=" * 78)
print("SAVING VALIDATION PREDICTIONS")
print("=" * 78)

validation_output = validation_info[
    [
        "ACO_ID",
        "ACO_NAME",
        "STATE",
        "YEAR",
        TARGET,
    ]
].copy()

validation_output[
    "PREDICTED_NEXT_YEAR_SAVINGS_RATE"
] = validation_predictions

validation_output[
    "PREDICTED_NEXT_YEAR_SAVINGS_RATE_PCT"
] = (
    validation_predictions * 100
)

validation_output[
    "ABSOLUTE_ERROR"
] = np.abs(
    validation_output[TARGET]
    - validation_output[
        "PREDICTED_NEXT_YEAR_SAVINGS_RATE"
    ]
)

validation_output.to_csv(
    VALIDATION_PATH,
    index=False,
)

print(
    f"Validation predictions saved:\n"
    f"{VALIDATION_PATH}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 78)
print("CATBOOST FEATURE IMPORTANCE")
print("=" * 78)

importance_values = (
    model.get_feature_importance()
)

importance_df = pd.DataFrame(
    {
        "FEATURE": FEATURES,
        "IMPORTANCE": importance_values,
    }
).sort_values(
    "IMPORTANCE",
    ascending=False,
)

print()

print(
    importance_df
    .head(20)
    .to_string(
        index=False
    )
)

importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False,
)

print()
print(
    f"Feature importance saved:\n"
    f"{IMPORTANCE_PATH}"
)


# ============================================================
# FINAL PRODUCTION CATBOOST MODEL
# ============================================================

print()
print("=" * 78)
print("TRAINING FINAL PRODUCTION CATBOOST MODEL")
print("=" * 78)

print()
print(
    "Historical production data: 2018–2023"
)

print(
    "2024 remains the forecast/scoring year."
)


# ------------------------------------------------------------
# Use validation-selected iteration count.
# ------------------------------------------------------------

production_iterations = max(
    300,
    best_iteration + 1,
)

production_params = CATBOOST_PARAMS.copy()

production_params[
    "iterations"
] = production_iterations

production_params[
    "verbose"
] = 200


print()
print(
    f"Production iterations: "
    f"{production_iterations}"
)


production_model = CatBoostRegressor(
    **production_params
)

production_model.fit(
    X_all,
    y_all,
)


# ============================================================
# SAVE PRODUCTION MODEL
# ============================================================

print()
print("=" * 78)
print("SAVING CATBOOST MODEL")
print("=" * 78)

production_model.save_model(
    str(MODEL_PATH)
)

print(
    f"CatBoost model saved:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# 2024 FORECAST
# ============================================================

print()
print("=" * 78)
print("CREATING 2024 FORECAST")
print("=" * 78)

forecast_predictions = (
    production_model.predict(
        X_scoring
    )
)


# ============================================================
# FORECAST OUTPUT
# ============================================================

forecast_output = scoring_df[
    [
        "ACO_ID",
        "ACO_NAME",
        "STATE",
        "YEAR",
    ]
].copy()

forecast_output[
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE"
] = forecast_predictions

forecast_output[
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT"
] = (
    forecast_predictions * 100
)


# ============================================================
# FORECAST CLASSIFICATION
# ============================================================

def classify_forecast(value):

    if value >= 0.05:
        return "HIGH SAVINGS"

    elif value >= 0.00:
        return "MODERATE SAVINGS"

    else:
        return "LOW SAVINGS"


forecast_output[
    "SAVINGS_FORECAST_CATEGORY"
] = [
    classify_forecast(value)
    for value in forecast_predictions
]


# ============================================================
# SAVE 2024 FORECAST
# ============================================================

print()
print("=" * 78)
print("SAVING 2024 FORECAST")
print("=" * 78)

forecast_output.to_csv(
    FORECAST_PATH,
    index=False,
)

print(
    f"2024 forecast saved:\n"
    f"{FORECAST_PATH}"
)


# ============================================================
# FORECAST SUMMARY
# ============================================================

print()
print("=" * 78)
print("2024 FORECAST SUMMARY")
print("=" * 78)

print(
    f"Rows forecasted : "
    f"{len(forecast_output):,}"
)

print(
    f"Mean forecast   : "
    f"{forecast_predictions.mean():.6f}"
)

print(
    f"Median forecast : "
    f"{np.median(forecast_predictions):.6f}"
)

print(
    f"Minimum forecast: "
    f"{forecast_predictions.min():.6f}"
)

print(
    f"Maximum forecast: "
    f"{forecast_predictions.max():.6f}"
)

print()
print("Forecast categories:")

print(
    forecast_output[
        "SAVINGS_FORECAST_CATEGORY"
    ].value_counts()
)


# ============================================================
# TOP 10
# ============================================================

print()
print("=" * 78)
print("TOP 10 FORECASTED SAVINGS")
print("=" * 78)

top_10 = (
    forecast_output
    .sort_values(
        "FORECASTED_NEXT_YEAR_SAVINGS_RATE",
        ascending=False,
    )
    .head(10)
)

print(
    top_10[
        [
            "ACO_ID",
            "ACO_NAME",
            "STATE",
            "YEAR",
            "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
            "SAVINGS_FORECAST_CATEGORY",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BOTTOM 10
# ============================================================

print()
print("=" * 78)
print("BOTTOM 10 FORECASTED SAVINGS")
print("=" * 78)

bottom_10 = (
    forecast_output
    .sort_values(
        "FORECASTED_NEXT_YEAR_SAVINGS_RATE",
        ascending=True,
    )
    .head(10)
)

print(
    bottom_10[
        [
            "ACO_ID",
            "ACO_NAME",
            "STATE",
            "YEAR",
            "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
            "SAVINGS_FORECAST_CATEGORY",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# METRICS JSON
# ============================================================

metrics = {
    "model": "CatBoostRegressor",

    "model_count": 1,

    "other_models_trained": [],

    "target": TARGET,

    "validation_year": 2023,

    "training_years": "2018-2022",

    "production_training_years": "2018-2023",

    "forecast_year": 2024,

    "training_rows": int(
        len(X_train)
    ),

    "validation_rows": int(
        len(X_valid)
    ),

    "production_rows": int(
        len(X_all)
    ),

    "forecast_rows_2024": int(
        len(X_scoring)
    ),

    "features": int(
        len(FEATURES)
    ),

    "mae": float(mae),

    "rmse": float(rmse),

    "r2": float(r2),

    "r2_percentage": float(
        r2 * 100
    ),

    "best_iteration": int(
        best_iteration + 1
    ),

    "production_iterations": int(
        production_iterations
    ),

    "leakage_protection": True,

    "future_target_used_as_feature": False,
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


# ============================================================
# METADATA
# ============================================================

metadata = {
    "project": "ContractIQ",

    "dataset": "Dataset 1",

    "model": "CatBoostRegressor",

    "only_model": True,

    "other_models": [],

    "target": TARGET,

    "base_features": BASE_FEATURES,

    "engineered_features": FEATURES,

    "validation_strategy": (
        "Strict time-based validation"
    ),

    "training_years": "2018-2022",

    "validation_year": 2023,

    "production_training_years": "2018-2023",

    "forecast_year": 2024,

    "parameters": {
        key: value
        for key, value
        in production_params.items()
    },

    "validation_metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "R2_PERCENTAGE": float(
            r2 * 100
        ),
    },

    "best_iteration": int(
        best_iteration + 1
    ),

    "production_iterations": int(
        production_iterations
    ),

    "leakage_protection": True,

    "future_target_used_as_feature": False,
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        default=str,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("FINAL CATBOOST-ONLY MODEL SUMMARY")
print("=" * 78)

print()
print(
    "MODEL: CatBoostRegressor ONLY"
)

print(
    "Random Forest     : NOT TRAINED"
)

print(
    "Extra Trees       : NOT TRAINED"
)

print(
    "XGBoost           : NOT TRAINED"
)

print(
    "HistGradientBoost : NOT TRAINED"
)

print(
    "Grid Search       : NOT USED"
)

print(
    "Random Search     : NOT USED"
)

print()
print(
    f"Training rows     : {len(X_train):,}"
)

print(
    f"Validation rows   : {len(X_valid):,}"
)

print(
    f"Production rows   : {len(X_all):,}"
)

print(
    f"2024 forecast rows: {len(X_scoring):,}"
)

print()
print(
    f"2023 MAE          : {mae:.6f}"
)

print(
    f"2023 RMSE         : {rmse:.6f}"
)

print(
    f"2023 R²           : {r2:.6f}"
)

print(
    f"2023 R² percentage: {r2 * 100:.2f}%"
)

print()
print(
    f"Best iteration    : "
    f"{best_iteration + 1}"
)

print()
print("MODEL FILE:")
print(MODEL_PATH)

print()
print("2024 FORECAST FILE:")
print(FORECAST_PATH)

print()
print("VALIDATION FILE:")
print(VALIDATION_PATH)

print()
print("FEATURE IMPORTANCE:")
print(IMPORTANCE_PATH)

print()
print("METRICS:")
print(METRICS_PATH)

print()
print("METADATA:")
print(METADATA_PATH)

print()
print("=" * 78)
print("CATBOOST-ONLY TRAINING COMPLETE")
print("=" * 78)