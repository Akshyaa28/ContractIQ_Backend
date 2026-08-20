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

print()
print("IMPORTANT")
print("This script trains ONLY CatBoost.")
print("Random Forest is NOT used.")
print("Extra Trees is NOT used.")
print("XGBoost is NOT used.")
print("HistGradientBoosting is NOT used.")
print("GridSearch is NOT used.")
print("RandomizedSearch is NOT used.")


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# preprocessing/dataset1/forecasting/train_forecast_model.py
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
# FEATURES
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
# CHECK INPUTS
# ============================================================

print()
print("=" * 78)
print("CHECKING INPUT DATA")
print("=" * 78)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Historical dataset not found:\n{INPUT_PATH}"
    )

if not SCORING_PATH.exists():
    raise FileNotFoundError(
        f"2024 scoring dataset not found:\n{SCORING_PATH}"
    )

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Historical dataset found.")
print("2024 scoring dataset found.")


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 78)
print("LOADING DATA")
print("=" * 78)

df = pd.read_csv(INPUT_PATH)
scoring_df = pd.read_csv(SCORING_PATH)

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

print(
    f"Historical rows : {len(df):,}"
)

print(
    f"2024 rows       : {len(scoring_df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_hist = (
    IDENTIFIERS
    + BASE_FEATURES
    + [TARGET]
)

required_score = (
    IDENTIFIERS
    + BASE_FEATURES
)

missing_hist = [
    c for c in required_hist
    if c not in df.columns
]

missing_score = [
    c for c in required_score
    if c not in scoring_df.columns
]

if missing_hist:
    raise ValueError(
        "Missing historical columns:\n"
        + "\n".join(missing_hist)
    )

if missing_score:
    raise ValueError(
        "Missing scoring columns:\n"
        + "\n".join(missing_score)
    )

print("All required columns are present.")


# ============================================================
# TYPE CONVERSION
# ============================================================

print()
print("=" * 78)
print("VALIDATING DATA TYPES")
print("=" * 78)

df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce",
)

scoring_df["YEAR"] = pd.to_numeric(
    scoring_df["YEAR"],
    errors="coerce",
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

for col in BASE_FEATURES:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

    scoring_df[col] = pd.to_numeric(
        scoring_df[col],
        errors="coerce",
    )


if df["YEAR"].isna().any():
    raise ValueError("Historical YEAR contains invalid values.")

if scoring_df["YEAR"].isna().any():
    raise ValueError("Scoring YEAR contains invalid values.")

df["YEAR"] = df["YEAR"].astype(int)
scoring_df["YEAR"] = scoring_df["YEAR"].astype(int)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 78)
print("DATA VALIDATION")
print("=" * 78)

if df[TARGET].isna().any():
    raise ValueError(
        "Historical target contains missing values."
    )

if np.isinf(df[TARGET]).any():
    raise ValueError(
        "Historical target contains infinite values."
    )

for col in BASE_FEATURES:

    if df[col].isna().any():
        raise ValueError(
            f"Missing historical values in {col}"
        )

    if scoring_df[col].isna().any():
        raise ValueError(
            f"Missing scoring values in {col}"
        )

    if np.isinf(df[col]).any():
        raise ValueError(
            f"Infinite historical values in {col}"
        )

    if np.isinf(scoring_df[col]).any():
        raise ValueError(
            f"Infinite scoring values in {col}"
        )

print(
    f"Historical year range : "
    f"{df['YEAR'].min()}–{df['YEAR'].max()}"
)

print(
    f"Scoring years         : "
    f"{sorted(scoring_df['YEAR'].unique().tolist())}"
)

print(
    f"Target mean           : "
    f"{df[TARGET].mean():.6f}"
)

print(
    f"Target std            : "
    f"{df[TARGET].std():.6f}"
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_rows = df.duplicated().sum()

duplicate_aco_year = df.duplicated(
    subset=["ACO_ID", "YEAR"]
).sum()

print()
print(
    f"Exact duplicate rows : {duplicate_rows}"
)

print(
    f"Duplicate ACO-year   : {duplicate_aco_year}"
)

if duplicate_rows:
    raise ValueError(
        "Exact duplicate rows detected."
    )

if duplicate_aco_year:
    raise ValueError(
        "Duplicate ACO-year rows detected."
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print()
print("=" * 78)
print("CREATING CATBOOST FORECAST FEATURES")
print("=" * 78)


def engineer_features(data):
    """
    Leakage-safe feature engineering.

    All lag/rolling features are constructed from information
    available before the target year.

    NEXT_YEAR_SAVINGS_RATE is NEVER used as a predictor.
    """

    out = data.copy()

    # --------------------------------------------------------
    # Current-level aliases
    # --------------------------------------------------------

    out["SAVINGS_CURRENT"] = (
        out["PREVIOUS_SAVINGS_RATE"]
    )

    out["QUALITY_CURRENT"] = (
        out["PREVIOUS_QUALITY_SCORE"]
    )

    out["PERFORMANCE_CURRENT"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    # --------------------------------------------------------
    # Nonlinear terms
    # --------------------------------------------------------

    nonlinear_columns = [
        "PREVIOUS_SAVINGS_RATE",
        "PREVIOUS_QUALITY_SCORE",
        "PREVIOUS_PERFORMANCE_GAP_PCT",
        "EXPENDITURE_GROWTH_PCT",
        "BENCHMARK_GROWTH_PCT",
        "BENEFICIARY_GROWTH_PCT",
        "QUALITY_CHANGE",
        "N_AB",
    ]

    for col in nonlinear_columns:

        out[f"{col}_SQ"] = (
            out[col] ** 2
        )

        out[f"{col}_ABS"] = (
            np.abs(out[col])
        )

    # --------------------------------------------------------
    # Cubic savings / quality effects
    # --------------------------------------------------------

    out["SAVINGS_CUBE"] = (
        out["PREVIOUS_SAVINGS_RATE"] ** 3
    )

    out["QUALITY_CUBE"] = (
        out["PREVIOUS_QUALITY_SCORE"] ** 3
    )

    out["PERFORMANCE_CUBE"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"] ** 3
    )

    # --------------------------------------------------------
    # Important interactions
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

    out["SAVINGS_QUALITY_CHANGE"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        * out["QUALITY_CHANGE"]
    )

    out["PERFORMANCE_QUALITY_CHANGE"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
        * out["QUALITY_CHANGE"]
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

    # --------------------------------------------------------
    # Difference features
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

    out["QUALITY_PERFORMANCE_GAP"] = (
        out["PREVIOUS_QUALITY_SCORE"]
        - out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    # --------------------------------------------------------
    # Composite indicators
    # --------------------------------------------------------

    out["GROWTH_COMPOSITE"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        + out["BENCHMARK_GROWTH_PCT"]
        + out["BENEFICIARY_GROWTH_PCT"]
    ) / 3.0

    out["GROWTH_ABSOLUTE"] = (
        np.abs(out["EXPENDITURE_GROWTH_PCT"])
        + np.abs(out["BENCHMARK_GROWTH_PCT"])
        + np.abs(out["BENEFICIARY_GROWTH_PCT"])
    ) / 3.0

    out["QUALITY_COMPOSITE"] = (
        out["PREVIOUS_QUALITY_SCORE"]
        + out["QUALITY_CHANGE"]
    ) / 2.0

    out["SAVINGS_MOMENTUM_CURRENT"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        + out["QUALITY_CHANGE"]
        - out["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    out["PERFORMANCE_QUALITY_RATIO"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
        /
        (
            np.abs(
                out["PREVIOUS_QUALITY_SCORE"]
            )
            + 1e-4
        )
    )

    out["SAVINGS_QUALITY_RATIO"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        /
        (
            np.abs(
                out["PREVIOUS_QUALITY_SCORE"]
            )
            + 1e-4
        )
    )

    out["EXPENDITURE_BENCHMARK_RATIO"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        /
        (
            np.abs(
                out["BENCHMARK_GROWTH_PCT"]
            )
            + 1e-4
        )
    )

    # --------------------------------------------------------
    # Sort chronologically
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
    # LAG FEATURES
    # --------------------------------------------------------

    lag_columns = {
        "SAVINGS": "PREVIOUS_SAVINGS_RATE",
        "QUALITY": "PREVIOUS_QUALITY_SCORE",
        "PERFORMANCE": "PREVIOUS_PERFORMANCE_GAP_PCT",
        "EXPENDITURE": "EXPENDITURE_GROWTH_PCT",
        "BENCHMARK": "BENCHMARK_GROWTH_PCT",
        "BENEFICIARY": "BENEFICIARY_GROWTH_PCT",
        "QUALITY_CHANGE": "QUALITY_CHANGE",
    }

    for name, source in lag_columns.items():

        out[f"{name}_LAG_1"] = (
            grouped[source]
            .shift(1)
        )

        out[f"{name}_LAG_2"] = (
            grouped[source]
            .shift(2)
        )

        out[f"{name}_LAG_3"] = (
            grouped[source]
            .shift(3)
        )

    # --------------------------------------------------------
    # Year-over-year changes
    # --------------------------------------------------------

    out["SAVINGS_CHANGE_1"] = (
        out["PREVIOUS_SAVINGS_RATE"]
        - out["SAVINGS_LAG_1"]
    )

    out["SAVINGS_CHANGE_2"] = (
        out["SAVINGS_LAG_1"]
        - out["SAVINGS_LAG_2"]
    )

    out["SAVINGS_ACCELERATION"] = (
        out["SAVINGS_CHANGE_1"]
        - out["SAVINGS_CHANGE_2"]
    )

    out["QUALITY_CHANGE_1"] = (
        out["PREVIOUS_QUALITY_SCORE"]
        - out["QUALITY_LAG_1"]
    )

    out["QUALITY_CHANGE_2"] = (
        out["QUALITY_LAG_1"]
        - out["QUALITY_LAG_2"]
    )

    out["PERFORMANCE_CHANGE_1"] = (
        out["PREVIOUS_PERFORMANCE_GAP_PCT"]
        - out["PERFORMANCE_LAG_1"]
    )

    out["EXPENDITURE_CHANGE_1"] = (
        out["EXPENDITURE_GROWTH_PCT"]
        - out["EXPENDITURE_LAG_1"]
    )

    out["BENCHMARK_CHANGE_1"] = (
        out["BENCHMARK_GROWTH_PCT"]
        - out["BENCHMARK_LAG_1"]
    )

    out["BENEFICIARY_CHANGE_1"] = (
        out["BENEFICIARY_GROWTH_PCT"]
        - out["BENEFICIARY_LAG_1"]
    )

    # --------------------------------------------------------
    # Rolling historical statistics
    # --------------------------------------------------------

    source_columns = {
        "SAVINGS": "PREVIOUS_SAVINGS_RATE",
        "QUALITY": "PREVIOUS_QUALITY_SCORE",
        "PERFORMANCE": "PREVIOUS_PERFORMANCE_GAP_PCT",
        "EXPENDITURE": "EXPENDITURE_GROWTH_PCT",
        "BENCHMARK": "BENCHMARK_GROWTH_PCT",
        "BENEFICIARY": "BENEFICIARY_GROWTH_PCT",
    }

    for name, source in source_columns.items():

        shifted = grouped[source].shift(1)

        out[f"{name}_ROLL_MEAN_2"] = (
            shifted
            .groupby(out["ACO_ID"])
            .transform(
                lambda x:
                x.rolling(
                    2,
                    min_periods=1,
                ).mean()
            )
        )

        out[f"{name}_ROLL_MEAN_3"] = (
            shifted
            .groupby(out["ACO_ID"])
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=1,
                ).mean()
            )
        )

        out[f"{name}_ROLL_STD_3"] = (
            shifted
            .groupby(out["ACO_ID"])
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=2,
                ).std()
            )
        )

        out[f"{name}_ROLL_MIN_3"] = (
            shifted
            .groupby(out["ACO_ID"])
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=1,
                ).min()
            )
        )

        out[f"{name}_ROLL_MAX_3"] = (
            shifted
            .groupby(out["ACO_ID"])
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=1,
                ).max()
            )
        )

    # --------------------------------------------------------
    # ACO age / historical depth
    # --------------------------------------------------------

    out["ACO_HISTORY_COUNT"] = (
        grouped.cumcount()
    )

    # --------------------------------------------------------
    # Calendar trend
    # --------------------------------------------------------

    out["YEAR_INDEX"] = (
        out["YEAR"] - out["YEAR"].min()
    )

    # --------------------------------------------------------
    # Clean infinities
    # --------------------------------------------------------

    numeric_columns = out.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_columns:

        out[col] = (
            out[col]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return out


# ============================================================
# ENGINEER HISTORICAL DATA
# ============================================================

print()
print("Engineering historical dataset...")

historical = engineer_features(df)

print(
    f"Historical engineered rows: "
    f"{len(historical):,}"
)


# ============================================================
# ENGINEER SCORING DATA CORRECTLY
# ============================================================

print()
print("Engineering 2024 scoring dataset...")

#
# IMPORTANT:
#
# The 2024 scoring file alone does not contain the historical
# rows needed to construct multi-year ACO lags.
#
# Therefore we concatenate historical + scoring data,
# engineer everything chronologically, and then extract
# 2024.
#

combined = pd.concat(
    [
        df.assign(_SCORING_ROW=False),
        scoring_df.assign(_SCORING_ROW=True),
    ],
    ignore_index=True,
    sort=False,
)

combined["YEAR"] = pd.to_numeric(
    combined["YEAR"],
    errors="coerce",
).astype(int)

combined_features = engineer_features(
    combined
)

all_scoring = combined_features[
    combined_features["_SCORING_ROW"] == True
].copy()

historical = combined_features[
    combined_features["_SCORING_ROW"] == False
].copy()

print(
    f"2024 engineered rows: "
    f"{len(all_scoring):,}"
)


# ============================================================
# DETERMINE FEATURES
# ============================================================

excluded_columns = set(
    IDENTIFIERS
    + [
        TARGET,
        "_SCORING_ROW",
    ]
)

FEATURES = [
    c
    for c in historical.columns
    if c not in excluded_columns
    and pd.api.types.is_numeric_dtype(
        historical[c]
    )
]


print()
print("=" * 78)
print("FINAL FEATURE SET")
print("=" * 78)

print(
    f"Total model features: {len(FEATURES)}"
)

for i, feature in enumerate(
    FEATURES,
    start=1,
):
    print(
        f"{i:3d}. {feature}"
    )


# ============================================================
# FILL MISSING ENGINEERED VALUES
# ============================================================

print()
print("=" * 78)
print("CLEANING ENGINEERED FEATURES")
print("=" * 78)

#
# We calculate medians using TRAINING YEARS ONLY.
# This avoids using validation-year distribution when
# constructing the model matrix.
#

training_mask = (
    historical["YEAR"] <= 2022
)

for feature in FEATURES:

    median_value = historical.loc[
        training_mask,
        feature,
    ].median()

    if pd.isna(median_value):
        median_value = 0.0

    historical[feature] = (
        historical[feature]
        .fillna(median_value)
        .astype(np.float32)
    )

    all_scoring[feature] = (
        all_scoring[feature]
        .fillna(median_value)
        .astype(np.float32)
    )


# ============================================================
# TIME SPLIT
# ============================================================

print()
print("=" * 78)
print("TIME-BASED VALIDATION")
print("=" * 78)

train_df = historical[
    historical["YEAR"] <= 2022
].copy()

valid_df = historical[
    historical["YEAR"] == 2023
].copy()

if len(train_df) == 0:
    raise ValueError(
        "No training rows found."
    )

if len(valid_df) == 0:
    raise ValueError(
        "No 2023 validation rows found."
    )


X_train = train_df[
    FEATURES
].astype(
    np.float32
)

y_train = train_df[
    TARGET
].astype(
    np.float32
)

X_valid = valid_df[
    FEATURES
].astype(
    np.float32
)

y_valid = valid_df[
    TARGET
].astype(
    np.float32
)


print(
    f"Training rows   : {len(X_train):,}"
)

print(
    f"Validation rows : {len(X_valid):,}"
)

print(
    f"Training years  : 2018–2022"
)

print(
    f"Validation year : 2023"
)


# ============================================================
# CATBOOST ONLY
# ============================================================

print()
print("=" * 78)
print("CATBOOST CONFIGURATION")
print("=" * 78)

CATBOOST_PARAMS = {

    # More iterations + low learning rate.
    "iterations": 5000,

    # Deeper trees capture nonlinear relationships.
    "depth": 9,

    "learning_rate": 0.025,

    # Regularization.
    "l2_leaf_reg": 7.0,

    # Randomness / Bayesian bootstrap.
    "random_strength": 0.8,

    "bagging_temperature": 0.5,

    # Objective.
    "loss_function": "RMSE",

    # Evaluation metric.
    "eval_metric": "R2",

    # Random seed.
    "random_seed": 42,

    # CPU only.
    "task_type": "CPU",

    # IMPORTANT:
    # 4 threads prevents the PC from becoming unusable.
    "thread_count": 4,

    # Early stopping.
    "od_type": "Iter",

    "od_wait": 250,

    # Do not create CatBoost temp files.
    "allow_writing_files": False,

    # Print every 250 iterations.
    "verbose": 250,
}


print(
    f"Iterations        : "
    f"{CATBOOST_PARAMS['iterations']}"
)

print(
    f"Depth             : "
    f"{CATBOOST_PARAMS['depth']}"
)

print(
    f"Learning rate     : "
    f"{CATBOOST_PARAMS['learning_rate']}"
)

print(
    f"L2 regularization : "
    f"{CATBOOST_PARAMS['l2_leaf_reg']}"
)

print(
    f"Threads           : "
    f"{CATBOOST_PARAMS['thread_count']}"
)

print()
print("MODEL: CatBoostRegressor ONLY")


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 78)
print("TRAINING CATBOOST")
print("=" * 78)

print()
print("Training started.")
print("There will be NO Random Forest.")
print("There will be NO Extra Trees.")
print("There will be NO XGBoost.")
print("There will be NO hyperparameter search.")
print()

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
print("2023 VALIDATION RESULTS")
print("=" * 78)

pred_valid = model.predict(
    X_valid
)

mae = mean_absolute_error(
    y_valid,
    pred_valid,
)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        pred_valid,
    )
)

r2 = r2_score(
    y_valid,
    pred_valid,
)


print()
print(
    f"MAE       : {mae:.6f}"
)

print(
    f"RMSE      : {rmse:.6f}"
)

print(
    f"R²        : {r2:.6f}"
)

print(
    f"R² %      : {r2 * 100:.2f}%"
)

print()

if r2 >= 0.85:

    print(
        "TARGET STATUS: R² >= 85% ACHIEVED"
    )

elif r2 >= 0.80:

    print(
        "TARGET STATUS: STRONG — 80%+ R²"
    )

else:

    print(
        "TARGET STATUS: Below 85%"
    )

    print(
        "The dataset may not contain enough "
        "predictive information to honestly reach 85%."
    )


# ============================================================
# VALIDATION OUTPUT
# ============================================================

validation_output = valid_df[
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
] = pred_valid

validation_output[
    "ABSOLUTE_ERROR"
] = np.abs(
    validation_output[TARGET]
    -
    validation_output[
        "PREDICTED_NEXT_YEAR_SAVINGS_RATE"
    ]
)

validation_output[
    "ERROR_PERCENTAGE"
] = (
    validation_output["ABSOLUTE_ERROR"]
    * 100
)

validation_output.to_csv(
    VALIDATION_PATH,
    index=False,
)

print()
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

importance = model.get_feature_importance()

importance_df = pd.DataFrame(
    {
        "FEATURE": FEATURES,
        "IMPORTANCE": importance,
    }
)

importance_df = (
    importance_df
    .sort_values(
        "IMPORTANCE",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

print()

print(
    importance_df.head(30).to_string(
        index=False
    )
)

importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False,
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

production_iterations = max(
    500,
    best_iteration + 1,
)

print()
print(
    f"Best validation iteration : "
    f"{best_iteration + 1}"
)

print(
    f"Production iterations     : "
    f"{production_iterations}"
)


# ============================================================
# FINAL PRODUCTION TRAINING
# ============================================================

print()
print("=" * 78)
print("TRAINING FINAL CATBOOST PRODUCTION MODEL")
print("=" * 78)

#
# Production model uses all available historical data:
#
# 2018–2023
#
# The validation-derived iteration count is retained.
#

X_full = historical[
    FEATURES
].astype(
    np.float32
)

y_full = historical[
    TARGET
].astype(
    np.float32
)

production_params = CATBOOST_PARAMS.copy()

production_params[
    "iterations"
] = production_iterations

production_params[
    "verbose"
] = 500

production_model = CatBoostRegressor(
    **production_params
)

production_model.fit(
    X_full,
    y_full,
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 78)
print("SAVING CATBOOST MODEL")
print("=" * 78)

production_model.save_model(
    str(MODEL_PATH)
)

print(
    f"Model saved:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# 2024 FORECAST
# ============================================================

print()
print("=" * 78)
print("CREATING 2024 FORECAST")
print("=" * 78)

X_2024 = all_scoring[
    FEATURES
].astype(
    np.float32
)

forecast_predictions = (
    production_model.predict(
        X_2024
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
# FORECAST CATEGORY
# ============================================================

def classify_forecast(value):

    if value >= 0.05:
        return "HIGH SAVINGS"

    if value >= 0.00:
        return "MODERATE SAVINGS"

    return "LOW SAVINGS"


forecast_output[
    "SAVINGS_FORECAST_CATEGORY"
] = [
    classify_forecast(
        x
    )
    for x in forecast_predictions
]


# ============================================================
# SAVE FORECAST
# ============================================================

forecast_output.to_csv(
    FORECAST_PATH,
    index=False,
)

print()
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
    f"Rows       : "
    f"{len(forecast_output):,}"
)

print(
    f"Mean       : "
    f"{forecast_predictions.mean():.6f}"
)

print(
    f"Median     : "
    f"{np.median(forecast_predictions):.6f}"
)

print(
    f"Minimum    : "
    f"{forecast_predictions.min():.6f}"
)

print(
    f"Maximum    : "
    f"{forecast_predictions.max():.6f}"
)

print()
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

top10 = (
    forecast_output
    .sort_values(
        "FORECASTED_NEXT_YEAR_SAVINGS_RATE",
        ascending=False,
    )
    .head(10)
)

print(
    top10[
        [
            "ACO_ID",
            "ACO_NAME",
            "STATE",
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

bottom10 = (
    forecast_output
    .sort_values(
        "FORECASTED_NEXT_YEAR_SAVINGS_RATE",
        ascending=True,
    )
    .head(10)
)

print(
    bottom10[
        [
            "ACO_ID",
            "ACO_NAME",
            "STATE",
            "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
            "SAVINGS_FORECAST_CATEGORY",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# METRICS
# ============================================================

metrics = {

    "model": "CatBoostRegressor",

    "models_trained": [
        "CatBoostRegressor"
    ],

    "random_forest_used": False,

    "extra_trees_used": False,

    "xgboost_used": False,

    "hist_gradient_boosting_used": False,

    "hyperparameter_search": False,

    "target": TARGET,

    "validation_year": 2023,

    "training_years": "2018-2022",

    "production_years": "2018-2023",

    "forecast_year": 2024,

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

    "training_rows": int(
        len(X_train)
    ),

    "validation_rows": int(
        len(X_valid)
    ),

    "production_rows": int(
        len(X_full)
    ),

    "forecast_rows": int(
        len(X_2024)
    ),

    "feature_count": int(
        len(FEATURES)
    ),
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ============================================================
# METADATA
# ============================================================

metadata = {

    "model": "CatBoostRegressor",

    "models_trained": [
        "CatBoostRegressor"
    ],

    "target": TARGET,

    "base_features": BASE_FEATURES,

    "engineered_features": FEATURES,

    "validation_strategy": (
        "2018-2022 training, "
        "2023 time-based validation"
    ),

    "production_training": (
        "2018-2023"
    ),

    "forecast_year": 2024,

    "parameters": {
        k: v
        for k, v in production_params.items()
    },

    "validation_metrics": {

        "MAE": float(mae),

        "RMSE": float(rmse),

        "R2": float(r2),

        "R2_PERCENTAGE": float(
            r2 * 100
        ),
    },

    "feature_count": len(FEATURES),

    "best_iteration": int(
        best_iteration + 1
    ),

    "leakage_protection": True,

    "future_target_used_as_feature": False,

    "random_forest_used": False,

    "extra_trees_used": False,

    "xgboost_used": False,

    "hyperparameter_search": False,
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        default=str,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("FINAL CATBOOST-ONLY SUMMARY")
print("=" * 78)

print()
print(
    "MODEL                : CatBoostRegressor"
)

print(
    "RANDOM FOREST        : NOT USED"
)

print(
    "EXTRA TREES          : NOT USED"
)

print(
    "XGBOOST              : NOT USED"
)

print(
    "HIST GRADIENT BOOST  : NOT USED"
)

print(
    "GRID SEARCH          : NOT USED"
)

print(
    "RANDOM SEARCH        : NOT USED"
)

print()
print(
    f"Training rows        : {len(X_train):,}"
)

print(
    f"Validation rows      : {len(X_valid):,}"
)

print(
    f"Production rows      : {len(X_full):,}"
)

print(
    f"2024 forecast rows   : {len(X_2024):,}"
)

print()
print(
    f"2023 MAE             : {mae:.6f}"
)

print(
    f"2023 RMSE            : {rmse:.6f}"
)

print(
    f"2023 R²              : {r2:.6f}"
)

print(
    f"2023 R² percentage   : {r2 * 100:.2f}%"
)

print()
print(
    f"Best iteration       : "
    f"{best_iteration + 1}"
)

print()
print("MODEL:")
print(MODEL_PATH)

print()
print("2024 FORECAST:")
print(FORECAST_PATH)

print()
print("VALIDATION:")
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

print()
print("STATUS: SUCCESS")

print()
print("ONLY CATBOOST WAS TRAINED.")

print("=" * 78)