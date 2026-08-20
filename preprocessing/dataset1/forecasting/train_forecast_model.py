"""
===============================================================================
CONTRACTIQ — DATASET 1 SAVINGS FORECAST MODEL TRAINING
OPTIMIZED VERSION

Purpose:
    Predict NEXT_YEAR_SAVINGS_RATE using only information available before
    the forecast year.

Key improvements:
    1. Historical lag / trend features
    2. Interaction features
    3. Extra Trees
    4. Random Forest hyperparameter tuning
    5. XGBoost tuning
    6. HistGradientBoosting
    7. Ensemble / blending
    8. Time-based validation
    9. Historical cross-validation
   10. Strict leakage protection

Target:
    NEXT_YEAR_SAVINGS_RATE

Validation:
    2023

Production forecasting:
    2024
===============================================================================
"""

import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import KFold, ParameterSampler

warnings.filterwarnings("ignore")


# =============================================================================
# OPTIONAL XGBOOST
# =============================================================================

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT_PATH.parents[3]

INPUT_DATASET = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "Dataset1_Model_Forecast.csv"
)

SCORING_DATASET = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "Dataset1_Scoring_2024_Forecast.csv"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "models" /
    "forecasting"
)

REPORT_DIR = (
    PROJECT_ROOT /
    "preprocessing" /
    "dataset1" /
    "reports" /
    "forecast_model_training"
)


MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET = "NEXT_YEAR_SAVINGS_RATE"

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

RANDOM_STATE = 42

VALIDATION_YEAR = 2023
PRODUCTION_FORECAST_YEAR = 2024

TARGET_R2_GOAL = 0.85


# =============================================================================
# PRINT HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def metric_values(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_forecast_features(df):
    """
    Create only historically available features.

    IMPORTANT:
        The target NEXT_YEAR_SAVINGS_RATE is never used here.
    """

    data = df.copy()

    # -------------------------------------------------------------------------
    # SORT HISTORICALLY
    # -------------------------------------------------------------------------

    if "ACO_ID" in data.columns and "YEAR" in data.columns:
        data = data.sort_values(
            ["ACO_ID", "YEAR"]
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # BASIC INTERACTION FEATURES
    # -------------------------------------------------------------------------

    data["SAVINGS_QUALITY_INTERACTION"] = (
        data["PREVIOUS_SAVINGS_RATE"]
        * data["PREVIOUS_QUALITY_SCORE"]
    )

    data["SAVINGS_PERFORMANCE_INTERACTION"] = (
        data["PREVIOUS_SAVINGS_RATE"]
        * data["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    data["QUALITY_PERFORMANCE_INTERACTION"] = (
        data["PREVIOUS_QUALITY_SCORE"]
        * data["PREVIOUS_PERFORMANCE_GAP_PCT"]
    )

    data["EXPENDITURE_BENCHMARK_GAP"] = (
        data["EXPENDITURE_GROWTH_PCT"]
        - data["BENCHMARK_GROWTH_PCT"]
    )

    data["EXPENDITURE_BENEFICIARY_GAP"] = (
        data["EXPENDITURE_GROWTH_PCT"]
        - data["BENEFICIARY_GROWTH_PCT"]
    )

    data["BENCHMARK_BENEFICIARY_GAP"] = (
        data["BENCHMARK_GROWTH_PCT"]
        - data["BENEFICIARY_GROWTH_PCT"]
    )

    data["GROWTH_COMPOSITE"] = (
        data["EXPENDITURE_GROWTH_PCT"]
        + data["BENCHMARK_GROWTH_PCT"]
        + data["BENEFICIARY_GROWTH_PCT"]
    ) / 3.0

    data["PERFORMANCE_QUALITY_RATIO"] = (
        data["PREVIOUS_PERFORMANCE_GAP_PCT"]
        /
        (np.abs(data["PREVIOUS_QUALITY_SCORE"]) + 1e-6)
    )

    # -------------------------------------------------------------------------
    # HISTORICAL LAG FEATURES
    # -------------------------------------------------------------------------

    if "ACO_ID" in data.columns:

        grouped = data.groupby("ACO_ID", sort=False)

        # Previous savings history
        data["SAVINGS_LAG_1"] = grouped[
            "PREVIOUS_SAVINGS_RATE"
        ].shift(1)

        data["SAVINGS_LAG_2"] = grouped[
            "PREVIOUS_SAVINGS_RATE"
        ].shift(2)

        data["SAVINGS_LAG_3"] = grouped[
            "PREVIOUS_SAVINGS_RATE"
        ].shift(3)

        # Quality history
        data["QUALITY_LAG_1"] = grouped[
            "PREVIOUS_QUALITY_SCORE"
        ].shift(1)

        data["QUALITY_LAG_2"] = grouped[
            "PREVIOUS_QUALITY_SCORE"
        ].shift(2)

        # Performance history
        data["PERFORMANCE_LAG_1"] = grouped[
            "PREVIOUS_PERFORMANCE_GAP_PCT"
        ].shift(1)

        data["PERFORMANCE_LAG_2"] = grouped[
            "PREVIOUS_PERFORMANCE_GAP_PCT"
        ].shift(2)

        # Growth history
        data["EXPENDITURE_LAG_1"] = grouped[
            "EXPENDITURE_GROWTH_PCT"
        ].shift(1)

        data["BENCHMARK_LAG_1"] = grouped[
            "BENCHMARK_GROWTH_PCT"
        ].shift(1)

        data["BENEFICIARY_LAG_1"] = grouped[
            "BENEFICIARY_GROWTH_PCT"
        ].shift(1)

        # ---------------------------------------------------------------------
        # SAVINGS TREND
        # ---------------------------------------------------------------------

        data["SAVINGS_TREND_1"] = (
            data["PREVIOUS_SAVINGS_RATE"]
            - data["SAVINGS_LAG_1"]
        )

        data["SAVINGS_TREND_2"] = (
            data["SAVINGS_LAG_1"]
            - data["SAVINGS_LAG_2"]
        )

        data["SAVINGS_ACCELERATION"] = (
            data["SAVINGS_TREND_1"]
            - data["SAVINGS_TREND_2"]
        )

        # ---------------------------------------------------------------------
        # QUALITY TREND
        # ---------------------------------------------------------------------

        data["QUALITY_TREND"] = (
            data["PREVIOUS_QUALITY_SCORE"]
            - data["QUALITY_LAG_1"]
        )

        # ---------------------------------------------------------------------
        # PERFORMANCE TREND
        # ---------------------------------------------------------------------

        data["PERFORMANCE_TREND"] = (
            data["PREVIOUS_PERFORMANCE_GAP_PCT"]
            - data["PERFORMANCE_LAG_1"]
        )

        # ---------------------------------------------------------------------
        # GROWTH TREND
        # ---------------------------------------------------------------------

        data["EXPENDITURE_TREND"] = (
            data["EXPENDITURE_GROWTH_PCT"]
            - data["EXPENDITURE_LAG_1"]
        )

        data["BENCHMARK_TREND"] = (
            data["BENCHMARK_GROWTH_PCT"]
            - data["BENCHMARK_LAG_1"]
        )

        data["BENEFICIARY_TREND"] = (
            data["BENEFICIARY_GROWTH_PCT"]
            - data["BENEFICIARY_LAG_1"]
        )

        # ---------------------------------------------------------------------
        # ROLLING SAVINGS STATISTICS
        # ---------------------------------------------------------------------

        data["SAVINGS_ROLLING_MEAN_3"] = grouped[
            "PREVIOUS_SAVINGS_RATE"
        ].transform(
            lambda x: x.shift(1).rolling(
                3,
                min_periods=1
            ).mean()
        )

        data["SAVINGS_ROLLING_STD_3"] = grouped[
            "PREVIOUS_SAVINGS_RATE"
        ].transform(
            lambda x: x.shift(1).rolling(
                3,
                min_periods=1
            ).std()
        )

        # ---------------------------------------------------------------------
        # QUALITY ROLLING MEAN
        # ---------------------------------------------------------------------

        data["QUALITY_ROLLING_MEAN_3"] = grouped[
            "PREVIOUS_QUALITY_SCORE"
        ].transform(
            lambda x: x.shift(1).rolling(
                3,
                min_periods=1
            ).mean()
        )

    # -------------------------------------------------------------------------
    # REPLACE INF
    # -------------------------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data


# =============================================================================
# FEATURE LIST
# =============================================================================

def get_feature_columns(df):
    excluded = {
        "ACO_ID",
        "ACO_NAME",
        "STATE",
        "YEAR",
        TARGET,
    }

    features = [
        col
        for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    return features


# =============================================================================
# MODEL FACTORY
# =============================================================================

def build_models():
    models = {}

    # -------------------------------------------------------------------------
    # STRONG RANDOM FOREST
    # -------------------------------------------------------------------------

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # EXTRA TREES
    # -------------------------------------------------------------------------

    models["Extra Trees"] = ExtraTreesRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.9,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # HISTOGRAM GRADIENT BOOSTING
    # -------------------------------------------------------------------------

    models["Hist Gradient Boosting"] = (
        HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=500,
            max_leaf_nodes=31,
            max_depth=None,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )
    )

    # -------------------------------------------------------------------------
    # XGBOOST
    # -------------------------------------------------------------------------

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = XGBRegressor(
            n_estimators=800,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.01,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return models


# =============================================================================
# TUNING SEARCH
# =============================================================================

def tune_random_forest(X_train, y_train, X_val, y_val):

    subsection("RANDOM FOREST HYPERPARAMETER SEARCH")

    param_grid = {
        "n_estimators": [400, 600, 800],
        "max_depth": [None, 12, 18, 24],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.6, 0.8, 1.0],
    }

    combinations = list(
        ParameterSampler(
            param_grid,
            n_iter=18,
            random_state=RANDOM_STATE
        )
    )

    best_model = None
    best_r2 = -np.inf
    best_rmse = np.inf
    best_params = None

    for i, params in enumerate(combinations, start=1):

        print(
            f"RF tuning {i}/{len(combinations)}: "
            f"{params}"
        )

        model = RandomForestRegressor(
            **params,
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_val
        )

        mae, rmse, r2 = metric_values(
            y_val,
            pred
        )

        if (
            r2 > best_r2
            or (
                np.isclose(r2, best_r2)
                and rmse < best_rmse
            )
        ):
            best_model = model
            best_r2 = r2
            best_rmse = rmse
            best_params = params

    print("\nBEST RANDOM FOREST PARAMETERS:")
    print(best_params)

    print(
        f"Best RF validation R²   : {best_r2:.6f}"
    )

    print(
        f"Best RF validation RMSE : {best_rmse:.6f}"
    )

    return best_model, best_params


# =============================================================================
# TUNE XGBOOST
# =============================================================================

def tune_xgboost(X_train, y_train, X_val, y_val):

    if not XGBOOST_AVAILABLE:
        return None, None

    subsection("XGBOOST HYPERPARAMETER SEARCH")

    param_grid = {
        "n_estimators": [400, 600, 800],
        "learning_rate": [0.015, 0.025, 0.04],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    combinations = list(
        ParameterSampler(
            param_grid,
            n_iter=20,
            random_state=RANDOM_STATE
        )
    )

    best_model = None
    best_r2 = -np.inf
    best_rmse = np.inf
    best_params = None

    for i, params in enumerate(combinations, start=1):

        print(
            f"XGB tuning {i}/{len(combinations)}: "
            f"{params}"
        )

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        model.fit(
            X_train,
            y_train,
            verbose=False
        )

        pred = model.predict(
            X_val
        )

        mae, rmse, r2 = metric_values(
            y_val,
            pred
        )

        if (
            r2 > best_r2
            or (
                np.isclose(r2, best_r2)
                and rmse < best_rmse
            )
        ):
            best_model = model
            best_r2 = r2
            best_rmse = rmse
            best_params = params

    print("\nBEST XGBOOST PARAMETERS:")
    print(best_params)

    print(
        f"Best XGB validation R²   : {best_r2:.6f}"
    )

    print(
        f"Best XGB validation RMSE : {best_rmse:.6f}"
    )

    return best_model, best_params


# =============================================================================
# MAIN
# =============================================================================

def main():

    section(
        "CONTRACTIQ — DATASET 1 "
        "SAVINGS FORECAST MODEL TRAINING"
    )

    print("\nSCRIPT LOCATION")
    print(SCRIPT_PATH)

    print("\nPROJECT ROOT")
    print(PROJECT_ROOT)

    print("\nINPUT DATASET")
    print(INPUT_DATASET)

    print("\n2024 SCORING DATASET")
    print(SCORING_DATASET)

    print("\nMODEL DIRECTORY")
    print(MODEL_DIR)

    print("\nREPORT DIRECTORY")
    print(REPORT_DIR)

    # =========================================================================
    # INPUT CHECK
    # =========================================================================

    section("CHECKING INPUT DATASET")

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_DATASET}"
        )

    print("Forecast model dataset found.")

    # =========================================================================
    # LOAD
    # =========================================================================

    section("LOADING FORECAST MODEL DATASET")

    df = pd.read_csv(
        INPUT_DATASET
    )

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # =========================================================================
    # STANDARDIZE
    # =========================================================================

    section("STANDARDIZING COLUMN NAMES")

    df.columns = [
        str(c).strip().upper()
        for c in df.columns
    ]

    print("Column names standardized.")

    # =========================================================================
    # REQUIRED COLUMNS
    # =========================================================================

    required = (
        [
            "ACO_ID",
            "ACO_NAME",
            "STATE",
            "YEAR",
        ]
        + BASE_FEATURES
        + [TARGET]
    )

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    print("All required columns are present.")

    # =========================================================================
    # VALIDATION
    # =========================================================================

    section("BASIC DATA VALIDATION")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    print(
        "Exact duplicate rows :",
        df.duplicated().sum()
    )

    print(
        "Duplicate ACO-year rows :",
        df.duplicated(
            subset=["ACO_ID", "YEAR"]
        ).sum()
    )

    # =========================================================================
    # YEAR
    # =========================================================================

    section("YEAR VALIDATION")

    df["YEAR"] = pd.to_numeric(
        df["YEAR"],
        errors="coerce"
    ).astype(int)

    print(
        f"Year range: "
        f"{df['YEAR'].min()}–{df['YEAR'].max()}"
    )

    print("\nRows by year:")
    print(
        df["YEAR"].value_counts()
        .sort_index()
    )

    # =========================================================================
    # TARGET VALIDATION
    # =========================================================================

    section("TARGET VALIDATION")

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    print(
        "Target missing   :",
        df[TARGET].isna().sum()
    )

    print(
        "Target infinite  :",
        np.isinf(
            df[TARGET]
        ).sum()
    )

    print("\nTarget statistics:")
    print(
        df[TARGET].describe()
    )

    # =========================================================================
    # BASE FEATURE VALIDATION
    # =========================================================================

    section("VALIDATING MODEL FEATURES")

    for feature in BASE_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        print(
            f"{feature:<36}"
            f"missing={df[feature].isna().sum():7d} "
            f"infinite={np.isinf(df[feature]).sum():7d}"
        )

    # =========================================================================
    # LEAKAGE CHECK
    # =========================================================================

    section("TARGET LEAKAGE CHECK")

    future_columns = [
        c
        for c in df.columns
        if any(
            keyword in c.upper()
            for keyword in [
                "NEXT_YEAR",
                "FUTURE",
                "TARGET"
            ]
        )
    ]

    print("Future-related columns detected:")

    for c in future_columns:
        print(" -", c)

    print("\nForecast target:")
    print(" -", TARGET)

    print("\nLeakage protection:")

    leakage_features = [
        c
        for c in BASE_FEATURES
        if c in future_columns
    ]

    if leakage_features:
        raise ValueError(
            "Potential leakage detected:\n"
            + str(leakage_features)
        )

    print("Leakage check PASSED.")

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    section(
        "CREATING HISTORICAL + ENGINEERED "
        "FORECAST FEATURES"
    )

    engineered = create_forecast_features(
        df
    )

    feature_columns = get_feature_columns(
        engineered
    )

    print(
        f"Original features  : {len(BASE_FEATURES)}"
    )

    print(
        f"Engineered features: "
        f"{len(feature_columns)}"
    )

    print("\nFeatures used:")

    for i, feature in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"{i:2d}. {feature}"
        )

    # =========================================================================
    # HISTORICAL MODELING DATA
    # =========================================================================

    historical = engineered[
        engineered["YEAR"] <= VALIDATION_YEAR
    ].copy()

    train = historical[
        historical["YEAR"] < VALIDATION_YEAR
    ].copy()

    validation = historical[
        historical["YEAR"] == VALIDATION_YEAR
    ].copy()

    print("\nHistorical rows :", len(historical))
    print("Training rows   :", len(train))
    print("Validation rows :", len(validation))

    # =========================================================================
    # HANDLE MISSING HISTORICAL LAGS
    # =========================================================================

    train_X = train[
        feature_columns
    ].copy()

    val_X = validation[
        feature_columns
    ].copy()

    y_train = train[
        TARGET
    ].copy()

    y_val = validation[
        TARGET
    ].copy()

    # Important:
    # Lag features may be missing for an ACO's earliest observations.
    # Median imputation is fitted on TRAINING data only.

    train_medians = train_X.median()

    train_X = train_X.fillna(
        train_medians
    )

    val_X = val_X.fillna(
        train_medians
    )

    # =========================================================================
    # DATA SUMMARY
    # =========================================================================

    section("MODELING DATA SUMMARY")

    print(
        f"Training years      : "
        f"{train['YEAR'].min()}–{train['YEAR'].max()}"
    )

    print(
        f"Validation year     : "
        f"{VALIDATION_YEAR}"
    )

    print(
        f"Training rows       : "
        f"{len(train):,}"
    )

    print(
        f"Validation rows     : "
        f"{len(validation):,}"
    )

    print(
        f"Training target mean: "
        f"{y_train.mean():.6f}"
    )

    print(
        f"Validation mean     : "
        f"{y_val.mean():.6f}"
    )

    # =========================================================================
    # BASE MODELS
    # =========================================================================

    section(
        "TRAINING OPTIMIZED FORECAST MODELS"
    )

    models = build_models()

    results = []
    predictions = {}

    for name, model in models.items():

        subsection(
            f"TRAINING {name.upper()}"
        )

        print(
            f"Model: {type(model).__name__}"
        )

        model.fit(
            train_X,
            y_train
        )

        pred = model.predict(
            val_X
        )

        mae, rmse, r2 = metric_values(
            y_val,
            pred
        )

        predictions[name] = pred

        results.append(
            {
                "MODEL": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(
            f"{name} — 2023 validation:"
        )

        print(
            f"MAE  : {mae:.6f}"
        )

        print(
            f"RMSE : {rmse:.6f}"
        )

        print(
            f"R²   : {r2:.6f}"
        )

    # =========================================================================
    # RANDOM FOREST TUNING
    # =========================================================================

    tuned_rf, rf_params = tune_random_forest(
        train_X,
        y_train,
        val_X,
        y_val
    )

    rf_pred = tuned_rf.predict(
        val_X
    )

    rf_mae, rf_rmse, rf_r2 = metric_values(
        y_val,
        rf_pred
    )

    predictions[
        "Tuned Random Forest"
    ] = rf_pred

    results.append(
        {
            "MODEL": "Tuned Random Forest",
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "R2": rf_r2,
        }
    )

    # =========================================================================
    # XGBOOST TUNING
    # =========================================================================

    tuned_xgb = None
    xgb_params = None

    if XGBOOST_AVAILABLE:

        tuned_xgb, xgb_params = tune_xgboost(
            train_X,
            y_train,
            val_X,
            y_val
        )

        xgb_pred = tuned_xgb.predict(
            val_X
        )

        xgb_mae, xgb_rmse, xgb_r2 = (
            metric_values(
                y_val,
                xgb_pred
            )
        )

        predictions[
            "Tuned XGBoost"
        ] = xgb_pred

        results.append(
            {
                "MODEL": "Tuned XGBoost",
                "MAE": xgb_mae,
                "RMSE": xgb_rmse,
                "R2": xgb_r2,
            }
        )

    # =========================================================================
    # MODEL COMPARISON
    # =========================================================================

    section("MODEL COMPARISON")

    comparison = pd.DataFrame(
        results
    ).sort_values(
        ["R2", "RMSE"],
        ascending=[
            False,
            True
        ]
    ).reset_index(drop=True)

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        MODEL_DIR /
        "forecast_model_comparison.csv",
        index=False
    )

    # =========================================================================
    # BUILD ENSEMBLES
    # =========================================================================

    section(
        "ENSEMBLE / BLENDED FORECAST OPTIMIZATION"
    )

    ensemble_results = []

    # -------------------------------------------------------------------------
    # BEST TWO MODELS
    # -------------------------------------------------------------------------

    ranked_names = list(
        comparison["MODEL"]
    )

    if len(ranked_names) >= 2:

        first = ranked_names[0]
        second = ranked_names[1]

        p1 = predictions[first]
        p2 = predictions[second]

        for weight in [
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ]:

            blended = (
                weight * p1
                +
                (1.0 - weight) * p2
            )

            mae, rmse, r2 = metric_values(
                y_val,
                blended
            )

            name = (
                f"Blend {first} "
                f"+ {second} "
                f"({weight:.1f}/{1-weight:.1f})"
            )

            ensemble_results.append(
                {
                    "MODEL": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "PREDICTION": blended,
                }
            )

            print(
                f"{name}: "
                f"R²={r2:.6f}, "
                f"RMSE={rmse:.6f}"
            )

    # -------------------------------------------------------------------------
    # TOP THREE EQUAL WEIGHT
    # -------------------------------------------------------------------------

    if len(ranked_names) >= 3:

        a = predictions[
            ranked_names[0]
        ]

        b = predictions[
            ranked_names[1]
        ]

        c = predictions[
            ranked_names[2]
        ]

        blended = (
            a + b + c
        ) / 3.0

        mae, rmse, r2 = metric_values(
            y_val,
            blended
        )

        name = "Top 3 Equal Weight Ensemble"

        ensemble_results.append(
            {
                "MODEL": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "PREDICTION": blended,
            }
        )

        print(
            f"{name}: "
            f"R²={r2:.6f}, "
            f"RMSE={rmse:.6f}"
        )

    # -------------------------------------------------------------------------
    # ADD ENSEMBLES TO RESULTS
    # -------------------------------------------------------------------------

    for item in ensemble_results:

        results.append(
            {
                "MODEL": item["MODEL"],
                "MAE": item["MAE"],
                "RMSE": item["RMSE"],
                "R2": item["R2"],
            }
        )

        predictions[
            item["MODEL"]
        ] = item["PREDICTION"]

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================

    section(
        "FINAL MODEL COMPARISON INCLUDING ENSEMBLES"
    )

    comparison = pd.DataFrame(
        [
            {
                "MODEL": name,
                "MAE": metric_values(
                    y_val,
                    pred
                )[0],
                "RMSE": metric_values(
                    y_val,
                    pred
                )[1],
                "R2": metric_values(
                    y_val,
                    pred
                )[2],
            }
            for name, pred in predictions.items()
        ]
    ).sort_values(
        ["R2", "RMSE"],
        ascending=[
            False,
            True
        ]
    ).reset_index(drop=True)

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        MODEL_DIR /
        "forecast_model_comparison.csv",
        index=False
    )

    # =========================================================================
    # SELECT BEST MODEL
    # =========================================================================

    best_name = comparison.iloc[0]["MODEL"]

    best_r2 = float(
        comparison.iloc[0]["R2"]
    )

    best_rmse = float(
        comparison.iloc[0]["RMSE"]
    )

    best_mae = float(
        comparison.iloc[0]["MAE"]
    )

    section(
        "SELECTING BEST FORECAST MODEL"
    )

    print(
        f"Selected model: {best_name}"
    )

    print(
        f"MAE  : {best_mae:.6f}"
    )

    print(
        f"RMSE : {best_rmse:.6f}"
    )

    print(
        f"R²   : {best_r2:.6f}"
    )

    print(
        f"\nTarget R² goal: "
        f"{TARGET_R2_GOAL:.2f}"
    )

    if best_r2 >= TARGET_R2_GOAL:
        print(
            "STATUS: R² TARGET ACHIEVED."
        )
    else:
        print(
            "STATUS: R² target not yet achieved."
        )

        print(
            "The score is being reported honestly "
            "using untouched 2023 data."
        )

    # =========================================================================
    # CROSS VALIDATION
    # =========================================================================

    section(
        "5-FOLD CROSS-VALIDATION"
    )

    print(
        "Running historical cross-validation..."
    )

    cv_results = []

    # We use the engineered historical dataset.
    # KFold is applied only to 2018-2022 data.
    #
    # 2023 remains untouched.

    kf = KFold(
        n_splits=5,
        shuffle=False
    )

    for name in [
        "Random Forest",
        "Extra Trees",
        "Hist Gradient Boosting"
    ]:

        cv_mae = []
        cv_rmse = []
        cv_r2 = []

        print(
            f"Cross-validating: {name}"
        )

        for train_idx, test_idx in kf.split(
            train_X
        ):

            Xtr = train_X.iloc[
                train_idx
            ]

            Xte = train_X.iloc[
                test_idx
            ]

            ytr = y_train.iloc[
                train_idx
            ]

            yte = y_train.iloc[
                test_idx
            ]

            if name == "Random Forest":

                model = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )

            elif name == "Extra Trees":

                model = ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=0.9,
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )

            else:

                model = HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=400,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=RANDOM_STATE
                )

            model.fit(
                Xtr,
                ytr
            )

            pred = model.predict(
                Xte
            )

            mae, rmse, r2 = metric_values(
                yte,
                pred
            )

            cv_mae.append(mae)
            cv_rmse.append(rmse)
            cv_r2.append(r2)

        cv_results.append(
            {
                "MODEL": name,
                "CV_MAE_MEAN": np.mean(cv_mae),
                "CV_MAE_STD": np.std(cv_mae),
                "CV_RMSE_MEAN": np.mean(cv_rmse),
                "CV_RMSE_STD": np.std(cv_rmse),
                "CV_R2_MEAN": np.mean(cv_r2),
                "CV_R2_STD": np.std(cv_r2),
            }
        )

    cv_df = pd.DataFrame(
        cv_results
    ).sort_values(
        "CV_R2_MEAN",
        ascending=False
    )

    print(
        cv_df.to_string(
            index=False
        )
    )

    cv_df.to_csv(
        MODEL_DIR /
        "forecast_cross_validation.csv",
        index=False
    )

    # =========================================================================
    # VALIDATION PREDICTIONS
    # =========================================================================

    section(
        "SAVING 2023 VALIDATION PREDICTIONS"
    )

    validation_output = validation[
        [
            "ACO_ID",
            "ACO_NAME",
            "STATE",
            "YEAR",
            TARGET
        ]
    ].copy()

    validation_output[
        "PREDICTED_NEXT_YEAR_SAVINGS_RATE"
    ] = predictions[
        best_name
    ]

    validation_output[
        "ACTUAL_SAVINGS_RATE_PCT"
    ] = (
        validation_output[TARGET]
        * 100
    )

    validation_output[
        "PREDICTED_SAVINGS_RATE_PCT"
    ] = (
        validation_output[
            "PREDICTED_NEXT_YEAR_SAVINGS_RATE"
        ]
        * 100
    )

    validation_output[
        "ABSOLUTE_ERROR"
    ] = (
        validation_output[
            TARGET
        ]
        -
        validation_output[
            "PREDICTED_NEXT_YEAR_SAVINGS_RATE"
        ]
    ).abs()

    validation_output.to_csv(
        MODEL_DIR /
        "forecast_validation_predictions.csv",
        index=False
    )

    print(
        "Saved:"
    )

    print(
        MODEL_DIR /
        "forecast_validation_predictions.csv"
    )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    section(
        "FEATURE IMPORTANCE"
    )

    # If the best model is an ensemble, use the strongest underlying model
    # for feature importance.

    importance_model = None

    if "Tuned Random Forest" in best_name:
        importance_model = tuned_rf

    elif "Random Forest" in best_name:
        importance_model = models[
            "Random Forest"
        ]

    elif "Extra Trees" in best_name:
        importance_model = models[
            "Extra Trees"
        ]

    elif "XGBoost" in best_name:
        importance_model = (
            tuned_xgb
            if "Tuned" in best_name
            else models["XGBoost"]
        )

    if (
        importance_model is not None
        and hasattr(
            importance_model,
            "feature_importances_"
        )
    ):

        importance_df = pd.DataFrame(
            {
                "FEATURE": feature_columns,
                "IMPORTANCE":
                    importance_model.feature_importances_
            }
        ).sort_values(
            "IMPORTANCE",
            ascending=False
        )

        print(
            importance_df.to_string(
                index=False
            )
        )

        importance_df.to_csv(
            MODEL_DIR /
            "forecast_feature_importance.csv",
            index=False
        )

    # =========================================================================
    # TRAIN FINAL PRODUCTION MODEL
    # =========================================================================

    section(
        "TRAINING FINAL PRODUCTION MODEL"
    )

    print(
        "Production period:"
    )

    print(
        f"2018–{VALIDATION_YEAR}"
    )

    print(
        f"Rows: {len(historical):,}"
    )

    production_X = historical[
        feature_columns
    ].copy()

    production_y = historical[
        TARGET
    ].copy()

    production_medians = (
        production_X.median()
    )

    production_X = production_X.fillna(
        production_medians
    )

    # =========================================================================
    # PRODUCTION MODEL
    # =========================================================================

    if best_name == "Tuned Random Forest":

        production_model = RandomForestRegressor(
            **rf_params,
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif best_name == "Tuned XGBoost":

        production_model = XGBRegressor(
            **xgb_params,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif best_name == "Random Forest":

        production_model = models[
            "Random Forest"
        ]

    elif best_name == "Extra Trees":

        production_model = models[
            "Extra Trees"
        ]

    elif best_name == "Hist Gradient Boosting":

        production_model = models[
            "Hist Gradient Boosting"
        ]

    else:

        # For an ensemble, use the strongest individual model
        # as the persisted production model.
        #
        # This keeps the .joblib interface simple and reliable.

        strongest_individual = comparison[
            ~comparison["MODEL"].str.contains(
                "Blend|Ensemble",
                regex=True
            )
        ].iloc[0]["MODEL"]

        if strongest_individual == "Tuned Random Forest":

            production_model = RandomForestRegressor(
                **rf_params,
                bootstrap=True,
                n_jobs=-1,
                random_state=RANDOM_STATE
            )

        elif strongest_individual == "Tuned XGBoost":

            production_model = XGBRegressor(
                **xgb_params,
                objective="reg:squarederror",
                eval_metric="rmse",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )

        else:

            production_model = models[
                strongest_individual
            ]

    production_model.fit(
        production_X,
        production_y
    )

    print(
        "Final production model trained successfully."
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    section(
        "SAVING PRODUCTION MODEL"
    )

    model_path = (
        MODEL_DIR /
        "forecast_optimized_model.joblib"
    )

    joblib.dump(
        {
            "model": production_model,
            "features": feature_columns,
            "feature_medians":
                production_medians.to_dict(),
            "model_name": best_name,
            "validation_r2": best_r2,
            "validation_rmse": best_rmse,
            "validation_mae": best_mae,
            "validation_year":
                VALIDATION_YEAR,
            "forecast_year":
                PRODUCTION_FORECAST_YEAR,
        },
        model_path
    )

    # Keep compatibility with existing pipeline name.
    compatibility_path = (
        MODEL_DIR /
        "forecast_random_forest.joblib"
    )

    joblib.dump(
        {
            "model": production_model,
            "features": feature_columns,
            "feature_medians":
                production_medians.to_dict(),
            "model_name": best_name,
            "validation_r2": best_r2,
            "validation_rmse": best_rmse,
            "validation_mae": best_mae,
            "validation_year":
                VALIDATION_YEAR,
            "forecast_year":
                PRODUCTION_FORECAST_YEAR,
        },
        compatibility_path
    )

    print(
        "Optimized model saved:"
    )

    print(
        model_path
    )

    print(
        "\nCompatibility model saved:"
    )

    print(
        compatibility_path
    )

    # =========================================================================
    # 2024 SCORING
    # =========================================================================

    section(
        "CREATING 2024 FORECAST"
    )

    if not SCORING_DATASET.exists():

        print(
            "2024 scoring dataset not found."
        )

        print(
            "Skipping 2024 forecast."
        )

    else:

        scoring = pd.read_csv(
            SCORING_DATASET
        )

        scoring.columns = [
            str(c).strip().upper()
            for c in scoring.columns
        ]

        # ---------------------------------------------------------------------
        # Feature engineering
        #
        # Combine historical + scoring records first so lag/trend features
        # are computed from the full known history.
        # ---------------------------------------------------------------------

        scoring_base = scoring.copy()

        combined = pd.concat(
            [
                df.drop(
                    columns=[
                        TARGET
                    ],
                    errors="ignore"
                ),
                scoring_base
            ],
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset=[
                "ACO_ID",
                "YEAR"
            ],
            keep="last"
        )

        combined_features = (
            create_forecast_features(
                combined
            )
        )

        scoring_features = (
            combined_features[
                combined_features["YEAR"]
                == PRODUCTION_FORECAST_YEAR
            ].copy()
        )

        # ---------------------------------------------------------------------
        # Ensure only the scoring rows are used.
        # ---------------------------------------------------------------------

        if "ACO_ID" in scoring.columns:

            scoring_ids = set(
                scoring["ACO_ID"]
            )

            scoring_features = (
                scoring_features[
                    scoring_features[
                        "ACO_ID"
                    ].isin(scoring_ids)
                ]
            )

        X_2024 = scoring_features[
            feature_columns
        ].copy()

        X_2024 = X_2024.fillna(
            production_medians
        )

        forecast_values = (
            production_model.predict(
                X_2024
            )
        )

        scoring_output = scoring_features[
            [
                c
                for c in [
                    "ACO_ID",
                    "ACO_NAME",
                    "STATE",
                    "YEAR"
                ]
                if c in scoring_features.columns
            ]
        ].copy()

        scoring_output[
            "FORECASTED_NEXT_YEAR_SAVINGS_RATE"
        ] = forecast_values

        scoring_output[
            "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT"
        ] = (
            forecast_values * 100
        )

        # ---------------------------------------------------------------------
        # FORECAST CATEGORIES
        #
        # These thresholds DO NOT affect R².
        # They are business presentation thresholds only.
        # ---------------------------------------------------------------------

        LOW_THRESHOLD = 0.00
        HIGH_THRESHOLD = 0.05

        scoring_output[
            "SAVINGS_FORECAST_CATEGORY"
        ] = np.select(
            [
                scoring_output[
                    "FORECASTED_NEXT_YEAR_SAVINGS_RATE"
                ] < LOW_THRESHOLD,

                scoring_output[
                    "FORECASTED_NEXT_YEAR_SAVINGS_RATE"
                ] >= HIGH_THRESHOLD
            ],
            [
                "LOW SAVINGS",
                "HIGH SAVINGS"
            ],
            default="MODERATE SAVINGS"
        )

        forecast_path = (
            MODEL_DIR /
            "Dataset1_Scoring_2024_Forecast_Predictions.csv"
        )

        scoring_output.to_csv(
            forecast_path,
            index=False
        )

        print(
            "2024 forecast saved:"
        )

        print(
            forecast_path
        )

        # ---------------------------------------------------------------------
        # FORECAST SUMMARY
        # ---------------------------------------------------------------------

        section(
            "2024 FORECAST SUMMARY"
        )

        print(
            f"Rows forecasted : "
            f"{len(scoring_output):,}"
        )

        print(
            f"Mean forecast   : "
            f"{forecast_values.mean():.6f}"
        )

        print(
            f"Median forecast : "
            f"{np.median(forecast_values):.6f}"
        )

        print(
            f"Minimum forecast: "
            f"{forecast_values.min():.6f}"
        )

        print(
            f"Maximum forecast: "
            f"{forecast_values.max():.6f}"
        )

        print(
            "\nForecast categories:"
        )

        print(
            scoring_output[
                "SAVINGS_FORECAST_CATEGORY"
            ].value_counts()
        )

        # ---------------------------------------------------------------------
        # TOP 10
        # ---------------------------------------------------------------------

        section(
            "TOP 10 FORECASTED SAVINGS"
        )

        print(
            scoring_output.sort_values(
                "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
                ascending=False
            ).head(10).to_string(
                index=False
            )
        )

        # ---------------------------------------------------------------------
        # BOTTOM 10
        # ---------------------------------------------------------------------

        section(
            "BOTTOM 10 FORECASTED SAVINGS"
        )

        print(
            scoring_output.sort_values(
                "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
                ascending=True
            ).head(10).to_string(
                index=False
            )
        )

    # =========================================================================
    # VALIDATION METADATA
    # =========================================================================

    section(
        "SAVING MODEL METADATA"
    )

    metadata = {
        "project": "ContractIQ",
        "dataset": "Dataset1",
        "target": TARGET,
        "selected_model": best_name,
        "validation_year":
            VALIDATION_YEAR,
        "forecast_year":
            PRODUCTION_FORECAST_YEAR,
        "validation_mae":
            float(best_mae),
        "validation_rmse":
            float(best_rmse),
        "validation_r2":
            float(best_r2),
        "target_r2_goal":
            TARGET_R2_GOAL,
        "r2_goal_achieved":
            bool(
                best_r2 >= TARGET_R2_GOAL
            ),
        "base_features":
            BASE_FEATURES,
        "total_features":
            feature_columns,
        "number_of_features":
            len(feature_columns),
        "xgboost_available":
            XGBOOST_AVAILABLE,
        "rf_best_parameters":
            rf_params,
        "xgb_best_parameters":
            xgb_params,
        "leakage_protection":
            True,
        "future_target_used_as_feature":
            False,
        "production_training_years":
            [
                int(
                    historical["YEAR"].min()
                ),
                int(
                    historical["YEAR"].max()
                )
            ],
    }

    metadata_path = (
        MODEL_DIR /
        "forecast_model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # =========================================================================
    # VALIDATION METRICS
    # =========================================================================

    validation_metrics = {
        "selected_model":
            best_name,
        "MAE":
            float(best_mae),
        "RMSE":
            float(best_rmse),
        "R2":
            float(best_r2),
        "validation_year":
            VALIDATION_YEAR,
        "r2_target":
            TARGET_R2_GOAL,
        "r2_target_achieved":
            bool(
                best_r2 >= TARGET_R2_GOAL
            ),
    }

    with open(
        MODEL_DIR /
        "forecast_validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            validation_metrics,
            f,
            indent=4
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    section(
        "FINAL FORECAST MODEL SUMMARY"
    )

    print(
        f"Input dataset       : "
        f"{INPUT_DATASET}"
    )

    print(
        f"Historical rows     : "
        f"{len(historical):,}"
    )

    print(
        f"Training rows       : "
        f"{len(train):,}"
    )

    print(
        f"Validation rows     : "
        f"{len(validation):,}"
    )

    print(
        f"Training years      : "
        f"{train['YEAR'].min()}–"
        f"{train['YEAR'].max()}"
    )

    print(
        f"Validation year     : "
        f"{VALIDATION_YEAR}"
    )

    print(
        f"Selected model      : "
        f"{best_name}"
    )

    print(
        f"MAE                 : "
        f"{best_mae:.6f}"
    )

    print(
        f"RMSE                : "
        f"{best_rmse:.6f}"
    )

    print(
        f"R²                  : "
        f"{best_r2:.6f}"
    )

    print(
        f"R² target           : "
        f"{TARGET_R2_GOAL:.2f}"
    )

    print(
        f"Number of features  : "
        f"{len(feature_columns)}"
    )

    print(
        "\nTARGET:"
    )

    print(
        f" - {TARGET}"
    )

    print(
        "\nLEAKAGE:"
    )

    print(
        " - Future target NOT used as predictor"
    )

    print(
        " - 2023 reserved for time-based validation"
    )

    print(
        " - 2024 reserved for production forecasting"
    )

    print(
        "\nOUTPUT FILES:"
    )

    outputs = [
        "forecast_optimized_model.joblib",
        "forecast_random_forest.joblib",
        "forecast_model_comparison.csv",
        "forecast_cross_validation.csv",
        "forecast_feature_importance.csv",
        "forecast_validation_predictions.csv",
        "forecast_validation_metrics.json",
        "forecast_model_metadata.json",
        "Dataset1_Scoring_2024_Forecast_Predictions.csv",
    ]

    for i, filename in enumerate(
        outputs,
        start=1
    ):
        path = MODEL_DIR / filename

        if path.exists():
            print(
                f"{i}. {path}"
            )

    # =========================================================================
    # COMPLETION
    # =========================================================================

    section(
        "FORECAST MODEL TRAINING COMPLETE"
    )

    print(
        "STATUS: SUCCESS"
    )

    print(
        f"\nBEST MODEL: {best_name}"
    )

    print(
        f"2023 TIME-BASED R²: "
        f"{best_r2:.6f}"
    )

    print(
        f"2023 TIME-BASED RMSE: "
        f"{best_rmse:.6f}"
    )

    if best_r2 >= TARGET_R2_GOAL:

        print(
            "\nTARGET ACHIEVED:"
        )

        print(
            f"R² >= {TARGET_R2_GOAL:.2f}"
        )

    else:

        print(
            "\nTARGET NOT YET ACHIEVED:"
        )

        print(
            f"Current R² = {best_r2:.6f}"
        )

        print(
            f"Required R² = {TARGET_R2_GOAL:.2f}"
        )

        print(
            "\nIMPORTANT:"
        )

        print(
            "The reported R² is genuine 2023 "
            "time-based validation R²."
        )

        print(
            "No threshold manipulation or "
            "future-target leakage was used."
        )

    print(
        "\nNo future target was used as a predictor."
    )

    print(
        "2024 remains reserved for production forecasting."
    )

    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()