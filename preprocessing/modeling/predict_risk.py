"""
==============================================================================
CONTRACTIQ — RISK PREDICTION PIPELINE
==============================================================================

Purpose:
    Load the validated/frozen Random Forest risk model and generate
    production risk predictions.

Model:
    risk_random_forest_85_87.joblib

Production threshold:
    0.23

Features:
    1. N_AB
    2. PREVIOUS_SAVINGS_RATE
    3. PREVIOUS_QUALITY_SCORE
    4. PREVIOUS_PERFORMANCE_GAP_PCT
    5. EXPENDITURE_GROWTH_PCT
    6. BENCHMARK_GROWTH_PCT
    7. BENEFICIARY_GROWTH_PCT
    8. QUALITY_CHANGE

Target:
    NEXT_YEAR_RISK

Modes:
    1. Single ACO prediction
    2. CSV batch prediction

==============================================================================
"""

from pathlib import Path
import sys
import json
import argparse

import joblib
import numpy as np
import pandas as pd


# ==============================================================================
# PROJECT PATHS
# ==============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "risk_prediction"

MODEL_PATH = MODEL_DIR / "risk_random_forest_85_87.joblib"

DEFAULT_SCORING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Scoring_2024.csv"
)

OUTPUT_DIR = MODEL_DIR

DEFAULT_OUTPUT_FILE = (
    OUTPUT_DIR
    / "Dataset1_Scoring_2024_Risk_Predictions_Final.csv"
)


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

PRODUCTION_THRESHOLD = 0.23

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

IDENTIFIER_COLUMNS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]


# ==============================================================================
# DISPLAY HELPERS
# ==============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ==============================================================================
# LOAD MODEL
# ==============================================================================

def load_model():

    print_section("LOADING PRODUCTION RISK MODEL")

    print(f"Model path:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"""
Production Random Forest model was not found.

Expected:
{MODEL_PATH}

Train the model first:
python train_risk_model.py
"""
        )

    model = joblib.load(MODEL_PATH)

    print()
    print("Model loaded successfully.")
    print(f"Model type: {type(model).__name__}")

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not support probability prediction."
        )

    return model


# ==============================================================================
# VALIDATE MODEL FEATURES
# ==============================================================================

def validate_model_features(model):

    print_section("VALIDATING MODEL FEATURE CONFIGURATION")

    if hasattr(model, "feature_names_in_"):

        trained_features = list(model.feature_names_in_)

        print("Features stored in trained model:")

        for i, feature in enumerate(trained_features, start=1):
            print(f"{i}. {feature}")

        if trained_features != FEATURES:

            raise ValueError(
                "\nFeature mismatch detected.\n\n"
                f"Expected:\n{FEATURES}\n\n"
                f"Model contains:\n{trained_features}"
            )

    else:

        print(
            "WARNING: Model does not contain feature_names_in_."
        )

        print(
            "Using the frozen ContractIQ feature order."
        )

    print()
    print("Feature configuration PASSED.")


# ==============================================================================
# VALIDATE INPUT DATA
# ==============================================================================

def validate_input_dataframe(df):

    print_section("VALIDATING INPUT DATA")

    df = df.copy()

    # --------------------------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    print("Column names standardized.")

    # --------------------------------------------------------------------------
    # Required feature check
    # --------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "\nMissing required model features:\n"
            + "\n".join(
                f" - {feature}"
                for feature in missing_features
            )
        )

    print()
    print("All required model features are present.")

    # --------------------------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------------------------

    print()
    print("Checking numeric model features...")

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        missing_count = int(df[feature].isna().sum())

        infinite_count = int(
            np.isinf(df[feature].to_numpy()).sum()
        )

        print(
            f"{feature:<38}"
            f"missing={missing_count:>6}"
            f" infinite={infinite_count:>6}"
        )

        if missing_count > 0:
            raise ValueError(
                f"\nMissing/non-numeric values detected in {feature}."
            )

        if infinite_count > 0:
            raise ValueError(
                f"\nInfinite values detected in {feature}."
            )

    print()
    print("Input feature validation PASSED.")

    return df


# ==============================================================================
# GENERATE PREDICTIONS
# ==============================================================================

def generate_predictions(model, df):

    print_section("GENERATING RISK PREDICTIONS")

    X = df[FEATURES].copy()

    print(f"Rows to score : {len(X):,}")
    print(f"Features      : {len(FEATURES)}")
    print(f"Threshold     : {PRODUCTION_THRESHOLD}")

    # --------------------------------------------------------------------------
    # Probability
    # --------------------------------------------------------------------------

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] != 2:

        raise ValueError(
            "Expected binary classification model with two probability columns."
        )

    risk_probability = probabilities[:, 1]

    # --------------------------------------------------------------------------
    # Classification using production threshold
    # --------------------------------------------------------------------------

    risk_prediction = (
        risk_probability >= PRODUCTION_THRESHOLD
    ).astype(int)

    result = df.copy()

    result["RISK_PROBABILITY"] = risk_probability

    result["RISK_PREDICTION"] = risk_prediction

    result["RISK_LABEL"] = np.where(
        risk_prediction == 1,
        "HIGH RISK",
        "LOW RISK"
    )

    # --------------------------------------------------------------------------
    # Risk percentage
    # --------------------------------------------------------------------------

    result["RISK_PROBABILITY_PCT"] = (
        result["RISK_PROBABILITY"] * 100
    ).round(2)

    # --------------------------------------------------------------------------
    # Sort highest risk first
    # --------------------------------------------------------------------------

    result = result.sort_values(
        by="RISK_PROBABILITY",
        ascending=False
    ).reset_index(drop=True)

    return result


# ==============================================================================
# DISPLAY SUMMARY
# ==============================================================================

def display_prediction_summary(result):

    print_section("PREDICTION SUMMARY")

    total_rows = len(result)

    risk_cases = int(
        (result["RISK_PREDICTION"] == 1).sum()
    )

    non_risk_cases = int(
        (result["RISK_PREDICTION"] == 0).sum()
    )

    risk_rate = (
        risk_cases / total_rows
        if total_rows > 0
        else 0
    )

    print(f"Rows scored       : {total_rows:,}")
    print(f"Predicted risk    : {risk_cases:,}")
    print(f"Predicted non-risk: {non_risk_cases:,}")
    print(f"Predicted risk rate: {risk_rate:.4%}")

    print()
    print("Risk probability statistics:")

    print(
        result["RISK_PROBABILITY"].describe()
    )

    # --------------------------------------------------------------------------
    # Show top 10
    # --------------------------------------------------------------------------

    print_section("TOP 10 HIGHEST-RISK ACOs")

    display_columns = [
        column
        for column in [
            "ACO_ID",
            "ACO_NAME",
            "STATE",
            "YEAR",
            "RISK_PROBABILITY_PCT",
            "RISK_LABEL",
        ]
        if column in result.columns
    ]

    if display_columns:

        print(
            result[
                display_columns
            ].head(10).to_string(index=False)
        )

    else:

        print(
            result[
                [
                    "RISK_PROBABILITY_PCT",
                    "RISK_LABEL",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


# ==============================================================================
# SAVE PREDICTIONS
# ==============================================================================

def save_predictions(result, output_path):

    print_section("SAVING PREDICTIONS")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_path,
        index=False
    )

    print("Predictions saved successfully:")
    print(output_path)

    print()
    print(f"Rows saved: {len(result):,}")


# ==============================================================================
# SAVE METADATA
# ==============================================================================

def save_prediction_metadata(
    result,
    input_path,
    output_path,
):

    risk_cases = int(
        (result["RISK_PREDICTION"] == 1).sum()
    )

    non_risk_cases = int(
        (result["RISK_PREDICTION"] == 0).sum()
    )

    total_rows = len(result)

    metadata = {

        "project": "ContractIQ",

        "model": {
            "type": "RandomForestClassifier",
            "file": str(MODEL_PATH),
            "classification_threshold": PRODUCTION_THRESHOLD,
        },

        "features": FEATURES,

        "input": str(input_path),

        "output": str(output_path),

        "rows_scored": total_rows,

        "predicted_risk_cases": risk_cases,

        "predicted_non_risk_cases": non_risk_cases,

        "predicted_risk_rate": (
            risk_cases / total_rows
            if total_rows > 0
            else 0
        ),

        "notes": [
            "Production model is the validated Random Forest.",
            "Production classification threshold is 0.23.",
            "No target labels are modified during prediction.",
            "NEXT_YEAR_SAVINGS_RATE is not used as a predictor.",
            "Prediction uses the same eight model features used during training.",
        ],
    }

    metadata_path = (
        Path(output_path).with_suffix(".json")
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print()
    print("Prediction metadata saved:")
    print(metadata_path)


# ==============================================================================
# BATCH CSV MODE
# ==============================================================================

def run_csv_prediction(
    model,
    input_path,
    output_path,
):

    print_header(
        "CONTRACTIQ — BATCH RISK PREDICTION"
    )

    print_section("PROJECT CONFIGURATION")

    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Input:")
    print(input_path)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Output:")
    print(output_path)

    # --------------------------------------------------------------------------
    # Input existence
    # --------------------------------------------------------------------------

    if not Path(input_path).exists():

        raise FileNotFoundError(
            f"""
Input CSV was not found.

Expected:
{input_path}
"""
        )

    # --------------------------------------------------------------------------
    # Load
    # --------------------------------------------------------------------------

    print_section("LOADING INPUT CSV")

    df = pd.read_csv(input_path)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------------------

    df = validate_input_dataframe(df)

    # --------------------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------------------

    result = generate_predictions(
        model,
        df
    )

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    display_prediction_summary(result)

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------

    save_predictions(
        result,
        output_path
    )

    save_prediction_metadata(
        result,
        input_path,
        output_path
    )

    print_header(
        "BATCH RISK PREDICTION COMPLETE"
    )


# ==============================================================================
# SINGLE ACO MODE
# ==============================================================================

def get_float_input(prompt):

    while True:

        value = input(prompt).strip()

        try:

            number = float(value)

            if not np.isfinite(number):
                raise ValueError

            return number

        except ValueError:

            print(
                "Please enter a valid numeric value."
            )


def run_single_prediction(model):

    print_header(
        "CONTRACTIQ — SINGLE ACO RISK PREDICTION"
    )

    print()
    print(
        "Enter the current ACO values."
    )

    print(
        "These must correspond to the same eight features used during training."
    )

    print()

    # --------------------------------------------------------------------------
    # Optional identifiers
    # --------------------------------------------------------------------------

    aco_id = input(
        "ACO ID [optional]: "
    ).strip()

    aco_name = input(
        "ACO Name [optional]: "
    ).strip()

    state = input(
        "State [optional]: "
    ).strip()

    # --------------------------------------------------------------------------
    # Features
    # --------------------------------------------------------------------------

    print()
    print("MODEL FEATURES")
    print("-" * 78)

    values = {}

    values["N_AB"] = get_float_input(
        "N_AB: "
    )

    values["PREVIOUS_SAVINGS_RATE"] = get_float_input(
        "PREVIOUS_SAVINGS_RATE: "
    )

    values["PREVIOUS_QUALITY_SCORE"] = get_float_input(
        "PREVIOUS_QUALITY_SCORE: "
    )

    values["PREVIOUS_PERFORMANCE_GAP_PCT"] = get_float_input(
        "PREVIOUS_PERFORMANCE_GAP_PCT: "
    )

    values["EXPENDITURE_GROWTH_PCT"] = get_float_input(
        "EXPENDITURE_GROWTH_PCT: "
    )

    values["BENCHMARK_GROWTH_PCT"] = get_float_input(
        "BENCHMARK_GROWTH_PCT: "
    )

    values["BENEFICIARY_GROWTH_PCT"] = get_float_input(
        "BENEFICIARY_GROWTH_PCT: "
    )

    values["QUALITY_CHANGE"] = get_float_input(
        "QUALITY_CHANGE: "
    )

    # --------------------------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------------------------

    df = pd.DataFrame(
        [values]
    )

    df = validate_input_dataframe(df)

    # --------------------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------------------

    result = generate_predictions(
        model,
        df
    )

    row = result.iloc[0]

    # --------------------------------------------------------------------------
    # Display
    # --------------------------------------------------------------------------

    print_section(
        "CONTRACTIQ RISK ASSESSMENT"
    )

    if aco_id:
        print(f"ACO ID                : {aco_id}")

    if aco_name:
        print(f"ACO Name              : {aco_name}")

    if state:
        print(f"State                 : {state}")

    print()
    print(
        f"Risk probability      : "
        f"{row['RISK_PROBABILITY_PCT']:.2f}%"
    )

    print(
        f"Production threshold  : "
        f"{PRODUCTION_THRESHOLD:.2f}"
    )

    print(
        f"Risk prediction       : "
        f"{row['RISK_PREDICTION']}"
    )

    print(
        f"Risk classification   : "
        f"{row['RISK_LABEL']}"
    )

    print()

    if row["RISK_PREDICTION"] == 1:

        print(
            "RESULT: HIGH RISK"
        )

        print(
            "The predicted risk probability is at or above "
            "the production threshold."
        )

    else:

        print(
            "RESULT: LOW RISK"
        )

        print(
            "The predicted risk probability is below "
            "the production threshold."
        )

    print_section(
        "SINGLE ACO PREDICTION COMPLETE"
    )


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "ContractIQ production risk prediction pipeline."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "csv",
        ],
        default="csv",
        help=(
            "Prediction mode: single ACO or CSV batch."
        ),
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_SCORING_FILE),
        help=(
            "Input CSV for batch prediction."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help=(
            "Output CSV path."
        ),
    )

    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    args = parse_arguments()

    try:

        # ----------------------------------------------------------------------
        # Load model
        # ----------------------------------------------------------------------

        model = load_model()

        # ----------------------------------------------------------------------
        # Validate model
        # ----------------------------------------------------------------------

        validate_model_features(
            model
        )

        # ----------------------------------------------------------------------
        # Prediction mode
        # ----------------------------------------------------------------------

        if args.mode == "single":

            run_single_prediction(
                model
            )

        else:

            run_csv_prediction(
                model=model,
                input_path=Path(args.input),
                output_path=Path(args.output),
            )

        print()
        print("=" * 78)
        print("STATUS: SUCCESS")
        print("=" * 78)

    except Exception as exc:

        print()
        print("=" * 78)
        print("ERROR")
        print("=" * 78)
        print()
        print(str(exc))
        print()

        sys.exit(1)


if __name__ == "__main__":
    main()