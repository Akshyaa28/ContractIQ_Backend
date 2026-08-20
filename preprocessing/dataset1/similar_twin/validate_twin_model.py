from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

HISTORICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Model_Twin.csv"
)

SCORING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Scoring_2024_Twin.csv"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Top5_Similar_Twins_2024.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Top5_Twin_Summary_2024.csv"
)

OUTPERFORMER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Outperforming_Twins_2024.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "similar_twin"
)

SCALER_PATH = MODEL_DIR / "twin_scaler.joblib"
NEIGHBOR_MODEL_PATH = MODEL_DIR / "twin_nearest_neighbors.joblib"
FEATURE_WEIGHTS_PATH = MODEL_DIR / "twin_feature_weights.json"
METADATA_PATH = MODEL_DIR / "twin_model_metadata.json"


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5
SCORING_YEAR = 2024

TARGET = "NEXT_YEAR_SAVINGS_RATE"

TWIN_FEATURES = [
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
]


# ============================================================
# REQUIRED RESULT COLUMNS
# ============================================================

REQUIRED_RESULT_COLUMNS = [
    "TARGET_ACO_ID",
    "TARGET_ACO_NAME",
    "TARGET_STATE",
    "TARGET_YEAR",
    "TWIN_RANK",
    "TWIN_ACO_ID",
    "TWIN_ACO_NAME",
    "TWIN_STATE",
    "TWIN_YEAR",
    "DISTANCE",
    "SIMILARITY_SCORE",
    "TWIN_NEXT_YEAR_SAVINGS_RATE",
    "TWIN_NEXT_YEAR_SAVINGS_RATE_PCT",
    "OUTPERFORMING_TWIN",
    "OUTPERFORMANCE_GAP",
    "OUTPERFORMANCE_GAP_PCT",
    "TOP5_AVG_SAVINGS_RATE",
    "TOP5_MEDIAN_SAVINGS_RATE",
    "TOP5_WEIGHTED_SAVINGS_RATE",
    "TOP5_BEST_SAVINGS_RATE",
    "TOP5_WORST_SAVINGS_RATE",
    "TOP5_OUTPERFORMER_COUNT",
    "TOP5_OUTPERFORMER_RATE",
]


# ============================================================
# HELPER
# ============================================================

def fail(message):
    raise RuntimeError(message)


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = [
    HISTORICAL_PATH,
    SCORING_PATH,
    RESULT_PATH,
    SUMMARY_PATH,
    OUTPERFORMER_PATH,
    SCALER_PATH,
    NEIGHBOR_MODEL_PATH,
    FEATURE_WEIGHTS_PATH,
    METADATA_PATH,
]

for path in required_files:
    if not path.exists():
        fail(f"Missing required file: {path}")


# ============================================================
# LOAD DATA
# ============================================================

historical = pd.read_csv(HISTORICAL_PATH)
scoring = pd.read_csv(SCORING_PATH)
results = pd.read_csv(RESULT_PATH)
summary = pd.read_csv(SUMMARY_PATH)
outperformers = pd.read_csv(OUTPERFORMER_PATH)

historical.columns = (
    historical.columns.str.strip().str.upper()
)

scoring.columns = (
    scoring.columns.str.strip().str.upper()
)

results.columns = (
    results.columns.str.strip().str.upper()
)

summary.columns = (
    summary.columns.str.strip().str.upper()
)

outperformers.columns = (
    outperformers.columns.str.strip().str.upper()
)


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

if len(scoring) == 0:
    fail("2024 scoring dataset is empty.")

if len(historical) == 0:
    fail("Historical Twin dataset is empty.")

if len(results) == 0:
    fail("Twin result dataset is empty.")


# ============================================================
# RESULT COLUMN VALIDATION
# ============================================================

missing_columns = [
    column
    for column in REQUIRED_RESULT_COLUMNS
    if column not in results.columns
]

if missing_columns:
    fail(
        "Missing result columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# 2024 SCORING VALIDATION
# ============================================================

scoring["YEAR"] = pd.to_numeric(
    scoring["YEAR"],
    errors="coerce",
)

if scoring["YEAR"].isna().any():
    fail("Invalid YEAR values in scoring data.")

if not (
    scoring["YEAR"] == SCORING_YEAR
).all():
    fail("Scoring data contains non-2024 records.")


# ============================================================
# HISTORICAL YEAR VALIDATION
# ============================================================

historical["YEAR"] = pd.to_numeric(
    historical["YEAR"],
    errors="coerce",
)

if historical["YEAR"].isna().any():
    fail("Invalid YEAR values in historical data.")

if not (
    historical["YEAR"] <= 2023
).all():
    fail(
        "Historical Twin reference contains 2024 or later records."
    )


# ============================================================
# TOP 5 ROW COUNT
# ============================================================

expected_rows = len(scoring) * TOP_K

if len(results) != expected_rows:
    fail(
        f"Expected {expected_rows:,} result rows, "
        f"found {len(results):,}."
    )


# ============================================================
# EXACTLY 5 TWINS PER ACO
# ============================================================

rank_counts = (
    results
    .groupby("TARGET_ACO_ID")["TWIN_RANK"]
    .nunique()
)

if not (
    rank_counts == TOP_K
).all():
    fail(
        "Not every target ACO has exactly 5 Twin ranks."
    )


row_counts = (
    results
    .groupby("TARGET_ACO_ID")
    .size()
)

if not (
    row_counts == TOP_K
).all():
    fail(
        "Not every target ACO has exactly 5 Twin rows."
    )


# ============================================================
# RANK VALIDATION
# ============================================================

valid_ranks = set(range(1, TOP_K + 1))

for aco_id, group in results.groupby(
    "TARGET_ACO_ID"
):

    ranks = set(
        group["TWIN_RANK"].astype(int)
    )

    if ranks != valid_ranks:
        fail(
            f"Invalid Twin ranks for ACO {aco_id}."
        )


# ============================================================
# UNIQUE TWIN VALIDATION
# ============================================================

twin_counts = (
    results
    .groupby("TARGET_ACO_ID")["TWIN_ACO_ID"]
    .nunique()
)

if not (
    twin_counts == TOP_K
).all():
    fail(
        "Duplicate Twin ACOs detected."
    )


# ============================================================
# SELF-MATCH VALIDATION
# ============================================================

self_matches = (
    results["TARGET_ACO_ID"].astype(str)
    ==
    results["TWIN_ACO_ID"].astype(str)
)

if self_matches.any():
    fail(
        "Self-match detected."
    )


# ============================================================
# TWIN YEAR VALIDATION
# ============================================================

results["TWIN_YEAR"] = pd.to_numeric(
    results["TWIN_YEAR"],
    errors="coerce",
)

if results["TWIN_YEAR"].isna().any():
    fail("Invalid TWIN_YEAR values.")

if not (
    results["TWIN_YEAR"] <= 2023
).all():
    fail(
        "A Twin contains a future/scoring-year record."
    )


# ============================================================
# TARGET YEAR VALIDATION
# ============================================================

results["TARGET_YEAR"] = pd.to_numeric(
    results["TARGET_YEAR"],
    errors="coerce",
)

if not (
    results["TARGET_YEAR"] == SCORING_YEAR
).all():
    fail(
        "Result dataset contains non-2024 target records."
    )


# ============================================================
# DISTANCE VALIDATION
# ============================================================

numeric_columns = [
    "DISTANCE",
    "SIMILARITY_SCORE",
    "TWIN_NEXT_YEAR_SAVINGS_RATE",
    "TWIN_NEXT_YEAR_SAVINGS_RATE_PCT",
    "OUTPERFORMANCE_GAP",
    "OUTPERFORMANCE_GAP_PCT",
]

for column in numeric_columns:

    results[column] = pd.to_numeric(
        results[column],
        errors="coerce",
    )

    if results[column].isna().any():
        fail(
            f"{column} contains invalid values."
        )

    if not np.isfinite(
        results[column].to_numpy()
    ).all():
        fail(
            f"{column} contains infinite values."
        )


if (
    results["DISTANCE"] < 0
).any():
    fail(
        "Negative similarity distance detected."
    )


# ============================================================
# SIMILARITY SCORE VALIDATION
# ============================================================

expected_similarity = (
    100.0
    / (1.0 + results["DISTANCE"])
)

similarity_difference = np.abs(
    results["SIMILARITY_SCORE"]
    - expected_similarity
)

if (
    similarity_difference > 1e-6
).any():
    fail(
        "Similarity scores do not match the distance formula."
    )


if (
    results["SIMILARITY_SCORE"] < 0
).any():
    fail(
        "Negative similarity score detected."
    )

if (
    results["SIMILARITY_SCORE"] > 100
).any():
    fail(
        "Similarity score above 100 detected."
    )


# ============================================================
# RANK ORDER VALIDATION
# ============================================================

for aco_id, group in results.groupby(
    "TARGET_ACO_ID"
):

    ordered = (
        group
        .sort_values("TWIN_RANK")
    )

    distances = (
        ordered["DISTANCE"]
        .to_numpy()
    )

    if not np.all(
        distances[:-1]
        <= distances[1:] + 1e-10
    ):
        fail(
            f"Distance ranking is invalid for ACO {aco_id}."
        )


# ============================================================
# HISTORICAL TWIN EXISTENCE VALIDATION
# ============================================================

historical_ids = set(
    historical["ACO_ID"].astype(str)
)

result_twin_ids = set(
    results["TWIN_ACO_ID"].astype(str)
)

missing_twin_ids = (
    result_twin_ids - historical_ids
)

if missing_twin_ids:
    fail(
        "Some selected Twins do not exist in historical data."
    )


# ============================================================
# TARGET ACO COVERAGE
# ============================================================

scoring_ids = set(
    scoring["ACO_ID"].astype(str)
)

result_target_ids = set(
    results["TARGET_ACO_ID"].astype(str)
)

if scoring_ids != result_target_ids:
    missing_targets = (
        scoring_ids - result_target_ids
    )

    extra_targets = (
        result_target_ids - scoring_ids
    )

    fail(
        "Target ACO coverage mismatch. "
        f"Missing={len(missing_targets)}, "
        f"Extra={len(extra_targets)}."
    )


# ============================================================
# OPTION C VALIDATION
# ============================================================

results["OUTPERFORMING_TWIN"] = (
    results["OUTPERFORMING_TWIN"]
    .astype(bool)
)

for aco_id, group in results.groupby(
    "TARGET_ACO_ID"
):

    median_rate = float(
        group["TWIN_NEXT_YEAR_SAVINGS_RATE"]
        .median()
    )

    expected_flag = (
        group["TWIN_NEXT_YEAR_SAVINGS_RATE"]
        > median_rate
    )

    actual_flag = (
        group["OUTPERFORMING_TWIN"]
    )

    if not (
        expected_flag.to_numpy()
        ==
        actual_flag.to_numpy()
    ).all():
        fail(
            f"Option C validation failed for ACO {aco_id}."
        )


# ============================================================
# OPTION C GAP VALIDATION
# ============================================================

for aco_id, group in results.groupby(
    "TARGET_ACO_ID"
):

    median_rate = float(
        group["TWIN_NEXT_YEAR_SAVINGS_RATE"]
        .median()
    )

    expected_gap = (
        group["TWIN_NEXT_YEAR_SAVINGS_RATE"]
        - median_rate
    )

    actual_gap = (
        group["OUTPERFORMANCE_GAP"]
    )

    if not np.allclose(
        expected_gap.to_numpy(),
        actual_gap.to_numpy(),
        atol=1e-8,
    ):
        fail(
            f"Option C gap validation failed for ACO {aco_id}."
        )


# ============================================================
# OUTPERFORMER FILE VALIDATION
# ============================================================

expected_outperformers = results[
    results["OUTPERFORMING_TWIN"]
].copy()

if len(outperformers) != len(
    expected_outperformers
):
    fail(
        "Outperformer output row count does not match."
    )


expected_pairs = set(
    zip(
        expected_outperformers[
            "TARGET_ACO_ID"
        ].astype(str),
        expected_outperformers[
            "TWIN_ACO_ID"
        ].astype(str),
    )
)

actual_pairs = set(
    zip(
        outperformers[
            "TARGET_ACO_ID"
        ].astype(str),
        outperformers[
            "TWIN_ACO_ID"
        ].astype(str),
    )
)

if expected_pairs != actual_pairs:
    fail(
        "Outperformer output does not match Option C results."
    )


# ============================================================
# SUMMARY VALIDATION
# ============================================================

if len(summary) != len(scoring):
    fail(
        "Summary does not contain exactly one row per 2024 ACO."
    )

summary_ids = set(
    summary["TARGET_ACO_ID"].astype(str)
)

if summary_ids != scoring_ids:
    fail(
        "Summary ACO coverage does not match scoring data."
    )


# ============================================================
# MODEL ARTIFACT VALIDATION
# ============================================================

scaler = joblib.load(
    SCALER_PATH
)

neighbor_model = joblib.load(
    NEIGHBOR_MODEL_PATH
)

if not hasattr(
    scaler,
    "transform",
):
    fail(
        "Saved scaler is invalid."
    )

if not hasattr(
    neighbor_model,
    "kneighbors",
):
    fail(
        "Saved nearest-neighbor model is invalid."
    )


# ============================================================
# FEATURE WEIGHTS VALIDATION
# ============================================================

with open(
    FEATURE_WEIGHTS_PATH,
    "r",
    encoding="utf-8",
) as file:

    weights_metadata = json.load(file)


saved_features = (
    weights_metadata.get("features", [])
)

if saved_features != TWIN_FEATURES:
    fail(
        "Saved feature list does not match Twin features."
    )

if weights_metadata.get(
    "target_used_for_similarity"
) is not False:
    fail(
        "Target leakage detected in saved metadata."
    )


# ============================================================
# METADATA VALIDATION
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)


if metadata.get("top_k") != TOP_K:
    fail(
        "Saved metadata TOP_K does not equal 5."
    )

if metadata.get("scoring_year") != SCORING_YEAR:
    fail(
        "Saved metadata scoring year is incorrect."
    )

if metadata.get(
    "target_used_for_similarity"
) is not False:
    fail(
        "Metadata indicates target leakage."
    )

if metadata.get(
    "same_aco_excluded"
) is not True:
    fail(
        "Metadata does not confirm same-ACO exclusion."
    )


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

historical_target = pd.to_numeric(
    historical[TARGET],
    errors="coerce",
)

selected_target = pd.to_numeric(
    results["TWIN_NEXT_YEAR_SAVINGS_RATE"],
    errors="coerce",
)

historical_mean = float(
    historical_target.mean()
)

selected_mean = float(
    selected_target.mean()
)

historical_median = float(
    historical_target.median()
)

selected_median = float(
    selected_target.median()
)

performance_difference = (
    selected_mean
    - historical_mean
)

performance_lift_pct = (
    performance_difference * 100.0
)


# ============================================================
# OPTION C COVERAGE
# ============================================================

aco_count = len(scoring)

outperforming_aco_count = (
    results.loc[
        results["OUTPERFORMING_TWIN"],
        "TARGET_ACO_ID",
    ]
    .nunique()
)

outperforming_coverage = (
    outperforming_aco_count
    / aco_count
)


# ============================================================
# FINAL VALIDATION RESULT
# ============================================================

print(
    "Twin validation PASSED."
)

print(
    f"2024 ACOs: {aco_count:,}"
)

print(
    f"Top 5 rows: {len(results):,}"
)

print(
    f"Outperforming Twins: {len(outperformers):,}"
)

print(
    f"Outperformer coverage: "
    f"{outperforming_coverage * 100:.2f}%"
)

print(
    f"Selected Twin mean savings: "
    f"{selected_mean * 100:.3f}%"
)

print(
    f"Historical mean savings: "
    f"{historical_mean * 100:.3f}%"
)

print(
    f"Selected vs historical: "
    f"{performance_lift_pct:+.3f} percentage points"
)

print(
    "Leakage checks: PASSED"
)

print(
    "Artifact checks: PASSED"
)

print(
    "Status: SUCCESS"
)