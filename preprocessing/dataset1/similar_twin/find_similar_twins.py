from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

HISTORICAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "Dataset1_Model_Twin.csv"
)

SCORING_PATH = (
    PROJECT_ROOT / "data" / "processed" / "Dataset1_Scoring_2024_Twin.csv"
)

MODEL_DIR = (
    PROJECT_ROOT / "models" / "similar_twin"
)

OUTPUT_PATH = (
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

SCALER_PATH = MODEL_DIR / "twin_scaler.joblib"

NEIGHBOR_MODEL_PATH = (
    MODEL_DIR / "twin_nearest_neighbors.joblib"
)

FEATURE_WEIGHTS_PATH = (
    MODEL_DIR / "twin_feature_weights.json"
)

METADATA_PATH = (
    MODEL_DIR / "twin_model_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5
CANDIDATE_K = 50

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

IDENTIFIERS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]


FEATURE_WEIGHTS = {
    "N_AB": 0.90,
    "PREVIOUS_SAVINGS_RATE": 1.25,
    "PREVIOUS_QUALITY_SCORE": 1.15,
    "PREVIOUS_PERFORMANCE_GAP_PCT": 1.15,
    "EXPENDITURE_GROWTH_PCT": 1.00,
    "BENCHMARK_GROWTH_PCT": 1.00,
    "BENEFICIARY_GROWTH_PCT": 0.90,
    "QUALITY_CHANGE": 1.05,
}


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

if not HISTORICAL_PATH.exists():
    raise FileNotFoundError(
        f"Missing file: {HISTORICAL_PATH}"
    )

if not SCORING_PATH.exists():
    raise FileNotFoundError(
        f"Missing file: {SCORING_PATH}"
    )


historical = pd.read_csv(HISTORICAL_PATH)
scoring = pd.read_csv(SCORING_PATH)

historical.columns = (
    historical.columns.str.strip().str.upper()
)

scoring.columns = (
    scoring.columns.str.strip().str.upper()
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_historical = (
    IDENTIFIERS + TWIN_FEATURES + [TARGET]
)

required_scoring = (
    IDENTIFIERS + TWIN_FEATURES
)

missing_historical = [
    column
    for column in required_historical
    if column not in historical.columns
]

missing_scoring = [
    column
    for column in required_scoring
    if column not in scoring.columns
]

if missing_historical:
    raise ValueError(
        "Missing historical columns: "
        + ", ".join(missing_historical)
    )

if missing_scoring:
    raise ValueError(
        "Missing scoring columns: "
        + ", ".join(missing_scoring)
    )


# ============================================================
# VALIDATE YEARS
# ============================================================

historical["YEAR"] = pd.to_numeric(
    historical["YEAR"],
    errors="coerce",
)

scoring["YEAR"] = pd.to_numeric(
    scoring["YEAR"],
    errors="coerce",
)

if historical["YEAR"].isna().any():
    raise ValueError(
        "Historical YEAR contains invalid values."
    )

if scoring["YEAR"].isna().any():
    raise ValueError(
        "Scoring YEAR contains invalid values."
    )

historical["YEAR"] = historical["YEAR"].astype(int)
scoring["YEAR"] = scoring["YEAR"].astype(int)

if not (scoring["YEAR"] == 2024).all():
    raise ValueError(
        "Scoring dataset must contain only 2024 records."
    )


# ============================================================
# VALIDATE TARGET
# ============================================================

historical[TARGET] = pd.to_numeric(
    historical[TARGET],
    errors="coerce",
)

if historical[TARGET].isna().any():
    raise ValueError(
        f"{TARGET} contains missing values."
    )

if not np.isfinite(
    historical[TARGET].to_numpy()
).all():
    raise ValueError(
        f"{TARGET} contains infinite values."
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

for feature in TWIN_FEATURES:

    historical[feature] = pd.to_numeric(
        historical[feature],
        errors="coerce",
    )

    scoring[feature] = pd.to_numeric(
        scoring[feature],
        errors="coerce",
    )

    if historical[feature].isna().any():
        raise ValueError(
            f"Historical feature contains missing values: {feature}"
        )

    if scoring[feature].isna().any():
        raise ValueError(
            f"Scoring feature contains missing values: {feature}"
        )

    if not np.isfinite(
        historical[feature].to_numpy()
    ).all():
        raise ValueError(
            f"Historical feature contains infinite values: {feature}"
        )

    if not np.isfinite(
        scoring[feature].to_numpy()
    ).all():
        raise ValueError(
            f"Scoring feature contains infinite values: {feature}"
        )


# ============================================================
# HISTORICAL REFERENCE DATA
# ============================================================

reference = historical[
    historical["YEAR"] <= 2023
].copy()

reference.reset_index(
    drop=True,
    inplace=True,
)

if reference.empty:
    raise ValueError(
        "Historical reference dataset is empty."
    )


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

reference_matrix = scaler.fit_transform(
    reference[TWIN_FEATURES]
).astype(np.float32)

scoring_matrix = scaler.transform(
    scoring[TWIN_FEATURES]
).astype(np.float32)


# ============================================================
# APPLY FEATURE WEIGHTS
# ============================================================

weight_vector = np.array(
    [
        FEATURE_WEIGHTS[feature]
        for feature in TWIN_FEATURES
    ],
    dtype=np.float32,
)

weight_vector /= weight_vector.mean()

weighted_reference = (
    reference_matrix * weight_vector
).astype(np.float32)

weighted_scoring = (
    scoring_matrix * weight_vector
).astype(np.float32)


# ============================================================
# NEAREST NEIGHBOR MODEL
# ============================================================

neighbor_count = min(
    CANDIDATE_K,
    len(reference),
)

nn_model = NearestNeighbors(
    n_neighbors=neighbor_count,
    metric="euclidean",
    algorithm="auto",
    n_jobs=-1,
)

nn_model.fit(weighted_reference)

distances, indices = nn_model.kneighbors(
    weighted_scoring
)


# ============================================================
# SIMILARITY FUNCTION
# ============================================================

def distance_to_similarity(distance):
    return 100.0 / (1.0 + float(distance))


# ============================================================
# FIND TOP 5 TWINS
# ============================================================

results = []

for target_position in range(len(scoring)):

    target_row = scoring.iloc[target_position]

    target_aco_id = target_row["ACO_ID"]

    selected_twins = []

    for candidate_position in range(
        len(indices[target_position])
    ):

        reference_position = int(
            indices[target_position][candidate_position]
        )

        twin_row = reference.iloc[
            reference_position
        ]

        twin_aco_id = twin_row["ACO_ID"]

        if str(twin_aco_id) == str(target_aco_id):
            continue

        if any(
            str(item["TWIN_ACO_ID"])
            == str(twin_aco_id)
            for item in selected_twins
        ):
            continue

        distance = float(
            distances[
                target_position
            ][candidate_position]
        )

        similarity = distance_to_similarity(
            distance
        )

        selected_twins.append(
            {
                "TWIN_ACO_ID": twin_row["ACO_ID"],
                "TWIN_ACO_NAME": twin_row["ACO_NAME"],
                "TWIN_STATE": twin_row["STATE"],
                "TWIN_YEAR": int(twin_row["YEAR"]),
                "DISTANCE": distance,
                "SIMILARITY_SCORE": similarity,
                "TWIN_NEXT_YEAR_SAVINGS_RATE": float(
                    twin_row[TARGET]
                ),
            }
        )

        if len(selected_twins) == TOP_K:
            break

    if len(selected_twins) != TOP_K:
        raise RuntimeError(
            f"Could not find {TOP_K} unique Twins "
            f"for ACO {target_aco_id}."
        )

    twin_rates = np.array(
        [
            twin["TWIN_NEXT_YEAR_SAVINGS_RATE"]
            for twin in selected_twins
        ],
        dtype=float,
    )

    similarities = np.array(
        [
            twin["SIMILARITY_SCORE"]
            for twin in selected_twins
        ],
        dtype=float,
    )

    mean_rate = float(
        np.mean(twin_rates)
    )

    median_rate = float(
        np.median(twin_rates)
    )

    best_rate = float(
        np.max(twin_rates)
    )

    worst_rate = float(
        np.min(twin_rates)
    )

    weighted_rate = float(
        np.sum(
            twin_rates
            * (similarities / similarities.sum())
        )
    )

    outperformer_count = int(
        np.sum(twin_rates > median_rate)
    )

    outperformer_rate = float(
        np.mean(twin_rates > median_rate)
    )

    for rank, twin in enumerate(
        selected_twins,
        start=1,
    ):

        twin_rate = twin[
            "TWIN_NEXT_YEAR_SAVINGS_RATE"
        ]

        twin.update(
            {
                "TARGET_ACO_ID": target_aco_id,
                "TARGET_ACO_NAME": target_row["ACO_NAME"],
                "TARGET_STATE": target_row["STATE"],
                "TARGET_YEAR": int(target_row["YEAR"]),
                "TWIN_RANK": rank,

                "TWIN_NEXT_YEAR_SAVINGS_RATE_PCT":
                    twin_rate * 100.0,

                "OUTPERFORMING_TWIN":
                    bool(twin_rate > median_rate),

                "OUTPERFORMANCE_GAP":
                    twin_rate - median_rate,

                "OUTPERFORMANCE_GAP_PCT":
                    (twin_rate - median_rate) * 100.0,

                "TOP5_AVG_SAVINGS_RATE":
                    mean_rate,

                "TOP5_MEDIAN_SAVINGS_RATE":
                    median_rate,

                "TOP5_WEIGHTED_SAVINGS_RATE":
                    weighted_rate,

                "TOP5_BEST_SAVINGS_RATE":
                    best_rate,

                "TOP5_WORST_SAVINGS_RATE":
                    worst_rate,

                "TOP5_OUTPERFORMER_COUNT":
                    outperformer_count,

                "TOP5_OUTPERFORMER_RATE":
                    outperformer_rate,
            }
        )

        results.append(twin)


# ============================================================
# RESULT DATAFRAME
# ============================================================

result_columns = [
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

results_df = pd.DataFrame(results)

results_df = results_df[
    result_columns
]


# ============================================================
# VALIDATE TOP 5
# ============================================================

expected_rows = (
    len(scoring) * TOP_K
)

if len(results_df) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"got {len(results_df)}."
    )

rank_counts = (
    results_df
    .groupby("TARGET_ACO_ID")["TWIN_RANK"]
    .nunique()
)

if not (rank_counts == TOP_K).all():
    raise RuntimeError(
        "Some ACOs do not have exactly 5 Twin ranks."
    )

twin_counts = (
    results_df
    .groupby("TARGET_ACO_ID")["TWIN_ACO_ID"]
    .nunique()
)

if not (twin_counts == TOP_K).all():
    raise RuntimeError(
        "Duplicate Twin ACOs detected."
    )

self_matches = (
    results_df["TARGET_ACO_ID"].astype(str)
    == results_df["TWIN_ACO_ID"].astype(str)
)

if self_matches.any():
    raise RuntimeError(
        "Self-match detected."
    )


# ============================================================
# SAVE TOP 5 RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary_df = (
    results_df
    .sort_values(
        ["TARGET_ACO_ID", "TWIN_RANK"]
    )
    .groupby(
        [
            "TARGET_ACO_ID",
            "TARGET_ACO_NAME",
            "TARGET_STATE",
            "TARGET_YEAR",
        ],
        as_index=False,
    )
    .agg(
        TOP5_AVG_SAVINGS_RATE=(
            "TOP5_AVG_SAVINGS_RATE",
            "first",
        ),
        TOP5_MEDIAN_SAVINGS_RATE=(
            "TOP5_MEDIAN_SAVINGS_RATE",
            "first",
        ),
        TOP5_WEIGHTED_SAVINGS_RATE=(
            "TOP5_WEIGHTED_SAVINGS_RATE",
            "first",
        ),
        TOP5_BEST_SAVINGS_RATE=(
            "TOP5_BEST_SAVINGS_RATE",
            "first",
        ),
        TOP5_WORST_SAVINGS_RATE=(
            "TOP5_WORST_SAVINGS_RATE",
            "first",
        ),
        TOP5_OUTPERFORMER_COUNT=(
            "TOP5_OUTPERFORMER_COUNT",
            "first",
        ),
        TOP5_OUTPERFORMER_RATE=(
            "TOP5_OUTPERFORMER_RATE",
            "first",
        ),
        BEST_TWIN_SIMILARITY=(
            "SIMILARITY_SCORE",
            "max",
        ),
    )
)

for column in [
    "TOP5_AVG_SAVINGS_RATE",
    "TOP5_MEDIAN_SAVINGS_RATE",
    "TOP5_WEIGHTED_SAVINGS_RATE",
    "TOP5_BEST_SAVINGS_RATE",
    "TOP5_WORST_SAVINGS_RATE",
]:

    summary_df[
        f"{column}_PCT"
    ] = (
        summary_df[column] * 100.0
    )


summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# OPTION C — OUTPERFORMING TWINS
# ============================================================

outperformers_df = results_df[
    results_df["OUTPERFORMING_TWIN"]
].copy()

outperformers_df.to_csv(
    OUTPERFORMER_PATH,
    index=False,
)


# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH,
)

joblib.dump(
    nn_model,
    NEIGHBOR_MODEL_PATH,
)


# ============================================================
# SAVE FEATURE WEIGHTS
# ============================================================

weights_metadata = {
    "features": TWIN_FEATURES,
    "weights": {
        feature: float(weight)
        for feature, weight in zip(
            TWIN_FEATURES,
            weight_vector,
        )
    },
    "method": (
        "StandardScaler followed by "
        "feature-weighted Euclidean distance"
    ),
    "target_used_for_similarity": False,
    "target": TARGET,
}

with open(
    FEATURE_WEIGHTS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        weights_metadata,
        file,
        indent=4,
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": "NearestNeighbors",
    "method": "Weighted Euclidean Similarity",
    "top_k": TOP_K,
    "candidate_k": CANDIDATE_K,
    "historical_reference_years": "2018-2023",
    "scoring_year": 2024,
    "features": TWIN_FEATURES,
    "feature_weights": {
        feature: float(weight)
        for feature, weight in zip(
            TWIN_FEATURES,
            weight_vector,
        )
    },
    "target": TARGET,
    "target_used_for_similarity": False,
    "same_aco_excluded": True,
    "one_record_per_twin_aco": True,
    "option_c": {
        "enabled": True,
        "description": (
            "Historically outperforming Twin is defined "
            "as a Twin whose next-year savings rate is "
            "above the median of the selected Top 5 Twins."
        ),
    },
    "historical_rows": len(reference),
    "scoring_rows": len(scoring),
    "output_rows": len(results_df),
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
    )


# ============================================================
# SIMPLE TERMINAL OUTPUT
# ============================================================

print(
    f"Done. {len(scoring):,} ACOs × {TOP_K} Twins."
)

print(
    f"Saved: {OUTPUT_PATH}"
)

print(
    f"Outperformers: {len(outperformers_df):,}"
)