from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

RECOMMENDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Twin_Recommendations_2024.csv"
)

TOP5_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Top5_Similar_Twins_2024.csv"
)


# ---------------------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",

    "TWIN_1_ACO_ID",
    "TWIN_1_ACO_NAME",
    "TWIN_1_SIMILARITY_SCORE",
    "TWIN_1_SAVINGS_RATE_PCT",

    "TWIN_2_ACO_ID",
    "TWIN_2_ACO_NAME",
    "TWIN_2_SIMILARITY_SCORE",
    "TWIN_2_SAVINGS_RATE_PCT",

    "TWIN_3_ACO_ID",
    "TWIN_3_ACO_NAME",
    "TWIN_3_SIMILARITY_SCORE",
    "TWIN_3_SAVINGS_RATE_PCT",

    "TWIN_4_ACO_ID",
    "TWIN_4_ACO_NAME",
    "TWIN_4_SIMILARITY_SCORE",
    "TWIN_4_SAVINGS_RATE_PCT",

    "TWIN_5_ACO_ID",
    "TWIN_5_ACO_NAME",
    "TWIN_5_SIMILARITY_SCORE",
    "TWIN_5_SAVINGS_RATE_PCT",

    "TOP5_AVG_SAVINGS_RATE_PCT",
    "TOP5_MEDIAN_SAVINGS_RATE_PCT",
    "TOP5_WEIGHTED_SAVINGS_RATE_PCT",
    "TOP5_BEST_SAVINGS_RATE_PCT",
    "TOP5_WORST_SAVINGS_RATE_PCT",

    "OUTPERFORMING_TWIN_COUNT",
    "OUTPERFORMING_TWIN_RATE",

    "BEST_OUTPERFORMING_TWIN_ID",
    "BEST_OUTPERFORMING_TWIN_NAME",
    "BEST_OUTPERFORMING_TWIN_SIMILARITY",
    "BEST_OUTPERFORMING_TWIN_SAVINGS_RATE_PCT",
    "BEST_OUTPERFORMING_TWIN_GAP_PCT",

    "TWIN_RECOMMENDATION",
]


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

if not RECOMMENDATION_PATH.exists():
    raise FileNotFoundError(
        f"Recommendation file not found:\n{RECOMMENDATION_PATH}"
    )

if not TOP5_PATH.exists():
    raise FileNotFoundError(
        f"Top-5 Twin file not found:\n{TOP5_PATH}"
    )

recommendations = pd.read_csv(RECOMMENDATION_PATH)
top5 = pd.read_csv(TOP5_PATH)


# ---------------------------------------------------------------------------
# COLUMN VALIDATION
# ---------------------------------------------------------------------------

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in recommendations.columns
]

if missing_columns:
    raise ValueError(
        "Missing required recommendation columns:\n"
        + "\n".join(missing_columns)
    )


# ---------------------------------------------------------------------------
# BASIC ROW VALIDATION
# ---------------------------------------------------------------------------

if len(recommendations) != 7166:
    raise ValueError(
        f"Expected 7,166 recommendation rows, "
        f"found {len(recommendations):,}."
    )

if recommendations["ACO_ID"].nunique() != 7166:
    raise ValueError(
        "Recommendation output does not contain "
        "exactly one row per 2024 ACO."
    )

if recommendations["ACO_ID"].duplicated().any():
    raise ValueError(
        "Duplicate ACO_ID values found."
    )


# ---------------------------------------------------------------------------
# YEAR VALIDATION
# ---------------------------------------------------------------------------

if not (recommendations["YEAR"] == 2024).all():
    raise ValueError(
        "Recommendation output contains non-2024 rows."
    )


# ---------------------------------------------------------------------------
# MISSING VALUE VALIDATION
# ---------------------------------------------------------------------------

critical_columns = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
]

for rank in range(1, 6):
    critical_columns.extend(
        [
            f"TWIN_{rank}_ACO_ID",
            f"TWIN_{rank}_SIMILARITY_SCORE",
            f"TWIN_{rank}_SAVINGS_RATE_PCT",
        ]
    )

critical_missing = (
    recommendations[critical_columns]
    .isna()
    .sum()
    .sum()
)

if critical_missing != 0:
    raise ValueError(
        f"Critical missing values found: {critical_missing}"
    )


# ---------------------------------------------------------------------------
# TWIN VALIDATION
# ---------------------------------------------------------------------------

for rank in range(1, 6):

    twin_id = f"TWIN_{rank}_ACO_ID"
    similarity = f"TWIN_{rank}_SIMILARITY_SCORE"

    if (
        recommendations["ACO_ID"].astype(str)
        == recommendations[twin_id].astype(str)
    ).any():

        raise ValueError(
            f"Self-match detected in {twin_id}."
        )

    if not np.isfinite(
        pd.to_numeric(
            recommendations[similarity],
            errors="coerce",
        )
    ).all():

        raise ValueError(
            f"Invalid similarity values in {similarity}."
        )

    if (
        pd.to_numeric(
            recommendations[similarity],
            errors="coerce",
        )
        < 0
    ).any():

        raise ValueError(
            f"Negative similarity values in {similarity}."
        )


# ---------------------------------------------------------------------------
# UNIQUE TOP-5 TWIN VALIDATION
# ---------------------------------------------------------------------------

twin_columns = [
    f"TWIN_{rank}_ACO_ID"
    for rank in range(1, 6)
]

for _, row in recommendations.iterrows():

    twins = [
        str(row[column])
        for column in twin_columns
    ]

    if len(set(twins)) != 5:
        raise ValueError(
            f"Duplicate Twin ACOs found for "
            f"{row['ACO_ID']}."
        )


# ---------------------------------------------------------------------------
# TOP-5 ORDER VALIDATION
# ---------------------------------------------------------------------------

for _, row in recommendations.iterrows():

    similarities = [
        float(
            row[
                f"TWIN_{rank}_SIMILARITY_SCORE"
            ]
        )
        for rank in range(1, 6)
    ]

    if similarities != sorted(
        similarities,
        reverse=True,
    ):
        raise ValueError(
            f"Top-5 Twin similarity order is invalid "
            f"for {row['ACO_ID']}."
        )


# ---------------------------------------------------------------------------
# OPTION C VALIDATION
# ---------------------------------------------------------------------------

outperformer_count = pd.to_numeric(
    recommendations["OUTPERFORMING_TWIN_COUNT"],
    errors="coerce",
)

outperformer_rate = pd.to_numeric(
    recommendations["OUTPERFORMING_TWIN_RATE"],
    errors="coerce",
)

if outperformer_count.isna().any():
    raise ValueError(
        "Invalid OUTPERFORMING_TWIN_COUNT values."
    )

if (
    (outperformer_count < 0)
    | (outperformer_count > 5)
).any():

    raise ValueError(
        "OUTPERFORMING_TWIN_COUNT must be between 0 and 5."
    )

if outperformer_rate.isna().any():
    raise ValueError(
        "Invalid OUTPERFORMING_TWIN_RATE values."
    )

if (
    (outperformer_rate < 0)
    | (outperformer_rate > 1)
).any():

    raise ValueError(
        "OUTPERFORMING_TWIN_RATE must be between 0 and 1."
    )


expected_rate = (
    outperformer_count / 5
)

if not np.allclose(
    outperformer_rate,
    expected_rate,
):

    raise ValueError(
        "OUTPERFORMING_TWIN_RATE is inconsistent "
        "with OUTPERFORMING_TWIN_COUNT."
    )


# ---------------------------------------------------------------------------
# BEST OUTPERFORMING TWIN VALIDATION
# ---------------------------------------------------------------------------

for _, row in recommendations.iterrows():

    count = int(
        row["OUTPERFORMING_TWIN_COUNT"]
    )

    best_id = row[
        "BEST_OUTPERFORMING_TWIN_ID"
    ]

    best_name = row[
        "BEST_OUTPERFORMING_TWIN_NAME"
    ]

    best_savings = row[
        "BEST_OUTPERFORMING_TWIN_SAVINGS_RATE_PCT"
    ]

    if count > 0:

        if pd.isna(best_id) or pd.isna(best_name):
            raise ValueError(
                f"Missing best outperforming Twin for "
                f"{row['ACO_ID']}."
            )

        if pd.isna(best_savings):
            raise ValueError(
                f"Missing best outperforming Twin savings "
                f"for {row['ACO_ID']}."
            )

    else:

        if not pd.isna(best_id):
            raise ValueError(
                f"Best outperforming Twin exists even though "
                f"count is zero for {row['ACO_ID']}."
            )


# ---------------------------------------------------------------------------
# RECOMMENDATION VALIDATION
# ---------------------------------------------------------------------------

valid_recommendations = {
    "Use outperforming Twin as benchmark",
    "Use TOP-5 Twin benchmark",
}

invalid_recommendations = (
    ~recommendations["TWIN_RECOMMENDATION"].isin(
        valid_recommendations
    )
)

if invalid_recommendations.any():
    raise ValueError(
        "Invalid TWIN_RECOMMENDATION values found."
    )


for _, row in recommendations.iterrows():

    count = int(
        row["OUTPERFORMING_TWIN_COUNT"]
    )

    recommendation = row[
        "TWIN_RECOMMENDATION"
    ]

    if count > 0:
        expected = (
            "Use outperforming Twin as benchmark"
        )
    else:
        expected = (
            "Use TOP-5 Twin benchmark"
        )

    if recommendation != expected:
        raise ValueError(
            f"Invalid recommendation logic for "
            f"{row['ACO_ID']}."
        )


# ---------------------------------------------------------------------------
# TOP-5 SOURCE CONSISTENCY
# ---------------------------------------------------------------------------

top5_required = [
    "TARGET_ACO_ID",
    "TWIN_RANK",
    "TWIN_ACO_ID",
]

missing_top5 = [
    column
    for column in top5_required
    if column not in top5.columns
]

if missing_top5:
    raise ValueError(
        "Top-5 source is missing required columns:\n"
        + "\n".join(missing_top5)
    )


expected_top5_rows = 7166 * 5

if len(top5) != expected_top5_rows:
    raise ValueError(
        f"Expected {expected_top5_rows:,} Top-5 rows, "
        f"found {len(top5):,}."
    )


# ---------------------------------------------------------------------------
# LEAKAGE VALIDATION
# ---------------------------------------------------------------------------

for column in recommendations.columns:

    column_upper = column.upper()

    if column_upper in {
        "NEXT_YEAR_SAVINGS_RATE",
        "NEXT_YEAR_SAVINGS_RATE_PCT",
    }:

        raise ValueError(
            "Future target found in recommendation output: "
            f"{column}"
        )


# ---------------------------------------------------------------------------
# FINAL OUTPUT CHECK
# ---------------------------------------------------------------------------

if recommendations.empty:
    raise ValueError(
        "Recommendation dataset is empty."
    )


print("Twin recommendations validation PASSED.")
print(f"2024 ACOs: {len(recommendations):,}")
print(f"Recommendation rows: {len(recommendations):,}")
print(
    f"Duplicate ACOs: "
    f"{recommendations['ACO_ID'].duplicated().sum()}"
)
print("Invalid Twins: 0")
print("Self-matches: 0")
print(f"Missing critical values: {critical_missing}")
print("Leakage checks: PASSED")
print("Consistency checks: PASSED")
print("Status: SUCCESS")