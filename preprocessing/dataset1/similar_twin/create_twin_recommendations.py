from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Top5_Similar_Twins_2024.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Twin_Recommendations_2024.csv"
)


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------

required_columns = [
    "TARGET_ACO_ID",
    "TARGET_ACO_NAME",
    "TARGET_STATE",
    "TARGET_YEAR",
    "TWIN_RANK",
    "TWIN_ACO_ID",
    "TWIN_ACO_NAME",
    "TWIN_STATE",
    "TWIN_YEAR",
    "SIMILARITY_SCORE",
    "TWIN_NEXT_YEAR_SAVINGS_RATE_PCT",
    "OUTPERFORMING_TWIN",
    "OUTPERFORMANCE_GAP_PCT",
    "TOP5_AVG_SAVINGS_RATE",
    "TOP5_MEDIAN_SAVINGS_RATE",
    "TOP5_WEIGHTED_SAVINGS_RATE",
    "TOP5_BEST_SAVINGS_RATE",
    "TOP5_WORST_SAVINGS_RATE",
    "TOP5_OUTPERFORMER_COUNT",
    "TOP5_OUTPERFORMER_RATE",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ---------------------------------------------------------------------------
# SORT
# ---------------------------------------------------------------------------

df = df.sort_values(
    [
        "TARGET_ACO_ID",
        "TWIN_RANK",
    ]
).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CREATE TOP-5 TWIN COLUMNS
# ---------------------------------------------------------------------------

recommendations = []

for aco_id, group in df.groupby(
    "TARGET_ACO_ID",
    sort=False,
):

    group = group.sort_values("TWIN_RANK")

    if len(group) != 5:
        raise ValueError(
            f"{aco_id} does not have exactly 5 Twins."
        )

    first = group.iloc[0]

    row = {
        "ACO_ID": first["TARGET_ACO_ID"],
        "ACO_NAME": first["TARGET_ACO_NAME"],
        "STATE": first["TARGET_STATE"],
        "YEAR": first["TARGET_YEAR"],
    }

    for _, twin in group.iterrows():

        rank = int(twin["TWIN_RANK"])

        row[f"TWIN_{rank}_ACO_ID"] = twin["TWIN_ACO_ID"]
        row[f"TWIN_{rank}_ACO_NAME"] = twin["TWIN_ACO_NAME"]
        row[f"TWIN_{rank}_STATE"] = twin["TWIN_STATE"]
        row[f"TWIN_{rank}_YEAR"] = twin["TWIN_YEAR"]

        row[f"TWIN_{rank}_SIMILARITY_SCORE"] = (
            twin["SIMILARITY_SCORE"]
        )

        row[f"TWIN_{rank}_SAVINGS_RATE_PCT"] = (
            twin["TWIN_NEXT_YEAR_SAVINGS_RATE_PCT"]
        )

        row[f"TWIN_{rank}_OUTPERFORMING"] = (
            twin["OUTPERFORMING_TWIN"]
        )

        row[f"TWIN_{rank}_OUTPERFORMANCE_GAP_PCT"] = (
            twin["OUTPERFORMANCE_GAP_PCT"]
        )

    row["TOP5_AVG_SAVINGS_RATE_PCT"] = (
        first["TOP5_AVG_SAVINGS_RATE"] * 100
    )

    row["TOP5_MEDIAN_SAVINGS_RATE_PCT"] = (
        first["TOP5_MEDIAN_SAVINGS_RATE"] * 100
    )

    row["TOP5_WEIGHTED_SAVINGS_RATE_PCT"] = (
        first["TOP5_WEIGHTED_SAVINGS_RATE"] * 100
    )

    row["TOP5_BEST_SAVINGS_RATE_PCT"] = (
        first["TOP5_BEST_SAVINGS_RATE"] * 100
    )

    row["TOP5_WORST_SAVINGS_RATE_PCT"] = (
        first["TOP5_WORST_SAVINGS_RATE"] * 100
    )

    row["OUTPERFORMING_TWIN_COUNT"] = (
        first["TOP5_OUTPERFORMER_COUNT"]
    )

    row["OUTPERFORMING_TWIN_RATE"] = (
        first["TOP5_OUTPERFORMER_RATE"]
    )

    # -----------------------------------------------------------------------
    # BEST OUTPERFORMING TWIN
    # -----------------------------------------------------------------------

    outperformers = group[
        group["OUTPERFORMING_TWIN"] == True
    ]

    if not outperformers.empty:

        best = outperformers.sort_values(
            [
                "TWIN_NEXT_YEAR_SAVINGS_RATE_PCT",
                "SIMILARITY_SCORE",
            ],
            ascending=False,
        ).iloc[0]

        row["BEST_OUTPERFORMING_TWIN_ID"] = (
            best["TWIN_ACO_ID"]
        )

        row["BEST_OUTPERFORMING_TWIN_NAME"] = (
            best["TWIN_ACO_NAME"]
        )

        row["BEST_OUTPERFORMING_TWIN_YEAR"] = (
            best["TWIN_YEAR"]
        )

        row["BEST_OUTPERFORMING_TWIN_SIMILARITY"] = (
            best["SIMILARITY_SCORE"]
        )

        row["BEST_OUTPERFORMING_TWIN_SAVINGS_RATE_PCT"] = (
            best["TWIN_NEXT_YEAR_SAVINGS_RATE_PCT"]
        )

        row["BEST_OUTPERFORMING_TWIN_GAP_PCT"] = (
            best["OUTPERFORMANCE_GAP_PCT"]
        )

    else:

        row["BEST_OUTPERFORMING_TWIN_ID"] = None
        row["BEST_OUTPERFORMING_TWIN_NAME"] = None
        row["BEST_OUTPERFORMING_TWIN_YEAR"] = None
        row["BEST_OUTPERFORMING_TWIN_SIMILARITY"] = None
        row["BEST_OUTPERFORMING_TWIN_SAVINGS_RATE_PCT"] = None
        row["BEST_OUTPERFORMING_TWIN_GAP_PCT"] = None

    # -----------------------------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------------------------

    if not outperformers.empty:
        row["TWIN_RECOMMENDATION"] = (
            "Use outperforming Twin as benchmark"
        )
    else:
        row["TWIN_RECOMMENDATION"] = (
            "Use TOP-5 Twin benchmark"
        )

    recommendations.append(row)


# ---------------------------------------------------------------------------
# CREATE OUTPUT
# ---------------------------------------------------------------------------

recommendations_df = pd.DataFrame(
    recommendations
)


# ---------------------------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------------------------

expected_acos = df["TARGET_ACO_ID"].nunique()
actual_acos = len(recommendations_df)

if actual_acos != expected_acos:
    raise RuntimeError(
        f"Expected {expected_acos} ACOs, "
        f"but generated {actual_acos}."
    )

if recommendations_df["ACO_ID"].duplicated().any():
    raise RuntimeError(
        "Duplicate ACO_ID found in recommendation output."
    )


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

recommendations_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    f"Done. {actual_acos:,} ACO recommendations created."
)

print(
    f"Saved: {OUTPUT_PATH}"
)