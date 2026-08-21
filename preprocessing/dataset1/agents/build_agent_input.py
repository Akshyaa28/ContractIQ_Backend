from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AGENT_INPUT_DIR = PROJECT_ROOT / "data" / "agent_inputs"

MODEL_OUTPUT_PATH = (
    PROCESSED_DIR / "Dataset1_Model_Output_2024.csv"
)

OUTPUT_PATH = (
    AGENT_INPUT_DIR / "Dataset1_Agent_Input_2024.csv"
)


# ============================================================================
# CHECK INPUT
# ============================================================================

if not MODEL_OUTPUT_PATH.exists():
    raise FileNotFoundError(
        f"Model output not found:\n{MODEL_OUTPUT_PATH}"
    )


# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

AGENT_INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv(
    MODEL_OUTPUT_PATH
)


# ============================================================================
# STANDARDIZE COLUMN NAMES
# ============================================================================

df.columns = (
    df.columns
    .str.strip()
    .str.upper()
)


# ============================================================================
# REQUIRED COLUMN
# ============================================================================

if "ACO_ID" not in df.columns:
    raise ValueError(
        "Dataset1_Model_Output_2024.csv must contain ACO_ID."
    )


# ============================================================================
# STANDARDIZE ACO ID
# ============================================================================

df["ACO_ID"] = (
    df["ACO_ID"]
    .astype(str)
    .str.strip()
)


# ============================================================================
# VALIDATE ACO ID
# ============================================================================

if df["ACO_ID"].isna().any():
    raise ValueError(
        "Agent input contains missing ACO_ID values."
    )

if (df["ACO_ID"] == "").any():
    raise ValueError(
        "Agent input contains empty ACO_ID values."
    )

if df["ACO_ID"].duplicated().any():
    raise ValueError(
        "Dataset1_Model_Output_2024.csv must contain "
        "exactly one row per ACO."
    )


# ============================================================================
# VALIDATE YEAR
# ============================================================================

if "YEAR" in df.columns:

    df["YEAR"] = pd.to_numeric(
        df["YEAR"],
        errors="coerce",
    )

    if df["YEAR"].isna().any():
        raise ValueError(
            "Agent input contains invalid YEAR values."
        )

    df["YEAR"] = df["YEAR"].astype(int)

    if not (df["YEAR"] == 2024).all():
        raise ValueError(
            "Agent input must contain only 2024 records."
        )


# ============================================================================
# VALIDATE MISSING VALUES
# ============================================================================

critical_columns = [
    "ACO_ID",
]

if "ACO_NAME" in df.columns:
    critical_columns.append("ACO_NAME")

if "STATE" in df.columns:
    critical_columns.append("STATE")

missing_values = (
    df[critical_columns]
    .isna()
    .sum()
)

if missing_values.any():
    invalid = missing_values[
        missing_values > 0
    ]

    raise ValueError(
        "Critical agent input columns contain missing values:\n"
        + invalid.to_string()
    )


# ============================================================================
# SORT
# ============================================================================

df = df.sort_values(
    "ACO_ID"
).reset_index(
    drop=True
)


# ============================================================================
# SAVE
# ============================================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================================
# RESULT
# ============================================================================

print(
    f"Done. {len(df):,} ACO agent inputs created."
)

print(
    f"Saved: {OUTPUT_PATH}"
)