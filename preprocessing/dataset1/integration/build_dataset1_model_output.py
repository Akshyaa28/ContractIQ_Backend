"""
===============================================================================
CONTRACTIQ — BUILD Dataset1_Model_Output_2024.csv
===============================================================================

Purpose
-------
Merge the three 2024 model outputs into one canonical output file:
  1. Risk predictions   (models/risk_prediction/Dataset1_Scoring_2024_Risk_Predictions_Final.csv)
  2. Forecast           (models/forecasting/Dataset1_Scoring_2024_Forecast_Predictions.csv)
  3. Twin recommendations (data/processed/Dataset1_Twin_Recommendations_2024.csv)

All three cover exactly 7,166 unique ACOs for YEAR = 2024.

Output columns
--------------
Identifiers (4):
  ACO_ID, ACO_NAME, STATE, YEAR

Base features (8):
  N_AB, PREVIOUS_SAVINGS_RATE, PREVIOUS_QUALITY_SCORE,
  PREVIOUS_PERFORMANCE_GAP_PCT, EXPENDITURE_GROWTH_PCT,
  BENCHMARK_GROWTH_PCT, BENEFICIARY_GROWTH_PCT, QUALITY_CHANGE

Risk predictions (4):
  RISK_PROBABILITY, RISK_PREDICTION, RISK_LABEL, RISK_PROBABILITY_PCT

Forecast predictions (3):
  FORECASTED_NEXT_YEAR_SAVINGS_RATE,
  FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT,
  SAVINGS_FORECAST_CATEGORY

Twin recommendations (50):
  TWIN_1–5 details + aggregate stats + BEST_OUTPERFORMING_TWIN_* + TWIN_RECOMMENDATION

Output
------
  data/processed/Dataset1_Model_Output_2024.csv
===============================================================================
"""

from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_PATH  = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

RISK_PATH = (
    PROJECT_ROOT
    / "models"
    / "risk_prediction"
    / "Dataset1_Scoring_2024_Risk_Predictions_Final.csv"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "Dataset1_Scoring_2024_Forecast_Predictions.csv"
)

TWIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Twin_Recommendations_2024.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Model_Output_2024.csv"
)

EXPECTED_ROWS = 7_166
EXPECTED_YEAR = 2024


# ============================================================================
# HELPERS
# ============================================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_and_standardise(path: Path) -> pd.DataFrame:
    """Load CSV and uppercase all column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()
    df["ACO_ID"] = df["ACO_ID"].astype(str).str.strip()
    return df


# ============================================================================
# CHECK INPUTS EXIST
# ============================================================================

section("CHECKING INPUT FILES")

for label, path in [
    ("Risk predictions",       RISK_PATH),
    ("Forecast predictions",   FORECAST_PATH),
    ("Twin recommendations",   TWIN_PATH),
]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n  {path}")
    print(f"  ✓  {label}")
    print(f"       {path}")


# ============================================================================
# LOAD
# ============================================================================

section("LOADING DATA")

risk     = load_and_standardise(RISK_PATH)
forecast = load_and_standardise(FORECAST_PATH)
twin     = load_and_standardise(TWIN_PATH)

print(f"  Risk        : {len(risk):,} rows × {len(risk.columns)} columns")
print(f"  Forecast    : {len(forecast):,} rows × {len(forecast.columns)} columns")
print(f"  Twin        : {len(twin):,} rows × {len(twin.columns)} columns")


# ============================================================================
# VALIDATE EACH SOURCE
# ============================================================================

section("VALIDATING SOURCE FILES")

for label, df in [("Risk", risk), ("Forecast", forecast), ("Twin", twin)]:

    # Row count
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"{label}: expected {EXPECTED_ROWS:,} rows, got {len(df):,}."
        )

    # ACO_ID present
    if "ACO_ID" not in df.columns:
        raise ValueError(f"{label}: ACO_ID column is missing.")

    # Unique ACO_ID
    dups = df["ACO_ID"].duplicated().sum()
    if dups:
        raise ValueError(f"{label}: {dups:,} duplicate ACO_ID values found.")

    print(f"  ✓  {label}  — {len(df):,} rows, ACO_ID unique")

# YEAR = 2024 on risk file (which carries YEAR)
if "YEAR" in risk.columns:
    bad_years = risk[risk["YEAR"].astype(str).str.strip() != str(EXPECTED_YEAR)]
    if not bad_years.empty:
        raise ValueError(
            f"Risk file contains {len(bad_years):,} rows where YEAR ≠ {EXPECTED_YEAR}."
        )
    print(f"  ✓  All rows are YEAR = {EXPECTED_YEAR}")

# All three cover the same ACO set
risk_ids     = set(risk["ACO_ID"])
forecast_ids = set(forecast["ACO_ID"])
twin_ids     = set(twin["ACO_ID"])

if risk_ids != forecast_ids:
    diff = risk_ids.symmetric_difference(forecast_ids)
    raise ValueError(f"Risk vs Forecast ACO sets differ ({len(diff)} mismatches).")

if risk_ids != twin_ids:
    diff = risk_ids.symmetric_difference(twin_ids)
    raise ValueError(f"Risk vs Twin ACO sets differ ({len(diff)} mismatches).")

print(f"  ✓  All three files share the same {len(risk_ids):,} ACO_IDs")


# ============================================================================
# PREPARE RISK — keep base features + prediction columns only
# ============================================================================

section("PREPARING RISK COLUMNS")

RISK_KEEP = [
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
    "RISK_PROBABILITY",
    "RISK_PREDICTION",
    "RISK_LABEL",
    "RISK_PROBABILITY_PCT",
]

missing = [c for c in RISK_KEEP if c not in risk.columns]
if missing:
    raise ValueError(f"Risk file is missing expected columns: {missing}")

risk = risk[RISK_KEEP].copy()
print(f"  Risk columns kept: {len(RISK_KEEP)}")


# ============================================================================
# PREPARE FORECAST — drop redundant identifiers (keep only predictions)
# ============================================================================

section("PREPARING FORECAST COLUMNS")

FORECAST_PRED_COLS = [
    "ACO_ID",
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE",
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
    "SAVINGS_FORECAST_CATEGORY",
]

missing = [c for c in FORECAST_PRED_COLS if c not in forecast.columns]
if missing:
    raise ValueError(f"Forecast file is missing expected columns: {missing}")

forecast_slim = forecast[FORECAST_PRED_COLS].copy()
print(f"  Forecast prediction columns kept: {len(FORECAST_PRED_COLS) - 1}  (+ join key)")


# ============================================================================
# PREPARE TWIN — drop redundant identifiers (ACO_NAME, STATE, YEAR already in risk)
# ============================================================================

section("PREPARING TWIN COLUMNS")

# Drop columns that already exist in the risk output to avoid _x/_y suffixes
TWIN_DROP = ["ACO_NAME", "STATE", "YEAR"]
twin_slim  = twin.drop(
    columns=[c for c in TWIN_DROP if c in twin.columns]
).copy()

print(f"  Twin columns after drop: {len(twin_slim.columns)}")


# ============================================================================
# MERGE
# ============================================================================

section("MERGING")

# Step 1: risk + forecast  (7,166 × 16 + 3 prediction cols = 19)
output = risk.merge(
    forecast_slim,
    on="ACO_ID",
    how="inner",
    validate="one_to_one",
)

print(f"  After risk + forecast merge : {len(output):,} rows × {len(output.columns)} columns")

# Step 2: + twin recommendations
output = output.merge(
    twin_slim,
    on="ACO_ID",
    how="inner",
    validate="one_to_one",
)

print(f"  After + twin merge          : {len(output):,} rows × {len(output.columns)} columns")


# ============================================================================
# POST-MERGE VALIDATION
# ============================================================================

section("POST-MERGE VALIDATION")

if len(output) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Row count after merge is {len(output):,}, expected {EXPECTED_ROWS:,}."
    )

if output["ACO_ID"].duplicated().any():
    raise RuntimeError("Duplicate ACO_ID values found in merged output.")

if output["ACO_ID"].isna().any():
    raise RuntimeError("Null ACO_ID values found in merged output.")

print(f"  ✓  Row count     : {len(output):,}  (expected {EXPECTED_ROWS:,})")
print(f"  ✓  ACO_ID unique : True")
print(f"  ✓  No null ACO_ID")
print(f"  Total columns   : {len(output.columns)}")


# ============================================================================
# SORT — canonical order: ACO_ID ascending
# ============================================================================

output = output.sort_values("ACO_ID").reset_index(drop=True)


# ============================================================================
# SAVE
# ============================================================================

section("SAVING OUTPUT")

output.to_csv(OUTPUT_PATH, index=False)

print(f"  Saved : {OUTPUT_PATH}")
print(f"  Rows  : {len(output):,}")
print(f"  Cols  : {len(output.columns)}")
print()
print("  Columns:")
for i, col in enumerate(output.columns, 1):
    print(f"    {i:3d}. {col}")

print()
print("=" * 70)
print("BUILD COMPLETE")
print("=" * 70)
