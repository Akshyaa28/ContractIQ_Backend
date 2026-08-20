"""
===============================================================================
CONTRACTIQ — VALIDATE Dataset1_Model_Output_2024.csv
===============================================================================

Nine validation checks
----------------------
 1. Exactly 7,166 rows
 2. One row per ACO  (ACO_ID is the unique primary key)
 3. 2024 only        (YEAR column contains only 2024)
 4. Forecast prediction present   (value, pct, category — no nulls)
 5. Twin recommendation present   (TWIN_1–5 columns, TWIN_RECOMMENDATION — no nulls)
 6. No self-Twin                  (no TWIN_N_ACO_ID == ACO_ID in any rank)
 7. No missing critical values    (identifiers + base features + all predictions)
 8. No leakage                    (NEXT_YEAR_SAVINGS_RATE / NEXT_YEAR_RISK absent)
 9. Forecast and Twin coverage match (same 7,166 ACO_IDs in every prediction block)

Plus a final column audit: print every column name so you can eyeball the
complete schema.

Exit code
---------
  0  — all checks passed
  1  — one or more checks failed
===============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_PATH  = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Dataset1_Model_Output_2024.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPECTED_ROWS = 7_166
EXPECTED_YEAR = 2024

# Columns that must exist and must have zero nulls
CRITICAL_COLUMNS = [
    # Identifiers
    "ACO_ID",
    "ACO_NAME",
    "STATE",
    "YEAR",
    # Base features
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
    # Risk predictions
    "RISK_PROBABILITY",
    "RISK_PREDICTION",
    "RISK_LABEL",
    "RISK_PROBABILITY_PCT",
    # Forecast predictions
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE",
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
    "SAVINGS_FORECAST_CATEGORY",
    # Twin rank 1 (must always exist — best match)
    "TWIN_1_ACO_ID",
    "TWIN_1_SIMILARITY_SCORE",
    "TWIN_1_SAVINGS_RATE_PCT",
    "TWIN_1_OUTPERFORMING",
    # Twin recommendation label
    "TWIN_RECOMMENDATION",
]

# Columns that must NOT exist (leakage guard).
# These are EXACT column names (after uppercasing) that would indicate
# a future outcome has leaked into the feature set.
# Prediction-derived columns such as FORECASTED_NEXT_YEAR_SAVINGS_RATE
# or RISK_LABEL are legitimate outputs and are NOT in this list.
FORBIDDEN_COLUMNS = [
    "NEXT_YEAR_SAVINGS_RATE",   # raw future target — never a feature
    "NEXT_YEAR_RISK",           # raw future target — never a feature
    "ACTUAL_NEXT_YEAR_SAVINGS_RATE",
    "ACTUAL_NEXT_YEAR_RISK",
    "TARGET_NEXT_YEAR_SAVINGS_RATE",
    "TARGET_NEXT_YEAR_RISK",
]

# All five twin rank ID columns
TWIN_RANK_ID_COLS = [
    "TWIN_1_ACO_ID",
    "TWIN_2_ACO_ID",
    "TWIN_3_ACO_ID",
    "TWIN_4_ACO_ID",
    "TWIN_5_ACO_ID",
]

# All five twin rank core columns (must be present and non-null)
TWIN_RANK_CORE_COLS = []
for rank in range(1, 6):
    TWIN_RANK_CORE_COLS += [
        f"TWIN_{rank}_ACO_ID",
        f"TWIN_{rank}_SIMILARITY_SCORE",
        f"TWIN_{rank}_SAVINGS_RATE_PCT",
        f"TWIN_{rank}_OUTPERFORMING",
    ]


# ============================================================================
# HELPERS
# ============================================================================

PASSED = []
FAILED = []


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_pass(check_id: int, description: str, detail: str = "") -> None:
    msg = f"  [PASS] Check {check_id}: {description}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    PASSED.append(check_id)


def check_fail(check_id: int, description: str, detail: str = "") -> None:
    msg = f"  [FAIL] Check {check_id}: {description}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    FAILED.append(check_id)


# ============================================================================
# LOAD
# ============================================================================

section("LOADING FILE")

if not OUTPUT_PATH.exists():
    print(f"  [FATAL] File not found:\n  {OUTPUT_PATH}")
    sys.exit(1)

df = pd.read_csv(OUTPUT_PATH)
df.columns = df.columns.str.strip().str.upper()

print(f"  Loaded : {OUTPUT_PATH.name}")
print(f"  Rows   : {len(df):,}")
print(f"  Columns: {len(df.columns)}")


# ============================================================================
# CHECK 1 — Exactly 7,166 rows
# ============================================================================

section("CHECK 1 — Exactly 7,166 rows")

if len(df) == EXPECTED_ROWS:
    check_pass(1, f"Row count = {EXPECTED_ROWS:,}", f"Actual: {len(df):,}")
else:
    check_fail(1, f"Row count ≠ {EXPECTED_ROWS:,}", f"Actual: {len(df):,}")


# ============================================================================
# CHECK 2 — One row per ACO (ACO_ID is unique primary key)
# ============================================================================

section("CHECK 2 — One row per ACO (ACO_ID unique)")

if "ACO_ID" not in df.columns:
    check_fail(2, "ACO_ID column missing")
else:
    dup_count = int(df["ACO_ID"].duplicated().sum())
    null_count = int(df["ACO_ID"].isna().sum())
    unique_count = df["ACO_ID"].nunique()

    if dup_count == 0 and null_count == 0:
        check_pass(2, "ACO_ID is unique and non-null",
                   f"{unique_count:,} unique ACOs")
    else:
        detail = f"Duplicates: {dup_count}  |  Nulls: {null_count}  |  Unique: {unique_count:,}"
        check_fail(2, "ACO_ID has duplicates or nulls", detail)


# ============================================================================
# CHECK 3 — 2024 only (YEAR column)
# ============================================================================

section("CHECK 3 — YEAR = 2024 only")

if "YEAR" not in df.columns:
    check_fail(3, "YEAR column missing")
else:
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    wrong_years = df[df["YEAR"] != EXPECTED_YEAR]
    null_years  = df["YEAR"].isna().sum()

    if len(wrong_years) == 0 and null_years == 0:
        check_pass(3, f"All rows have YEAR = {EXPECTED_YEAR}",
                   f"{len(df):,} rows checked")
    else:
        detail = (
            f"Rows with YEAR ≠ {EXPECTED_YEAR}: {len(wrong_years):,}  |  "
            f"Null YEARs: {null_years:,}"
        )
        if not wrong_years.empty:
            other_vals = wrong_years["YEAR"].unique().tolist()
            detail += f"  |  Other values: {other_vals}"
        check_fail(3, f"YEAR column contains values ≠ {EXPECTED_YEAR}", detail)


# ============================================================================
# CHECK 4 — Forecast prediction present and non-null
# ============================================================================

section("CHECK 4 — Forecast prediction present (no nulls)")

FORECAST_REQUIRED = [
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE",
    "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
    "SAVINGS_FORECAST_CATEGORY",
]

missing_forecast_cols = [c for c in FORECAST_REQUIRED if c not in df.columns]

if missing_forecast_cols:
    check_fail(4, "Forecast columns missing", f"Missing: {missing_forecast_cols}")
else:
    null_counts = {
        col: int(df[col].isna().sum())
        for col in FORECAST_REQUIRED
    }
    any_nulls = any(v > 0 for v in null_counts.values())

    if not any_nulls:
        # Also check categories are valid
        valid_cats = {"HIGH SAVINGS", "MODERATE SAVINGS", "LOW SAVINGS"}
        actual_cats = set(df["SAVINGS_FORECAST_CATEGORY"].unique())
        invalid_cats = actual_cats - valid_cats

        cat_counts = df["SAVINGS_FORECAST_CATEGORY"].value_counts().to_dict()
        detail = "  |  ".join(f"{k}: {v:,}" for k, v in cat_counts.items())

        if not invalid_cats:
            check_pass(4, "All forecast columns present and non-null", detail)
        else:
            check_fail(4, f"Invalid SAVINGS_FORECAST_CATEGORY values: {invalid_cats}",
                       detail)
    else:
        null_detail = "  |  ".join(
            f"{col}: {n} nulls" for col, n in null_counts.items() if n > 0
        )
        check_fail(4, "Forecast columns contain nulls", null_detail)


# ============================================================================
# CHECK 5 — Twin recommendation present (TWIN_1–5 core cols + label, no nulls)
# ============================================================================

section("CHECK 5 — Twin recommendation present (no nulls in core twin columns)")

missing_twin_cols = [c for c in TWIN_RANK_CORE_COLS if c not in df.columns]

if missing_twin_cols:
    check_fail(5, "Twin rank columns missing", f"Missing: {missing_twin_cols}")
else:
    # Check TWIN_RECOMMENDATION
    if "TWIN_RECOMMENDATION" not in df.columns:
        check_fail(5, "TWIN_RECOMMENDATION column missing")
    else:
        cols_to_check = TWIN_RANK_CORE_COLS + ["TWIN_RECOMMENDATION"]
        null_counts = {
            col: int(df[col].isna().sum())
            for col in cols_to_check
        }
        any_nulls = any(v > 0 for v in null_counts.values())

        if not any_nulls:
            rec_counts = df["TWIN_RECOMMENDATION"].value_counts().to_dict()
            detail = "  |  ".join(f"{k}: {v:,}" for k, v in rec_counts.items())
            check_pass(5, "All twin rank core columns present and non-null", detail)
        else:
            null_detail = "  |  ".join(
                f"{col}: {n} nulls" for col, n in null_counts.items() if n > 0
            )
            check_fail(5, "Twin columns contain nulls", null_detail)


# ============================================================================
# CHECK 6 — No self-Twin (TWIN_N_ACO_ID ≠ ACO_ID for all ranks)
# ============================================================================

section("CHECK 6 — No self-Twin (no ACO matched to itself)")

present_twin_id_cols = [c for c in TWIN_RANK_ID_COLS if c in df.columns]

if not present_twin_id_cols:
    check_fail(6, "No TWIN_N_ACO_ID columns found to check")
else:
    self_twin_found = False
    self_twin_details = []

    for col in present_twin_id_cols:
        self_mask = (
            df[col].astype(str).str.strip()
            == df["ACO_ID"].astype(str).str.strip()
        )
        count = int(self_mask.sum())
        if count > 0:
            self_twin_found = True
            self_twin_details.append(f"{col}: {count} self-matches")

    if not self_twin_found:
        check_pass(6, "No ACO is matched to itself in any twin rank",
                   f"Checked {len(present_twin_id_cols)} rank columns")
    else:
        check_fail(6, "Self-twin detected", "  |  ".join(self_twin_details))


# ============================================================================
# CHECK 7 — No missing critical values
# ============================================================================

section("CHECK 7 — No missing critical values in key columns")

missing_cols = [c for c in CRITICAL_COLUMNS if c not in df.columns]

if missing_cols:
    check_fail(7, "Critical columns are absent from the file",
               f"Missing: {missing_cols}")
else:
    null_report = {}
    for col in CRITICAL_COLUMNS:
        n = int(df[col].isna().sum())
        if n > 0:
            null_report[col] = n

    if not null_report:
        check_pass(7, f"All {len(CRITICAL_COLUMNS)} critical columns present and non-null")
    else:
        detail = "  |  ".join(f"{col}: {n}" for col, n in null_report.items())
        check_fail(7, f"{len(null_report)} critical columns have nulls", detail)


# ============================================================================
# CHECK 8 — No leakage (forbidden future-outcome columns absent)
# ============================================================================

section("CHECK 8 — No leakage (future target columns absent)")

upper_cols = [c.upper() for c in df.columns]

found_forbidden = []
for forbidden in FORBIDDEN_COLUMNS:
    # Exact column-name match only (avoids false positives on
    # prediction columns whose names legitimately contain these strings)
    if forbidden in upper_cols:
        found_forbidden.append(forbidden)

if not found_forbidden:
    check_pass(8, "No forbidden future-outcome columns detected",
               f"Checked patterns: {FORBIDDEN_COLUMNS}")
else:
    check_fail(8, "Leakage: forbidden columns found",
               f"Found: {found_forbidden}")


# ============================================================================
# CHECK 9 — Forecast and Twin coverage match (same 7,166 ACOs everywhere)
# ============================================================================

section("CHECK 9 — Forecast and Twin coverage match")

if "ACO_ID" not in df.columns:
    check_fail(9, "ACO_ID column missing — cannot check coverage")
else:
    all_ids = set(df["ACO_ID"].astype(str).str.strip())
    total   = len(all_ids)

    # Every row that has a forecast must also have twin rank 1
    # (already enforced by the merge, but double-check via non-null counts)
    issues = []

    fc_non_null = int(df["FORECASTED_NEXT_YEAR_SAVINGS_RATE"].notna().sum()) \
        if "FORECASTED_NEXT_YEAR_SAVINGS_RATE" in df.columns else None

    twin1_non_null = int(df["TWIN_1_ACO_ID"].notna().sum()) \
        if "TWIN_1_ACO_ID" in df.columns else None

    if fc_non_null is not None and fc_non_null != EXPECTED_ROWS:
        issues.append(f"Forecast non-null rows: {fc_non_null:,} (expected {EXPECTED_ROWS:,})")

    if twin1_non_null is not None and twin1_non_null != EXPECTED_ROWS:
        issues.append(f"TWIN_1_ACO_ID non-null rows: {twin1_non_null:,} (expected {EXPECTED_ROWS:,})")

    if not issues:
        detail = (
            f"All {total:,} ACOs have both forecast and twin_1 values  |  "
            f"FC non-null: {fc_non_null:,}  |  TWIN_1 non-null: {twin1_non_null:,}"
        )
        check_pass(9, "Forecast and Twin coverage match exactly", detail)
    else:
        check_fail(9, "Coverage mismatch between forecast and twin",
                   "  |  ".join(issues))


# ============================================================================
# COLUMN AUDIT — print all 73 columns
# ============================================================================

section("COLUMN AUDIT — Final schema")

print(f"\n  Total columns: {len(df.columns)}\n")
BEST_OUTPERFORMING_TWIN_COLS = {
    "BEST_OUTPERFORMING_TWIN_ID",
    "BEST_OUTPERFORMING_TWIN_NAME",
    "BEST_OUTPERFORMING_TWIN_YEAR",
    "BEST_OUTPERFORMING_TWIN_SIMILARITY",
    "BEST_OUTPERFORMING_TWIN_SAVINGS_RATE_PCT",
    "BEST_OUTPERFORMING_TWIN_GAP_PCT",
}

for i, col in enumerate(df.columns, 1):
    dtype = str(df[col].dtype)
    null_count = int(df[col].isna().sum())
    if null_count > 0:
        if col in BEST_OUTPERFORMING_TWIN_COLS:
            null_flag = f"  ← {null_count} NULLS (expected — ACOs with no outperforming twin)"
        else:
            null_flag = f"  ← {null_count} NULLS"
    else:
        null_flag = ""
    print(f"  {i:3d}. {col:<45} [{dtype}]{null_flag}")


# ============================================================================
# SAMPLE ROWS
# ============================================================================

section("SAMPLE ROWS (first 3)")

display_cols = [
    "ACO_ID", "ACO_NAME", "STATE", "YEAR",
    "RISK_LABEL", "RISK_PROBABILITY_PCT",
    "SAVINGS_FORECAST_CATEGORY", "FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT",
    "TWIN_1_ACO_ID", "TWIN_1_SIMILARITY_SCORE", "TWIN_1_OUTPERFORMING",
    "TWIN_RECOMMENDATION",
]
available = [c for c in display_cols if c in df.columns]
print(df[available].head(3).to_string(index=False))


# ============================================================================
# SUMMARY
# ============================================================================

section("VALIDATION SUMMARY")

total_checks = len(PASSED) + len(FAILED)
print(f"\n  Total checks : {total_checks}")
print(f"  Passed       : {len(PASSED)}  {PASSED}")
print(f"  Failed       : {len(FAILED)}  {FAILED if FAILED else '—'}")

if not FAILED:
    print()
    print("  ✓  ALL CHECKS PASSED")
    print("  Dataset1_Model_Output_2024.csv is valid.")
    print("  Model pipeline is officially finished.")
    print()
    print("=" * 70)
    print("VALIDATION COMPLETE — PIPELINE FINISHED")
    print("=" * 70)
    sys.exit(0)
else:
    print()
    print("  ✗  VALIDATION FAILED — fix the issues above before proceeding.")
    print()
    print("=" * 70)
    print("VALIDATION FAILED")
    print("=" * 70)
    sys.exit(1)
