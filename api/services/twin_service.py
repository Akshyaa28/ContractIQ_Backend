from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "similar_twin"

SCALER_PATH   = MODEL_DIR / "twin_scaler.joblib"
NN_PATH       = MODEL_DIR / "twin_nearest_neighbors.joblib"
WEIGHTS_PATH  = MODEL_DIR / "twin_feature_weights.json"
METADATA_PATH = MODEL_DIR / "twin_model_metadata.json"

REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "Dataset1_Model_Twin.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FEATURES = [
    "N_AB",
    "PREVIOUS_SAVINGS_RATE",
    "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT",
    "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT",
    "BENEFICIARY_GROWTH_PCT",
    "QUALITY_CHANGE",
]

TOP_K = 5
CANDIDATE_K = 50
TARGET = "NEXT_YEAR_SAVINGS_RATE"


# ============================================================
# LAZY-LOADED GLOBALS
# ============================================================

_scaler = None
_nn_model = None
_weights = None
_reference_df = None
_weighted_reference = None


def _load_all():
    """Load all twin model artifacts once."""
    global _scaler, _nn_model, _weights, _reference_df, _weighted_reference

    if _scaler is not None:
        return  # already loaded

    for label, path in [
        ("Scaler", SCALER_PATH),
        ("NearestNeighbors", NN_PATH),
        ("Weights", WEIGHTS_PATH),
        ("Reference data", REFERENCE_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Twin {label} not found:\n{path}")

    _scaler = joblib.load(SCALER_PATH)
    _nn_model = joblib.load(NN_PATH)

    with open(WEIGHTS_PATH) as f:
        weights_data = json.load(f)
    _weights = np.array(
        [weights_data["weights"][feat] for feat in MODEL_FEATURES],
        dtype=np.float32,
    )

    # Load and prepare reference
    ref = pd.read_csv(REFERENCE_PATH)
    ref.columns = ref.columns.str.strip().str.upper()
    ref = ref[ref["YEAR"] <= 2023].copy().reset_index(drop=True)

    for col in MODEL_FEATURES:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")

    _reference_df = ref

    # Pre-compute weighted reference matrix
    ref_matrix = _scaler.transform(ref[MODEL_FEATURES].values).astype(np.float32)
    _weighted_reference = (ref_matrix * _weights).astype(np.float32)


def get_model():
    """Public accessor — triggers lazy load."""
    _load_all()
    return _nn_model


# ============================================================
# TWIN PREDICTION
# ============================================================

def find_similar_twins(input_data: dict) -> dict:
    """
    Given 8 features for a query ACO, find its top-5 similar twins
    from the historical reference set.

    Returns dict with twin details + aggregate stats.
    """
    _load_all()

    # Build query vector
    query_values = np.array(
        [[input_data[feat] for feat in MODEL_FEATURES]],
        dtype=np.float32,
    )

    # Scale + weight
    scaled = _scaler.transform(query_values).astype(np.float32)
    weighted_query = (scaled * _weights).astype(np.float32)

    # Find candidates
    distances, indices = _nn_model.kneighbors(weighted_query, n_neighbors=CANDIDATE_K)
    distances = distances[0]
    indices = indices[0]

    # Deduplicate: pick unique ACO_IDs, keep closest first
    seen_acos = set()
    query_aco = str(input_data.get("ACO_ID", "")).strip()
    twins = []

    for dist, idx in zip(distances, indices):
        row = _reference_df.iloc[idx]
        twin_aco = str(row["ACO_ID"]).strip()

        # Exclude self-match
        if twin_aco == query_aco:
            continue
        if twin_aco in seen_acos:
            continue

        seen_acos.add(twin_aco)
        similarity = round(float(100.0 / (1.0 + dist)), 6)
        twins.append({
            "rank": len(twins) + 1,
            "twin_aco_id": twin_aco,
            "twin_aco_name": str(row.get("ACO_NAME", "")),
            "twin_state": str(row.get("STATE", "")),
            "twin_year": int(row["YEAR"]),
            "similarity_score": similarity,
            "twin_savings_rate_pct": round(float(row[TARGET]) * 100, 4),
        })

        if len(twins) >= TOP_K:
            break

    # Compute aggregate stats
    savings_rates = [t["twin_savings_rate_pct"] for t in twins]
    median_savings = float(np.median(savings_rates)) if savings_rates else 0.0

    # Mark outperformers (above median of Top 5)
    for t in twins:
        t["outperforming"] = t["twin_savings_rate_pct"] > median_savings

    outperformer_count = sum(1 for t in twins if t["outperforming"])

    return {
        "twins": twins,
        "top5_avg_savings_rate_pct": round(float(np.mean(savings_rates)), 4) if savings_rates else 0.0,
        "top5_median_savings_rate_pct": round(median_savings, 4),
        "top5_best_savings_rate_pct": round(float(max(savings_rates)), 4) if savings_rates else 0.0,
        "top5_worst_savings_rate_pct": round(float(min(savings_rates)), 4) if savings_rates else 0.0,
        "outperformer_count": outperformer_count,
        "outperformer_rate": round(outperformer_count / max(len(twins), 1), 4),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> dict:
    """Return metadata about the twin model."""
    _load_all()

    return {
        "model_type": "NearestNeighbors (Weighted Euclidean)",
        "model_file": NN_PATH.name,
        "scaler_file": SCALER_PATH.name,
        "features": MODEL_FEATURES,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "reference_rows": len(_reference_df),
        "reference_years": "2018–2023",
        "target": TARGET,
        "target_used_for_similarity": False,
        "self_match_excluded": True,
    }
