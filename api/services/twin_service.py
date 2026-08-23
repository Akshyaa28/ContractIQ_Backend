"""
Twin Matching Service
=====================
Loads the NearestNeighbors model + StandardScaler and finds
the top-5 most similar historical ACO-year peers for a query ACO.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "similar_twin"
SCALER_PATH = MODEL_DIR / "twin_scaler.joblib"
NN_PATH = MODEL_DIR / "twin_nearest_neighbors.joblib"
WEIGHTS_PATH = MODEL_DIR / "twin_feature_weights.json"
REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "Dataset1_Model_Twin.csv"

MODEL_FEATURES = [
    "N_AB", "PREVIOUS_SAVINGS_RATE", "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT", "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT", "BENEFICIARY_GROWTH_PCT", "QUALITY_CHANGE",
]

TOP_K = 5
CANDIDATE_K = 50
TARGET = "NEXT_YEAR_SAVINGS_RATE"

_scaler = None
_nn_model = None
_weights = None
_reference_df = None


def _load_all():
    """Load all twin model artifacts (called once, cached)."""
    global _scaler, _nn_model, _weights, _reference_df

    if _scaler is not None:
        return

    for label, path in [("Scaler", SCALER_PATH), ("NN", NN_PATH),
                        ("Weights", WEIGHTS_PATH), ("Reference", REFERENCE_PATH)]:
        if not path.exists():
            raise FileNotFoundError(f"Twin {label} not found: {path}")

    _scaler = joblib.load(SCALER_PATH)
    _nn_model = joblib.load(NN_PATH)

    with open(WEIGHTS_PATH) as f:
        weights_data = json.load(f)
    _weights = np.array([weights_data["weights"][f] for f in MODEL_FEATURES], dtype=np.float32)

    ref = pd.read_csv(REFERENCE_PATH)
    ref.columns = ref.columns.str.strip().str.upper()
    ref = ref[ref["YEAR"] <= 2023].copy().reset_index(drop=True)
    for col in MODEL_FEATURES:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    _reference_df = ref


def get_model():
    """Public accessor — triggers lazy load."""
    _load_all()
    return _nn_model


def find_similar_twins(input_data: dict) -> dict:
    """Find top-5 similar historical ACO-year peers."""
    _load_all()

    query = np.array([[input_data[f] for f in MODEL_FEATURES]], dtype=np.float32)
    scaled = _scaler.transform(query).astype(np.float32)
    weighted = (scaled * _weights).astype(np.float32)

    distances, indices = _nn_model.kneighbors(weighted, n_neighbors=CANDIDATE_K)
    distances, indices = distances[0], indices[0]

    seen_acos = set()
    query_aco = str(input_data.get("ACO_ID", "")).strip()
    twins = []

    for dist, idx in zip(distances, indices):
        row = _reference_df.iloc[idx]
        twin_aco = str(row["ACO_ID"]).strip()

        if twin_aco == query_aco or twin_aco in seen_acos:
            continue

        seen_acos.add(twin_aco)
        twins.append({
            "rank": len(twins) + 1,
            "twin_aco_id": twin_aco,
            "twin_aco_name": str(row.get("ACO_NAME", "")),
            "twin_state": str(row.get("STATE", "")),
            "twin_year": int(row["YEAR"]),
            "similarity_score": round(float(100.0 / (1.0 + dist)), 6),
            "twin_savings_rate_pct": round(float(row[TARGET]) * 100, 4),
        })

        if len(twins) >= TOP_K:
            break

    savings = [t["twin_savings_rate_pct"] for t in twins]
    median = float(np.median(savings)) if savings else 0.0

    for t in twins:
        t["outperforming"] = t["twin_savings_rate_pct"] > median

    return {
        "twins": twins,
        "top5_avg_savings_rate_pct": round(float(np.mean(savings)), 4) if savings else 0.0,
        "top5_median_savings_rate_pct": round(median, 4),
        "top5_best_savings_rate_pct": round(float(max(savings)), 4) if savings else 0.0,
        "top5_worst_savings_rate_pct": round(float(min(savings)), 4) if savings else 0.0,
        "outperformer_count": sum(1 for t in twins if t["outperforming"]),
        "outperformer_rate": round(sum(1 for t in twins if t["outperforming"]) / max(len(twins), 1), 4),
    }


def get_model_info() -> dict:
    """Return twin model metadata."""
    _load_all()
    return {
        "model_type": "NearestNeighbors (Weighted Euclidean)",
        "model_file": NN_PATH.name,
        "scaler_file": SCALER_PATH.name,
        "features": MODEL_FEATURES,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "reference_rows": len(_reference_df),
        "reference_years": "2018-2023",
        "target": TARGET,
        "target_used_for_similarity": False,
        "self_match_excluded": True,
    }
