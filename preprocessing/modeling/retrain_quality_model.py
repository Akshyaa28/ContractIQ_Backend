"""
Retrain the quality score pipeline on the local sklearn version
so the saved joblib is compatible with the API.

Input : data/quality/enhanced_synthetic_aco_quality_data.csv
Output: models/quality_prediction/quality_score_pipeline.joblib
        models/quality_prediction/quality_model_metadata.json
        models/quality_prediction/quality_model_metrics.json
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "quality" / "enhanced_synthetic_aco_quality_data.csv"
MODEL_DIR    = PROJECT_ROOT / "models" / "quality_prediction"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL    = MODEL_DIR / "quality_score_pipeline.joblib"
OUTPUT_METADATA = MODEL_DIR / "quality_model_metadata.json"
OUTPUT_METRICS  = MODEL_DIR / "quality_model_metrics.json"

RANDOM_STATE  = 42
TARGET        = "target_Quality_Score_T1"
MIN_R2_GOAL   = 0.80
MAX_R2_GOAL   = 0.89

# ============================================================
# LOAD + CLEAN
# ============================================================

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna(subset=["ACO_ID", "Year_T", "Year_T1", TARGET]).copy()
df = df[df["Year_T1"].eq(df["Year_T"] + 1)].copy()
df = df[df[TARGET].between(0, 100)].copy()

print(f"Rows: {len(df):,}  |  ACOs: {df['ACO_ID'].nunique():,}  |  Years (T1): {sorted(df['Year_T1'].unique())}")

# ============================================================
# FEATURES
# ============================================================

excl = ["ACO_ID", "Year_T1", TARGET]
feature_columns = [c for c in df.columns if c not in excl]
numerical_features  = df[feature_columns].select_dtypes(include=np.number).columns.tolist()
categorical_features = df[feature_columns].select_dtypes(exclude=np.number).columns.tolist()

print(f"Features: {len(feature_columns)}  "
      f"(num={len(numerical_features)}, cat={len(categorical_features)})")

X      = df[feature_columns].copy()
y      = df[TARGET].copy()
groups = df["ACO_ID"].copy()

# ============================================================
# PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler",  StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("numeric",      numeric_pipeline,      numerical_features),
    ("categorical",  categorical_pipeline,  categorical_features),
])

selected_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model",        Ridge(alpha=10.0)),
])

# ============================================================
# TEMPORAL SPLIT — train on 2017-2023, test on 2024
# ============================================================

latest_year = int(df["Year_T1"].max())
dev_mask    = df["Year_T1"] < latest_year
test_mask   = df["Year_T1"] == latest_year

X_dev,  y_dev  = X.loc[dev_mask],  y.loc[dev_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]
groups_dev     = groups.loc[dev_mask]

print(f"Dev rows:  {len(X_dev):,}  |  Test rows: {len(X_test):,}  (year {latest_year})")

# ============================================================
# GROUPED CV
# ============================================================

print("\nRunning 5-fold grouped CV on dev set...")
cv_scores = cross_validate(
    selected_pipeline, X_dev, y_dev,
    groups=groups_dev,
    cv=GroupKFold(n_splits=5),
    scoring={"MAE": "neg_mean_absolute_error",
             "RMSE": "neg_root_mean_squared_error",
             "R2": "r2"},
    n_jobs=-1,
)
cv_r2   = cv_scores["test_R2"].mean()
cv_mae  = -cv_scores["test_MAE"].mean()
cv_rmse = -cv_scores["test_RMSE"].mean()
print(f"  CV R² = {cv_r2*100:.2f}%  MAE = {cv_mae:.3f}  RMSE = {cv_rmse:.3f}")

# ============================================================
# FIT ON DEV, EVALUATE ON 2024
# ============================================================

print("\nFitting on dev (2017-2023) and evaluating on 2024...")
selected_pipeline.fit(X_dev, y_dev)
test_preds = np.clip(selected_pipeline.predict(X_test), 0, 100)
absolute_errors = np.abs(y_test.to_numpy() - test_preds)

test_r2   = float(r2_score(y_test, test_preds))
test_mae  = float(mean_absolute_error(y_test, test_preds))
test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

print(f"  Test R²   = {test_r2*100:.2f}%")
print(f"  Test MAE  = {test_mae:.4f}")
print(f"  Test RMSE = {test_rmse:.4f}")
print(f"  Within 1 pt:  {np.mean(absolute_errors <= 1)*100:.1f}%")
print(f"  Within 2 pts: {np.mean(absolute_errors <= 2)*100:.1f}%")
print(f"  Within 3 pts: {np.mean(absolute_errors <= 3)*100:.1f}%")
print(f"  Within 5 pts: {np.mean(absolute_errors <= 5)*100:.1f}%")

if MIN_R2_GOAL <= test_r2 <= MAX_R2_GOAL:
    print(f"\n  GOAL MET: R² is within {MIN_R2_GOAL*100:.0f}–{MAX_R2_GOAL*100:.0f}% ✓")
else:
    print(f"\n  R² {test_r2*100:.2f}% is outside {MIN_R2_GOAL*100:.0f}–{MAX_R2_GOAL*100:.0f}%")

# ============================================================
# REFIT ON ALL DATA FOR PRODUCTION
# ============================================================

print("\nRefitting on full dataset (all years) for production model...")
production_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model",        Ridge(alpha=10.0)),
])
production_pipeline.fit(X, y)

# ============================================================
# SAVE
# ============================================================

joblib.dump(production_pipeline, OUTPUT_MODEL)
print(f"Saved: {OUTPUT_MODEL}")

# ---- metadata ----
metadata = {
    "project":             "ContractIQ",
    "dataset":             "ACO Quality Score Synthetic Dataset",
    "script":              "retrain_quality_model.py",
    "model_type":          "Ridge (sklearn Pipeline)",
    "target":              TARGET,
    "feature_columns":     feature_columns,
    "numerical_features":  numerical_features,
    "categorical_features": categorical_features,
    "total_features":      len(feature_columns),
    "rows":                len(df),
    "unique_acos":         int(df["ACO_ID"].nunique()),
    "years_T1":            sorted([int(v) for v in df["Year_T1"].unique()]),
    "train_years":         f"T1 2017–{latest_year-1}",
    "test_year":           latest_year,
    "train_rows":          int(len(X_dev)),
    "test_rows":           int(len(X_test)),
    "validation_strategy": "Strict future-year hold-out (2024) + 5-fold GroupKFold on dev",
    "leakage_protection":  True,
    "future_target_as_feature": False,
    "disclosure":          (
        "Trained on a SYNTHETIC ACO dataset created for prototype validation. "
        "Not real CMS outcome data. Must not be presented as evidence of "
        "effectiveness on real CMS data."
    ),
}
with open(OUTPUT_METADATA, "w") as f:
    json.dump(metadata, f, indent=4, default=str)
print(f"Saved: {OUTPUT_METADATA}")

# ---- metrics ----
metrics_out = {
    "model_type":            "Ridge",
    "target":                TARGET,
    "test_year":             latest_year,
    "train_years":           f"T1 2017–{latest_year-1}",
    "cv_r2":                 cv_r2,
    "cv_r2_percentage":      cv_r2 * 100,
    "cv_mae":                cv_mae,
    "cv_rmse":               cv_rmse,
    "test_r2":               test_r2,
    "test_r2_percentage":    test_r2 * 100,
    "test_mae":              test_mae,
    "test_rmse":             test_rmse,
    "within_1_pt_pct":       float(np.mean(absolute_errors <= 1) * 100),
    "within_2_pts_pct":      float(np.mean(absolute_errors <= 2) * 100),
    "within_3_pts_pct":      float(np.mean(absolute_errors <= 3) * 100),
    "within_5_pts_pct":      float(np.mean(absolute_errors <= 5) * 100),
    "r2_goal_met":           bool(MIN_R2_GOAL <= test_r2 <= MAX_R2_GOAL),
}
with open(OUTPUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=4)
print(f"Saved: {OUTPUT_METRICS}")

print()
print("=" * 60)
print("QUALITY MODEL RETRAINING COMPLETE")
print("=" * 60)
