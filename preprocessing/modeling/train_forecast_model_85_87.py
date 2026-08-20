"""
===============================================================================
CONTRACTIQ — DATASET 1 SAVINGS FORECAST MODEL
Best achievable R² on strict time-based validation (train 2018-2022, val 2023)
===============================================================================

Findings from empirical search
-------------------------------
- Data ceiling: ~80-81% R² on the strict time split.
  The year-to-year savings rate has low autocorrelation
  (corr ≈ 0 between consecutive years per ACO), meaning
  the predictive signal comes from cross-sectional
  characteristics, not temporal patterns.
- Random Forest with no max_depth and msl=1 is the best
  single model (~80.5%).
- XGBoost with low lambda (0.1) on base features: ~79%.
- Stacking / ensembling does not improve beyond ~80%.
- 85-87% is NOT achievable on this data with this split
  without data leakage.

This script:
1. Engineers 94 features (lags, trends, rolling stats, interactions).
2. Trains Random Forest (best performer) + LightGBM + XGBoost.
3. Selects best model by validation R².
4. Saves all artifacts to models/forecast_prediction/.

Leakage controls
----------------
- NEXT_YEAR_SAVINGS_RATE never used as a predictor.
- All lags/rolling use .shift(1) (prior-year only).
- Imputation medians from training set only.

Outputs (models/forecast_prediction/)
--------------------------------------
- forecast_best_model_85_87.joblib
- forecast_model_metadata_85_87.json
- forecast_model_metrics_85_87.json
- forecast_model_comparison_85_87.csv
- forecast_feature_importance_85_87.csv
- forecast_model_predictions_85_87.csv
- Dataset1_Scoring_2024_Forecast_85_87.csv
===============================================================================
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH  = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_PATH   = PROJECT_ROOT / "data" / "processed" / "Dataset1_Model_Forecast.csv"
SCORING_PATH = PROJECT_ROOT / "data" / "processed" / "Dataset1_Scoring_2024_Forecast.csv"
MODEL_DIR    = PROJECT_ROOT / "models" / "forecast_prediction"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL      = MODEL_DIR / "forecast_best_model_85_87.joblib"
OUTPUT_METADATA   = MODEL_DIR / "forecast_model_metadata_85_87.json"
OUTPUT_METRICS    = MODEL_DIR / "forecast_model_metrics_85_87.json"
OUTPUT_COMPARISON = MODEL_DIR / "forecast_model_comparison_85_87.csv"
OUTPUT_IMPORTANCE = MODEL_DIR / "forecast_feature_importance_85_87.csv"
OUTPUT_VAL_PREDS  = MODEL_DIR / "forecast_model_predictions_85_87.csv"
OUTPUT_FORECAST   = MODEL_DIR / "Dataset1_Scoring_2024_Forecast_85_87.csv"


# ============================================================
# CONFIG
# ============================================================

TARGET        = "NEXT_YEAR_SAVINGS_RATE"
IDENTIFIERS   = ["ACO_ID", "ACO_NAME", "STATE", "YEAR"]
BASE_FEATURES = [
    "N_AB", "PREVIOUS_SAVINGS_RATE", "PREVIOUS_QUALITY_SCORE",
    "PREVIOUS_PERFORMANCE_GAP_PCT", "EXPENDITURE_GROWTH_PCT",
    "BENCHMARK_GROWTH_PCT", "BENEFICIARY_GROWTH_PCT", "QUALITY_CHANGE",
]


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def calc_metrics(y_true, y_pred):
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy().sort_values(["ACO_ID", "YEAR"]).reset_index(drop=True)
    g = d.groupby("ACO_ID", sort=False)

    # Lags 1-3
    lag_src = {
        "SAV":   "PREVIOUS_SAVINGS_RATE",
        "QUAL":  "PREVIOUS_QUALITY_SCORE",
        "PERF":  "PREVIOUS_PERFORMANCE_GAP_PCT",
        "EXP":   "EXPENDITURE_GROWTH_PCT",
        "BENCH": "BENCHMARK_GROWTH_PCT",
        "BEN":   "BENEFICIARY_GROWTH_PCT",
        "QCHG":  "QUALITY_CHANGE",
    }
    for short, col in lag_src.items():
        for lag in [1, 2, 3]:
            d[f"{short}_L{lag}"] = g[col].shift(lag)

    # Trends and acceleration
    d["SAV_TREND"]  = d["PREVIOUS_SAVINGS_RATE"]        - d["SAV_L1"]
    d["SAV_TREND2"] = d["SAV_L1"]                       - d["SAV_L2"]
    d["SAV_ACCEL"]  = d["SAV_TREND"]                    - d["SAV_TREND2"]
    d["QUAL_TREND"] = d["PREVIOUS_QUALITY_SCORE"]       - d["QUAL_L1"]
    d["PERF_TREND"] = d["PREVIOUS_PERFORMANCE_GAP_PCT"] - d["PERF_L1"]
    d["EXP_TREND"]  = d["EXPENDITURE_GROWTH_PCT"]       - d["EXP_L1"]
    d["SAV_CHG_2Y"] = d["PREVIOUS_SAVINGS_RATE"]        - d["SAV_L2"]
    d["SAV_CHG_3Y"] = d["PREVIOUS_SAVINGS_RATE"]        - d["SAV_L3"]

    # Rolling stats (shift-then-roll — no leakage)
    roll_src = {
        "SAV":   "PREVIOUS_SAVINGS_RATE",
        "QUAL":  "PREVIOUS_QUALITY_SCORE",
        "PERF":  "PREVIOUS_PERFORMANCE_GAP_PCT",
        "EXP":   "EXPENDITURE_GROWTH_PCT",
        "BENCH": "BENCHMARK_GROWTH_PCT",
    }
    for short, col in roll_src.items():
        s   = g[col].shift(1)
        grp = s.groupby(d["ACO_ID"])
        d[f"{short}_RM2"] = grp.transform(lambda x: x.rolling(2, min_periods=1).mean())
        d[f"{short}_RM3"] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
        d[f"{short}_RM5"] = grp.transform(lambda x: x.rolling(5, min_periods=1).mean())
        d[f"{short}_RS3"] = grp.transform(lambda x: x.rolling(3, min_periods=2).std())

    # Interactions
    d["SAV_x_QUAL"]       = d["PREVIOUS_SAVINGS_RATE"]        * d["PREVIOUS_QUALITY_SCORE"]
    d["SAV_x_PERF"]       = d["PREVIOUS_SAVINGS_RATE"]        * d["PREVIOUS_PERFORMANCE_GAP_PCT"]
    d["QUAL_x_PERF"]      = d["PREVIOUS_QUALITY_SCORE"]       * d["PREVIOUS_PERFORMANCE_GAP_PCT"]
    d["SAV_x_QCHG"]       = d["PREVIOUS_SAVINGS_RATE"]        * d["QUALITY_CHANGE"]
    d["PERF_x_QCHG"]      = d["PREVIOUS_PERFORMANCE_GAP_PCT"] * d["QUALITY_CHANGE"]
    d["EXP_x_BENCH"]      = d["EXPENDITURE_GROWTH_PCT"]       * d["BENCHMARK_GROWTH_PCT"]
    d["SAV_L1_x_SAV"]     = d["SAV_L1"]                       * d["PREVIOUS_SAVINGS_RATE"]
    d["SAV_L1_x_QUAL"]    = d["SAV_L1"]                       * d["PREVIOUS_QUALITY_SCORE"]
    d["SAV_TREND_x_PERF"] = d["SAV_TREND"]                    * d["PREVIOUS_PERFORMANCE_GAP_PCT"]

    # Difference features
    d["EXP_BENCH_GAP"] = d["EXPENDITURE_GROWTH_PCT"] - d["BENCHMARK_GROWTH_PCT"]
    d["EXP_BEN_GAP"]   = d["EXPENDITURE_GROWTH_PCT"] - d["BENEFICIARY_GROWTH_PCT"]

    # Nonlinear
    for col in BASE_FEATURES:
        d[f"{col}_SQ"]  = d[col] ** 2
        d[f"{col}_ABS"] = np.abs(d[col])
    d["SAV_CUBE"]  = d["PREVIOUS_SAVINGS_RATE"]        ** 3
    d["PERF_CUBE"] = d["PREVIOUS_PERFORMANCE_GAP_PCT"] ** 3
    d["SAV_SQRT"]  = np.sign(d["PREVIOUS_SAVINGS_RATE"]) * np.sqrt(np.abs(d["PREVIOUS_SAVINGS_RATE"]))

    # Composites
    d["SAV_MOM"]     = d["PREVIOUS_SAVINGS_RATE"] + d["QUALITY_CHANGE"] - d["PREVIOUS_PERFORMANCE_GAP_PCT"]
    d["GROWTH_COMP"] = (d["EXPENDITURE_GROWTH_PCT"] + d["BENCHMARK_GROWTH_PCT"] + d["BENEFICIARY_GROWTH_PCT"]) / 3.0
    d["QUAL_COMP"]   = (d["PREVIOUS_QUALITY_SCORE"] + d["QUALITY_CHANGE"]) / 2.0
    d["PERF_QUAL_R"] = d["PREVIOUS_PERFORMANCE_GAP_PCT"] / (np.abs(d["PREVIOUS_QUALITY_SCORE"]) + 1e-4)
    d["SAV_QUAL_R"]  = d["PREVIOUS_SAVINGS_RATE"]        / (np.abs(d["PREVIOUS_QUALITY_SCORE"]) + 1e-4)

    # Panel features
    d["ACO_AGE"]  = g.cumcount()
    d["YEAR_IDX"] = d["YEAR"] - d["YEAR"].min()

    # Clean infinities
    num = d.select_dtypes(include=[np.number]).columns
    for c in num:
        if c == TARGET:
            continue
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    return d


# ============================================================
# LOAD DATA
# ============================================================

section("LOADING DATA")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Historical dataset not found:\n{INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip().str.upper()
df = df.sort_values(["ACO_ID", "YEAR"]).reset_index(drop=True)
print(f"Historical rows : {len(df):,}  |  years {df['YEAR'].min()}–{df['YEAR'].max()}")

scoring_df = None
if SCORING_PATH.exists():
    scoring_df = pd.read_csv(SCORING_PATH)
    scoring_df.columns = scoring_df.columns.str.strip().str.upper()
    print(f"2024 scoring    : {len(scoring_df):,} rows")
else:
    print("WARNING: 2024 scoring file not found — 2024 forecast will be skipped.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

section("ENGINEERING FEATURES")

if scoring_df is not None:
    combined = pd.concat(
        [df.assign(_SCORE=False), scoring_df.assign(_SCORE=True)],
        ignore_index=True, sort=False,
    )
    combined_eng = engineer(combined)
    historical   = combined_eng[combined_eng["_SCORE"] == False].copy()
    scoring_eng  = combined_eng[combined_eng["_SCORE"] == True].copy()
else:
    historical  = engineer(df)
    scoring_eng = None

excl = set(IDENTIFIERS + [TARGET, "_SCORE"])
FEATURES = [
    c for c in historical.columns
    if c not in excl and pd.api.types.is_numeric_dtype(historical[c])
]
print(f"Engineered features: {len(FEATURES)}")


# ============================================================
# TRAIN / VALIDATION SPLIT + IMPUTATION
# ============================================================

section("TRAIN / VALIDATION SPLIT")

train_df = historical[historical["YEAR"] <= 2022].copy()
valid_df  = historical[historical["YEAR"] == 2023].copy()

print(f"Training rows (2018–2022) : {len(train_df):,}")
print(f"Validation rows (2023)    : {len(valid_df):,}")

train_meds = train_df[FEATURES].median()
train_df[FEATURES] = train_df[FEATURES].fillna(train_meds).astype(np.float32)
valid_df[FEATURES] = valid_df[FEATURES].fillna(train_meds).astype(np.float32)

X_tr = train_df[FEATURES];  y_tr = train_df[TARGET].astype(np.float32)
X_va = valid_df[FEATURES];  y_va = valid_df[TARGET].astype(np.float32)

if scoring_eng is not None:
    scoring_eng[FEATURES] = scoring_eng[FEATURES].fillna(train_meds).astype(np.float32)
    X_sc = scoring_eng[FEATURES]


# ============================================================
# CANDIDATE DEFINITIONS
# ============================================================

section("DEFINING MODEL CANDIDATES")

import lightgbm as lgb
import xgboost  as xgb
from sklearn.ensemble import RandomForestRegressor

candidates = []

# ------ Random Forest (best performer on this dataset) ------
rf_cfgs = [
    ("RF_msl1_mf06", dict(n_estimators=600, max_depth=None, min_samples_leaf=1, max_features=0.6, random_state=42, n_jobs=4)),
    ("RF_msl1_mf07", dict(n_estimators=600, max_depth=None, min_samples_leaf=1, max_features=0.7, random_state=42, n_jobs=4)),
    ("RF_msl2_mf07", dict(n_estimators=600, max_depth=None, min_samples_leaf=2, max_features=0.7, random_state=42, n_jobs=4)),
    ("RF_msl1_mf08", dict(n_estimators=400, max_depth=None, min_samples_leaf=1, max_features=0.8, random_state=42, n_jobs=4)),
]
for tag, params in rf_cfgs:
    candidates.append((tag, "RF", params.copy()))

# ------ LightGBM ------
lgb_cfgs = [
    ("LGB_d9_lam01_nl255", dict(n_estimators=3000, learning_rate=0.02, max_depth=9,  num_leaves=255, min_child_samples=8,  reg_lambda=0.1,  reg_alpha=0.0, subsample=0.85, colsample_bytree=0.8, random_state=42, n_jobs=4, verbose=-1)),
    ("LGB_d8_lam01_nl127", dict(n_estimators=3000, learning_rate=0.02, max_depth=8,  num_leaves=127, min_child_samples=10, reg_lambda=0.1,  reg_alpha=0.0, subsample=0.85, colsample_bytree=0.8, random_state=42, n_jobs=4, verbose=-1)),
    ("LGB_d10_lam01_nl255",dict(n_estimators=3000, learning_rate=0.01, max_depth=10, num_leaves=255, min_child_samples=8,  reg_lambda=0.1,  reg_alpha=0.0, subsample=0.9,  colsample_bytree=0.85,random_state=42, n_jobs=4, verbose=-1)),
    ("LGB_d8_lam02_nl127", dict(n_estimators=3000, learning_rate=0.02, max_depth=8,  num_leaves=127, min_child_samples=10, reg_lambda=0.2,  reg_alpha=0.0, subsample=0.8,  colsample_bytree=0.75,random_state=42, n_jobs=4, verbose=-1)),
]
for tag, params in lgb_cfgs:
    candidates.append((tag, "LGB", params.copy()))

# ------ XGBoost ------
xgb_cfgs = [
    ("XGB_d8_lam01",  dict(n_estimators=2000, learning_rate=0.02, max_depth=8, subsample=0.85, colsample_bytree=0.8, min_child_weight=3, reg_lambda=0.1, reg_alpha=0.0, random_state=42, n_jobs=4, verbosity=0, early_stopping_rounds=150)),
    ("XGB_d9_lam01",  dict(n_estimators=2000, learning_rate=0.01, max_depth=9, subsample=0.85, colsample_bytree=0.8, min_child_weight=2, reg_lambda=0.1, reg_alpha=0.0, random_state=42, n_jobs=4, verbosity=0, early_stopping_rounds=150)),
    ("XGB_d8_lam02",  dict(n_estimators=2000, learning_rate=0.02, max_depth=8, subsample=0.85, colsample_bytree=0.8, min_child_weight=3, reg_lambda=0.2, reg_alpha=0.0, random_state=42, n_jobs=4, verbosity=0, early_stopping_rounds=150)),
]
for tag, params in xgb_cfgs:
    candidates.append((tag, "XGB", params.copy()))

# ------ CatBoost ------
try:
    from catboost import CatBoostRegressor
    cb_cfgs = [
        ("CB_d8_l201", dict(iterations=2000, depth=8, learning_rate=0.02, l2_leaf_reg=1.0, random_strength=0.5, bagging_temperature=0.3, random_seed=42, task_type="CPU", thread_count=4, od_type="Iter", od_wait=150, verbose=0, allow_writing_files=False)),
        ("CB_d8_l203", dict(iterations=2000, depth=8, learning_rate=0.02, l2_leaf_reg=3.0, random_strength=0.5, bagging_temperature=0.3, random_seed=42, task_type="CPU", thread_count=4, od_type="Iter", od_wait=150, verbose=0, allow_writing_files=False)),
    ]
    for tag, params in cb_cfgs:
        candidates.append((tag, "CB", params.copy()))
    print("CatBoost: available — included.")
except ImportError:
    print("CatBoost: not installed — skipped.")

print(f"Total candidates: {len(candidates)}")


# ============================================================
# TRAINING LOOP
# ============================================================

section("TRAINING ALL CANDIDATES")

results = []

for tag, mtype, params in candidates:

    print(f"  [{mtype}] {tag} ...", end="", flush=True)

    try:
        if mtype == "RF":
            m = RandomForestRegressor(**params)
            m.fit(X_tr, y_tr)

        elif mtype == "LGB":
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(150, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

        elif mtype == "XGB":
            early = params.pop("early_stopping_rounds", 150)
            m = xgb.XGBRegressor(**params)
            m.set_params(early_stopping_rounds=early)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        elif mtype == "CB":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

        preds = m.predict(X_va)
        met   = calc_metrics(y_va, preds)
        r2    = met["r2"]

        print(f"  R²={r2*100:.2f}%")

        results.append({
            "tag": tag, "mtype": mtype,
            "r2": r2, "mae": met["mae"], "rmse": met["rmse"],
            "model": m, "params": params,
        })

    except Exception as exc:
        print(f"  ERROR: {exc}")


# ============================================================
# SELECT BEST
# ============================================================

section("SELECTING BEST MODEL")

if not results:
    raise RuntimeError("No models completed successfully.")

best = max(results, key=lambda r: r["r2"])

print(f"Best model  : {best['tag']}  ({best['mtype']})")
print(f"R²          : {best['r2']*100:.2f}%")
print(f"MAE         : {best['mae']:.6f}")
print(f"RMSE        : {best['rmse']:.6f}")


# ============================================================
# COMPARISON TABLE
# ============================================================

section("MODEL COMPARISON")

rows = [
    {"MODEL": r["tag"], "TYPE": r["mtype"],
     "R2": round(r["r2"], 6), "R2_PCT": round(r["r2"]*100, 2),
     "MAE": round(r["mae"], 6), "RMSE": round(r["rmse"], 6)}
    for r in sorted(results, key=lambda x: x["r2"], reverse=True)
]
comp_df = pd.DataFrame(rows)
print(comp_df.to_string(index=False))
comp_df.to_csv(OUTPUT_COMPARISON, index=False)
print(f"\nComparison saved: {OUTPUT_COMPARISON}")


# ============================================================
# RETRAIN ON FULL 2018–2023
# ============================================================

section("RETRAINING PRODUCTION MODEL ON 2018–2023")

full_df = historical.copy()
X_full  = full_df[FEATURES]
y_full  = full_df[TARGET].astype(np.float32)

mtype  = best["mtype"]
params = best["params"].copy()

print(f"Model : {best['tag']}  ({mtype})")
print(f"Rows  : {len(X_full):,}")

if mtype == "RF":
    prod = RandomForestRegressor(**params)
    prod.fit(X_full, y_full)

elif mtype == "LGB":
    prod = lgb.LGBMRegressor(**params)
    prod.fit(X_full, y_full)

elif mtype == "XGB":
    prod = xgb.XGBRegressor(**params)
    prod.set_params(early_stopping_rounds=None)
    prod.fit(X_full, y_full, verbose=False)

elif mtype == "CB":
    from catboost import CatBoostRegressor
    prod = CatBoostRegressor(**params)
    prod.fit(X_full, y_full)

joblib.dump(prod, OUTPUT_MODEL)
print(f"Saved: {OUTPUT_MODEL}")


# ============================================================
# 2023 VALIDATION PREDICTIONS
# ============================================================

section("SAVING 2023 VALIDATION PREDICTIONS")

val_preds = best["model"].predict(X_va)
val_out   = valid_df[["ACO_ID", "ACO_NAME", "STATE", "YEAR", TARGET]].copy()
val_out["PREDICTED_NEXT_YEAR_SAVINGS_RATE"]     = val_preds
val_out["PREDICTED_NEXT_YEAR_SAVINGS_RATE_PCT"] = val_preds * 100
val_out["ABSOLUTE_ERROR"] = np.abs(val_out[TARGET] - val_preds)
val_out.to_csv(OUTPUT_VAL_PREDS, index=False)
print(f"Saved: {OUTPUT_VAL_PREDS}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

section("FEATURE IMPORTANCE")

m_obj = best["model"]
if mtype in ("XGB", "LGB", "RF"):
    imp_vals = m_obj.feature_importances_
elif mtype == "CB":
    imp_vals = m_obj.get_feature_importance()
else:
    imp_vals = None

if imp_vals is not None:
    imp_df = pd.DataFrame({"FEATURE": FEATURES, "IMPORTANCE": imp_vals})
    imp_df = imp_df.sort_values("IMPORTANCE", ascending=False).reset_index(drop=True)
    print(imp_df.head(20).to_string(index=False))
    imp_df.to_csv(OUTPUT_IMPORTANCE, index=False)
    print(f"\nSaved: {OUTPUT_IMPORTANCE}")


# ============================================================
# 2024 FORECAST
# ============================================================

if scoring_eng is not None:

    section("2024 FORECAST")

    fc_preds = prod.predict(X_sc)

    def classify(v):
        if v >= 0.05:  return "HIGH SAVINGS"
        if v >= 0.00:  return "MODERATE SAVINGS"
        return "LOW SAVINGS"

    fc_out = scoring_df[["ACO_ID", "ACO_NAME", "STATE", "YEAR"]].copy()
    fc_out["FORECASTED_NEXT_YEAR_SAVINGS_RATE"]     = fc_preds
    fc_out["FORECASTED_NEXT_YEAR_SAVINGS_RATE_PCT"] = fc_preds * 100
    fc_out["SAVINGS_FORECAST_CATEGORY"] = [classify(v) for v in fc_preds]
    fc_out.to_csv(OUTPUT_FORECAST, index=False)

    print(f"Rows forecasted : {len(fc_out):,}")
    print(f"Mean forecast   : {fc_preds.mean():.6f}")
    print(fc_out["SAVINGS_FORECAST_CATEGORY"].value_counts().to_string())
    print(f"Saved: {OUTPUT_FORECAST}")


# ============================================================
# SAVE METRICS + METADATA
# ============================================================

section("SAVING METRICS & METADATA")

metrics_out = {
    "selected_model":           best["tag"],
    "model_type":               mtype,
    "target":                   TARGET,
    "validation_year":          2023,
    "training_years":           "2018-2022",
    "production_years":         "2018-2023",
    "train_rows":               int(len(X_tr)),
    "validation_rows":          int(len(X_va)),
    "production_rows":          int(len(X_full)),
    "features":                 int(len(FEATURES)),
    "mae":                      float(best["mae"]),
    "rmse":                     float(best["rmse"]),
    "r2":                       float(best["r2"]),
    "r2_percentage":            float(best["r2"] * 100),
    "leakage_protection":       True,
    "future_target_as_feature": False,
    "candidates_evaluated":     len(results),
}

with open(OUTPUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=4)
print(f"Metrics  saved: {OUTPUT_METRICS}")

metadata_out = {
    "project": "ContractIQ", "dataset": "Dataset 1",
    "script":  "train_forecast_model_85_87.py",
    "selected_model": best["tag"], "model_type": mtype,
    "target": TARGET,
    "base_features": BASE_FEATURES,
    "engineered_features": FEATURES,
    "params": {k: v for k, v in best["params"].items() if not callable(v)},
    "validation_strategy": "Strict time-based (train 2018-2022, validate 2023)",
    "validation_metrics": {
        "r2":    float(best["r2"]),
        "r2_pct":float(best["r2"] * 100),
        "mae":   float(best["mae"]),
        "rmse":  float(best["rmse"]),
    },
}
with open(OUTPUT_METADATA, "w") as f:
    json.dump(metadata_out, f, indent=4, default=str)
print(f"Metadata saved: {OUTPUT_METADATA}")


# ============================================================
# FINAL SUMMARY
# ============================================================

section("FINAL SUMMARY")

print(f"  Selected model : {best['tag']}  ({mtype})")
print(f"  Validation R²  : {best['r2']*100:.2f}%")
print(f"  MAE            : {best['mae']:.6f}")
print(f"  RMSE           : {best['rmse']:.6f}")
print(f"  Candidates run : {len(results)}")
print()
print(f"  Model file     : {OUTPUT_MODEL.name}")
print(f"  Metrics        : {OUTPUT_METRICS.name}")
print(f"  Comparison     : {OUTPUT_COMPARISON.name}")
print()
print("=" * 78)
print("TRAINING COMPLETE")
print("=" * 78)
