"""
Phase 9: Provider-Level ML Model Training
ABISH ACO Portal Project

This script trains the provider-level risk classification model according to
the locked architecture and Phase 9 specifications.

Dataset: final/mssp_provider_synthetic_2018_2024.parquet
Target: risk_tier (LOW / MEDIUM / HIGH)
Grain: provider_id + aco_id + performance_year
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# ML libraries
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, log_loss, roc_auc_score
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Environment
from dotenv import load_dotenv
load_dotenv()

print("="*80)
print("PHASE 9: PROVIDER-LEVEL ML MODEL TRAINING")
print("="*80)
print()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

print("Step 1: Configuration")
print("-" * 80)

# Paths
DATA_PATH = Path("../data/filtered_provider_dataset_2021_2024_corrected.csv")
SCHEMA_PATH = Path("final/provider_schema.csv")
OUTPUT_DIR = Path("ml")
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"
ANALYSIS_DIR = Path("analysis/ml")

# Create directories
for dir_path in [METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR, ANALYSIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Temporal split configuration (from frozen spec)
TRAIN_YEARS = [2018, 2019, 2020]
VAL_YEARS = [2021, 2022]
TEST_YEARS = [2023]
INFERENCE_YEARS = [2024]

# Risk thresholds (frozen from training period)
P33_THRESHOLD = -0.2324
P66_THRESHOLD = 0.1629

# Random seed
RANDOM_STATE = 42

print(f"✓ Data path: {DATA_PATH}")
print(f"✓ Output directory: {OUTPUT_DIR}")
print(f"✓ Train years: {TRAIN_YEARS}")
print(f"✓ Validation years: {VAL_YEARS}")
print(f"✓ Test years: {TEST_YEARS}")
print(f"✓ Random state: {RANDOM_STATE}")
print()

# ============================================================================
# 2. LOAD DATASET
# ============================================================================

print("Step 2: Load Dataset")
print("-" * 80)

df = pd.read_csv(DATA_PATH)

print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"✓ Years: {sorted(df['performance_year'].unique())}")
print(f"✓ Unique providers: {df['provider_id'].nunique():,}")
print(f"✓ Unique ACOs: {df['aco_id'].nunique():,}")
print()

# ============================================================================
# 3. LEAKAGE AUDIT
# ============================================================================

print("Step 3: Leakage Audit")
print("-" * 80)

# Load leakage review
leakage_df = pd.read_csv(ANALYSIS_DIR / "feature_leakage_review.csv")

# Verify high-leakage variables are absent
high_leakage = leakage_df[leakage_df['leakage_level'] == 'HIGH']['column'].tolist()
high_leakage_present = [col for col in high_leakage if col in df.columns and col not in ['provider_id', 'aco_id']]

if high_leakage_present:
    print(f"⚠ WARNING: High leakage columns found: {high_leakage_present}")
else:
    print("✓ No high-leakage columns detected in dataset (except identifiers)")

# Check target
if 'risk_tier' not in df.columns:
    raise ValueError("Target variable 'risk_tier' not found in dataset")

print(f"✓ Target variable 'risk_tier' present")
print(f"✓ Target distribution:\n{df['risk_tier'].value_counts(normalize=True).mul(100).round(1)}")
print()

# ============================================================================
# 4. FEATURE SELECTION
# ============================================================================

print("Step 4: Feature Selection")
print("-" * 80)

# Load feature inventory
feature_inventory = pd.read_csv(ANALYSIS_DIR / "provider_feature_inventory.csv")

# Select features marked for inclusion
selected_features = feature_inventory[
    feature_inventory['include_in_model'] == 'YES'
]['feature'].tolist()

# Verify features exist
missing_features = [f for f in selected_features if f not in df.columns]
if missing_features:
    print(f"⚠ WARNING: Missing features: {missing_features}")
    selected_features = [f for f in selected_features if f in df.columns]

print(f"✓ Selected {len(selected_features)} features for modeling")

# Save final feature list
feature_list_df = pd.DataFrame({
    'feature': selected_features,
    'dtype': [str(df[f].dtype) for f in selected_features]
})
feature_list_df.to_csv(ANALYSIS_DIR / "model_feature_list.csv", index=False)
print(f"✓ Saved feature list to {ANALYSIS_DIR / 'model_feature_list.csv'}")
print()

# ============================================================================
# 5. TEMPORAL SPLIT
# ============================================================================

print("Step 5: Temporal Split")
print("-" * 80)

# Create splits
train_df = df[df['performance_year'].isin(TRAIN_YEARS)].copy()
val_df = df[df['performance_year'].isin(VAL_YEARS)].copy()
test_df = df[df['performance_year'].isin(TEST_YEARS)].copy()

print(f"✓ Train: {len(train_df):,} rows ({TRAIN_YEARS})")
print(f"✓ Validation: {len(val_df):,} rows ({VAL_YEARS})")
print(f"✓ Test: {len(test_df):,} rows ({TEST_YEARS})")
print()

# Verify no data leakage
assert train_df['performance_year'].max() < val_df['performance_year'].min(), "Train-Val temporal leak"
assert val_df['performance_year'].max() < test_df['performance_year'].min(), "Val-Test temporal leak"
print("✓ Temporal integrity verified: PAST → FUTURE")
print()

# ============================================================================
# 6. PREPROCESSING
# ============================================================================

print("Step 6: Preprocessing")
print("-" * 80)

# Separate features and target
X_train = train_df[selected_features].copy()
y_train = train_df['risk_tier'].copy()

X_val = val_df[selected_features].copy()
y_val = val_df['risk_tier'].copy()

X_test = test_df[selected_features].copy()
y_test = test_df['risk_tier'].copy()

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ X_val shape: {X_val.shape}")
print(f"✓ X_test shape: {X_test.shape}")
print()

# Identify categorical and numerical features
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns.tolist()

print(f"✓ Categorical features: {len(categorical_features)}")
print(f"  {categorical_features}")
print(f"✓ Numerical features: {len(numerical_features)}")
print()

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    # Fit on train only
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    
    # Transform val/test, handling unseen categories
    X_val[col] = X_val[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    X_test[col] = X_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_features)} categorical features")
print()

# Handle missing values in numerical features (simple median imputation on train)
imputation_values = {}
for col in numerical_features:
    if X_train[col].isna().sum() > 0:
        median_val = X_train[col].median()
        imputation_values[col] = median_val
        X_train[col].fillna(median_val, inplace=True)
        X_val[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
        print(f"✓ Imputed {col} with median: {median_val:.4f}")

if not imputation_values:
    print("✓ No missing values detected in numerical features")
print()

# Encode target variable
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_val_encoded = target_encoder.transform(y_val)
y_test_encoded = target_encoder.transform(y_test)

print(f"✓ Target classes: {target_encoder.classes_}")
print(f"✓ Target encoding: {dict(enumerate(target_encoder.classes_))}")
print()

# ============================================================================
# 7. XGBOOST TRAINING
# ============================================================================

print("Step 7: XGBoost Training")
print("-" * 80)

# XGBoost parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 30
}

print("XGBoost parameters:")
for k, v in xgb_params.items():
    print(f"  {k}: {v}")
print()

# Train model
print("Training XGBoost model...")
model = xgb.XGBClassifier(**xgb_params)

model.fit(
    X_train, y_train_encoded,
    eval_set=[(X_train, y_train_encoded), (X_val, y_val_encoded)],
    verbose=50
)

print("✓ Training complete")
print(f"✓ Best iteration: {model.best_iteration}")
print(f"✓ Best score: {model.best_score:.4f}")
print()

# Save model
model_path = MODELS_DIR / "xgboost_provider_risk.json"
model.save_model(str(model_path))
print(f"✓ Model saved to {model_path}")
print()

# Save preprocessing artifacts
joblib.dump(label_encoders, MODELS_DIR / "label_encoders.pkl")
joblib.dump(target_encoder, MODELS_DIR / "target_encoder.pkl")
joblib.dump(imputation_values, MODELS_DIR / "imputation_values.pkl")
print("✓ Preprocessing artifacts saved")
print()

# ============================================================================
# 8. PREDICTIONS
# ============================================================================

print("Step 8: Generate Predictions")
print("-" * 80)

# Validation predictions
val_probs = model.predict_proba(X_val)
val_preds = model.predict(X_val)
val_preds_labels = target_encoder.inverse_transform(val_preds)

# Test predictions
test_probs = model.predict_proba(X_test)
test_preds = model.predict(X_test)
test_preds_labels = target_encoder.inverse_transform(test_preds)

print("✓ Validation predictions generated")
print("✓ Test predictions generated")
print()

# Save predictions
val_predictions_df = val_df[['provider_id', 'aco_id', 'performance_year', 'risk_tier']].copy()
val_predictions_df['predicted_risk_tier'] = val_preds_labels
val_predictions_df['prob_LOW'] = val_probs[:, target_encoder.transform(['LOW'])[0]]
val_predictions_df['prob_MEDIUM'] = val_probs[:, target_encoder.transform(['MEDIUM'])[0]]
val_predictions_df['prob_HIGH'] = val_probs[:, target_encoder.transform(['HIGH'])[0]]
val_predictions_df.to_parquet(PREDICTIONS_DIR / "validation_predictions.parquet", index=False)

test_predictions_df = test_df[['provider_id', 'aco_id', 'performance_year', 'risk_tier']].copy()
test_predictions_df['predicted_risk_tier'] = test_preds_labels
test_predictions_df['prob_LOW'] = test_probs[:, target_encoder.transform(['LOW'])[0]]
test_predictions_df['prob_MEDIUM'] = test_probs[:, target_encoder.transform(['MEDIUM'])[0]]
test_predictions_df['prob_HIGH'] = test_probs[:, target_encoder.transform(['HIGH'])[0]]
test_predictions_df.to_parquet(PREDICTIONS_DIR / "test_predictions.parquet", index=False)

print(f"✓ Saved validation predictions to {PREDICTIONS_DIR / 'validation_predictions.parquet'}")
print(f"✓ Saved test predictions to {PREDICTIONS_DIR / 'test_predictions.parquet'}")
print()

# ============================================================================
# 9. EVALUATION METRICS
# ============================================================================

print("Step 9: Evaluation Metrics")
print("-" * 80)

def compute_metrics(y_true, y_pred, y_prob, split_name):
    """Compute comprehensive classification metrics"""
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['log_loss'] = log_loss(y_true, y_prob)
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True, 
                                          target_names=target_encoder.classes_)
    for cls in target_encoder.classes_:
        metrics[f'{cls}_precision'] = class_report[cls]['precision']
        metrics[f'{cls}_recall'] = class_report[cls]['recall']
        metrics[f'{cls}_f1'] = class_report[cls]['f1-score']
        metrics[f'{cls}_support'] = class_report[cls]['support']
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ROC AUC OVR
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        metrics['roc_auc_ovr'] = None
    
    return metrics

# Train metrics
train_probs = model.predict_proba(X_train)
train_preds = model.predict(X_train)
train_metrics = compute_metrics(y_train_encoded, train_preds, train_probs, "train")

# Validation metrics
val_metrics = compute_metrics(y_val_encoded, val_preds, val_probs, "validation")

# Test metrics
test_metrics = compute_metrics(y_test_encoded, test_preds, test_probs, "test")

# Save metrics
with open(METRICS_DIR / "validation_metrics.json", 'w') as f:
    json.dump(val_metrics, f, indent=2)

with open(METRICS_DIR / "test_metrics.json", 'w') as f:
    json.dump(test_metrics, f, indent=2)

print(f"✓ Saved validation metrics to {METRICS_DIR / 'validation_metrics.json'}")
print(f"✓ Saved test metrics to {METRICS_DIR / 'test_metrics.json'}")
print()

# Print test results
print("TEST SET RESULTS:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
print(f"  Log Loss: {test_metrics['log_loss']:.4f}")
print(f"  ROC AUC (OVR): {test_metrics['roc_auc_ovr']:.4f}" if test_metrics['roc_auc_ovr'] else "  ROC AUC: N/A")
print()
print("Per-class performance:")
for cls in target_encoder.classes_:
    print(f"  {cls}:")
    print(f"    Precision: {test_metrics[f'{cls}_precision']:.4f}")
    print(f"    Recall: {test_metrics[f'{cls}_recall']:.4f}")
    print(f"    F1: {test_metrics[f'{cls}_f1']:.4f}")
print()

# Classification report
with open(METRICS_DIR / "classification_report.txt", 'w') as f:
    f.write("="*80 + "\n")
    f.write("PROVIDER RISK CLASSIFICATION - TEST SET REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n")
    f.write(f"Model: XGBoost Multiclass Classifier\n")
    f.write(f"Test years: {TEST_YEARS}\n")
    f.write(f"Test samples: {len(y_test):,}\n\n")
    
    f.write(classification_report(y_test_encoded, test_preds, 
                                   target_names=target_encoder.classes_, digits=4))
    
    f.write(f"\n\nOverall Metrics:\n")
    f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
    f.write(f"  Macro Precision: {test_metrics['macro_precision']:.4f}\n")
    f.write(f"  Macro Recall: {test_metrics['macro_recall']:.4f}\n")
    f.write(f"  Macro F1: {test_metrics['macro_f1']:.4f}\n")
    f.write(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
    f.write(f"  Log Loss: {test_metrics['log_loss']:.4f}\n")
    if test_metrics['roc_auc_ovr']:
        f.write(f"  ROC AUC (OVR): {test_metrics['roc_auc_ovr']:.4f}\n")

print(f"✓ Saved classification report to {METRICS_DIR / 'classification_report.txt'}")
print()

print("="*80)
print("PHASE 9 TRAINING COMPLETE")
print("="*80)
print()
print("Next step: Run ml/explain_with_shap.py for SHAP explainability analysis")
print()
