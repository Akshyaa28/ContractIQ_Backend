# ACO Provider Risk Assessment API - Deployment Package

## 🎯 Overview
Complete deployment package for the ACO Provider Risk ML model with FastAPI backend.

## ✅ Features
- XGBoost ML model (91.3% accuracy)
- FastAPI REST API with automatic docs
- What-If simulation capabilities  
- Corrected dataset (A00002 ACO format)
- 2021-2024 data (58,933 providers)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start API Server
```bash
cd api
python main.py
```

### 3. Access API
- **API URL:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## 🔌 API Endpoints

### Provider Lookup & Risk Assessment
```bash
POST http://localhost:8000/api/v1/lookup/provider
Content-Type: application/json

{
  "provider_id": "PRV_SYN_ACO_000653_2801046",
  "performance_year": 2024
}
```

### What-If Risk Simulation
```bash
POST http://localhost:8000/api/v1/simulate/what-if
Content-Type: application/json

{
  "provider_id": "PRV_SYN_ACO_000653_2801046", 
  "performance_year": 2024,
  "modifications": {
    "ed_visits_vs_aco": -100,
    "quality_vs_aco": 2.0,
    "per_capita_vs_aco": -5000,
    "admissions_vs_aco": -50,
    "snf_admission_vs_aco": -20
  }
}
```

### Get Simulation Configuration
```bash
GET http://localhost:8000/api/v1/simulation-config
```

## 📁 Package Contents
```
deployment_package/
├── api/
│   └── main.py              # FastAPI application
├── models/
│   ├── xgboost_provider_risk.joblib    # Trained ML model
│   ├── target_encoder.pkl              # Risk tier encoder  
│   ├── label_encoders.pkl              # Category encoders
│   ├── imputation_values.pkl           # Missing value handlers
│   └── metadata.json                   # Model metadata
├── data/
│   ├── filtered_provider_dataset_2021_2024_corrected.csv
│   └── model_feature_list.csv          # Required 41 features
├── config/
│   ├── what_if_simulator_config.json   # Simulation settings
│   └── aco_id_mapping.json             # ACO ID transformations
├── requirements.txt                    # Python dependencies
└── README.md                          # This documentation
```

## 🎯 Model Details
- **Algorithm:** XGBoost Classifier
- **Accuracy:** 91.3%  
- **Input Features:** 41 engineered features
- **Output Classes:** HIGH, MEDIUM, LOW risk
- **Training Data:** 58,933 providers (2021-2024)
- **ACO ID Format:** A00002 (corrected format)

## 🔄 What-If Simulation
Modify 5 key metrics that control 46.3% of model decisions:

1. **Emergency Room Visits vs ACO** (`ed_visits_vs_aco`)
2. **Quality Score vs ACO** (`quality_vs_aco`)
3. **Per Capita Cost vs ACO** (`per_capita_vs_aco`) 
4. **Hospital Admissions vs ACO** (`admissions_vs_aco`)
5. **Skilled Nursing Facility vs ACO** (`snf_admission_vs_aco`)

## 📊 Test Examples

### HIGH Risk Provider (shows simulate button)
```json
{
  "provider_id": "PRV_SYN_ACO_000002_001383",
  "performance_year": 2021
}
```

### LOW Risk Provider (no simulate button)
```json
{
  "provider_id": "PRV_SYN_ACO_000653_2801046", 
  "performance_year": 2024
}
```

## 🛡️ Production Notes
- Add authentication/authorization as needed
- Configure environment variables for security
- Use gunicorn/uvicorn for production WSGI
- Set up proper logging and monitoring
- Consider rate limiting for public APIs

## ✅ Ready for Deployment
This package is completely self-contained and ready to run on any system with Python 3.8+!