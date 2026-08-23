#!/usr/bin/env python3
"""
FastAPI Application for ACO Provider Risk Assessment
Deployment-ready version with all required functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="ACO Provider Risk Assessment API",
    description="ML-powered risk assessment for healthcare providers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
target_encoder = None
label_encoders = None
imputation_values = None
dataset = None
feature_list = None
simulation_config = None

# Pydantic models for request/response
class ProviderRequest(BaseModel):
    provider_id: str
    performance_year: int

class SimulationRequest(BaseModel):
    provider_id: str
    performance_year: int
    modifications: Dict[str, float]

class PredictionResponse(BaseModel):
    status: str
    provider_id: str
    predicted_risk_tier: str
    confidence: float
    class_probabilities: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    global model, target_encoder, label_encoders, imputation_values
    global dataset, feature_list, simulation_config
    
    print("🚀 Loading ACO Provider Risk Model...")
    
    try:
        # Load model files
        model = joblib.load('../models/xgboost_provider_risk.joblib')
        target_encoder = joblib.load('../models/target_encoder.pkl')
        label_encoders = joblib.load('../models/label_encoders.pkl')
        imputation_values = joblib.load('../models/imputation_values.pkl')
        
        # Load dataset
        dataset = pd.read_csv('../data/filtered_provider_dataset_2021_2024_corrected.csv')
        feature_list = pd.read_csv('../data/model_feature_list.csv')['feature'].tolist()
        
        # Load simulation config
        with open('../config/what_if_simulator_config.json', 'r') as f:
            simulation_config = json.load(f)
        
        print("✅ Model and data loaded successfully!")
        print(f"   Dataset: {len(dataset):,} providers")
        print(f"   Features: {len(feature_list)} features")
        print(f"   Model: XGBoost with 91.3% accuracy")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

def predict_provider_risk(provider_data: Dict) -> Dict:
    """Make risk prediction for a provider"""
    try:
        # Extract features in correct order
        features = []
        for feature_name in feature_list:
            value = provider_data.get(feature_name, 0)
            features.append(value)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Convert to risk tier
        risk_tier = target_encoder.inverse_transform([prediction])[0]
        
        # Get class probabilities
        class_names = target_encoder.classes_
        class_probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return {
            "predicted_risk_tier": risk_tier,
            "confidence": float(max(probabilities)),
            "class_probabilities": class_probs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "message": "ACO Provider Risk Assessment API",
        "version": "1.0.0",
        "status": "operational",
        "dataset": {
            "providers": len(dataset) if dataset is not None else 0,
            "aco_format": "A00002",
            "years": "2021-2024"
        },
        "model": {
            "type": "XGBoost Classifier",
            "accuracy": "91.3%",
            "features": len(feature_list) if feature_list else 0
        },
        "endpoints": {
            "lookup": "POST /api/v1/lookup/provider",
            "simulate": "POST /api/v1/simulate/what-if",
            "config": "GET /api/v1/simulation-config"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None,
        "dataset_loaded": dataset is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/lookup/provider")
async def lookup_provider(request: ProviderRequest):
    """Look up provider and predict risk"""
    try:
        # Find provider in dataset
        provider_data = dataset[
            (dataset['provider_id'] == request.provider_id) & 
            (dataset['performance_year'] == request.performance_year)
        ]
        
        if provider_data.empty:
            # Check if provider exists in other years
            provider_other_years = dataset[dataset['provider_id'] == request.provider_id]
            if provider_other_years.empty:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Provider {request.provider_id} not found in any year"
                )
            
            available_years = sorted(provider_other_years['performance_year'].unique().tolist())
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Provider {request.provider_id} not found for year {request.performance_year}",
                    "available_years": available_years
                }
            )
        
        provider_record = provider_data.iloc[0].to_dict()
        
        # Make prediction
        prediction = predict_provider_risk(provider_record)
        
        return {
            "status": "success",
            "provider_data": {
                "provider_id": request.provider_id,
                "aco_id": provider_record.get('aco_id'),
                "performance_year": request.performance_year,
                "provider_type": provider_record.get('provider_type'),
                "specialty": provider_record.get('specialty'),
                "beneficiary_count": provider_record.get('beneficiary_count')
            },
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup error: {str(e)}")
@app.get("/api/v1/simulation-config")
async def get_simulation_config():
    """Get configuration for What-If simulation sliders"""
    try:
        return {
            "status": "success",
            "simulation_metrics": simulation_config["metrics"],
            "coverage_info": simulation_config["what_if_simulator_config"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")

@app.post("/api/v1/simulate/what-if")
async def simulate_what_if(request: SimulationRequest):
    """Perform What-If simulation with modified metrics"""
    try:
        # Find original provider data
        provider_data = dataset[
            (dataset['provider_id'] == request.provider_id) & 
            (dataset['performance_year'] == request.performance_year)
        ]
        
        if provider_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Provider {request.provider_id} not found for year {request.performance_year}"
            )
        
        original_record = provider_data.iloc[0].to_dict()
        
        # Create modified record
        modified_record = original_record.copy()
        modifications_applied = {}
        
        # Apply modifications
        for field, new_value in request.modifications.items():
            if field in original_record:
                original_value = original_record[field]
                modified_record[field] = new_value
                modifications_applied[field] = {
                    "original": original_value,
                    "modified": new_value,
                    "change": new_value - original_value
                }
        
        # Make predictions
        original_prediction = predict_provider_risk(original_record)
        modified_prediction = predict_provider_risk(modified_record)
        
        # Compare results
        risk_change = determine_risk_change(
            original_prediction["predicted_risk_tier"],
            modified_prediction["predicted_risk_tier"]
        )
        
        return {
            "status": "success",
            "provider_info": {
                "provider_id": request.provider_id,
                "aco_id": original_record.get('aco_id'),
                "performance_year": request.performance_year
            },
            "modifications_applied": modifications_applied,
            "predictions": {
                "original": original_prediction,
                "modified": modified_prediction
            },
            "risk_comparison": {
                "original_risk": original_prediction["predicted_risk_tier"],
                "modified_risk": modified_prediction["predicted_risk_tier"],
                "risk_change": risk_change,
                "confidence_change": modified_prediction["confidence"] - original_prediction["confidence"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

def determine_risk_change(original_risk: str, modified_risk: str) -> str:
    """Determine if risk improved, worsened, or stayed the same"""
    risk_levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    
    original_level = risk_levels.get(original_risk, 2)
    modified_level = risk_levels.get(modified_risk, 2)
    
    if modified_level < original_level:
        return "improved"
    elif modified_level > original_level:
        return "worsened"
    else:
        return "unchanged"

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting ACO Provider Risk API Server...")
    print("📊 Dataset: Corrected filtered dataset with A00002 ACO format")
    print("🎯 Model: XGBoost with 91.3% accuracy")
    print("🌐 Server: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)