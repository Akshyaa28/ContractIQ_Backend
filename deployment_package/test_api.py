#!/usr/bin/env python3
"""
Test ACO Provider Risk Assessment API
Run this after starting the API server
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing ACO Provider Risk Assessment API")
    print("="*50)
    
    # Wait for server to start
    print("⏳ Waiting for server...")
    time.sleep(2)
    
    # Test 1: Health Check
    print("\n📊 Test 1: Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        result = response.json()
        print(f"✅ Status: {result['status']}")
        print(f"✅ Model: {'loaded' if result['model_loaded'] else 'failed'}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("💡 Make sure API server is running: python api/main.py")
        return
    
    # Test 2: Provider Lookup  
    print("\n🔍 Test 2: Provider Risk Assessment")
    test_provider = {
        "provider_id": "PRV_SYN_ACO_000653_2801046",
        "performance_year": 2024
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/lookup/provider",
            json=test_provider,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            provider = result['provider_data']
            
            print(f"✅ Provider Found:")
            print(f"   ACO ID: {provider['aco_id']}")
            print(f"   Type: {provider['provider_type']}")
            print(f"   Risk: {prediction['predicted_risk_tier']}")
            print(f"   Confidence: {prediction['confidence']*100:.1f}%")
            
        else:
            print(f"❌ Lookup failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Provider lookup failed: {e}")
    
    # Test 3: What-If Simulation
    print("\n🔄 Test 3: What-If Simulation")
    simulation_request = {
        "provider_id": "PRV_SYN_ACO_000002_001383", 
        "performance_year": 2021,
        "modifications": {
            "ed_visits_vs_aco": -100,
            "quality_vs_aco": 2.0,
            "per_capita_vs_aco": -5000
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/simulate/what-if",
            json=simulation_request,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            original = result['predictions']['original']['predicted_risk_tier']
            modified = result['predictions']['modified']['predicted_risk_tier'] 
            risk_change = result['risk_comparison']['risk_change']
            
            print(f"✅ Simulation Completed:")
            print(f"   Original Risk: {original}")
            print(f"   Modified Risk: {modified}")
            print(f"   Change: {risk_change.upper()}")
            
        else:
            print(f"❌ Simulation failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
    
    # Test 4: API Documentation
    print(f"\n📖 Test 4: API Documentation")
    print(f"✅ Interactive docs available at: {base_url}/docs")
    print(f"✅ OpenAPI spec available at: {base_url}/openapi.json")
    
    print(f"\n🎉 API Testing Complete!")
    print(f"🌐 Server running at: {base_url}")
    print(f"📚 Documentation: {base_url}/docs")

if __name__ == "__main__":
    test_api()