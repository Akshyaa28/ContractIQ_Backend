# ContractIQ — AWS Deployment Guide

## Critical Constraints for AWS Lambda

| Constraint | Lambda Limit | Your Project |
|---|---|---|
| Deployment package (zipped) | 50 MB | ~15 MB (code + deps without models) |
| Unzipped package | 250 MB | ~180 MB (sklearn + pandas + numpy) |
| /tmp storage | 512 MB | 308 MB (all 4 model files) |
| Memory | Up to 10 GB | Recommend 2–4 GB |
| Timeout | 15 min max | Cold start ~15–30s with model loading |

## The Problem

Your **forecast model alone is 263 MB**. Lambda's unzipped package limit is 250 MB.
You CANNOT bundle the models inside the Lambda deployment package.

## Recommended Architecture

### Option A: Lambda + S3 (simplest)

```
┌─────────────────────┐     ┌────────┐     ┌──────────────┐
│  API Gateway (REST) │────▶│ Lambda │────▶│ S3 (models)  │
└─────────────────────┘     └────────┘     └──────────────┘
                                │
                          Downloads models
                          to /tmp on cold start
```

1. Store `.joblib` files in S3
2. Lambda downloads them to `/tmp/` on cold start
3. After first load, models stay in memory until Lambda recycles

**Cold start:** ~15–30 seconds (downloading 308 MB from S3)
**Warm requests:** ~50–200 ms

### Option B: ECS / Fargate (recommended for production)

```
┌─────────────────────┐     ┌──────────────────────┐
│  ALB (Load Balancer)│────▶│ ECS Fargate Container │
└─────────────────────┘     │  (Docker + models)    │
                            └──────────────────────┘
```

- No size limits — bundle everything in the Docker image
- Always warm — no cold start penalty
- Auto-scaling based on CPU/memory
- **Best for this project** given the 308 MB model payload

### Option C: Lambda + EFS

- Mount an EFS filesystem with models pre-loaded
- Lambda reads models directly from EFS (no S3 download)
- Faster cold start than Option A (~5–10 seconds)
- Slightly more complex setup

---

## Option A — Lambda + S3 Setup

### 1. Upload models to S3

```bash
aws s3 cp models/risk_prediction/risk_random_forest_85_87.joblib \
    s3://contractiq-models/production/risk_random_forest_85_87.joblib

aws s3 cp models/forecasting/forecast_random_forest.joblib \
    s3://contractiq-models/production/forecast_random_forest.joblib

aws s3 cp models/quality_prediction/quality_score_pipeline.joblib \
    s3://contractiq-models/production/quality_score_pipeline.joblib

aws s3 cp models/similar_twin/twin_nearest_neighbors.joblib \
    s3://contractiq-models/production/twin_nearest_neighbors.joblib

aws s3 cp models/similar_twin/twin_scaler.joblib \
    s3://contractiq-models/production/twin_scaler.joblib

aws s3 cp models/similar_twin/twin_feature_weights.json \
    s3://contractiq-models/production/twin_feature_weights.json

aws s3 cp data/processed/Dataset1_Model_Twin.csv \
    s3://contractiq-models/production/Dataset1_Model_Twin.csv
```

### 2. Lambda handler (`lambda_handler.py`)

```python
from mangum import Mangum
from api.main import app

handler = Mangum(app, lifespan="off")
```

### 3. Modify services to load from /tmp

Each service's `get_model()` function needs a check:
- If model exists locally → load it
- If not → download from S3 to /tmp → load it

### 4. Lambda configuration

- **Runtime:** Python 3.12
- **Memory:** 3008 MB (minimum for loading all models)
- **Timeout:** 120 seconds (for cold start)
- **Layers:** Use Lambda layers for numpy/scipy/sklearn or package with Docker

---

## Option B — ECS/Fargate Docker Setup

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY models/ models/
COPY data/processed/Dataset1_Model_Twin.csv data/processed/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and deploy

```bash
docker build -t contractiq-api .
docker tag contractiq-api:latest <account>.dkr.ecr.<region>.amazonaws.com/contractiq-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/contractiq-api:latest
```

---

## Package Compatibility with AWS Lambda Python 3.12

| Package | Version | Lambda Compatible | Notes |
|---|---|---|---|
| fastapi | 0.115.0 | ✅ | Pure Python |
| uvicorn | 0.30.6 | ✅ | Pure Python (not used in Lambda) |
| pydantic | 2.8.2 | ✅ | Has compiled extensions but provides wheels |
| scikit-learn | 1.5.1 | ✅ | Provides manylinux wheels for Lambda |
| pandas | 2.2.3 | ✅ | Provides manylinux wheels |
| numpy | 1.26.4 | ✅ | Provides manylinux wheels |
| scipy | 1.13.1 | ✅ | Provides manylinux wheels |
| joblib | 1.2.0 | ✅ | Pure Python |
| mangum | 0.19.0 | ✅ | Pure Python (Lambda ASGI adapter) |

**All production dependencies are Lambda-compatible.**

IMPORTANT: Use `scikit-learn==1.5.1` (not 1.8.0) to match the version that
trained the models. Version mismatch can cause `AttributeError` on unpickle.

---

## Files to push to GitHub

```
contractiq/
├── api/                    ← Push (code)
├── models/                 ← DO NOT push .joblib > 100 MB (use Git LFS or S3)
│   ├── quality_prediction/ ← Push (small: 8 KB)
│   └── similar_twin/       ← Push (small: 4.6 MB total)
├── data/                   ← DO NOT push CSVs (use S3)
├── preprocessing/          ← Push (scripts for reproducibility)
├── requirements.txt        ← Push
├── .env.example            ← Push
├── .gitignore              ← Push
├── DEPLOYMENT.md           ← Push
└── lambda_handler.py       ← Push (if using Lambda)
```

For files > 100 MB, use:
- **Git LFS** (`git lfs track "*.joblib"`) — stores in Git LFS storage
- **S3** — download at deploy time (recommended for Lambda)
