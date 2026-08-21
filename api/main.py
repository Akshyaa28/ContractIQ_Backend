from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables before anything else
load_dotenv()

from api.routes.risk     import router as risk_router
from api.routes.forecast import router as forecast_router
from api.routes.quality  import router as quality_router
from api.routes.twin     import router as twin_router
from api.routes.auth     import router as auth_router
from api.routes.predictions import router as predictions_router
from api.routes.quality_predict import router as quality_predict_router

from api.models.database import engine, Base
from api.models.user import User  # noqa: F401 — ensure model is registered
from api.models.prediction import PredictionInput, PredictionResult  # noqa: F401
from api.models.quality_data import QualityData, QualityPredictionInput, QualityPredictionResult  # noqa: F401

from api.services.risk_service import (
    get_model       as get_risk_model,
    MODEL_PATH      as RISK_MODEL_PATH,
    MODEL_FEATURES  as RISK_FEATURES,
    RISK_THRESHOLD,
)
from api.services.forecast_service import (
    get_model       as get_forecast_model,
    MODEL_PATH      as FORECAST_MODEL_PATH,
    MODEL_FEATURES  as FORECAST_FEATURES,
)
from api.services.quality_service import (
    get_model       as get_quality_model,
    MODEL_PATH      as QUALITY_MODEL_PATH,
    MODEL_FEATURES  as QUALITY_FEATURES,
)
from api.services.twin_service import (
    get_model       as get_twin_model,
    MODEL_DIR       as TWIN_MODEL_DIR,
    MODEL_FEATURES  as TWIN_FEATURES,
)


# ============================================================
# LIFESPAN — startup / shutdown
# ============================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Eagerly loads all models at startup and creates database tables.
    """

    project_root = Path(__file__).resolve().parents[1]

    print()
    print("=" * 64)
    print("ContractIQ API — starting up")
    print("=" * 64)
    print(f"  Project root : {project_root}")
    print()

    # ----------------------------------------------------------
    # 0. Database — create tables if they don't exist
    # ----------------------------------------------------------
    print("  [0/4] Creating database tables ...")
    Base.metadata.create_all(bind=engine)
    print("        Tables ready  ✓")

    # ----------------------------------------------------------
    # 1. Risk model
    # ----------------------------------------------------------
    print()
    print("  [1/4] Loading risk prediction model ...")
    print(f"        File      : {RISK_MODEL_PATH.name}")
    print(f"        Threshold : {RISK_THRESHOLD}")
    risk_model = get_risk_model()
    print(f"        Type      : {type(risk_model).__name__}  ✓")

    # ----------------------------------------------------------
    # 2. Forecast model
    # ----------------------------------------------------------
    print()
    print("  [2/3] Loading savings forecast model ...")
    print(f"        File     : {FORECAST_MODEL_PATH.name}")
    forecast_model = get_forecast_model()
    print(f"        Type     : {type(forecast_model).__name__}  ✓")

    # ----------------------------------------------------------
    # 3. Quality model
    # ----------------------------------------------------------
    print()
    print("  [3/4] Loading quality score model ...")
    print(f"        File     : {QUALITY_MODEL_PATH.name}")
    quality_model = get_quality_model()
    print(f"        Type     : Ridge (Pipeline)  ✓")
    _ = quality_model  # consumed for validation side-effect

    # ----------------------------------------------------------
    # 4. Twin matching model
    # ----------------------------------------------------------
    print()
    print("  [4/4] Loading twin matching model ...")
    print(f"        Dir      : {TWIN_MODEL_DIR}")
    twin_model = get_twin_model()
    print(f"        Type     : {type(twin_model).__name__}  ✓")

    print()
    print("  All 4 models loaded.  API is ready.")
    print("=" * 64)
    print()

    yield

    # ----------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------
    print()
    print("ContractIQ API — shutting down.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ContractIQ API",
    description=(
        "## ContractIQ — ACO Intelligence Platform\n\n"
        "Three independent ML models served from a single API:\n\n"
        "---\n\n"
        "### 1. Risk Prediction Model\n"
        "Classifies whether an ACO is at risk of **not meeting** its "
        "savings benchmark next year.\n\n"
        "| Property | Value |\n"
        "|---|---|\n"
        "| Algorithm | Random Forest (n_estimators=400, max_depth=12) |\n"
        "| Threshold | 0.23 | Accuracy | 85.55% | ROC-AUC | 0.7948 |\n"
        "| Training data | 2018–2023, 42 834 ACO-year rows |\n\n"
        "---\n\n"
        "### 2. Savings Forecast Model\n"
        "Regresses the **next-year savings rate** for an ACO.\n\n"
        "| Property | Value |\n"
        "|---|---|\n"
        "| Algorithm | Random Forest |\n"
        "| Validation R² | 88.82% (2023 hold-out) |\n"
        "| MAE | 0.008 (≈ 0.8 pp) |\n"
        "| Training data | 2018–2023, 42 834 ACO-year rows |\n\n"
        "---\n\n"
        "### 3. Quality Score Prediction Model\n"
        "Regresses the **next-year composite quality score** (0–100).\n\n"
        "| Property | Value |\n"
        "|---|---|\n"
        "| Algorithm | Ridge (sklearn Pipeline) |\n"
        "| Test R² | 83.38% (2024 hold-out) |\n"
        "| MAE | 2.46 quality points |\n"
        "| Training data | Synthetic, 24 000 ACO-year rows |\n"
        "| Disclosure | Trained on SYNTHETIC data — not real CMS data |\n\n"
        "---\n\n"
        "### Endpoints\n"
        "| Method | Path | Description |\n"
        "|---|---|---|\n"
        "| POST | `/risk/predict` | Single ACO risk prediction |\n"
        "| POST | `/risk/predict/batch` | Batch risk (≤500) |\n"
        "| GET  | `/risk/model/info` | Risk model metadata |\n"
        "| POST | `/forecast/predict` | Single ACO savings forecast |\n"
        "| POST | `/forecast/predict/batch` | Batch forecast (≤500) |\n"
        "| GET  | `/forecast/model/info` | Forecast model metadata |\n"
        "| POST | `/quality/predict` | Single ACO quality score prediction |\n"
        "| POST | `/quality/predict/batch` | Batch quality (≤500) |\n"
        "| GET  | `/quality/model/info` | Quality model metadata |\n"
        "| GET  | `/health` | Liveness check (all 3 models) |\n"
    ),
    version="1.0.0",
    contact={"name": "ContractIQ"},
    license_info={"name": "Private"},
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

# Replace "*" with your frontend URL before going to production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(auth_router)
app.include_router(predictions_router)
app.include_router(quality_predict_router)
app.include_router(risk_router)
app.include_router(forecast_router)
app.include_router(quality_router)
app.include_router(twin_router)


# ============================================================
# SYSTEM ENDPOINTS
# ============================================================

@app.get(
    "/health",
    tags=["System"],
    summary="Liveness check",
    response_class=JSONResponse,
)
def health_check():
    """
    Returns HTTP 200 while all three models are loaded and the
    service is ready to accept traffic.
    """

    risk_model     = get_risk_model()
    forecast_model = get_forecast_model()
    quality_model  = get_quality_model()
    twin_model     = get_twin_model()

    return {
        "status":  "healthy",
        "service": "ContractIQ API",
        "models": {
            "risk": {
                "type":      type(risk_model).__name__,
                "file":      RISK_MODEL_PATH.name,
                "threshold": RISK_THRESHOLD,
                "features":  len(RISK_FEATURES),
            },
            "forecast": {
                "type":     type(forecast_model).__name__,
                "file":     FORECAST_MODEL_PATH.name,
                "features": len(FORECAST_FEATURES),
            },
            "quality": {
                "type":     "Ridge (Pipeline)",
                "file":     QUALITY_MODEL_PATH.name,
                "features": len(QUALITY_FEATURES),
                "r2_pct":   83.38,
            },
            "twin": {
                "type":     type(twin_model).__name__,
                "features": len(TWIN_FEATURES),
                "top_k":    5,
            },
        },
    }


@app.get(
    "/",
    tags=["System"],
    summary="API root",
    response_class=JSONResponse,
)
def root():
    """Welcome message with quick-reference links to all endpoints."""

    return {
        "message": "ContractIQ API is running.",
        "version": "1.0.0",
        "docs":    "/docs",
        "redoc":   "/redoc",
        "endpoints": {
            "health":                  "/health",
            "risk_predict":            "/risk/predict",
            "risk_predict_batch":      "/risk/predict/batch",
            "risk_model_info":         "/risk/model/info",
            "forecast_predict":        "/forecast/predict",
            "forecast_batch":          "/forecast/predict/batch",
            "forecast_model_info":     "/forecast/model/info",
            "quality_predict":         "/quality/predict",
            "quality_predict_batch":   "/quality/predict/batch",
            "quality_model_info":      "/quality/model/info",
            "twin_find":               "/twin/find",
            "twin_model_info":         "/twin/model/info",
        },
    }
