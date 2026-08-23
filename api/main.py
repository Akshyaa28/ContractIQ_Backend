"""
ContractIQ API — Main Application Entry Point
==============================================
Single FastAPI application serving all endpoints:
- Authentication (signup, login, logout)
- ML Model predictions (risk, forecast, quality, twin)
- AI Agents (risk, forecast, twin, quality, recommendation)
- Chatbot
- Prediction history
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

# --- Routes ---
from api.routes.auth import router as auth_router
from api.routes.predictions import router as predictions_router
from api.routes.quality_predict import router as quality_predict_router
from api.routes.chat import router as chat_router
from api.routes.risk import router as risk_router
from api.routes.forecast import router as forecast_router
from api.routes.quality import router as quality_router
from api.routes.twin import router as twin_router
from api.routes.provider_risk import router as provider_risk_router
from agents.orchestrator.routes import router as agents_router

# --- Database models (registered for table creation) ---
from api.models.database import engine, Base
from api.models.user import User  # noqa: F401
from api.models.prediction import PredictionInput, PredictionResult  # noqa: F401
from api.models.quality_data import QualityData, QualityPredictionInput, QualityPredictionResult  # noqa: F401
from api.models.provider_data import ProviderData  # noqa: F401
from agents.services.agent_service import AgentResult  # noqa: F401

# --- ML Services ---
from api.services.risk_service import get_model as get_risk_model, MODEL_PATH as RISK_MODEL_PATH, MODEL_FEATURES as RISK_FEATURES, RISK_THRESHOLD
from api.services.forecast_service import get_model as get_forecast_model, MODEL_PATH as FORECAST_MODEL_PATH, MODEL_FEATURES as FORECAST_FEATURES
from api.services.quality_service import get_model as get_quality_model, MODEL_PATH as QUALITY_MODEL_PATH, MODEL_FEATURES as QUALITY_FEATURES
from api.services.twin_service import get_model as get_twin_model, MODEL_DIR as TWIN_MODEL_DIR, MODEL_FEATURES as TWIN_FEATURES

logger = logging.getLogger("contractiq")


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables and load all ML models at startup."""

    # Database
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")

    # Models
    get_risk_model()
    get_forecast_model()
    get_quality_model()
    get_twin_model()
    logger.info("All ML models loaded")

    yield

    logger.info("Shutting down")


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="ContractIQ API",
    description="ACO Intelligence Platform — Risk, Forecast, Quality, Twin ML models + AI Agents + Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

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
app.include_router(chat_router)
app.include_router(agents_router)
app.include_router(risk_router)
app.include_router(forecast_router)
app.include_router(quality_router)
app.include_router(twin_router)
app.include_router(provider_risk_router)


# ============================================================
# SYSTEM ENDPOINTS
# ============================================================

@app.get("/health", tags=["System"], response_class=JSONResponse)
def health_check():
    """Liveness check — confirms all models are loaded."""
    return {
        "status": "healthy",
        "service": "ContractIQ API",
        "models": {
            "risk":     {"type": type(get_risk_model()).__name__, "features": len(RISK_FEATURES)},
            "forecast": {"type": type(get_forecast_model()).__name__, "features": len(FORECAST_FEATURES)},
            "quality":  {"type": "Ridge (Pipeline)", "features": len(QUALITY_FEATURES)},
            "twin":     {"type": type(get_twin_model()).__name__, "features": len(TWIN_FEATURES)},
        },
    }


@app.get("/", tags=["System"], response_class=JSONResponse)
def root():
    """API root with endpoint directory."""
    return {
        "message": "ContractIQ API is running.",
        "version": "1.0.0",
        "docs": "/docs",
    }
