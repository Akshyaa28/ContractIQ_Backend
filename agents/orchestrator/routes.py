"""
Agent API endpoints — user-triggered, one agent at a time.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.models.user import User
from api.middleware.auth_middleware import get_current_user

from agents.schemas.agent_schemas import (
    AgentAnalyzeRequest,
    RecommendationRequest,
    AgentAnalyzeResponse,
)
from agents.services.prediction_service import (
    get_prediction_context,
    get_quality_prediction_context,
)
from agents.services.agent_service import store_agent_result, AgentResult
from agents.orchestrator.orchestrator import route_to_agent
from agents.recommendation_agent.agent import run_recommendation_agent


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/agents",
    tags=["AI Agents"],
)


# ============================================================
# GENERIC ORCHESTRATOR — auto-routes by analysis_type
# ============================================================

@router.post(
    "/analyze",
    response_model=AgentAnalyzeResponse,
    summary="Run agent analysis (auto-routed by analysis_type)",
)
def analyze(
    request: AgentAnalyzeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AgentAnalyzeResponse:
    """
    Retrieves prediction from DB, routes to the correct agent,
    stores the agent result, and returns the analysis.
    """
    try:
        context = get_prediction_context(db, request.input_id, str(user.id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    try:
        analysis = route_to_agent(context)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        )

    result_id = store_agent_result(
        db=db,
        input_id=request.input_id,
        user_id=str(user.id),
        aco_id=context["aco_id"],
        analysis_type=context["analysis_type"],
        agent_type=context["analysis_type"],
        result_json=analysis,
    )

    return AgentAnalyzeResponse(
        agent_result_id=result_id,
        agent_type=context["analysis_type"],
        aco_id=context["aco_id"],
        analysis=analysis,
    )


# ============================================================
# SPECIALIZED ENDPOINTS
# ============================================================

@router.post("/risk/analyze", response_model=AgentAnalyzeResponse, summary="Run Risk Agent only")
def analyze_risk(
    request: AgentAnalyzeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AgentAnalyzeResponse:
    return _run_specific_agent(request.input_id, "risk", db, user)


@router.post("/forecast/analyze", response_model=AgentAnalyzeResponse, summary="Run Forecast Agent only")
def analyze_forecast(
    request: AgentAnalyzeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AgentAnalyzeResponse:
    return _run_specific_agent(request.input_id, "forecast", db, user)


@router.post("/twin/analyze", response_model=AgentAnalyzeResponse, summary="Run Twin Agent only")
def analyze_twin(
    request: AgentAnalyzeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AgentAnalyzeResponse:
    return _run_specific_agent(request.input_id, "twin", db, user)


@router.post("/quality/analyze", response_model=AgentAnalyzeResponse, summary="Run Quality Agent only")
def analyze_quality(
    request: AgentAnalyzeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AgentAnalyzeResponse:
    """Quality uses a different DB table, so handle separately."""
    try:
        context = get_quality_prediction_context(db, request.input_id, str(user.id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    try:
        from agents.quality_agent.agent import run_quality_agent
        analysis = run_quality_agent(context)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality agent execution failed: {exc}",
        )

    result_id = store_agent_result(
        db=db,
        input_id=request.input_id,
        user_id=str(user.id),
        aco_id=context["aco_id"],
        analysis_type="quality",
        agent_type="quality",
        result_json=analysis,
    )

    return AgentAnalyzeResponse(
        agent_result_id=result_id,
        agent_type="quality",
        aco_id=context["aco_id"],
        analysis=analysis,
    )


# ============================================================
# RECOMMENDATION AGENT
# ============================================================

@router.post("/recommendation/analyze", response_model=AgentAnalyzeResponse, summary="Run Recommendation Agent")
def analyze_recommendation(
    request: RecommendationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AgentAnalyzeResponse:
    """
    Synthesizes available agent analyses into final recommendations.
    Retrieves existing agent_results for the given input_id.
    """
    # Get the base prediction context for aco_id
    try:
        context = get_prediction_context(db, request.input_id, str(user.id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # Collect existing agent results for this input_id
    agent_outputs = {}
    existing = db.query(AgentResult).filter(
        AgentResult.input_id == request.input_id,
        AgentResult.user_id == user.id,
    ).all()

    for r in existing:
        if r.agent_type == "risk" and request.include_risk:
            agent_outputs["risk"] = r.result_json
        elif r.agent_type == "forecast" and request.include_forecast:
            agent_outputs["forecast"] = r.result_json
        elif r.agent_type == "twin" and request.include_twin:
            agent_outputs["twin"] = r.result_json
        elif r.agent_type == "quality" and request.include_quality:
            agent_outputs["quality"] = r.result_json

    if not agent_outputs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No agent analyses found for this input_id. Run specialized agents first.",
        )

    try:
        analysis = run_recommendation_agent(context["aco_id"], agent_outputs)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation agent failed: {exc}",
        )

    result_id = store_agent_result(
        db=db,
        input_id=request.input_id,
        user_id=str(user.id),
        aco_id=context["aco_id"],
        analysis_type=context["analysis_type"],
        agent_type="recommendation",
        result_json=analysis,
    )

    return AgentAnalyzeResponse(
        agent_result_id=result_id,
        agent_type="recommendation",
        aco_id=context["aco_id"],
        analysis=analysis,
    )


# ============================================================
# HELPER
# ============================================================

def _run_specific_agent(
    input_id: str,
    expected_type: str,
    db: Session,
    user: User,
) -> AgentAnalyzeResponse:
    """Helper for specialized endpoints — validates type match."""
    try:
        context = get_prediction_context(db, input_id, str(user.id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if context["analysis_type"] != expected_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This input has analysis_type='{context['analysis_type']}', "
                   f"but you called the '{expected_type}' agent endpoint.",
        )

    try:
        analysis = route_to_agent(context)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        )

    result_id = store_agent_result(
        db=db,
        input_id=input_id,
        user_id=str(user.id),
        aco_id=context["aco_id"],
        analysis_type=expected_type,
        agent_type=expected_type,
        result_json=analysis,
    )

    return AgentAnalyzeResponse(
        agent_result_id=result_id,
        agent_type=expected_type,
        aco_id=context["aco_id"],
        analysis=analysis,
    )
