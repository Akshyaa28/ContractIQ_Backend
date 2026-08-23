"""
Agent Orchestrator
==================
Routes to the correct specialized agent based on analysis_type.
Never executes unrelated agents.
"""

from agents.risk_agent.agent import run_risk_agent
from agents.forecast_agent.agent import run_forecast_agent
from agents.twin_agent.agent import run_twin_agent
from agents.quality_agent.agent import run_quality_agent

AGENT_MAP = {
    "risk": run_risk_agent,
    "forecast": run_forecast_agent,
    "twin": run_twin_agent,
    "quality": run_quality_agent,
}


def route_to_agent(context: dict) -> dict:
    """Route to the correct agent based on analysis_type in context."""
    analysis_type = context.get("analysis_type")
    if analysis_type not in AGENT_MAP:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
    return AGENT_MAP[analysis_type](context)
