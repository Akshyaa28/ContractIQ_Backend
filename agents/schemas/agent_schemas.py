from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentAnalyzeRequest(BaseModel):
    """Request to trigger an agent analysis."""
    input_id: str = Field(..., description="UUID of the prediction input to analyze")


class RecommendationRequest(BaseModel):
    """Request for the recommendation agent."""
    input_id: str = Field(..., description="UUID of the prediction input")
    include_risk: bool = Field(True, description="Include risk agent analysis")
    include_forecast: bool = Field(True, description="Include forecast agent analysis")
    include_twin: bool = Field(True, description="Include twin agent analysis")
    include_quality: bool = Field(False, description="Include quality agent analysis")


class AgentAnalyzeResponse(BaseModel):
    """Response from any agent analysis."""
    agent_result_id: str = Field(..., description="UUID of stored agent result")
    agent_type: str = Field(..., description="Which agent ran")
    aco_id: str
    analysis: Dict[str, Any] = Field(..., description="The agent's structured analysis")
