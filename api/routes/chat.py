"""
Chat endpoint — Groq-powered ACO chatbot for CMS portal.
"""
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.services.chat_service import run_chat, client, CHAT_MODEL


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/chat",
    tags=["Chatbot"],
)


# ============================================================
# SCHEMAS
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    aco_id: Optional[str] = Field("", description="ACO ID to query (e.g. 'A00001')")
    message: str = Field(..., description="User's message")
    conversation_history: list[ChatMessage] = Field(
        default=[],
        description="Previous messages for context",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "aco_id": "A00001",
                "message": "How is this ACO performing on quality metrics?",
                "conversation_history": [],
            }
        }
    }


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Chatbot response")
    aco_id: str = Field(..., description="ACO that was queried")
    sources: list[str] = Field(..., description="Data sources used")
    suggested_questions: list[str] = Field(..., description="Follow-up suggestions")


# ============================================================
# ENDPOINTS
# ============================================================

@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with ContractIQ Assistant",
    description=(
        "Send a message to the ContractIQ chatbot. "
        "Optionally provide an ACO ID for data-specific answers. "
        "The chatbot uses the quality_data table for ACO context "
        "and Groq LLM for intelligent responses."
    ),
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Main chat endpoint. No auth required (public for the CMS portal).
    """
    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    result = run_chat(
        db=db,
        aco_id=request.aco_id or "",
        message=request.message,
        history=history,
    )

    return ChatResponse(**result)


@router.get(
    "/health",
    summary="Check chatbot health",
    description="Verifies Groq API connectivity.",
)
def chat_health():
    """Check if Groq API is reachable."""
    try:
        client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return {"status": "ok", "groq_status": "connected", "model": CHAT_MODEL}
    except Exception as e:
        return {"status": "error", "groq_status": "error", "message": str(e)}
