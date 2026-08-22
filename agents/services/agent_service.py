"""
Agent result storage service.
Stores agent analysis outputs separately from ML model outputs.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Session

from api.models.database import Base


class AgentResult(Base):
    """Stores agent analysis outputs — separate from ML model predictions."""
    __tablename__ = "agent_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    aco_id = Column(String(50), nullable=False)
    analysis_type = Column(String(20), nullable=False)
    agent_type = Column(String(30), nullable=False)  # risk/forecast/twin/quality/recommendation
    result_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AgentResult {self.agent_type} {self.aco_id}>"


def store_agent_result(
    db: Session,
    input_id: str,
    user_id: str,
    aco_id: str,
    analysis_type: str,
    agent_type: str,
    result_json: dict,
) -> str:
    """Store agent result and return its UUID."""
    record = AgentResult(
        input_id=input_id,
        user_id=user_id,
        aco_id=aco_id,
        analysis_type=analysis_type,
        agent_type=agent_type,
        result_json=result_json,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return str(record.id)
