import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB

from api.models.database import Base


class PredictionInput(Base):
    __tablename__ = "prediction_inputs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    aco_id = Column(String(50), nullable=False, index=True)
    analysis_type = Column(String(20), nullable=False)  # 'risk' / 'forecast' / 'twin'

    # 8 model input features
    n_ab = Column(Float, nullable=False)
    previous_savings_rate = Column(Float, nullable=False)
    previous_quality_score = Column(Float, nullable=False)
    previous_performance_gap_pct = Column(Float, nullable=False)
    expenditure_growth_pct = Column(Float, nullable=False)
    benchmark_growth_pct = Column(Float, nullable=False)
    beneficiary_growth_pct = Column(Float, nullable=False)
    quality_change = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PredictionInput {self.aco_id} {self.analysis_type}>"


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_id = Column(UUID(as_uuid=True), ForeignKey("prediction_inputs.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    aco_id = Column(String(50), nullable=False, index=True)
    analysis_type = Column(String(20), nullable=False)
    result_json = Column(JSONB, nullable=False)  # Full model output

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PredictionResult {self.aco_id} {self.analysis_type}>"
