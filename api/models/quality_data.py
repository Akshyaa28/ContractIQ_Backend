import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import UUID, JSONB

from api.models.database import Base


class QualityData(Base):
    """
    Lookup table — the full quality dataset (24,000 rows).
    Loaded once from enhanced_synthetic_aco_quality_data.csv.
    Frontend sends (aco_id, year_t) → backend looks up this table
    to get all 32 model features.
    """
    __tablename__ = "quality_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    aco_id = Column(String(50), nullable=False, index=True)
    year_t = Column(Integer, nullable=False, index=True)
    year_t1 = Column(Integer, nullable=False)
    primary_state = Column(String(50), nullable=False)
    revenue_category = Column(String(50), nullable=False)
    track = Column(String(50), nullable=False)
    agreement_period_num = Column(Integer, nullable=False)
    policy_version = Column(String(20), nullable=False)
    covid_period = Column(Integer, nullable=False)
    beneficiary_count = Column(Integer, nullable=False)
    hospital_count = Column(Integer, nullable=False)
    pcp_count = Column(Integer, nullable=False)
    specialist_count = Column(Integer, nullable=False)
    risk_score = Column(Float, nullable=False)
    chronic_disease_rate_pct = Column(Float, nullable=False)
    current_quality_score = Column(Float, nullable=False)
    previous_quality_score = Column(Float, nullable=True)  # NULL for first year
    readmission_rate_pct = Column(Float, nullable=False)
    admission_rate_per_1000 = Column(Float, nullable=False)
    ed_visit_rate_per_1000 = Column(Float, nullable=False)
    preventable_admission_rate_per_1000 = Column(Float, nullable=False)
    patient_experience_score = Column(Float, nullable=False)
    diabetes_control_rate_pct = Column(Float, nullable=False)
    blood_pressure_control_rate_pct = Column(Float, nullable=False)
    preventive_screening_rate_pct = Column(Float, nullable=False)
    followup_compliance_rate_pct = Column(Float, nullable=False)
    expenditure_per_beneficiary = Column(Float, nullable=False)
    benchmark_per_beneficiary = Column(Float, nullable=False)
    total_expenditure = Column(Float, nullable=False)
    benchmark_expenditure = Column(Float, nullable=False)
    savings_amount = Column(Float, nullable=False)
    savings_rate = Column(Float, nullable=False)
    final_share_rate = Column(Float, nullable=False)
    earned_savings_loss = Column(Float, nullable=False)
    target_quality_score_t1 = Column(Float, nullable=False)

    def __repr__(self):
        return f"<QualityData {self.aco_id} year_t={self.year_t}>"


class QualityPredictionInput(Base):
    """Records each quality prediction request."""
    __tablename__ = "quality_prediction_inputs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    aco_id = Column(String(50), nullable=False, index=True)
    year_t = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<QualityPredictionInput {self.aco_id} year_t={self.year_t}>"


class QualityPredictionResult(Base):
    """Stores the model output for each quality prediction."""
    __tablename__ = "quality_prediction_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    aco_id = Column(String(50), nullable=False, index=True)
    year_t = Column(Integer, nullable=False)
    features_json = Column(JSONB, nullable=False)  # The 32 features sent to model
    result_json = Column(JSONB, nullable=False)    # Model prediction output
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<QualityPredictionResult {self.aco_id} year_t={self.year_t}>"
