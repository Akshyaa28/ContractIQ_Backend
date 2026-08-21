"""
Load enhanced_synthetic_aco_quality_data.csv into the quality_data PostgreSQL table.
Run once: python scripts/load_quality_data.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from api.models.database import engine, Base, SessionLocal
from api.models.quality_data import QualityData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "quality" / "enhanced_synthetic_aco_quality_data.csv"


def main():
    print("=" * 60)
    print("Loading quality data into PostgreSQL")
    print("=" * 60)

    # Create table if not exists
    Base.metadata.create_all(bind=engine)
    print("  Tables ensured.")

    # Check if already loaded
    db = SessionLocal()
    existing = db.query(QualityData).count()
    if existing > 0:
        print(f"  Table already has {existing:,} rows. Skipping load.")
        print("  To reload: DELETE FROM quality_data; then run again.")
        db.close()
        return

    # Load CSV
    print(f"  Reading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df):,}")

    # Rename columns to match SQLAlchemy model (lowercase)
    column_map = {
        "ACO_ID": "aco_id",
        "Year_T": "year_t",
        "Year_T1": "year_t1",
        "Primary_State": "primary_state",
        "Revenue_Category": "revenue_category",
        "Track": "track",
        "Agreement_Period_Num": "agreement_period_num",
        "Policy_Version": "policy_version",
        "COVID_Period": "covid_period",
        "Beneficiary_Count": "beneficiary_count",
        "Hospital_Count": "hospital_count",
        "PCP_Count": "pcp_count",
        "Specialist_Count": "specialist_count",
        "Risk_Score": "risk_score",
        "Chronic_Disease_Rate_Pct": "chronic_disease_rate_pct",
        "Current_Quality_Score": "current_quality_score",
        "Previous_Quality_Score": "previous_quality_score",
        "Readmission_Rate_Pct": "readmission_rate_pct",
        "Admission_Rate_Per_1000": "admission_rate_per_1000",
        "ED_Visit_Rate_Per_1000": "ed_visit_rate_per_1000",
        "Preventable_Admission_Rate_Per_1000": "preventable_admission_rate_per_1000",
        "Patient_Experience_Score": "patient_experience_score",
        "Diabetes_Control_Rate_Pct": "diabetes_control_rate_pct",
        "Blood_Pressure_Control_Rate_Pct": "blood_pressure_control_rate_pct",
        "Preventive_Screening_Rate_Pct": "preventive_screening_rate_pct",
        "Followup_Compliance_Rate_Pct": "followup_compliance_rate_pct",
        "Expenditure_Per_Beneficiary": "expenditure_per_beneficiary",
        "Benchmark_Per_Beneficiary": "benchmark_per_beneficiary",
        "Total_Expenditure": "total_expenditure",
        "Benchmark_Expenditure": "benchmark_expenditure",
        "Savings_Amount": "savings_amount",
        "Savings_Rate": "savings_rate",
        "Final_Share_Rate": "final_share_rate",
        "Earned_Savings_Loss": "earned_savings_loss",
        "target_Quality_Score_T1": "target_quality_score_t1",
    }

    df = df.rename(columns=column_map)

    # Insert using pandas to_sql (fast bulk insert)
    print("  Inserting into PostgreSQL...")
    df.to_sql(
        "quality_data",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    # Verify
    count = db.query(QualityData).count()
    print(f"  Done. {count:,} rows in quality_data table.")
    db.close()

    print()
    print("=" * 60)
    print("LOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
