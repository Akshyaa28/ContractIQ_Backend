"""
Load filtered_provider_dataset_2021_2024_corrected.csv into PostgreSQL.
Run once: python scripts/load_provider_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from api.models.database import engine, Base, SessionLocal
from api.models.provider_data import ProviderData

CSV_PATH = Path(__file__).resolve().parents[1] / "deployment_package" / "data" / "filtered_provider_dataset_2021_2024_corrected.csv"


def main():
    print("Loading provider data into PostgreSQL...")

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    existing = db.query(ProviderData).count()
    if existing > 0:
        print(f"Table already has {existing:,} rows. Skipping.")
        db.close()
        return

    df = pd.read_csv(CSV_PATH)
    print(f"CSV rows: {len(df):,}")

    # Lowercase column names to match SQLAlchemy model
    df.columns = df.columns.str.lower()

    df.to_sql("provider_data", engine, if_exists="append", index=False, method="multi", chunksize=1000)

    count = db.query(ProviderData).count()
    print(f"Done. {count:,} rows loaded into provider_data table.")
    db.close()


if __name__ == "__main__":
    main()
