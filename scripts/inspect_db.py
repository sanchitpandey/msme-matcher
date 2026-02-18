from sqlalchemy import create_engine, text
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath("backend"))
from app.core.config import settings

def main():
    db_path = f"sqlite:///{settings.DATA_DIR}/app.db"
    engine = create_engine(db_path)
    
    with engine.connect() as conn:
        print("\n--- SEARCH LOGS ---")
        result = conn.execute(text("SELECT id, query_text, detected_category FROM search_logs ORDER BY id DESC LIMIT 5"))
        for row in result:
            print(f"ID: {row[0]} | Query: {row[1]} | Cat: {row[2]}")

        print("\n--- FEEDBACK LOGS ---")
        result = conn.execute(text("SELECT id, search_id, snp_id, action FROM feedback ORDER BY id DESC LIMIT 5"))
        for row in result:
            print(f"ID: {row[0]} | Search ID: {row[1]} | Action: {row[3]}")

if __name__ == "__main__":
    main()