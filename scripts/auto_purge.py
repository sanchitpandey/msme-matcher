import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Path Setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.core.config import settings
from app.models.sql_models import SearchLog, Feedback, Base

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AutoPurge")

def purge_old_data(days=30):
    """
    Deletes logs older than `days` to comply with Data Retention Policy.
    """
    db_url = f"sqlite:///{settings.DATA_DIR}/app.db"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    cutoff_date = datetime.utcnow() - timedelta(days=days)
    logger.info(f"Starting Purge. Deleting records older than: {cutoff_date}")

    try:
        # Delete old search logs
        deleted_logs = session.query(SearchLog).filter(SearchLog.timestamp < cutoff_date).delete()
        
        # Delete old feedback
        deleted_feedback = session.query(Feedback).filter(Feedback.timestamp < cutoff_date).delete()
        
        session.commit()
        logger.info(f"Compliance Check Complete.")
        logger.info(f"   - Deleted Search Logs: {deleted_logs}")
        logger.info(f"   - Deleted Feedback: {deleted_feedback}")
        
    except Exception as e:
        logger.error(f"Purge Failed: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    # In production, this runs via CRON daily.
    purge_old_data(days=30)