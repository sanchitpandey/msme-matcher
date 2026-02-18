import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

logger = logging.getLogger(__name__)

# Ensure the data directory exists
if not settings.DATA_DIR.exists():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite URL
SQLALCHEMY_DATABASE_URL = f"sqlite:///{settings.DATA_DIR}/app.db"

# connect_args check_same_thread=False is needed only for SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency for FastAPI to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()