from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.core.db import Base

class SearchLog(Base):
    """
    Audit Trail: Logs every query made to the system.
    Used for: Compliance auditing, analytics, and debugging.
    """
    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String, index=True)
    detected_category = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Store top result IDs (comma-separated) to reproduce the AI's decision later
    top_results_ids = Column(String)
    
    # Relationship to feedback
    feedback = relationship("Feedback", back_populates="search_log")

class Feedback(Base):
    """
    RLHF Data: Stores user reactions to search results.
    Used for: Retraining the LTR model (Ticket 1 & 4 scripts can use this later).
    """
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    search_id = Column(Integer, ForeignKey("search_logs.id"))
    
    # Which supplier did they interact with?
    snp_id = Column(String)
    
    # Action: 1 (Click/Call), -1 (Report/Irrelevant), 0 (Viewed)
    action = Column(Integer) 
    timestamp = Column(DateTime, default=datetime.utcnow)

    search_log = relationship("SearchLog", back_populates="feedback")