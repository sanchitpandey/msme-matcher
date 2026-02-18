from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# --- Request Schemas ---

class FeedbackCreate(BaseModel):
    search_id: int
    snp_id: str
    action: int  # 1=Positive, -1=Negative

# --- Response Schemas ---

class MatchResult(BaseModel):
    name: str
    snp_id: str
    location: str
    category: str
    capability_text: str
    score: float = 0.0
    ltr_score: float = 0.0
    price_tier: Optional[str] = "Med"

class MatchResponse(BaseModel):
    search_id: int
    count: int
    time_taken: str
    query_category: str
    matches: List[MatchResult]

class AnalyzeResponse(BaseModel):
    original_text: str
    predicted_category: str
    confidence: float
    extracted_attributes: Dict[str, Any]