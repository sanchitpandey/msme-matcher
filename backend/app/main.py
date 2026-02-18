import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Core & DB
from app.core.config import settings
from app.core.db import engine, Base, get_db
from app.models.sql_models import SearchLog, Feedback
from app.core.schemas import FeedbackCreate, MatchResponse, AnalyzeResponse

# Services
from app.services.retrieve import search, load_resources
from app.services.asr import transcribe_audio
from app.services.extract import extract_attributes
from app.services.classify import predict_category, load_classifier
from app.services.rank import re_rank_results, load_ranker
from app.services.features import extract_location_from_query

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create Database Tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME}...")
    try:
        load_resources()
        load_classifier()
        load_ranker()
        logger.info("All AI services ready.")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
    yield
    logger.info("Shutting down...")

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "System Operational", "status": "online"}

# --- AI ENDPOINTS ---

@app.post(f"{settings.API_V1_STR}/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        audio_bytes = await file.read()
        text = transcribe_audio(audio_bytes)
        return {"transcript": text}
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        raise HTTPException(status_code=500, detail="Audio processing failed")

@app.post(f"{settings.API_V1_STR}/analyze", response_model=AnalyzeResponse)
async def analyze_product(text: str = Form(...)):
    category, conf = predict_category(text)
    attrs = extract_attributes(text)
    return {
        "original_text": text,
        "predicted_category": category,
        "confidence": round(conf, 2),
        "extracted_attributes": attrs
    }

@app.post(f"{settings.API_V1_STR}/match", response_model=MatchResponse)
async def match_endpoint(
    query: str = Form(...), 
    db: Session = Depends(get_db)
):
    start = time.time()
    try:
        # 1. AI Pipeline
        query_cat, conf = predict_category(query)
        candidates = search(query, top_k=50)
        ranked_results = re_rank_results(query, query_cat, candidates)
        final_results = ranked_results[:10]
        
        # 2. Audit Logging
        top_ids = ",".join([r.get('snp_id', '') for r in final_results[:5]])
        log_entry = SearchLog(
            query_text=query,
            detected_category=query_cat,
            top_results_ids=top_ids
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        return {
            "search_id": log_entry.id,
            "count": len(final_results),
            "time_taken": f"{time.time() - start:.3f}s",
            "query_category": query_cat,
            "matches": final_results
        }
    except Exception as e:
        logger.error(f"Match error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/feedback")
async def submit_feedback(
    feedback: FeedbackCreate, 
    db: Session = Depends(get_db)
):
    try:
        fb_entry = Feedback(
            search_id=feedback.search_id,
            snp_id=feedback.snp_id,
            action=feedback.action
        )
        db.add(fb_entry)
        db.commit()
        return {"status": "success", "msg": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")
