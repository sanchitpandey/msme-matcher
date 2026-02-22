import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from deep_translator import GoogleTranslator

# Core & DB
from app.core.config import settings
from app.core.db import engine, Base, get_db
from app.models.sql_models import SearchLog, Feedback
from app.core.schemas import FeedbackCreate, MatchResponse, AnalyzeResponse
from app.core.schemas_ondc import ONDCSearchRequest, ONDCOnSearchResponse

# Services
from app.services.retrieve import search, load_resources
from app.services.asr import transcribe_audio
from app.services.extract import extract_attributes
from app.services.classify import predict_category, load_classifier
from app.services.rank import re_rank_results, load_ranker
from app.services.ondc_adapter import process_ondc_search
from app.services.ocr import extract_text_from_image, build_auto_filled_form

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
    return {
        "system": "IndiaAI MSME Intelligent Mapping System",
        "status": "operational",
        "modules": [
            "ASR Voice Registration",
            "OCR Document Processing",
            "Auto Form Filling",
            "Product Classification",
            "Semantic Matching",
            "LTR Ranking",
            "ONDC Integration",
            "Feedback Learning"
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# --- ONDC ENDPOINT ---

@app.post("/ondc/search", response_model=ONDCOnSearchResponse)
async def ondc_search_endpoint(request: ONDCSearchRequest):
    """
    ONDC Adapter Endpoint.
    Accepts ONDC Protocol JSON -> Returns ONDC Catalog JSON.
    """
    try:
        response = process_ondc_search(request)
        return response
    except Exception as e:
        logger.error(f"ONDC handler failed: {e}")
        raise HTTPException(status_code=500, detail="ONDC Adapter Error")

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

@app.post(f"{settings.API_V1_STR}/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    OCR + Auto Form Filling Endpoint.

    Accepts:
        GST certificate
        product catalog
        invoice image
        factory document

    Returns:
        structured auto-filled MSME registration data.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        file_bytes = await file.read()

        raw_text = extract_text_from_image(file_bytes)

        if not raw_text:
            return {
                "status": "failed",
                "message": "No readable text detected"
            }

        form = build_auto_filled_form(raw_text)

        return {
            "status": "success",
            "auto_filled_form": form
        }

    except Exception as e:
        logger.error(f"OCR endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="OCR processing failed")

@app.post(f"{settings.API_V1_STR}/match", response_model=MatchResponse)
async def match_endpoint(
    query: str = Form(...), 
    db: Session = Depends(get_db)
):
    start = time.time()
    try:
        search_query = query.strip()
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(search_query)
            
            if translated and translated.lower() != search_query.lower():
                logger.info(f"Text Translation Triggered: '{search_query}' -> '{translated}'")
                search_query = translated
        except Exception as e:
            logger.warning(f"Text translation failed (falling back to original): {e}")
        
        # 1. AI Pipeline
        query_cat, conf = predict_category(search_query)
        candidates = search(search_query, top_k=50)
        # Category Filter
        filtered = []
        for c in candidates:
            if query_cat.lower() in c.get("category", "").lower():
                filtered.append(c)

        # fallback if nothing matched
        if filtered:
            candidates = filtered
        ranked_results = re_rank_results(search_query, query_cat, candidates)
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