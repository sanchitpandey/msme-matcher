import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.services.retrieve import search, load_resources
from app.services.asr import transcribe_audio
from app.services.extract import extract_attributes
from app.services.classify import predict_category, load_classifier
from app.services.rank import re_rank_results, load_ranker

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifecycle Manager.
    Loads heavy models into memory on startup.
    """
    logger.info(f"Starting {settings.APP_NAME}...")
    try:
        load_resources()   # Retrieval Indices
        load_classifier()  # Category Model
        load_ranker()      # LTR Model
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME, 
    lifespan=lifespan,
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "System Operational", "version": "1.0.0"}

@app.post(f"{settings.API_V1_STR}/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    return {"transcript": text}

@app.post(f"{settings.API_V1_STR}/analyze")
async def analyze_product(text: str = Form(...)):
    category, conf = predict_category(text)
    attrs = extract_attributes(text)
    return {
        "original_text": text,
        "predicted_category": category,
        "confidence": round(conf, 2),
        "extracted_attributes": attrs
    }

@app.post(f"{settings.API_V1_STR}/match")
async def match_endpoint(query: str = Form(...)):
    start = time.time()
    
    try:
        # 1. Pipeline Execution
        query_cat, conf = predict_category(query)
        candidates = search(query, top_k=50)
        ranked_results = re_rank_results(query, query_cat, candidates)
        
        final_results = ranked_results[:10]
        
        return {
            "count": len(final_results),
            "time_taken": f"{time.time() - start:.3f}s",
            "query_category": query_cat,
            "matches": final_results
        }
    except Exception as e:
        logger.error(f"Match endpoint error: {e}")
        return {"error": "Internal Processing Error", "details": str(e)}