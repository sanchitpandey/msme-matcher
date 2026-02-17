from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import time

# Services
from app.services.retrieve import search, load_resources
from app.services.asr import transcribe_audio
from app.services.extract import extract_attributes
from app.services.classify import predict_category, load_classifier
from app.services.rank import re_rank_results, load_ranker

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up... loading AI models.")
    load_resources()   # Search Index
    load_classifier()  # Category Model
    load_ranker()      # LTR Model
    yield
    print("Shutting down...")

app = FastAPI(title="IndiaAI MSME Matching System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "IndiaAI MSME AI system running"}

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    return {"transcript": text}

@app.post("/api/match")
async def match_endpoint(query: str = Form(...)):
    """
    Full Pipeline:
    1. Understand (Classify)
    2. Retrieve (BM25 + FAISS)
    3. Re-Rank (LightGBM)
    """
    start = time.time()
    
    # Step 1: Understand
    query_cat, conf = predict_category(query)
    
    # Step 2: Retrieve
    candidates = search(query, top_k=50)
    
    # Step 3: Rank
    ranked_results = re_rank_results(query, query_cat, candidates)
    
    # Return top 10 most relevant
    final_results = ranked_results[:10]
    
    return {
        "count": len(final_results),
        "time_taken": f"{time.time() - start:.3f}s",
        "query_category": query_cat,
        "matches": final_results
    }

@app.post("/api/analyze")
async def analyze_product(text: str = Form(...)):
    category, conf = predict_category(text)
    attrs = extract_attributes(text)
    return {
        "original_text": text,
        "predicted_category": category,
        "confidence": round(conf, 2),
        "extracted_attributes": attrs
    }