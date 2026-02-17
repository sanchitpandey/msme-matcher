import json
import faiss
import numpy as np
import os
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/snp_profiles.json")
INDEX_PATH = os.path.join(BASE_DIR, "indices/faiss_snp.index")

# Global Cache
_bm25 = None
_sbert = None
_faiss_index = None
_data = []

def load_resources():
    global _bm25, _sbert, _faiss_index, _data
    
    print("Loading Retrieval Resources...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data missing at {DATA_PATH}")
        return
    
    with open(DATA_PATH, "r") as f:
        _data = json.load(f)

    # BM25
    corpus = [d.get("capability_text", "").lower().split() for d in _data]
    _bm25 = BM25Okapi(corpus)

    # SBERT
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # FAISS
    if os.path.exists(INDEX_PATH):
        _faiss_index = faiss.read_index(INDEX_PATH)
    else:
        print(f"Error: FAISS index missing at {INDEX_PATH}")

def search(query: str, top_k=50):
    """
    Returns candidates WITH scores for the Ranker.
    """
    if _bm25 is None: load_resources()
    
    # 1. BM25 Search
    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # 2. FAISS Search
    query_emb = _sbert.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    faiss_indices = []
    faiss_scores = []
    
    if _faiss_index:
        D, I = _faiss_index.search(query_emb, top_k)
        faiss_indices = I[0]
        faiss_scores = D[0]

    # 3. Merge & Attach Scores
    results_map = {}
    
    # Add FAISS results
    for i, idx in enumerate(faiss_indices):
        if idx >= len(_data) or idx < 0: continue
        item = _data[idx].copy()
        item['score'] = float(faiss_scores[i]) # Semantic Score
        item['source'] = 'semantic'
        results_map[item['snp_id']] = item
        
    # Add BM25 results
    for idx in bm25_indices:
        if idx >= len(_data) or idx < 0: continue
        item = _data[idx].copy()
        if item['snp_id'] not in results_map:
            item['score'] = 0.5 
            item['source'] = 'keyword'
            results_map[item['snp_id']] = item

    return list(results_map.values())