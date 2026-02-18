import json
import faiss
import numpy as np
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global Cache
_bm25 = None
_sbert = None
_faiss_index = None
_data = []

def load_resources():
    global _bm25, _sbert, _faiss_index, _data
    
    if _bm25 is not None:
        return

    logger.info("Loading Retrieval Resources...")
    
    if not settings.SNP_DATA_PATH.exists():
        logger.error(f"Data missing at {settings.SNP_DATA_PATH}")
        return
    
    try:
        with open(settings.SNP_DATA_PATH, "r") as f:
            _data = json.load(f)

        # BM25
        corpus = [d.get("capability_text", "").lower().split() for d in _data]
        _bm25 = BM25Okapi(corpus)

        # SBERT
        _sbert = SentenceTransformer(settings.SBERT_MODEL_NAME)

        # FAISS
        if settings.FAISS_INDEX_PATH.exists():
            _faiss_index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
            logger.info("Resources loaded successfully.")
        else:
            logger.error(f"FAISS index missing at {settings.FAISS_INDEX_PATH}")
            
    except Exception as e:
        logger.critical(f"Failed to initialize retrieval resources: {e}")

def search(query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Hybrid Search (BM25 + FAISS).
    Returns list of dicts with 'score' and 'source' fields.
    """
    if _bm25 is None: 
        load_resources()
    
    if not _data:
        return []

    try:
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

        # 3. Merge Results
        results_map = {}
        
        # Priority 1: Semantic matches
        for i, idx in enumerate(faiss_indices):
            if idx >= len(_data) or idx < 0: continue
            item = _data[idx].copy()
            item['score'] = float(faiss_scores[i])
            item['source'] = 'semantic'
            results_map[item['snp_id']] = item
            
        # Priority 2: Keyword matches
        for idx in bm25_indices:
            if idx >= len(_data) or idx < 0: continue
            item = _data[idx].copy()
            # Only add if not found via semantic search
            if item['snp_id'] not in results_map:
                item['score'] = 0.5 # Default score for keyword match
                item['source'] = 'keyword'
                results_map[item['snp_id']] = item

        return list(results_map.values())

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []