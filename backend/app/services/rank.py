import lightgbm as lgb
import numpy as np
import logging
from typing import List, Dict, Any

from app.core.config import settings
from app.services.features import compute_features

logger = logging.getLogger(__name__)

_ranker = None

def load_ranker():
    global _ranker
    if _ranker is not None:
        return

    if settings.LTR_MODEL_PATH.exists():
        try:
            _ranker = lgb.Booster(model_file=str(settings.LTR_MODEL_PATH))
            logger.info("LTR Ranker loaded.")
        except Exception as e:
            logger.error(f"Failed to load LTR model: {e}")
    else:
        logger.warning(f"LTR model not found at {settings.LTR_MODEL_PATH}. Skipping re-ranking.")

def re_rank_results(
    query_text: str, 
    query_category: str, 
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Re-orders retrieval candidates using LightGBM LambdaRank.
    """
    if not candidates:
        return []

    if _ranker is None:
        load_ranker()
    
    # Graceful fallback if model is missing
    if _ranker is None:
        return candidates 

    try:
        X_pred = []
        for cand in candidates:
            # Default score if missing (e.g., from pure keyword search)
            sem_score = cand.get('score', 0.5)
            
            features = compute_features(query_text, query_category, cand, sem_score)
            X_pred.append(features)
            
        X_pred_np = np.array(X_pred)
        
        # Predict relevance scores
        scores = _ranker.predict(X_pred_np)
        
        ranked_candidates = []
        for i, cand in enumerate(candidates):
            cand_copy = cand.copy() # Avoid mutating original list references
            cand_copy['ltr_score'] = float(scores[i])
            ranked_candidates.append(cand_copy)
            
        # Sort descending
        ranked_candidates.sort(key=lambda x: x['ltr_score'], reverse=True)
        
        return ranked_candidates

    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")
        return candidates