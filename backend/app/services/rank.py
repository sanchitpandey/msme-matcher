import lightgbm as lgb
import numpy as np
import logging
from typing import List, Dict, Any

from app.core.config import settings
from app.services.features import compute_features, extract_location_from_query
from app.services.geo import get_coordinates, haversine_distance

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
    
    if not candidates:
        return []

    if _ranker is None:
        load_ranker()

    if _ranker is None:
        for cand in candidates:
            cand.setdefault("ltr_score", 0.5)
            cand.setdefault("absolute_score", 0.5)
            cand.setdefault("explain", {})
        return candidates

    try:
        X_pred = []
        for cand in candidates:
            sem_score = cand.get("score", 0.0)
            features = compute_features(query_text, query_category, cand, sem_score)
            X_pred.append(features)
            
        X_pred_np = np.array(X_pred, dtype=np.float32)
        raw_scores = _ranker.predict(X_pred_np)

        ranked_candidates = []
        q_city = extract_location_from_query(query_text)
        q_coords = get_coordinates(q_city)

        for i, cand in enumerate(candidates):
            cand_copy = cand.copy()

            semantic_sim = float(cand.get("score", 0.0)) 
            cand_cat = cand.get("category", "")
            cat_match = 1.0 if query_category and cand_cat and query_category.lower() in cand_cat.lower() else 0.0
            capacity = float(cand.get("capacity_score", 0.0))

            cand_coords = get_coordinates(cand.get("location"))
            dist_km = float(haversine_distance(q_coords, cand_coords)) if (q_coords and cand_coords) else 2000.0

            confidence = semantic_sim 
            
            if cat_match == 1.0:
                confidence += 0.25  # Greater boost for right industry
            else:
                confidence -= 0.40  # Greater penalty for wrong industry
                
            if dist_km < 100:
                confidence += 0.10  # Local boost
            elif dist_km > 1000:
                confidence -= 0.15  # Out-of-state penalty
                
            # Floor it at 1% and cap it at 99.9%
            absolute_score = max(0.01, min(0.999, confidence))

            price_map = {"Low": 1, "Med": 2, "High": 3}
            price_factor = float(price_map.get(cand.get("price_tier", "Med"), 2))

            cand_copy["ltr_score"] = float(raw_scores[i]) if hasattr(raw_scores, "__len__") else float(raw_scores)
            cand_copy["absolute_score"] = float(absolute_score)

            cand_copy["explain"] = {
                "semantic_score": round(semantic_sim, 3),
                "category_match": float(cat_match),
                "capacity_score": round(capacity, 3),
                "distance_km": round(dist_km, 2),
                "price_factor": float(price_factor)
            }

            ranked_candidates.append(cand_copy)

        # Sort by the LightGBM machine ranking
        ranked_candidates.sort(key=lambda x: x.get("ltr_score", 0.0), reverse=True)
        
        return ranked_candidates

    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")
        return candidates