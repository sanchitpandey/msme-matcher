import numpy as np
import re
import logging
from typing import Optional, Dict, Any

from app.services.geo import get_coordinates, haversine_distance, load_geo_db

logger = logging.getLogger(__name__)

def extract_location_from_query(query_text: str) -> Optional[str]:
    """
    Scans query text for known Indian cities using the loaded Geo DB.
    """
    db = load_geo_db()
    if not db:
        return None
        
    cities = list(db.keys())
    query_lower = query_text.lower()
    
    # Sort by length to match "Navi Mumbai" before "Mumbai"
    sorted_cities = sorted(cities, key=len, reverse=True)
    
    for city in sorted_cities:
        # Use regex boundary \b to match whole words only
        if re.search(r'\b' + re.escape(city) + r'\b', query_lower):
            return city
            
    return None

def compute_features(
    query_text: str, 
    query_cat: str, 
    candidate: Dict[str, Any], 
    semantic_score: float
) -> np.ndarray:
    """
    Generates a 5-dimensional feature vector for the LTR Ranker.
    Order: [score, cat_match, capacity, log_dist, price]
    """
    try:
        # 1. Semantic Score
        f_score = float(semantic_score) * 2.0
        
        # 2. Category Match
        cand_cat = candidate.get("category", "")
        f_cat_match = 1.5 if query_cat and cand_cat and query_cat.lower() == cand_cat.lower() else -1.0
        
        # 3. Capacity Score
        f_capacity = float(candidate.get("capacity_score", 0.5))
        
        # 4. Geo Distance
        q_city = extract_location_from_query(query_text)
        cand_city = candidate.get("location", "")
        
        q_coords = get_coordinates(q_city)
        cand_coords = get_coordinates(cand_city)
        
        dist_km = haversine_distance(q_coords, cand_coords)
        f_dist = np.log1p(dist_km) # Log normalization
        
        # 5. Price Tier
        price_map = {"Low": 1, "Med": 2, "High": 3}
        f_price = float(price_map.get(candidate.get("price_tier", "Med"), 2))
    
        return np.array([f_score, f_cat_match, f_capacity, f_dist, f_price], dtype=np.float32)

    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        # Return a safe default vector
        return np.array([semantic_score, 0.0, 0.5, 8.0, 2.0], dtype=np.float32)