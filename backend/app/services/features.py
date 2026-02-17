import numpy as np
import re
from app.services.geo import get_coordinates, haversine_distance, load_geo_db

if not load_geo_db():
    pass

def extract_location_from_query(query_text):
    """
    Scans query text for any known Indian city.
    """
    db = load_geo_db()
    if not db:
        return None

    cities = db.keys()
    
    query_lower = query_text.lower()
    
    # Sort cities by length (descending)
    sorted_cities = sorted(cities, key=len, reverse=True)
    
    for city in sorted_cities:
        if re.search(r'\b' + re.escape(city) + r'\b', query_lower):
            return city
            
    return None

def compute_features(query_text, query_cat, candidate, semantic_score):
    """
    Generates a feature vector for the Ranker.
    Feature Vector (5-dim):
    [0] semantic_score (float)
    [1] category_match (0.0 or 1.0)
    [2] capacity_score (0.0 to 1.0)
    [3] log_geo_distance (float)
    [4] price_tier_val (1, 2, or 3)
    """
    # 1. Semantic Score
    f_score = float(semantic_score)
    
    # 2. Category Match
    cand_cat = candidate.get("category", "")
    f_cat_match = 1.0 if query_cat.lower() == cand_cat.lower() else 0.0
    
    # 3. Capacity Score
    f_capacity = float(candidate.get("capacity_score", 0.5))
    
    # 4. Geo Distance
    q_city = extract_location_from_query(query_text)
    cand_city = candidate.get("location", "")
    
    q_coords = get_coordinates(q_city)
    cand_coords = get_coordinates(cand_city)
    
    dist_km = haversine_distance(q_coords, cand_coords)
    f_dist = np.log1p(dist_km)
    
    # 5. Price Tier
    price_map = {"Low": 1, "Med": 2, "High": 3}
    f_price = price_map.get(candidate.get("price_tier", "Med"), 2)

    return np.array([f_score, f_cat_match, f_capacity, f_dist, f_price])