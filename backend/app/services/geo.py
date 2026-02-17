import json
import os
import math

# PATH to the generated DB
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
GEO_DB_PATH = os.path.join(BACKEND_DIR, "data", "taxonomy", "indian_locations.json")

_geo_cache = None

def load_geo_db():
    global _geo_cache
    if not os.path.exists(GEO_DB_PATH):
        print(f"Geo DB missing at {GEO_DB_PATH}")
        return {}
    
    with open(GEO_DB_PATH, "r") as f:
        _geo_cache = json.load(f)
    print(f"Loaded Geo DB with {len(_geo_cache)} locations.")
    return _geo_cache

def get_coordinates(location_name: str):
    """
    Returns (lat, lon) for a given city name.
    Case-insensitive. O(1) lookup.
    """
    if _geo_cache is None:
        load_geo_db()
        
    if not location_name:
        return None
        
    # normalize
    key = location_name.lower().strip()
    
    # Direct match
    if key in _geo_cache:
        data = _geo_cache[key]
        return (data['lat'], data['lon'])
    
    # Fuzzy fallback (simple substring check for production robustness)
    # In a real system, use Levenshtein distance here.
    for city, data in _geo_cache.items():
        if city in key or key in city:
             return (data['lat'], data['lon'])
             
    return None

def haversine_distance(coord1, coord2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    if not coord1 or not coord2:
        return 2000.0 # Default penalty (2000km) if unknown

    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r