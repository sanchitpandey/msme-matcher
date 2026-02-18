import json
import math
import logging
from typing import Dict, Optional, Tuple, Any

from app.core.config import settings

# Configure Logger
logger = logging.getLogger(__name__)

# Global Cache
_geo_cache: Optional[Dict[str, Any]] = None

def load_geo_db() -> Dict[str, Any]:
    """
    Loads the Indian Locations taxonomy into memory.
    """
    global _geo_cache
    if _geo_cache is not None:
        return _geo_cache

    if not settings.GEO_DB_PATH.exists():
        logger.error(f"Geo DB missing at {settings.GEO_DB_PATH}")
        return {}
    
    try:
        with open(settings.GEO_DB_PATH, "r", encoding="utf-8") as f:
            _geo_cache = json.load(f)
        logger.info(f"Loaded Geo DB with {len(_geo_cache)} locations.")
    except Exception as e:
        logger.exception(f"Failed to load Geo DB: {e}")
        return {}
        
    return _geo_cache

def get_coordinates(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Returns (lat, lon) for a given city name. Case-insensitive.
    """
    if _geo_cache is None:
        load_geo_db()
        
    if not location_name or _geo_cache is None:
        return None
        
    key = location_name.lower().strip()
    
    # Direct match
    if key in _geo_cache:
        data = _geo_cache[key]
        return (data['lat'], data['lon'])
    
    # Fallback: Check if city is part of a key (simple fuzzy match)
    for city, data in _geo_cache.items():
        if city == key: 
            return (data['lat'], data['lon'])
            
    return None

def haversine_distance(coord1: Optional[Tuple[float, float]], coord2: Optional[Tuple[float, float]]) -> float:
    """
    Calculate the great circle distance (km) between two points.
    Returns 2000.0 km if coordinates are invalid.
    """
    if not coord1 or not coord2:
        return 2000.0 

    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371.0 # Radius of earth in km
    
    return c * r