import sys
import json
import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Comprehensive list of Indian cities/towns with Lat/Lon
INDIAN_CITIES = {
    # Tier 1
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "delhi": {"lat": 28.7041, "lon": 77.1025, "state": "Delhi"},
    "bangalore": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    
    # Tier 2 & Industrial Hubs
    "surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat"},
    "kanpur": {"lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh"},
    "jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
    "indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh"},
    "thane": {"lat": 19.2183, "lon": 72.9781, "state": "Maharashtra"},
    "bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "pimpri-chinchwad": {"lat": 18.6298, "lon": 73.7997, "state": "Maharashtra"},
    "patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "vadodara": {"lat": 22.3072, "lon": 73.1812, "state": "Gujarat"},
    "ghaziabad": {"lat": 28.6692, "lon": 77.4538, "state": "Uttar Pradesh"},
    "ludhiana": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab"},
    "agra": {"lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh"},
    "nashik": {"lat": 19.9975, "lon": 73.7898, "state": "Maharashtra"},
    "faridabad": {"lat": 28.4089, "lon": 77.3178, "state": "Haryana"},
    "meerut": {"lat": 28.9845, "lon": 77.7064, "state": "Uttar Pradesh"},
    "rajkot": {"lat": 22.3039, "lon": 70.8022, "state": "Gujarat"},
    "kalyan-dombivli": {"lat": 19.2183, "lon": 73.1333, "state": "Maharashtra"},
    "vasai-virar": {"lat": 19.3919, "lon": 72.8397, "state": "Maharashtra"},
    "varanasi": {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh"},
    "srinagar": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir"},
    "aurangabad": {"lat": 19.8762, "lon": 75.3433, "state": "Maharashtra"},
    "dhanbad": {"lat": 23.7957, "lon": 86.4304, "state": "Jharkhand"},
    "amritsar": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab"},
    "navi mumbai": {"lat": 19.0330, "lon": 73.0297, "state": "Maharashtra"},
    "allahabad": {"lat": 25.4358, "lon": 81.8463, "state": "Uttar Pradesh"},
    "howrah": {"lat": 22.5958, "lon": 88.2636, "state": "West Bengal"},
    "ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand"},
    "gwalior": {"lat": 26.2183, "lon": 78.1828, "state": "Madhya Pradesh"},
    "jabalpur": {"lat": 23.1815, "lon": 79.9864, "state": "Madhya Pradesh"},
    "coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
    "vijayawada": {"lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh"},
    "jodhpur": {"lat": 26.2389, "lon": 73.0243, "state": "Rajasthan"},
    "madurai": {"lat": 9.9252, "lon": 78.1198, "state": "Tamil Nadu"},
    "raipur": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh"},
    "kota": {"lat": 25.2138, "lon": 75.8648, "state": "Rajasthan"},
    "chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Chandigarh"},
    "guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "solapur": {"lat": 17.6599, "lon": 75.9064, "state": "Maharashtra"},
    "hubli": {"lat": 15.3647, "lon": 75.1240, "state": "Karnataka"},
    "mysore": {"lat": 12.2958, "lon": 76.6394, "state": "Karnataka"},
    "tiruppur": {"lat": 11.1085, "lon": 77.3411, "state": "Tamil Nadu"},
    "gurgaon": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
    "noida": {"lat": 28.5355, "lon": 77.3910, "state": "Uttar Pradesh"},
    "kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "warangal": {"lat": 17.9689, "lon": 79.5941, "state": "Telangana"},
    "bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha"},
    "salem": {"lat": 11.6643, "lon": 78.1460, "state": "Tamil Nadu"},
    "bhilwara": {"lat": 25.3475, "lon": 74.6408, "state": "Rajasthan"},
    "siliguri": {"lat": 26.7271, "lon": 88.3953, "state": "West Bengal"}
}

def main():
    settings.GEO_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating Geo-DB with {len(INDIAN_CITIES)} locations...")
    try:
        with open(settings.GEO_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(INDIAN_CITIES, f, indent=2)
        logger.info(f"Geo-DB saved to {settings.GEO_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to save Geo-DB: {e}")

if __name__ == "__main__":
    main()