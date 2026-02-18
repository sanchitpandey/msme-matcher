import sys
import json
import random
import uuid
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
INPUT_FILE = settings.DATA_DIR / "raw" / "seed_snps.json"
TARGET_COUNT = 3000

CITIES = [
    "Mumbai", "Pune", "Nagpur", "Nashik", "Thane",
    "Delhi", "Noida", "Gurgaon", "Faridabad",
    "Bangalore", "Mysore", "Hubli",
    "Chennai", "Coimbatore", "Tiruppur", "Madurai",
    "Hyderabad", "Warangal", "Visakhapatnam",
    "Ahmedabad", "Surat", "Vadodara", "Rajkot",
    "Kolkata", "Howrah", "Siliguri",
    "Ludhiana", "Jalandhar", "Amritsar",
    "Jaipur", "Jodhpur", "Kota", "Bhilwara"
]

SUFFIXES = ["Pvt Ltd", "Enterprises", "Works", "Industries", "Solutions", "Traders", "& Sons", "Global", "Exports"]

def fuzz_text(text: str) -> str:
    """Adds noise to make synthetic data realistic."""
    variations = [
        lambda t: t.replace(" and ", " & "),
        lambda t: t.replace("Daily", "daily").replace("Monthly", "monthly"),
        lambda t: t + " [GST Registered]",
        lambda t: "Verified Supplier. " + t,
        lambda t: t + " (Bulk orders only)",
        lambda t: t.replace(".", ","),
        lambda t: t + " Transport extra.",
        lambda t: t.replace("kg", " kg").replace("pc", " pcs"),
        lambda t: t + " No credit.",
    ]
    
    if random.random() > 0.5:
        text = random.choice(variations)(text)
    if random.random() > 0.7:
        text = random.choice(variations)(text)
    return text

def main():
    if not INPUT_FILE.exists():
        logger.error(f"Seed file not found: {INPUT_FILE}")
        return

    logger.info(f"Loading seeds from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            seeds = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load seeds: {e}")
        return

    logger.info(f"Multiplying {len(seeds)} seeds to {TARGET_COUNT} profiles...")

    generated = []
    
    for _ in range(TARGET_COUNT):
        seed = random.choice(seeds)
        new_profile = seed.copy()
        new_profile["snp_id"] = str(uuid.uuid4())
        
        # 1. Randomize Name
        base_name = seed.get("name", "Factory").split(" ")[0] 
        new_profile["name"] = f"{base_name} {random.choice(SUFFIXES)}"
        
        # 2. Randomize Location
        new_profile["location"] = random.choice(CITIES)
        
        # 3. Jitter Capacity
        jitter = random.uniform(-0.15, 0.15)
        base_score = seed.get("capacity_score", 0.5)
        new_profile["capacity_score"] = round(max(0.1, min(1.0, base_score + jitter)), 2)
        
        # 4. Fuzz Text
        new_profile["capability_text"] = fuzz_text(seed.get("capability_text", ""))
        
        generated.append(new_profile)

    # Ensure output dir exists
    settings.SNP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(settings.SNP_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(generated, f, indent=2)

    logger.info(f"Success! Saved {len(generated)} profiles to {settings.SNP_DATA_PATH}")

if __name__ == "__main__":
    main()