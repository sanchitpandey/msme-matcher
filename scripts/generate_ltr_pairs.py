import sys
import json
import random
import logging
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.core.config import settings
from app.services.geo import get_coordinates, haversine_distance, load_geo_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Output Path
OUTPUT_PATH = settings.DATA_DIR / "processed" / "ltr_train.parquet"

def main():
    if not settings.SNP_DATA_PATH.exists():
        logger.error("Run generate_snp_profiles.py first.")
        return

    # Ensure Geo DB is loaded
    load_geo_db()
    
    # Load Cities for query generation
    geo_db = load_geo_db()
    if not geo_db:
        logger.error("Geo DB failed to load. Cannot generate location-aware pairs.")
        return
    
    cities = list(geo_db.keys())

    logger.info("Loading profiles...")
    with open(settings.SNP_DATA_PATH, "r", encoding="utf-8") as f:
        snps = json.load(f)

    logger.info(f"Loading SBERT model {settings.SBERT_MODEL_NAME}...")
    model = SentenceTransformer(settings.SBERT_MODEL_NAME)
    
    snp_texts = [s["capability_text"] for s in snps]
    snp_embeddings = model.encode(snp_texts, convert_to_tensor=True)

    queries = []
    
    # Base templates
    templates = [
        ("CNC turning job work in {city}", "Manufacturing (CNC, Metal)"),
        ("Textile fabric wholesale {city}", "Textiles"),
        ("Food packaging unit {city}", "Food Processing"),
        ("Steel fabrication {city}", "Manufacturing (CNC, Metal)")
    ]
    
    logger.info(f"Generating queries for {len(cities)} cities...")
    
    for city in cities:
        # Sample 2 templates per city to create diversity
        for t_text, t_cat in random.sample(templates, min(len(templates), 2)):
            q_text = t_text.format(city=city.title())
            queries.append({"text": q_text, "category": t_cat, "loc": city})

    ltr_data = []
    logger.info(f"Processing {len(queries)} queries to find matches...")
    
    for q in queries:
        q_emb = model.encode(q["text"], convert_to_tensor=True)
        scores = cos_sim(q_emb, snp_embeddings)[0]
        
        # Get top 50
        top_k_indices = scores.argsort(descending=True)[:50].tolist()
        
        for idx in top_k_indices:
            candidate = snps[idx]
            score = float(scores[idx])
            
            # LABELING LOGIC
            dist_val = 2000.0
            
            q_coords = get_coordinates(q["loc"])
            cand_coords = get_coordinates(candidate.get("location"))
            
            if q_coords and cand_coords:
                dist_val = haversine_distance(q_coords, cand_coords)
            
            relevance = 0
            
            if candidate.get("category") == q["category"]:
                # Tier 2: Excellent (Nearby + Match)
                if dist_val < 50: 
                    relevance = 2
                # Tier 1: Good (Match but far, OR nearby but weak score)
                elif dist_val < 200: 
                    relevance = 1
                elif score > 0.6: # High semantic match even if far
                    relevance = 1
            
            # Downsample negatives to keep dataset balanced
            if relevance > 0 or random.random() < 0.1:
                ltr_data.append({
                    "query": q["text"],
                    "doc_id": candidate["snp_id"],
                    "doc_text": candidate["capability_text"],
                    "label": relevance,
                    "semantic_score": score,
                    "capacity": candidate.get("capacity_score", 0),
                    "location": candidate.get("location", "")
                })

    df = pd.DataFrame(ltr_data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    
    logger.info(f"Saved {len(df)} pairs. Distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    main()