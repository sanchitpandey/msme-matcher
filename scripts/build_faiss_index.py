import sys
import json
import logging
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not settings.SNP_DATA_PATH.exists():
        logger.error(f"Input file missing: {settings.SNP_DATA_PATH}")
        return

    logger.info("Loading SNP profiles...")
    try:
        with open(settings.SNP_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    texts = [d.get("capability_text", "") for d in data]
    logger.info(f"Encoding {len(texts)} vectors using {settings.SBERT_MODEL_NAME}...")
    
    model = SentenceTransformer(settings.SBERT_MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    faiss.normalize_L2(embeddings)

    logger.info("Building FAISS Index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    settings.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(settings.FAISS_INDEX_PATH))
    logger.info(f"Success! Index saved to {settings.FAISS_INDEX_PATH}")

if __name__ == "__main__":
    main()