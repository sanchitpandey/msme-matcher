import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not settings.SNP_DATA_PATH.exists():
        logger.error(f"Data missing at {settings.SNP_DATA_PATH}")
        return

    logger.info("Loading data...")
    try:
        with open(settings.SNP_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return
    
    # Prepare Training Data
    texts = [d["capability_text"] for d in data]
    labels = [d["category"] for d in data]

    logger.info(f"Encoding {len(texts)} samples...")
    embedder = SentenceTransformer(settings.SBERT_MODEL_NAME)
    X = embedder.encode(texts, show_progress_bar=True)
    y = labels

    logger.info("Training Classifier (Logistic Regression)...")
    clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    clf.fit(X, y)

    # Evaluate
    preds = clf.predict(X)
    logger.info("\n--- Training Report ---")
    logger.info("\n" + classification_report(y, preds))

    # Save
    settings.CATEGORY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(settings.CATEGORY_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    
    logger.info(f"Model saved to {settings.CATEGORY_MODEL_PATH}")

if __name__ == "__main__":
    main()