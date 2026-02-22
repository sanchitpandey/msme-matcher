import pickle
import logging
import numpy as np
from typing import Tuple
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None
_embedder = None

_KEYWORD_CATEGORY_MAP = {
    "textile": "Textiles",
    "fabric": "Textiles",
    "cotton": "Textiles",
    "silk": "Textiles",
    "cnc": "Manufacturing (CNC, Metal)",
    "lathe": "Manufacturing (CNC, Metal)",
    "steel": "Manufacturing (CNC, Metal)",
    "metal": "Manufacturing (CNC, Metal)",
    "food": "Food Processing",
    "packaging": "Food Processing",
    "furniture": "Furniture",
    "printing": "Printing & Packaging"
}

def load_classifier():
    """Loads Logistic Regression model and SBERT embedder."""
    global _model, _embedder
    
    if _model is not None:
        return

    if not settings.CATEGORY_MODEL_PATH.exists():
        logger.error(f"Classifier not found at {settings.CATEGORY_MODEL_PATH}")
        return

    try:
        with open(settings.CATEGORY_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        
        # We reuse the SBERT model name from config
        _embedder = SentenceTransformer(settings.SBERT_MODEL_NAME)
        logger.info("Category Classifier loaded successfully.")
    except Exception as e:
        logger.exception(f"Error loading classifier: {e}")

def predict_category(text: str) -> Tuple[str, float]:
    """
    Predicts product category from text.
    Returns: (Category Name, Confidence Score)
    """
    if _model is None:
        load_classifier()
    
    if _model is None or _embedder is None:
        text_low = text.lower()
        for k, v in _KEYWORD_CATEGORY_MAP.items():
            if k in text_low:
                return v, 0.60
        return "Unknown", 0.0

    try:
        embedding = _embedder.encode([text])
        category = _model.predict(embedding)[0]
        probs = _model.predict_proba(embedding)[0]
        confidence = float(np.max(probs))
        return category, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Unknown", 0.0