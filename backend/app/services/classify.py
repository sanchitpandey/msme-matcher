import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# --- PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "category_classifier.pkl")

_model = None
_embedder = None

def load_classifier():
    global _model, _embedder
    if not os.path.exists(MODEL_PATH):
        print(f"Classifier model not found at {MODEL_PATH}")
        return

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Classifier loaded.")

def predict_category(text: str):
    if _model is None:
        load_classifier()
    
    if _model is None:
        return "Unknown", 0.0

    embedding = _embedder.encode([text])
    
    category = _model.predict(embedding)[0]
    probs = _model.predict_proba(embedding)[0]
    confidence = float(np.max(probs))

    return category, confidence