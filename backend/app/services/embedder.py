from sentence_transformers import SentenceTransformer
import numpy as np

# load once
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    emb = model.encode(text)
    return emb.tolist()
