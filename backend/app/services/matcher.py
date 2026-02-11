import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/snp_profiles.json", "r") as f:
    snp_db = json.load(f)

# precompute SNP embeddings
for snp in snp_db:
    snp["embedding"] = model.encode(snp["description"])

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_product(product_text: str):
    prod_emb = model.encode(product_text)

    scores = []
    for snp in snp_db:
        score = cosine(prod_emb, snp["embedding"])
        scores.append((snp["name"], float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]
