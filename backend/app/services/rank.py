import lightgbm as lgb
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models/ltr_model.txt")

_ranker = None

def load_ranker():
    global _ranker
    if os.path.exists(MODEL_PATH):
        _ranker = lgb.Booster(model_file=MODEL_PATH)
        print("LTR Ranker loaded.")
    else:
        print("LTR model not found. Run training/train_ltr.py")

def re_rank_results(query_text, query_category, candidates):
    if not candidates:
        return []

    if _ranker is None:
        load_ranker()
    
    if _ranker is None:
        return candidates 

    X_pred = []
    for cand in candidates:
        sem_score = cand.get('score', 0.5)
        
        features = compute_features(query_text, query_category, cand, sem_score)
        X_pred.append(features)
        
    X_pred = np.array(X_pred)
    
    scores = _ranker.predict(X_pred)
    
    ranked_candidates = []
    for i, cand in enumerate(candidates):
        cand['ltr_score'] = float(scores[i])
        ranked_candidates.append(cand)
        
    ranked_candidates.sort(key=lambda x: x['ltr_score'], reverse=True)
    
    return ranked_candidates