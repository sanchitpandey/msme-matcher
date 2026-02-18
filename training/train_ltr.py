import sys
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.core.config import settings
from app.services.geo import get_coordinates, haversine_distance, load_geo_db
from app.services.features import extract_location_from_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Input Data Path
DATA_PATH = settings.DATA_DIR / "processed" / "ltr_train.parquet"

def get_features_for_train(row):
    """
    Replicates compute_features logic for dataframe row.
    Feature Order: [semantic_score, cat_match, capacity, dist, price]
    """
    # 1. Semantic
    f_score = row['semantic_score']
    
    # 2. Category Match (Heuristic from label)
    f_cat_match = 1.0 if row['label'] > 0 else 0.0
    
    # 3. Capacity
    f_capacity = row['capacity']
    
    # 4. Geo Distance
    # Extract location from the *query text* in the training row
    q_city = extract_location_from_query(row['query'])
    cand_city = row['location']
    
    q_coords = get_coordinates(q_city)
    cand_coords = get_coordinates(cand_city)
    
    dist = haversine_distance(q_coords, cand_coords)
    f_dist = np.log1p(dist)
    
    # 5. Price (Default to Med/2 for training data)
    f_price = 2.0 

    return [f_score, f_cat_match, f_capacity, f_dist, f_price]

def main():
    # Ensure Geo DB is loaded for feature extraction
    load_geo_db()

    if not DATA_PATH.exists():
        logger.error(f"Data not found at {DATA_PATH}")
        logger.info("Run 'python scripts/generate_ltr_pairs.py' first.")
        return

    logger.info(f"Loading Training Data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Sort by query is required for LightGBM grouping
    df = df.sort_values(by="query")
    
    logger.info(f"Loaded {len(df)} rows.")
    logger.info(f"Label Distribution:\n{df['label'].value_counts()}")

    X = []
    y = []
    groups = []
    
    # Group data by query for LambdaRank
    query_groups = df.groupby("query")
    
    for name, group in query_groups:
        groups.append(len(group))
        for _, row in group.iterrows():
            X.append(get_features_for_train(row))
            y.append(row['label'])
            
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Training LightGBM on {len(X)} samples...")
    
    train_data = lgb.Dataset(X, label=y, group=groups)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 2,
        'verbose': -1
    }
    
    bst = lgb.train(params, train_data, num_boost_round=150)
    
    settings.LTR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(settings.LTR_MODEL_PATH))
    logger.info(f"LTR Model saved to {settings.LTR_MODEL_PATH}")

if __name__ == "__main__":
    main()