import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "IndiaAI MSME Matching System"
    API_V1_STR: str = "/api"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    INDEX_DIR: Path = BASE_DIR / "indices"
    
    # Specific File Paths
    SNP_DATA_PATH: Path = DATA_DIR / "processed" / "snp_profiles.json"
    GEO_DB_PATH: Path = DATA_DIR / "taxonomy" / "indian_locations.json"
    FAISS_INDEX_PATH: Path = INDEX_DIR / "faiss_snp.index"
    
    # Model Paths
    CATEGORY_MODEL_PATH: Path = MODEL_DIR / "category_classifier.pkl"
    LTR_MODEL_PATH: Path = MODEL_DIR / "ltr_model.txt"
    
    # Model Configurations
    SBERT_MODEL_NAME: str = "all-MiniLM-L6-v2"
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_COMPUTE_TYPE: str = "int8"

    class Config:
        case_sensitive = True

settings = Settings()