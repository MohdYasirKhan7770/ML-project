import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    APP_NAME = "Fake News Detector"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer.joblib")
    
    # Static files
    STATIC_DIR = os.getenv("STATIC_DIR", "static")

config = Config()
