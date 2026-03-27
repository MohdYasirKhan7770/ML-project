import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from predictor import predict_news, _load_model_and_vectorizer
from config_manager import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description="AI-powered Fake News Analysis API"
)

# Load model and tfidf on startup
try:
    logger.info("Loading ML model and vectorizer...")
    model, tfidf = _load_model_and_vectorizer()
    logger.info("ML assets loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ML assets: {str(e)}")
    model, tfidf = None, None

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": config.VERSION
    }

@app.get("/info")
async def info():
    return {
        "app_name": config.APP_NAME,
        "version": config.VERSION,
        "description": "Enterprise-ready Fake News Detection API"
    }

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")
    
    if model is None or tfidf is None:
        logger.error("Prediction attempt with failed model load.")
        raise HTTPException(status_code=503, detail="Model service unavailable.")

    try:
        logger.info(f"Analyzing text of length {len(req.text)}")
        result = predict_news(req.text, model=model, tfidf=tfidf)
        return result
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during text analysis.")

# Ensure static directory exists
os.makedirs(config.STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {config.APP_NAME} v{config.VERSION} on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
