import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config_manager import config
from predictor import _load_model_and_vectorizer, predict_news


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description="AI-powered Fake News Analysis API",
)

try:
    logger.info("Loading ML model and vectorizer...")
    model, tfidf = _load_model_and_vectorizer()
    logger.info("ML assets loaded successfully.")
except Exception as exc:
    logger.error(f"Failed to load ML assets: {exc}")
    model, tfidf = None, None


class PredictRequest(BaseModel):
    text: str


def build_agent_response(text: str, result: dict) -> dict:
    """Wrap classifier output in a more conversational agent reply."""
    confidence = result.get("confidence")
    confidence_pct = round((confidence or 0.0) * 100, 1)
    is_real = result["raw_prediction"] == 1

    verdict = "likely reliable" if is_real else "likely misleading"
    summary = (
        "The article follows patterns that are common in factual reporting."
        if is_real
        else "The article shows patterns often associated with sensational or unreliable claims."
    )
    next_steps = [
        "Treat this as a screening signal, not final proof.",
        "Check the publication, author, and date before sharing.",
        "Look for corroboration from multiple established outlets.",
    ]
    if not is_real:
        next_steps = [
            "Pause before sharing this claim.",
            "Search for matching reporting from trusted outlets.",
            "Check whether the headline is exaggerated compared with the body text.",
        ]

    excerpt = " ".join(text.split())
    excerpt = excerpt[:220] + ("..." if len(excerpt) > 220 else "")

    return {
        "message": (
            f"I reviewed the article and my current assessment is {verdict}. "
            f"Model confidence is {confidence_pct}%."
        ),
        "verdict": result["label"],
        "confidence_percent": confidence_pct,
        "summary": summary,
        "excerpt": excerpt,
        "next_steps": next_steps,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": config.VERSION,
    }


@app.get("/info")
async def info():
    return {
        "app_name": config.APP_NAME,
        "version": config.VERSION,
        "description": "Enterprise-ready Fake News Detection API",
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
        return predict_news(req.text, model=model, tfidf=tfidf)
    except Exception as exc:
        logger.error(f"Analysis error: {exc}")
        raise HTTPException(status_code=500, detail="An error occurred during text analysis.")


@app.post("/agent")
async def agent(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")

    if model is None or tfidf is None:
        logger.error("Agent attempt with failed model load.")
        raise HTTPException(status_code=503, detail="Model service unavailable.")

    try:
        prediction = predict_news(req.text, model=model, tfidf=tfidf)
        return {"prediction": prediction, "agent": build_agent_response(req.text, prediction)}
    except Exception as exc:
        logger.error(f"Agent analysis error: {exc}")
        raise HTTPException(status_code=500, detail="An error occurred during agent analysis.")


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


os.makedirs(config.STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {config.APP_NAME} v{config.VERSION} on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
