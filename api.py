import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from predictor import predict_news, _load_model_and_vectorizer

app = FastAPI(title="Fake News Detector API")

# Load model and tfidf on startup
model, tfidf = _load_model_and_vectorizer()

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Empty text provided."})
    
    result = predict_news(req.text, model=model, tfidf=tfidf)
    return result

# Attempt to mount static directory
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
