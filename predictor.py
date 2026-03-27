"""
============================================================================
  Predictor Module — Fake News Detection System
============================================================================
  Provides:
    • `predict_news(text)` — classify a single article as Real / Fake
    • `predict_batch(texts)` — classify a list of articles
    • CLI interactive loop for real-time prediction
============================================================================
"""

import sys

import joblib
import numpy as np

import config
from data_pipeline import preprocess_text


# ──────────────────────────────────────────────────────────────────────────
# 1. LOAD SAVED ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────
def _load_model_and_vectorizer():
    """Load the persisted model and TF-IDF vectorizer."""
    model = joblib.load(config.BEST_MODEL_FILE)
    tfidf = joblib.load(config.TFIDF_VECTORIZER_FILE)
    return model, tfidf


# ──────────────────────────────────────────────────────────────────────────
# 2. SINGLE PREDICTION
# ──────────────────────────────────────────────────────────────────────────
def predict_news(text: str, model=None, tfidf=None) -> dict:
    """
    Classify a single news article.

    Parameters
    ----------
    text  : str  — raw article text
    model : optional pre-loaded model (avoids reloading from disk)
    tfidf : optional pre-loaded vectorizer

    Returns
    -------
    dict with keys: label (str), confidence (float), raw_prediction (int)
    """
    if model is None or tfidf is None:
        model, tfidf = _load_model_and_vectorizer()

    clean = preprocess_text(text)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]

    # Confidence (probability of predicted class)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        confidence = float(np.max(proba))
    else:
        confidence = None

    label = "✅ REAL News" if prediction == 1 else "❌ FAKE News"

    return {
        "label": label,
        "confidence": confidence,
        "raw_prediction": int(prediction),
    }


# ──────────────────────────────────────────────────────────────────────────
# 3. BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────
def predict_batch(texts: list, model=None, tfidf=None) -> list:
    """
    Classify a list of news articles.

    Returns
    -------
    list of dicts (same format as `predict_news`)
    """
    if model is None or tfidf is None:
        model, tfidf = _load_model_and_vectorizer()

    return [predict_news(t, model=model, tfidf=tfidf) for t in texts]


# ──────────────────────────────────────────────────────────────────────────
# 4. CLI INTERACTIVE LOOP
# ──────────────────────────────────────────────────────────────────────────
def interactive_cli():
    """
    Launch a command-line interface for real-time fake news detection.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "═" * 60)
    print("  🔍  FAKE NEWS DETECTOR — Interactive CLI")
    print("═" * 60)
    print("  Type or paste a news article and press Enter.")
    print("  Type 'quit' or 'exit' to stop.\n")

    model, tfidf = _load_model_and_vectorizer()

    while True:
        try:
            text = input("📰  Enter news text:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not text:
            print("⚠️  Empty input. Please enter some text.\n")
            continue

        result = predict_news(text, model=model, tfidf=tfidf)
        conf_str = f" (confidence: {result['confidence']:.2%})" if result["confidence"] else ""
        print(f"\n  → {result['label']}{conf_str}\n")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    interactive_cli()
