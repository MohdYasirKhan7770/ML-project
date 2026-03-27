"""
============================================================================
  Streamlit UI — Fake News Detection System
============================================================================
  A polished web interface for real-time fake news classification.
  Run with:  streamlit run app.py
============================================================================
"""

import streamlit as st
import joblib
import os
import config
from data_pipeline import preprocess_text
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    /* Title styling */
    .title-container {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .title-container h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin: 0;
    }
    .title-container p {
        color: #b8b8d4;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .result-real {
        background: linear-gradient(135deg, #0a3d0a, #1a6b1a);
        border-left: 6px solid #2ecc71;
    }
    .result-fake {
        background: linear-gradient(135deg, #4a0000, #8b1a1a);
        border-left: 6px solid #e74c3c;
    }
    .result-card h2 {
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0 0 0.5rem 0;
    }
    .result-card p {
        color: #e0e0e0;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Confidence bar */
    .confidence-bar {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        height: 12px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .confidence-fill-real { background: linear-gradient(90deg, #2ecc71, #27ae60); }
    .confidence-fill-fake { background: linear-gradient(90deg, #e74c3c, #c0392b); }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-container">
    <h1>🔍 Fake News Detector</h1>
    <p>Powered by Machine Learning & NLP</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved model and vectorizer."""
    if not os.path.exists(config.BEST_MODEL_FILE):
        return None, None
    model = joblib.load(config.BEST_MODEL_FILE)
    tfidf = joblib.load(config.TFIDF_VECTORIZER_FILE)
    return model, tfidf


model, tfidf = load_model()

if model is None:
    st.error("⚠️ No trained model found! Please run the training notebook first "
             "(`main_notebook.ipynb`) to train and save a model.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# INPUT AREA
# ──────────────────────────────────────────────────────────────────────────
st.markdown("### 📰 Paste a news article below")
user_input = st.text_area(
    label="News text",
    height=200,
    placeholder="Type or paste the news article text here…",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔎  Analyze", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────────────────
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing…"):
            clean = preprocess_text(user_input)
            vector = tfidf.transform([clean])
            prediction = model.predict(vector)[0]

            # Confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vector)[0]
                confidence = float(np.max(proba))
            else:
                confidence = None

        # Display result
        if prediction == 1:
            css_class = "result-real"
            icon = "✅"
            label = "REAL News"
            fill_class = "confidence-fill-real"
        else:
            css_class = "result-fake"
            icon = "❌"
            label = "FAKE News"
            fill_class = "confidence-fill-fake"

        conf_pct = f"{confidence:.1%}" if confidence else "N/A"
        conf_width = f"{confidence * 100:.0f}%" if confidence else "0%"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <h2>{icon} {label}</h2>
            <p>Confidence: {conf_pct}</p>
            <div class="confidence-bar">
                <div class="confidence-fill {fill_class}" style="width: {conf_width};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show details in expander
        with st.expander("🔬 Details"):
            st.write(f"**Preprocessed text (first 500 chars):** {clean[:500]}…")
            if confidence:
                st.write(f"**Probability (Real):** {proba[1]:.4f}")
                st.write(f"**Probability (Fake):** {proba[0]:.4f}")


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — ABOUT
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown(
        "This application uses a Machine Learning model trained on "
        "a large corpus of labelled news articles to detect fake news."
    )
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- **NLP**: NLTK (lemmatization, stopwords)")
    st.markdown("- **Vectorization**: TF-IDF")
    st.markdown("- **Models**: NB / LR / RF (GridSearchCV)")
    st.markdown("- **UI**: Streamlit")

    st.markdown("---")
    st.markdown(
        '<p class="footer">Fake News Detector v1.0</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Built with ❤️ using Python, scikit-learn & Streamlit</div>',
    unsafe_allow_html=True,
)
