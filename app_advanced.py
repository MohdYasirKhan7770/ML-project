import streamlit as st
import streamlit.components.v1 as components

import advanced_config
from explain import ExplainabilityEngine
from model import FakeNewsTransformer
from validation import RealTimeValidator


st.set_page_config(page_title="Advanced Fake News Detector", page_icon="A", layout="wide")

st.markdown(
    """
<style>
    .stApp {
        background-color: #0E1117;
    }

    .title-container {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 16px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .title-container h1 {
        color: white;
        font-family: sans-serif;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-size: 2.8rem;
    }
    .title-container p {
        color: #e0e6ed;
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    .metric-card {
        background: #1A1F2B;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4A90E2;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.95rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0.2rem 0;
        line-height: 1.2;
    }
    .result-fake { border-left-color: #FF4B4B; }
    .result-real { border-left-color: #00CC96; }

    .evidence-card {
        background: #1A1F2B;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .evidence-card h4 {
        color: #63B3ED;
        margin-top: 0;
        font-size: 1.1rem;
        line-height: 1.4;
    }
    .evidence-source {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.4rem 0.8rem;
        background: rgba(74, 144, 226, 0.1);
        color: #63B3ED;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    .evidence-source:hover {
        background: rgba(74, 144, 226, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system():
    model = FakeNewsTransformer(load_saved=True)
    validator = RealTimeValidator(use_sbert=True)
    explainer = ExplainabilityEngine()
    return model, validator, explainer


try:
    with st.spinner("Initializing deep learning engine..."):
        model, validator, explainer = load_system()
except Exception as exc:
    st.error(f"Error loading models: {exc}")
    st.stop()

st.markdown(
    """
<div class="title-container">
    <h1>PrismaTruth AI</h1>
    <p>Advanced misinformation detection via DistilBERT, document similarity, and LIME explainability</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Input Article / Claim")
user_input = st.text_area("Paste the text you want to analyze:", height=180, label_visibility="collapsed")

if st.button("Analyze Authenticity", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Running DistilBERT analysis..."):
            pred_class, probs = model.predict(user_input)
            model_confidence = probs[pred_class]

        with st.spinner("Fetching related web evidence..."):
            sim_score, related_articles = validator.validate(user_input)

        with st.spinner("Generating LIME explanation..."):
            exp = explainer.explain_prediction(user_input, model.predict_proba)
            html_exp = explainer.get_html(exp) if exp else ""

        p_real_model = probs[1]
        p_real_final = (
            p_real_model * advanced_config.DECISION_WEIGHT_MODEL
            + sim_score * advanced_config.DECISION_WEIGHT_SIMILARITY
        )

        final_prediction = "REAL" if p_real_final > 0.5 else "FAKE"
        final_confidence = p_real_final if final_prediction == "REAL" else (1 - p_real_final)

        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### Verification Dashboard")

        col1, col2, col3 = st.columns(3)

        sys_color = "result-real" if pred_class == 1 else "result-fake"
        sys_text = "REAL" if pred_class == 1 else "FAKE"
        sys_val_color = "#00CC96" if pred_class == 1 else "#FF4B4B"

        final_color = "result-real" if final_prediction == "REAL" else "result-fake"
        final_val_color = "#00CC96" if final_prediction == "REAL" else "#FF4B4B"

        with col1:
            st.markdown(
                f"""
            <div class="metric-card {sys_color}">
                <div class="metric-label">Neural Assessment</div>
                <div class="metric-value" style="color: {sys_val_color}">{sys_text}</div>
                <div style="color: #A0AEC0">{model_confidence:.1%} BERT Confidence</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Corroboration Score</div>
                <div class="metric-value" style="color: #4A90E2">{sim_score:.2f}</div>
                <div style="color: #A0AEC0">Cosine Similarity (0.0 to 1.0)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-card {final_color}">
                <div class="metric-label">Final Verification</div>
                <div class="metric-value" style="color: {final_val_color}">{final_prediction}</div>
                <div style="color: #A0AEC0">Combined Trust: {final_confidence:.1%}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        col_exp, col_news = st.columns([1.2, 1])

        with col_exp:
            st.markdown("### Interpretability (LIME)")
            st.caption("Highlights specific words that pushed the model toward a real or fake decision.")
            if html_exp:
                with st.container():
                    components.html(html_exp, height=450, scrolling=True)
            else:
                st.info("Input text is too short to generate a meaningful LIME explanation.")

        with col_news:
            st.markdown("### Live Evidence")
            st.caption("Latest highly related articles returned by the search validator.")
            if related_articles:
                for index, article in enumerate(related_articles, start=1):
                    st.markdown(
                        f"""
                    <div class="evidence-card">
                        <h4>[{index}] {article['title']}</h4>
                        <p style="color: #CBD5E0; font-size: 0.95rem;">{article['snippet']}...</p>
                        <a href="{article['url']}" target="_blank" class="evidence-source">Read Source</a>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No highly similar recent articles were found for this claim.")

st.markdown(
    """
<div style="text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #2D3748; color: #718096; font-size: 0.9rem;">
    Advanced ML Fake News Detector | Built with PyTorch, SentenceTransformers, and Streamlit
</div>
""",
    unsafe_allow_html=True,
)
