import streamlit as st
import numpy as np
import time
import streamlit.components.v1 as components

# Import our custom modules
from model import FakeNewsTransformer
from validation import RealTimeValidator
from explain import ExplainabilityEngine
import advanced_config

# Page Config
st.set_page_config(page_title="Advanced Fake News Detector", page_icon="🛡️", layout="wide")

# Init models in cache
@st.cache_resource
def load_system():
    # Attempt to load saved model, fallback to base model
    model = FakeNewsTransformer(load_saved=True)
    validator = RealTimeValidator(use_sbert=True)
    explainer = ExplainabilityEngine()
    return model, validator, explainer

try:
    with st.spinner("Loading Transformer Models... this may take a minute"):
        model, validator, explainer = load_system()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("🛡️ Advanced Fake News Detection System")
st.markdown("Powered by **DistilBERT**, **LIME**, and **DuckDuckGo Real-Time Validation**.")

user_input = st.text_area("Enter News Article / Claim here:", height=150)

if st.button("Analyze News", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing with DistilBERT..."):
            # 1. Model Prediction
            pred_class, probs = model.predict(user_input) # 0 = Fake, 1 = Real
            model_confidence = probs[pred_class]
            
        with st.spinner("Validating locally & via Real-Time Web Search..."):
            # 2. Validation
            sim_score, related_articles = validator.validate(user_input)
            
        with st.spinner("Generating Explainability Report..."):
            # 3. Explainability
            exp = explainer.explain_prediction(user_input, model.predict_proba)
            html_exp = explainer.get_html(exp) if exp else ""

        # 4. Decision Engine
        # Here we blend model probability with similarity score to make a final call
        # Assuming class 1 is Real. P(Real) = probs[1]
        p_real_model = probs[1] 
        p_real_final = (p_real_model * advanced_config.DECISION_WEIGHT_MODEL) + (sim_score * advanced_config.DECISION_WEIGHT_SIMILARITY)
        
        final_prediction = "REAL" if p_real_final > 0.5 else "FAKE"
        final_confidence = p_real_final if final_prediction == "REAL" else (1 - p_real_final)

        # UI rendering
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        col1.metric("DistilBERT System", "REAL" if pred_class == 1 else "FAKE", f"{model_confidence:.1%} confidence")
        col2.metric("News API Support", f"{sim_score:.2f} Similarity Score", "+ Real Evidence" if sim_score > 0.5 else "- Weak Evidence")
        col3.metric("Final System Verdict", final_prediction, f"Combined Confidence: {final_confidence:.1%}")

        st.divider()
        
        col_exp, col_news = st.columns(2)
        
        with col_exp:
            st.subheader("🔍 LIME Explainability Engine")
            st.markdown("Highlights which words contributed to 'Fake' or 'Real' verdict.")
            if html_exp:
                components.html(html_exp, height=350, scrolling=True)
            else:
                st.info("Not enough text to generate LIME explanation.")

        with col_news:
            st.subheader("🌐 Real-Time Evidence (DuckDuckGo)")
            if related_articles:
                for i, article in enumerate(related_articles):
                    st.markdown(f"**[{i+1}] {article['title']}**")
                    st.write(article['snippet'])
                    st.write(f"[Read Source]({article['url']})")
                    st.markdown("---")
            else:
                st.warning("No highly similar recent articles found to support this claim.")
