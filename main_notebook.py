"""
╔══════════════════════════════════════════════════════════════════════════╗
║           FAKE NEWS DETECTION SYSTEM — Main Orchestrator               ║
║                                                                        ║
║  This script is designed to be run as a Jupyter Notebook (.py to .ipynb ║
║  or directly in VS Code's interactive mode with # %% cell markers).    ║
║                                                                        ║
║  Pipeline stages:                                                      ║
║    1. Data Loading & Cleaning                                          ║
║    2. NLP Preprocessing                                                ║
║    3. TF-IDF Vectorization                                             ║
║    4. Model Training (NB, LR, RF) with GridSearchCV                    ║
║    5. Cross-Validation                                                 ║
║    6. Evaluation (Metrics, Confusion Matrix, Comparison Chart)         ║
║    7. Save Best Model                                                  ║
║    8. Real-Time Prediction Demo                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# %% [markdown]
# # 🔍 Fake News Detection System
# ---
# An end-to-end ML pipeline for classifying news articles as **Real** or **Fake**.

# %% ──────────────────────── IMPORTS ────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project modules
import config
from data_pipeline import (
    load_data,
    clean_data,
    preprocess_dataframe,
    build_tfidf,
    run_pipeline,
)
from model_trainer import (
    train_all_models,
    save_artifacts,
)
from evaluator import (
    evaluate_all,
    plot_model_comparison,
)
from predictor import predict_news

print("✅ All modules imported successfully.")

# %% [markdown]
# ## 1️⃣ Data Pipeline
# Load the **True.csv** and **Fake.csv** datasets, clean, preprocess, and vectorize.

# %% ──────────────────────── DATA PIPELINE ─────────────────────────────

X_train, X_test, y_train, y_test, tfidf_vectorizer = run_pipeline()

print(f"\n📊 Training set shape : {X_train.shape}")
print(f"📊 Test set shape     : {X_test.shape}")
print(f"📊 Label distribution (train):")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    label = "Real" if u == 1 else "Fake"
    print(f"   {label}: {c:,} ({c / len(y_train):.1%})")

# %% [markdown]
# ## 2️⃣ Model Training
# Train **Naive Bayes**, **Logistic Regression**, and **Random Forest**
# using GridSearchCV with 5-fold cross-validation.

# %% ──────────────────────── TRAINING ──────────────────────────────────

results = train_all_models(X_train, y_train)

# Display summary table
print("\n\n📋  Training Summary")
print("─" * 65)
print(f"{'Model':<22} {'Best CV F1':>12} {'Best Params'}")
print("─" * 65)
for r in results:
    print(f"  {r['name']:<20} {r['best_cv_score']:>10.4f}   {r['best_params']}")
print("─" * 65)

# %% [markdown]
# ## 3️⃣ Evaluation
# Evaluate all models on the held-out test set and compare them.

# %% ──────────────────────── EVALUATION ────────────────────────────────

all_metrics = evaluate_all(results, X_test, y_test)

# %% ──────────────────────── COMPARISON CHART ──────────────────────────

plot_model_comparison(all_metrics)

# %% [markdown]
# ## 4️⃣ Save Best Model
# Persist the best-performing model and its TF-IDF vectorizer.

# %% ──────────────────────── SAVE ──────────────────────────────────────

best_model = results[0]["best_model"]
save_artifacts(best_model, tfidf_vectorizer)

print(f"\n🏆 Saved model: {results[0]['name']}")

# %% [markdown]
# ## 5️⃣ Real-Time Prediction Demo
# Try the model on a few sample texts.

# %% ──────────────────────── PREDICTION DEMO ──────────────────────────

sample_texts = [
    "Breaking: Scientists discover new vaccine that cures all diseases overnight!",
    "The president held a press conference today to discuss the new infrastructure bill "
    "that was passed by the Senate with bipartisan support.",
    "SHOCKING: Aliens land in New York City and demand world leaders meet them immediately!",
    "The Federal Reserve announced a 0.25% interest rate increase, citing steady economic growth.",
]

print("\n" + "═" * 60)
print("  📰  PREDICTION DEMO")
print("═" * 60)

for i, text in enumerate(sample_texts, 1):
    result = predict_news(text, model=best_model, tfidf=tfidf_vectorizer)
    conf_str = f" ({result['confidence']:.1%})" if result['confidence'] else ""
    print(f"\n  [{i}] {text[:80]}…")
    print(f"      → {result['label']}{conf_str}")

# %% [markdown]
# ## 6️⃣ Next Steps
# - Run the **Streamlit UI**: `streamlit run app.py`
# - Run the **CLI predictor**: `python predictor.py`
# - Tune hyperparameters in `config.py`

# %% ──────────────────────── END ───────────────────────────────────────
print("\n✅ Pipeline complete! All reports saved to:", config.REPORT_DIR)
