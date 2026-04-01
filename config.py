"""
============================================================================
  Configuration Module — Fake News Detection System
============================================================================
  Centralizes all hyperparameters, paths, and constants so every other
  module reads from a single source of truth.
============================================================================
"""

import os

# ──────────────────────────────────────────────────────────────────────────
# 1. DIRECTORY / FILE PATHS
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
for _dir in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Raw data filenames (place CSV files in the data/ folder)
TRUE_NEWS_FILE = os.path.join(DATA_DIR, "True.csv")
FAKE_NEWS_FILE = os.path.join(DATA_DIR, "Fake.csv")

# ──────────────────────────────────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────
MAX_FEATURES_TFIDF = 12_000          # Upper bound for word-level TF-IDF features
NGRAM_RANGE = (1, 2)                 # Word unigrams + bigrams
CHAR_MAX_FEATURES_TFIDF = 18_000     # Extra character-level features for short snippets
CHAR_NGRAM_RANGE = (3, 5)            # Character n-grams improve robustness on headlines
TEST_SIZE = 0.20                     # Train / test split ratio
RANDOM_STATE = 42                    # Reproducibility seed

# ──────────────────────────────────────────────────────────────────────────
# 3. MODEL HYPERPARAMETER GRIDS  (for GridSearchCV)
# ──────────────────────────────────────────────────────────────────────────
NB_PARAM_GRID = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
}

LR_PARAM_GRID = {
    "C": [0.1, 1, 5, 10],
    "penalty": ["l2"],
    "class_weight": [None, "balanced"],
    "max_iter": [1500],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 30, 50],
    "min_samples_split": [2, 5],
}

# ──────────────────────────────────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────────
CV_FOLDS = 5                         # Number of cross-validation folds
SCORING_METRIC = "f1"                # Primary metric for GridSearchCV

# ──────────────────────────────────────────────────────────────────────────
# 5. SAVED ARTIFACT NAMES
# ──────────────────────────────────────────────────────────────────────────
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.joblib")
TFIDF_VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")
