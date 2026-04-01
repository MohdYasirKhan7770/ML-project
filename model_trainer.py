"""
============================================================================
  Model Trainer Module — Fake News Detection System
============================================================================
  Responsibilities:
    • Train multiple ML classifiers (Naive Bayes, Logistic Regression,
      Random Forest) with GridSearchCV for hyperparameter optimisation.
    • Perform k-fold cross-validation.
    • Compare all models and select the best one.
    • Persist the best model + vectorizer to disk via joblib.
============================================================================
"""

import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB

import config

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────
# 1. MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────
def get_model_catalogue():
    """
    Return a list of (name, estimator, param_grid) tuples.
    Easy to extend — just add another entry to the list.
    """
    return [
        ("Naive Bayes",          MultinomialNB(),                          config.NB_PARAM_GRID),
        ("Logistic Regression",  LogisticRegression(random_state=config.RANDOM_STATE), config.LR_PARAM_GRID),
        ("Random Forest",        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1), config.RF_PARAM_GRID),
    ]


# ──────────────────────────────────────────────────────────────────────────
# 2. TRAIN A SINGLE MODEL  (with GridSearchCV)
# ──────────────────────────────────────────────────────────────────────────
def train_model(name, estimator, param_grid, X_train, y_train):
    """
    Run GridSearchCV on the given estimator and return the best model
    along with training metadata.

    Returns
    -------
    dict with keys: name, best_model, best_params, best_cv_score, train_time
    """
    print(f"\n{'─' * 60}")
    print(f"  Training: {name}")
    print(f"{'─' * 60}")

    start = time.time()

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=config.CV_FOLDS,
        scoring=config.SCORING_METRIC,
        n_jobs=1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    elapsed = time.time() - start

    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV {config.SCORING_METRIC.upper()}: {grid.best_score_:.4f}")
    print(f"  Time        : {elapsed:.1f}s")

    return {
        "name": name,
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "train_time": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATION REPORT
# ──────────────────────────────────────────────────────────────────────────
def cross_validate_model(model, X, y, cv=config.CV_FOLDS):
    """
    Run stratified k-fold cross-validation and return per-fold scores.
    """
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=config.SCORING_METRIC, n_jobs=1)
    print(f"  CV Scores ({cv} folds): {np.round(scores, 4)}")
    print(f"  Mean ± Std : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ──────────────────────────────────────────────────────────────────────────
# 4. TRAIN ALL MODELS & SELECT BEST
# ──────────────────────────────────────────────────────────────────────────
def train_all_models(X_train, y_train):
    """
    Train every model in the catalogue, run cross-validation on each,
    and return a list of result dicts sorted by CV score (descending).
    """
    catalogue = get_model_catalogue()
    results = []

    for name, estimator, param_grid in catalogue:
        result = train_model(name, estimator, param_grid, X_train, y_train)

        # Additional cross-val on best estimator
        print(f"\n  Cross-validation on best {name}:")
        cv_scores = cross_validate_model(result["best_model"], X_train, y_train)
        result["cv_scores"] = cv_scores

        results.append(result)

    # Sort by best_cv_score descending
    results.sort(key=lambda r: r["best_cv_score"], reverse=True)
    print(f"\n{'═' * 60}")
    print(f"  🏆  Best model: {results[0]['name']}  "
          f"(CV {config.SCORING_METRIC.upper()} = {results[0]['best_cv_score']:.4f})")
    print(f"{'═' * 60}")

    return results


# ──────────────────────────────────────────────────────────────────────────
# 5. PERSIST MODEL & VECTORIZER
# ──────────────────────────────────────────────────────────────────────────
def save_artifacts(best_model, tfidf_vectorizer):
    """
    Save the best trained model and the TF-IDF vectorizer to disk.
    """
    joblib.dump(best_model, config.BEST_MODEL_FILE)
    joblib.dump(tfidf_vectorizer, config.TFIDF_VECTORIZER_FILE)
    print(f"\n[SAVE] Model  → {config.BEST_MODEL_FILE}")
    print(f"[SAVE] TF-IDF → {config.TFIDF_VECTORIZER_FILE}")


def load_artifacts():
    """
    Load the saved model and vectorizer from disk.
    """
    model = joblib.load(config.BEST_MODEL_FILE)
    tfidf = joblib.load(config.TFIDF_VECTORIZER_FILE)
    print("[LOAD] Model and vectorizer loaded successfully.")
    return model, tfidf
