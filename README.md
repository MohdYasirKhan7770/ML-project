# PrismaTruth AI — Fake News Detection System

> A production-ready, end-to-end machine learning system for classifying news articles as **Real** or **Fake**, served through a modern web interface and a FastAPI backend.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why This Project?](#2-why-this-project)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Core Concepts Explained](#4-core-concepts-explained)
   - [Natural Language Processing (NLP)](#41-natural-language-processing-nlp)
   - [TF-IDF Vectorization](#42-tf-idf-vectorization)
   - [Machine Learning Classifiers](#43-machine-learning-classifiers)
   - [Hyperparameter Tuning with GridSearchCV](#44-hyperparameter-tuning-with-gridsearchcv)
   - [Cross-Validation](#45-cross-validation)
   - [Evaluation Metrics](#46-evaluation-metrics)
   - [Model Persistence with Joblib](#47-model-persistence-with-joblib)
5. [Project Structure](#5-project-structure)
6. [Module-by-Module Breakdown](#6-module-by-module-breakdown)
   - [config.py](#61-configpy)
   - [data_pipeline.py](#62-data_pipelinepy)
   - [model_trainer.py](#63-model_trainerpy)
   - [evaluator.py](#64-evaluatorpy)
   - [predictor.py](#65-predictorpy)
   - [api.py (FastAPI Backend)](#66-apipy-fastapi-backend)
   - [config_manager.py](#67-config_managerpy)
   - [Frontend (static/)](#68-frontend-static)
7. [Tech Stack](#7-tech-stack)
8. [ML Pipeline Flow](#8-ml-pipeline-flow)
9. [API Endpoints](#9-api-endpoints)
10. [Getting Started](#10-getting-started)
    - [Prerequisites](#101-prerequisites)
    - [Local Setup](#102-local-setup)
    - [Docker Deployment](#103-docker-deployment)
11. [Data Format](#11-data-format)
12. [Training the Model](#12-training-the-model)
13. [Configuration Reference](#13-configuration-reference)
14. [Design Decisions & Rationale](#14-design-decisions--rationale)
15. [Limitations & Future Work](#15-limitations--future-work)
16. [License](#16-license)

---

## 1. Project Overview

**PrismaTruth AI** is a complete, production-grade fake news detection system built with Python. It takes raw news article text as input and returns a binary classification:

- ✅ **REAL News** — The article follows linguistic patterns found in factual reporting.
- 🚩 **FAKE News** — The article exhibits patterns commonly seen in sensational, misleading, or fabricated content.

The system is not just a model — it is a full product: a trained ML classifier, a REST API, a conversational "agent" response layer, and a modern browser UI.

---

## 2. Why This Project?

Misinformation spreads faster than corrections. Manual fact-checking is slow and doesn't scale. This project demonstrates how Machine Learning and NLP can be applied to automatically screen news content, acting as a **first-pass filter** that can:

- Help journalists quickly triage suspicious articles.
- Assist readers in evaluating content credibility.
- Serve as a foundation for more sophisticated media literacy tools.

The project was also designed as a **learning vehicle** to master:
- The full ML pipeline from raw data to deployed API.
- Text feature engineering with TF-IDF.
- Model selection and hyperparameter optimization.
- Building production-quality Python backends with FastAPI.
- Creating modern web UIs served from a Python server.

---

## 3. Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                     USER'S BROWSER                       │
│  (index.html + style.css + script.js)                    │
│                                                          │
│  ┌──────────────┐    POST /agent     ┌────────────────┐  │
│  │  Textarea    │ ─────────────────► │  FastAPI       │  │
│  │  (article)   │                   │  (api.py)      │  │
│  └──────────────┘ ◄───────────────── └────────────────┘  │
│  ┌────────────────────────────────┐         │            │
│  │  Result Card                   │         │            │
│  │  Verdict + Confidence + Steps  │         ▼            │
│  └────────────────────────────────┘  ┌────────────────┐  │
│                                      │  predictor.py  │  │
│                                      │  (ML Core)     │  │
│                                      └────────────────┘  │
│                                             │            │
│                                    ┌────────┴────────┐   │
│                               ┌────┴────┐       ┌────┴──┐│
│                               │ TF-IDF  │       │ Model ││
│                               │.joblib  │       │.joblib││
│                               └─────────┘       └───────┘│
└──────────────────────────────────────────────────────────┘
```

---

## 4. Core Concepts Explained

### 4.1 Natural Language Processing (NLP)

**What it is:** NLP is the branch of AI that enables computers to understand, interpret, and generate human language.

**How it's used here:**
The raw news text must be transformed into a numerical form before any ML model can process it. This project uses the following NLP preprocessing steps inside `data_pipeline.py`:

1. **Lowercasing** — Ensures `"President"` and `"president"` are treated as the same word.
2. **URL removal** — Strips `http://...` links, which are noise for classification.
3. **HTML tag removal** — Cleans any `<div>`, `<p>` etc. that may appear in scraped text.
4. **Whitespace normalization** — Collapses multiple spaces into one.

**Why keep it lightweight?** Early experiments with heavy NLP (stemming, stopword removal, lemmatization via NLTK) were found to reduce accuracy on the hybrid TF-IDF setup, because even "stop words" like `"the"` and `"a"` carry signal in certain sensationalist headlines. The pipeline therefore uses minimal, targeted preprocessing.

---

### 4.2 TF-IDF Vectorization

**What it is:** TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It converts raw text into a numerical matrix where each value reflects how important a word is to a particular document relative to the whole corpus.

**Formula:**
```
TF-IDF(t, d) = TF(t, d) × log(N / DF(t))
```
- `TF(t, d)` — How often term `t` appears in document `d`.
- `N` — Total number of documents.
- `DF(t)` — Number of documents containing term `t`.

A word that appears frequently in one document but rarely across many documents gets a **high score** — it's distinctive. Common words like "the" appear everywhere and get a **low score**.

**Why a Hybrid (Word + Character) TF-IDF?**

This project uses a `FeatureUnion` of two TF-IDF vectorizers:

| Vectorizer | Analyzer | N-gram Range | Max Features | Purpose |
|---|---|---|---|---|
| Word TF-IDF | word | (1, 2) — unigrams + bigrams | 12,000 | Captures full words and two-word phrases |
| Char TF-IDF | char_wb | (3, 5) — 3 to 5 character sequences | 18,000 | Handles typos, new words, short headlines |

**Why bigrams?** Two-word phrases like `"breaking news"` or `"secret government"` carry much stronger fake/real signals than individual words alone.

**Why character n-grams?** Fake news often uses unusual spelling, ALL-CAPS, excessive punctuation, or invented words. Character n-grams are resilient to these stylistic tricks and work well even on short headlines.

---

### 4.3 Machine Learning Classifiers

Three classifiers are trained, evaluated, and compared:

#### Naive Bayes (`MultinomialNB`)
**Concept:** Based on Bayes' theorem, it assumes all features (words) are independent of each other (the "naive" assumption). It calculates the probability that a document belongs to each class.

**Why use it?**
- Extremely fast to train.
- Works surprisingly well for text classification.
- Acts as a strong baseline.
- Requires non-negative features, making it compatible with TF-IDF.

**Hyperparameter tuned:** `alpha` (Laplace smoothing — prevents zero probability for unseen words).

---

#### Logistic Regression (`LogisticRegression`)
**Concept:** Despite the name, this is a classification algorithm. It learns a set of weights for each feature and applies the sigmoid function to produce a probability between 0 and 1.

**Why use it?**
- Highly interpretable — you can inspect which words most strongly predict "fake" or "real".
- Generalizes well with regularization.
- Usually the best performer on TF-IDF features.

**Hyperparameters tuned:** `C` (inverse regularization strength), `penalty` (L2), `class_weight` (handles imbalanced classes).

---

#### Random Forest (`RandomForestClassifier`)
**Concept:** An ensemble of many Decision Trees, each trained on a random subset of samples and features. Final prediction is made by majority vote.

**Why use it?**
- Handles non-linear decision boundaries.
- Robust to overfitting due to ensemble averaging.
- No need for feature scaling.

**Hyperparameters tuned:** `n_estimators` (number of trees), `max_depth` (max tree depth), `min_samples_split` (controls leaf granularity).

---

### 4.4 Hyperparameter Tuning with GridSearchCV

**What it is:** `GridSearchCV` exhaustively tries every combination of hyperparameters from a defined grid, using cross-validation to score each combination.

**Why not just pick defaults?** Default hyperparameters are general-purpose starting points. Tuning, for example, the regularization strength `C` of Logistic Regression from `[0.1, 1, 5, 10]` can meaningfully improve F1 score, especially on imbalanced text datasets.

**Scoring metric:** The primary metric is `f1` (F1 Score), not accuracy. F1 balances precision and recall, which is essential when the cost of missing a real fake article (false negative) is high.

---

### 4.5 Cross-Validation

**What it is:** The dataset is split into `k` equal "folds". The model is trained `k` times, each time using a different fold as the validation set and the remaining `k-1` folds as training data.

**Configuration:** `CV_FOLDS = 5` (5-fold cross-validation).

**Why not just a single train/test split?**
- A single split can be "lucky" or "unlucky" depending on how the data falls.
- Cross-validation gives a much more reliable estimate of real-world performance.
- It also helps detect overfitting — if training accuracy is much higher than CV accuracy, the model has overfit.

**Stratified splitting** is used throughout to ensure each fold has the same ratio of real vs. fake articles as the full dataset.

---

### 4.6 Evaluation Metrics

The system calculates four metrics on the held-out test set:

| Metric | Formula | What it measures |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of all predicted fake, how many actually were? |
| **Recall** | TP / (TP + FN) | Of all actually fake, how many did we catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

**Why F1 over accuracy?** If the dataset has 90% real news, a model predicting "REAL" for everything would achieve 90% accuracy but be completely useless. F1 balances the mistake types.

Visualizations generated:
- **Confusion Matrix** — Heatmap showing true positives, false positives, true negatives, and false negatives.
- **Model Comparison Chart** — Grouped bar chart comparing all four metrics across all three models.

---

### 4.7 Model Persistence with Joblib

After training, the best model and the fitted TF-IDF vectorizer are saved to disk using **Joblib**:

```
models/
├── best_model.joblib        # Trained classifier (e.g., Logistic Regression)
└── tfidf_vectorizer.joblib  # Fitted FeatureUnion (word + char TF-IDF)
```

**Why Joblib?** Joblib uses memory-mapped files and is optimized for large NumPy arrays and sparse matrices, making it significantly faster than Python's built-in `pickle` for scientific objects.

**Why save the vectorizer separately?** The vectorizer was fitted on the training data. At inference time, any new text must be transformed by the *same* vectorizer with the *same* vocabulary. Saving them together ensures reproducibility.

---

## 5. Project Structure

```
ML-project/
│
├── 📁 data/                    # Raw CSV datasets (True.csv, Fake.csv)
├── 📁 models/                  # Saved model artifacts (.joblib files)
├── 📁 reports/                 # Generated plots (confusion matrices, charts)
├── 📁 static/                  # Frontend assets served by FastAPI
│   ├── index.html              # Main web UI (PrismaTruth AI)
│   ├── style.css               # Glassmorphic dark-theme styles
│   └── script.js               # Vanilla JS — fetches API, renders results
│
├── config.py                   # Hyperparameters, paths, constants (ML training)
├── config_manager.py           # Runtime config loaded from .env (API server)
├── data_pipeline.py            # Data loading, cleaning, preprocessing, TF-IDF
├── model_trainer.py            # GridSearchCV training + model selection
├── evaluator.py                # Metrics, confusion matrix, comparison charts
├── predictor.py                # Inference logic (single + batch prediction)
├── api.py                      # FastAPI server — REST endpoints + static serving
│
├── app.py                      # Streamlit alternative UI (standalone)
├── main_notebook.py            # End-to-end pipeline runner
├── data_loader.py              # Lightweight data loading utility
├── explain.py                  # Model explainability hooks (LIME)
├── validation.py               # Input validation utilities
├── test_model.py               # Quick prediction test script
│
├── .env                        # Environment variables (HOST, PORT, paths)
├── requirements.txt            # All Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose for one-command deployment
├── render.yaml                 # Render.com deployment config
└── README.md                   # This file
```

---

## 6. Module-by-Module Breakdown

### 6.1 `config.py`
**Role:** Single source of truth for all constants used during training.

Key settings:
```python
MAX_FEATURES_TFIDF = 12_000   # Word-level TF-IDF vocabulary size
CHAR_MAX_FEATURES_TFIDF = 18_000  # Character-level vocabulary size
NGRAM_RANGE = (1, 2)          # Unigrams + bigrams
CHAR_NGRAM_RANGE = (3, 5)     # 3- to 5-char sequences
CV_FOLDS = 5                  # Cross-validation splits
SCORING_METRIC = "f1"         # Primary optimization target
TEST_SIZE = 0.20              # 80/20 train/test split
RANDOM_STATE = 42             # Reproducibility seed
```

---

### 6.2 `data_pipeline.py`
**Role:** Handles the entire data journey — from raw CSVs to numerical feature matrices ready for training.

**Steps:**
1. `load_data()` — Reads `True.csv` and `Fake.csv`, labels them (1 = real, 0 = fake), shuffles, and merges.
2. `clean_data()` — Drops nulls/duplicates, combines `title` + `text` columns, and strips location/publisher prefixes (e.g., `"WASHINGTON (Reuters) - "`).
3. **Seed examples injection** — Injects 10 known-real and 10 known-fake seed examples to bias the model toward clear real-world patterns. This improves robustness.
4. `preprocess_text()` — Lightweight normalization (lowercase, URL removal, whitespace collapse).
5. `build_tfidf()` — Fits the word + character TF-IDF `FeatureUnion`.
6. `run_pipeline()` — Calls all of the above in order and returns train/test matrices.

---

### 6.3 `model_trainer.py`
**Role:** Trains all classifiers using GridSearchCV and selects the best one.

**Flow:**
1. `get_model_catalogue()` — Returns a list of (name, estimator, param grid) tuples. Easy to extend with new models.
2. `train_model()` — Runs `GridSearchCV` for a single model, prints results.
3. `cross_validate_model()` — Runs additional k-fold CV on the winner.
4. `train_all_models()` — Orchestrates training of all models, sorts by CV F1 score.
5. `save_artifacts()` — Persists the best model + vectorizer via Joblib.

---

### 6.4 `evaluator.py`
**Role:** Generates all evaluation artifacts for scientific analysis.

Outputs per model:
- Accuracy, Precision, Recall, F1 printed to console.
- Full `classification_report` (sklearn format).
- `confusion_matrix_{ModelName}.png` saved to `reports/`.
- `model_comparison.png` — grouped bar chart saved to `reports/`.

Uses `matplotlib.use("Agg")` to disable any GUI window, so it runs safely in headless/server environments.

---

### 6.5 `predictor.py`
**Role:** Core inference engine. Wraps the saved model + vectorizer for use at runtime.

Functions:
- `_load_model_and_vectorizer()` — Loads artifacts from disk once.
- `predict_news(text)` — Preprocesses input, transforms with TF-IDF, runs model, returns `{label, confidence, raw_prediction}`.
- `predict_batch(texts)` — Runs `predict_news` for a list of articles.
- `interactive_cli()` — Command-line REPL for manual testing.

---

### 6.6 `api.py` (FastAPI Backend)
**Role:** The production server. Exposes REST endpoints and serves the frontend.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{status, model_loaded, version}` |
| `/info` | GET | Returns app name, version, description |
| `/predict` | POST | Raw prediction: `{label, confidence, raw_prediction}` |
| `/agent` | POST | Full agent response: prediction + conversational explanation |
| `/` | GET | Serves `static/index.html` (the UI) |

The `/agent` endpoint is the primary one used by the UI. It wraps the model output in a human-readable response including:
- A verdict summary sentence.
- A plain-language explanation of the signal.
- Recommended next steps for the user.

---

### 6.7 `config_manager.py`
**Role:** Runtime configuration for the API server, loaded from `.env`.

Variables loaded:
- `HOST` — Server bind address (default: `0.0.0.0`)
- `PORT` — Server port (default: `8000`)
- `MODEL_PATH` — Path to `best_model.joblib`
- `VECTORIZER_PATH` — Path to `tfidf_vectorizer.joblib`
- `STATIC_DIR` — Directory for frontend files

Separation of training config (`config.py`) and runtime config (`config_manager.py`) follows the **12-factor app** principle: configuration that changes between environments should come from the environment, not hardcoded in source.

---

### 6.8 Frontend (`static/`)

**`index.html`** — Structure of the PrismaTruth AI web interface:
- **Hero section** — Branding, description, live model status badge.
- **Agent Console** — Textarea for input, example chips (pre-filled prompts), Analyze button.
- **Agent Response Card** — Verdict, confidence %, confidence bar, agent message, recommended steps.
- **Workflow Timeline** — Explains the 3-step process to non-technical users.
- **History section** — Shows last 8 analyses stored in `localStorage`.

**`script.js`** — Vanilla JavaScript (no frameworks):
- `checkHealth()` — Polls `/health` on page load to show live model status.
- `analyzeBtn` click → POSTs to `/agent` → calls `showResult()`.
- `addToHistory()` / `renderHistory()` — Manages session history in `localStorage`.
- `escapeHtml()` — Sanitizes user input to prevent XSS.
- Example chips populate the textarea on click.

**`style.css`** — Dark glassmorphic theme:
- CSS custom properties (variables) for colors and spacing.
- `backdrop-filter: blur()` for glass card effects.
- Gradient animated glows in the page background.
- Responsive layout with CSS Grid.
- Google Fonts: **Space Grotesk** + **Manrope**.

---

## 7. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **ML / NLP** | scikit-learn, NLTK, NumPy, pandas | Industry-standard, fast, well-documented |
| **Feature Engineering** | TF-IDF (word + char), FeatureUnion | Proven effective for text classification |
| **Model Persistence** | Joblib | Optimized for large NumPy/sparse arrays |
| **API Backend** | FastAPI + Uvicorn | Async, fast, auto-docs, type-safe with Pydantic |
| **Frontend** | Vanilla HTML + CSS + JavaScript | Zero dependencies, fast, full control |
| **Fonts** | Google Fonts (Space Grotesk, Manrope) | Modern, highly legible typography |
| **Alt UI** | Streamlit | Quick interactive prototyping |
| **Configuration** | python-dotenv | 12-factor app, environment-based config |
| **Containerization** | Docker + Docker Compose | Reproducible deployments |
| **Visualization** | Matplotlib + Seaborn | Standard scientific plotting |

---

## 8. ML Pipeline Flow

```
Raw CSVs (True.csv + Fake.csv)
           │
           ▼
    load_data()  ──────────────────────────── Label: 1=Real, 0=Fake
           │                                  Shuffle & merge datasets
           ▼
    clean_data()  ─────────────────────────── Drop nulls, duplicates
           │                                  Combine title + text
           │                                  Remove publisher prefixes
           │                                  Inject seed examples
           ▼
  preprocess_text()  ────────────────────── Lowercase, remove URLs/HTML
           │
           ▼
   train_test_split()  ─────────────────── 80% train / 20% test (stratified)
           │
           ▼
    build_tfidf()  ─────────────────────── Fit FeatureUnion (word + char)
           │                               on TRAINING data only
           ▼
   GridSearchCV × 3 models  ────────────── Naive Bayes
           │                               Logistic Regression
           │                               Random Forest
           ▼
    Select best CV F1 model
           │
           ▼
    evaluate_all()  ────────────────────── Accuracy, Precision, Recall, F1
           │                               Confusion matrices, comparison chart
           ▼
    save_artifacts()  ─────────────────── best_model.joblib
                                          tfidf_vectorizer.joblib
```

---

## 9. API Endpoints

### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `POST /predict`
**Request body:**
```json
{ "text": "Scientists discover hidden alien base under Antarctica." }
```
**Response:**
```json
{
  "label": "FAKE News",
  "confidence": 0.9412,
  "raw_prediction": 0
}
```

### `POST /agent`
**Request body:**
```json
{ "text": "The central bank raised interest rates by 0.25%." }
```
**Response:**
```json
{
  "prediction": {
    "label": "REAL News",
    "confidence": 0.8871,
    "raw_prediction": 1
  },
  "agent": {
    "message": "I reviewed the article and my current assessment is likely reliable. Model confidence is 88.7%.",
    "verdict": "REAL News",
    "confidence_percent": 88.7,
    "summary": "The article follows patterns that are common in factual reporting.",
    "excerpt": "The central bank raised interest rates by 0.25%.",
    "next_steps": [
      "Treat this as a screening signal, not final proof.",
      "Check the publication, author, and date before sharing.",
      "Look for corroboration from multiple established outlets."
    ]
  }
}
```

---

## 10. Getting Started

### 10.1 Prerequisites

- Python 3.10 or higher
- `pip` package manager
- (Optional) Docker Desktop

### 10.2 Local Setup

**Step 1: Clone the repository**
```bash
git clone <repository-url>
cd ML-project
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Download NLTK data** (auto-handled by `data_pipeline.py`, but can be done manually)
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

**Step 4: Place your data** — Put `True.csv` and `Fake.csv` in the `data/` directory.
> Dataset source: [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

**Step 5: Train the model** *(skip if `models/` already has `.joblib` files)*
```bash
python main_notebook.py
```

**Step 6: Run the API server**
```bash
python api.py
```

**Step 7: Access the UI** at `http://127.0.0.1:8000`

---

**Alternative: Run the Streamlit UI**
```bash
streamlit run app.py
# Access at http://127.0.0.1:8501
```

**Alternative: Test via CLI**
```bash
python predictor.py
```

### 10.3 Docker Deployment

**Build the image:**
```bash
docker build -t prismatruth-ai .
```

**Run the container:**
```bash
docker run -p 8000:8000 prismatruth-ai
```

**Or with Docker Compose:**
```bash
docker-compose up
```

---

## 11. Data Format

The model expects two CSV files in the `data/` directory:

**`True.csv`** — Real news articles
```
title,text,subject,date
"Government passes new bill...","WASHINGTON (Reuters) - ...","politicsNews","January 21, 2018"
```

**`Fake.csv`** — Fake/fabricated news articles
```
title,text,subject,date
"SHOCKING: Secret base found...","Scientists have discovered...","News","December 12, 2017"
```

Both files must have at least `title` or `text` columns (ideally both).

---

## 12. Training the Model

Run the full end-to-end training pipeline:

```bash
python main_notebook.py
```

This will:
1. Load and clean the data.
2. Run TF-IDF vectorization.
3. Train Naive Bayes, Logistic Regression, and Random Forest with GridSearchCV.
4. Run cross-validation on each.
5. Evaluate all models on the test set.
6. Save confusion matrix PNGs to `reports/`.
7. Save the best model and vectorizer to `models/`.

Expected output (abbreviated):
```
[DATA] Loaded 21,417 real + 23,481 fake = 44,898 total articles.
[CLEAN] Removed 892 duplicates -> 44,026 rows remain.
[TFIDF] Feature extractor fitted with 29,342 combined features.
[SPLIT] Train: 35,220  |  Test: 8,806

Training: Logistic Regression
  Best params : {'C': 5, 'class_weight': None, 'max_iter': 1500, 'penalty': 'l2'}
  Best CV F1:   0.9871

🏆  Best model: Logistic Regression  (CV F1 = 0.9871)

[SAVE] Model  → models/best_model.joblib
[SAVE] TF-IDF → models/tfidf_vectorizer.joblib
```

---

## 13. Configuration Reference

### `.env` (Runtime / API)
```env
HOST=0.0.0.0
PORT=8000
DEBUG=False
MODEL_PATH=models/best_model.joblib
VECTORIZER_PATH=models/tfidf_vectorizer.joblib
STATIC_DIR=static
```

### `config.py` (Training)
| Variable | Default | Description |
|---|---|---|
| `MAX_FEATURES_TFIDF` | `12000` | Word TF-IDF vocabulary cap |
| `CHAR_MAX_FEATURES_TFIDF` | `18000` | Char TF-IDF vocabulary cap |
| `NGRAM_RANGE` | `(1, 2)` | Word n-gram range |
| `CHAR_NGRAM_RANGE` | `(3, 5)` | Character n-gram range |
| `TEST_SIZE` | `0.2` | Fraction of data for testing |
| `RANDOM_STATE` | `42` | Seed for reproducibility |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `SCORING_METRIC` | `"f1"` | GridSearchCV optimization target |

---

## 14. Design Decisions & Rationale

| Decision | Why |
|---|---|
| **Hybrid TF-IDF (word + char)** | Word n-grams capture meaning; char n-grams handle typos, slang, and short headlines better. |
| **F1 as scoring metric** | More robust than accuracy for potentially imbalanced datasets. |
| **Separate training & runtime configs** | Follows 12-factor app principles; production settings should come from environment, not code. |
| **FastAPI over Flask** | Async-native, Pydantic validation, auto-generated `/docs`, faster performance. |
| **No frontend framework** | Zero build pipeline, instant load, full CSS/JS control — appropriate for a single-page AI tool. |
| **localStorage for history** | No database needed; keeps the deployment simple without sacrificing UX. |
| **Seed examples in training data** | Ensures the model correctly handles common short-form real-world inputs, countering distribution shift from headline-only inputs. |
| **Agent response layer** | Raw model probabilities are not meaningful to non-technical users. The `/agent` endpoint translates output into actionable language. |
| **Joblib over Pickle** | Faster serialization for large NumPy arrays and sparse matrices. |
| **Matplotlib Agg backend** | Allows chart generation in headless server environments (Docker, cloud VMs) without a display. |

---

## 15. Limitations & Future Work

### Current Limitations
- **Static training data** — The model is frozen at training time. It cannot learn about new events or evolving misinformation patterns without retraining.
- **English only** — TF-IDF features are language-dependent; non-English text will not be classified reliably.
- **Context-free** — The model only sees text patterns, not real-world truth. Well-written fake news may slip through.
- **Binary classification** — Real-world credibility exists on a spectrum; a binary label oversimplifies.

### Potential Improvements
- [ ] **Transformer models** (e.g., BERT, DistilBERT) for deeper semantic understanding.
- [ ] **Source/metadata signals** — Domain reputation, author, publication date.
- [ ] **Active learning** — Allow users to flag errors to improve the model over time.
- [ ] **Multi-language support** — Multilingual BERT or language detection.
- [ ] **Explainability** — LIME/SHAP integration to highlight which words influenced the verdict.
- [ ] **Database logging** — Store predictions in PostgreSQL for analytics.
- [ ] **Auth / rate limiting** — For public deployment.

---

## 16. License

MIT License — Free to use, modify, and distribute with attribution.

---

*Built with Python, scikit-learn, FastAPI, and Vanilla JS.*  
*PrismaTruth AI — Screening misinformation, one article at a time.*
