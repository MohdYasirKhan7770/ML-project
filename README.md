# 🔍 Fake News Detection System

An **end-to-end Machine Learning pipeline** for classifying news articles as **Real** or **Fake** using NLP and scikit-learn.

---

## 📁 Project Structure

```
ML-project/
├── config.py              # Centralized configuration & hyperparameters
├── data_pipeline.py       # Data loading, cleaning, NLP preprocessing, TF-IDF
├── model_trainer.py       # Training with GridSearchCV & cross-validation
├── evaluator.py           # Metrics, confusion matrices, comparison charts
├── predictor.py           # Single/batch prediction & CLI interface
├── app.py                 # Streamlit web UI
├── main_notebook.py       # Main orchestrator (Jupyter-compatible)
├── requirements.txt       # Python dependencies
├── README.md
├── data/                  # Place True.csv & Fake.csv here
│   ├── True.csv
│   └── Fake.csv
├── models/                # Saved model & vectorizer (auto-created)
└── reports/               # Confusion matrices & comparison charts (auto-created)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download the **Fake and Real News Dataset** from Kaggle:
- https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `True.csv` and `Fake.csv` into the `data/` folder.

### 3. Run the pipeline
```bash
python main_notebook.py
```
Or open `main_notebook.py` in VS Code / Jupyter and run cell-by-cell (uses `# %%` markers).

### 4. Launch the Streamlit UI
```bash
streamlit run app.py
```

### 5. Use the CLI predictor
```bash
python predictor.py
```

---

## 🧠 Models Trained

| Model                | Technique              |
| -------------------- | ---------------------- |
| Multinomial Naive Bayes | GridSearchCV (alpha)  |
| Logistic Regression  | GridSearchCV (C, penalty) |
| Random Forest        | GridSearchCV (n_estimators, max_depth) |

All models are compared on **Accuracy, Precision, Recall, and F1 Score**.

---

## 📊 Features

- ✅ Modular, production-ready code
- ✅ NLP preprocessing (NLTK: lemmatization, stopword removal)
- ✅ TF-IDF vectorization with bigrams
- ✅ GridSearchCV hyperparameter optimization
- ✅ 5-fold cross-validation
- ✅ Confusion matrix heatmaps
- ✅ Model comparison bar charts
- ✅ Best model saved with joblib
- ✅ CLI for real-time prediction
- ✅ Streamlit web UI
- ✅ Well-commented, clean code

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **NLTK** — Text preprocessing
- **scikit-learn** — ML models, TF-IDF, evaluation
- **matplotlib / seaborn** — Visualization
- **joblib** — Model serialization
- **Streamlit** — Web interface
