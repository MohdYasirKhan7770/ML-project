"""
============================================================================
  Data Pipeline Module — Fake News Detection System
============================================================================
  Handles every stage of the data journey:
    1. Loading raw CSVs (True.csv / Fake.csv)
    2. Labelling & merging
    3. Cleaning (drop NaN, remove duplicates)
    4. Text preprocessing (NLP via NLTK)
    5. TF-IDF vectorization
    6. Train / test split
============================================================================
"""

import re
import string
import warnings

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import config

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# NLTK RESOURCES  (download once, cached thereafter)
# ──────────────────────────────────────────────────────────────────────────
for _pkg in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ──────────────────────────────────────────────────────────────────────────
# 1. LOAD & MERGE
# ──────────────────────────────────────────────────────────────────────────
def load_data(true_path: str = config.TRUE_NEWS_FILE,
              fake_path: str = config.FAKE_NEWS_FILE) -> pd.DataFrame:
    """
    Load True.csv and Fake.csv, label them (1 = real, 0 = fake),
    merge, shuffle, and return a single DataFrame.
    """
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 1          # Real news
    df_fake["label"] = 0          # Fake news

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    print(f"[DATA] Loaded {len(df_true):,} real + {len(df_fake):,} fake "
          f"= {len(df):,} total articles.")
    return df


# ──────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ──────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop missing values and duplicate rows.
    Combine 'title' and 'text' into a single 'content' column.
    """
    df = df.copy()

    # Combine title + text (use empty string if column is missing)
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    elif "title" in df.columns:
        df["content"] = df["title"].fillna("")
    else:
        raise ValueError("Dataset must have at least a 'title' or 'text' column.")

    # Drop rows where content is empty after stripping
    df["content"] = df["content"].str.strip()
    df = df[df["content"].astype(bool)].reset_index(drop=True)

    # REMOVE PUBLISHER PREFIXES (Target Leakage Fix)
    # The True.csv dataset has headers like "WASHINGTON (Reuters) -"
    # This regex removes up to ~70 characters before the first " - " to strip the publisher.
    # Without this, the model just learns the word 'Reuters'.
    import re
    def remove_publisher(text):
        # Look for patterns like "CITY (Publisher) - " at the start of the string
        # We only apply this matching locally if it finds it near the beginning
        match = re.search(r'^.*? - ', text)
        if match and match.end() < 80:
            return text[match.end():]
        return text

    df["content"] = df["content"].apply(remove_publisher)

    # Remove duplicates
    initial_len = len(df)
    df.drop_duplicates(subset=["content"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[CLEAN] Removed {initial_len - len(df):,} duplicates → {len(df):,} rows remain.")

    return df[["content", "label"]]


# ──────────────────────────────────────────────────────────────────────────
# 3. NLP PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Apply full NLP pipeline to a single text string:
      • Lowercase
      • Remove URLs, HTML tags, numbers
      • Remove punctuation
      • Tokenize
      • Remove stopwords
      • Lemmatize
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)     # URLs
    text = re.sub(r"<.*?>", "", text)                         # HTML
    text = re.sub(r"\d+", "", text)                           # Numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Punctuation

    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `preprocess_text` to the entire 'content' column."""
    df = df.copy()
    print("[NLP]  Preprocessing text (this may take a minute)…")
    df["clean_content"] = df["content"].apply(preprocess_text)
    print("[NLP]  Done.")
    return df


# ──────────────────────────────────────────────────────────────────────────
# 4. TF-IDF VECTORIZATION
# ──────────────────────────────────────────────────────────────────────────
def build_tfidf(train_texts: pd.Series,
                max_features: int = config.MAX_FEATURES_TFIDF,
                ngram_range: tuple = config.NGRAM_RANGE) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the training corpus and return it.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,              # Apply log-scaling to TF
        strip_accents="unicode",
    )
    tfidf.fit(train_texts)
    print(f"[TFIDF] Vectorizer fitted with {len(tfidf.vocabulary_):,} features.")
    return tfidf


# ──────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline():
    """
    End-to-end data pipeline:
      Load → Clean → Preprocess NLP → Split → Vectorize
    Returns:
      X_train, X_test   (sparse TF-IDF matrices)
      y_train, y_test   (label arrays)
      tfidf_vectorizer   (fitted vectorizer for inference)
    """
    df = load_data()
    df = clean_data(df)
    df = preprocess_dataframe(df)

    # Train / test split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean_content"], df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"[SPLIT] Train: {len(X_train_txt):,}  |  Test: {len(X_test_txt):,}")

    # TF-IDF
    tfidf = build_tfidf(X_train_txt)
    X_train = tfidf.transform(X_train_txt)
    X_test = tfidf.transform(X_test_txt)

    return X_train, X_test, y_train, y_test, tfidf


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tfidf = run_pipeline()
    print(f"\nTrain matrix shape: {X_train.shape}")
    print(f"Test  matrix shape: {X_test.shape}")
