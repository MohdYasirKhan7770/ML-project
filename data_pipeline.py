"""
============================================================================
  Data Pipeline Module - Fake News Detection System
============================================================================
  Handles every stage of the data journey:
    1. Loading raw CSVs (True.csv / Fake.csv)
    2. Labelling and merging
    3. Cleaning (drop empty rows, remove duplicates)
    4. Lightweight normalization for inference and training
    5. Hybrid word + character TF-IDF vectorization
    6. Train / test split
============================================================================
"""

import re
import warnings

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

import config

warnings.filterwarnings("ignore")

for _pkg in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    nltk.download(_pkg, quiet=True)

SEED_REAL_EXAMPLES = [
    "NASA confirmed a successful deployment milestone for the James Webb Space Telescope during an official mission update.",
    "The World Health Organization said it is monitoring a new virus variant and urged countries to continue surveillance.",
    "The Federal Reserve announced an interest-rate adjustment after reviewing inflation and employment data.",
    "The Centers for Disease Control and Prevention released updated public health guidance based on current evidence.",
    "Researchers at a major university published peer-reviewed findings showing incremental progress in cancer treatment.",
    "The United Nations reported renewed humanitarian aid deliveries following negotiations with regional officials.",
    "A national weather agency issued storm warnings as satellite data showed a rapidly strengthening system offshore.",
    "Officials said the infrastructure bill cleared its final legislative vote after bipartisan negotiations.",
    "The European Space Agency shared new telescope imagery captured during a scheduled observation campaign.",
    "Government data showed consumer prices rising more slowly than the previous quarter, according to the official release.",
]

SEED_FAKE_EXAMPLES = [
    "Scientists secretly proved that aliens control world governments from a hidden lunar headquarters.",
    "Doctors reveal a miracle kitchen ingredient that cures every cancer overnight and is being suppressed.",
    "A leaked memo shows all citizens will be forced to accept tracking microchips by next month.",
    "Experts warn that drinking a homemade chemical mixture can instantly prevent every known virus.",
    "A secret underground city in Antarctica was discovered by independent researchers using forbidden maps.",
    "Breaking reports claim world leaders already met time travelers and agreed to hide the evidence.",
    "Anonymous insiders say a new law will ban all private pets unless they are government monitored.",
    "A viral post claims one vitamin completely replaces vaccines and makes medical treatment unnecessary.",
    "Witnesses say a hidden energy weapon created the recent earthquake as part of a global plot.",
    "A shocking report claims the moon landing footage contains proof of an alien military alliance.",
]


def load_data(
    true_path: str = config.TRUE_NEWS_FILE,
    fake_path: str = config.FAKE_NEWS_FILE,
) -> pd.DataFrame:
    """
    Load True.csv and Fake.csv, label them (1 = real, 0 = fake),
    merge, shuffle, and return a single DataFrame.
    """
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 1
    df_fake["label"] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    print(
        f"[DATA] Loaded {len(df_true):,} real + {len(df_fake):,} fake "
        f"= {len(df):,} total articles."
    )
    return df


def remove_publisher_prefix(text: str) -> str:
    """
    Remove location/publisher prefixes such as
    'WASHINGTON (Reuters) -' near the start of the text.
    """
    match = re.search(r"^.*? - ", text)
    if match and match.end() < 80:
        return text[match.end():]
    return text


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop empty values and duplicate content.
    Combine 'title' and 'text' into a single 'content' column.
    """
    df = df.copy()

    if "title" in df.columns and "text" in df.columns:
        combined_content = df["title"].fillna("").str.strip() + " " + df["text"].fillna("").str.strip()
        combined_df = pd.DataFrame({
            "content": combined_content,
            "label": df["label"],
        })

        title_only = df["title"].fillna("").str.strip()
        title_df = pd.DataFrame({
            "content": title_only,
            "label": df["label"],
        })
        df = pd.concat([combined_df, title_df], ignore_index=True)
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    elif "title" in df.columns:
        df["content"] = df["title"].fillna("")
    else:
        raise ValueError("Dataset must have at least a 'title' or 'text' column.")

    df["content"] = df["content"].str.strip()
    df = df[df["content"].astype(bool)].reset_index(drop=True)
    df["content"] = df["content"].apply(remove_publisher_prefix)

    initial_len = len(df)
    df.drop_duplicates(subset=["content"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[CLEAN] Removed {initial_len - len(df):,} duplicates -> {len(df):,} rows remain.")

    seed_df = pd.DataFrame(
        [{"content": text, "label": 1} for text in SEED_REAL_EXAMPLES]
        + [{"content": text, "label": 0} for text in SEED_FAKE_EXAMPLES]
    )
    df = pd.concat([df[["content", "label"]], seed_df], ignore_index=True)
    df.drop_duplicates(subset=["content"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[["content", "label"]]


def preprocess_text(text: str) -> str:
    """
    Apply lightweight normalization while preserving words that matter
    for short headlines and real-world snippets.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `preprocess_text` to the entire 'content' column."""
    df = df.copy()
    print("[NLP]  Preprocessing text (this may take a minute)...")
    df["clean_content"] = df["content"].apply(preprocess_text)
    print("[NLP]  Done.")
    return df


def build_tfidf(
    train_texts: pd.Series,
    max_features: int = config.MAX_FEATURES_TFIDF,
    ngram_range: tuple = config.NGRAM_RANGE,
):
    """
    Fit a hybrid word + character TF-IDF extractor on the training corpus.
    """
    word_tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
        dtype=np.float32,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.CHAR_NGRAM_RANGE,
        max_features=config.CHAR_MAX_FEATURES_TFIDF,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    feature_extractor = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])
    feature_extractor.fit(train_texts)

    feature_count = sum(
        len(vectorizer.vocabulary_)
        for _, vectorizer in feature_extractor.transformer_list
    )
    print(f"[TFIDF] Feature extractor fitted with {feature_count:,} combined features.")
    return feature_extractor


def run_pipeline():
    """
    End-to-end data pipeline:
      Load -> Clean -> Preprocess -> Split -> Vectorize
    Returns:
      X_train, X_test   (sparse feature matrices)
      y_train, y_test   (label arrays)
      tfidf_vectorizer  (fitted feature extractor for inference)
    """
    df = load_data()
    df = clean_data(df)
    df = preprocess_dataframe(df)

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean_content"],
        df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"[SPLIT] Train: {len(X_train_txt):,}  |  Test: {len(X_test_txt):,}")

    tfidf = build_tfidf(X_train_txt)
    X_train = tfidf.transform(X_train_txt)
    X_test = tfidf.transform(X_test_txt)

    return X_train, X_test, y_train, y_test, tfidf


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tfidf = run_pipeline()
    print(f"\nTrain matrix shape: {X_train.shape}")
    print(f"Test  matrix shape: {X_test.shape}")
