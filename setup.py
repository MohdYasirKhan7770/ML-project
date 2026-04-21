"""
============================================================================
  setup.py — PrismaTruth AI: One-command environment bootstrap
============================================================================
  Run this ONCE after installing requirements to download NLTK data
  and verify that the model artifacts are present.

  Usage:
      python setup.py
============================================================================
"""

import os
import sys


def check_python_version():
    if sys.version_info < (3, 10):
        print("❌  Python 3.10 or higher is required.")
        sys.exit(1)
    print(f"✅  Python {sys.version_info.major}.{sys.version_info.minor} detected.")


def download_nltk_data():
    print("\n📥  Downloading NLTK resources...")
    import nltk
    for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    print("✅  NLTK data downloaded.")


def check_model_artifacts():
    model_path = os.path.join("models", "best_model.joblib")
    tfidf_path = os.path.join("models", "tfidf_vectorizer.joblib")

    print("\n🔍  Checking for model artifacts...")
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        model_size = os.path.getsize(model_path) / 1024
        tfidf_size = os.path.getsize(tfidf_path) / 1024
        print(f"✅  best_model.joblib     ({model_size:.1f} KB)")
        print(f"✅  tfidf_vectorizer.joblib ({tfidf_size:.1f} KB)")
    else:
        print("⚠️   Model artifacts NOT found in models/")
        print("     Run the training pipeline first:")
        print("       python main_notebook.py")


def check_data():
    print("\n🔍  Checking for dataset files...")
    true_path = os.path.join("data", "True.csv")
    fake_path = os.path.join("data", "Fake.csv")

    if os.path.exists(true_path) and os.path.exists(fake_path):
        true_size = os.path.getsize(true_path) / (1024 * 1024)
        fake_size = os.path.getsize(fake_path) / (1024 * 1024)
        print(f"✅  True.csv  ({true_size:.1f} MB)")
        print(f"✅  Fake.csv  ({fake_size:.1f} MB)")
    else:
        print("⚠️   Dataset CSV files not found in data/")
        print("     Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("     Place True.csv and Fake.csv in the data/ folder.")


def print_summary():
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print("\n  To start the web application:")
    print("    python api.py")
    print("    → Open http://127.0.0.1:8000\n")
    print("  To run the Streamlit UI:")
    print("    streamlit run app.py")
    print("    → Open http://127.0.0.1:8501\n")
    print("  To run the CLI predictor:")
    print("    python predictor.py\n")
    print("  To run the full ML training pipeline:")
    print("    python main_notebook.py\n")
    print("  To run prediction tests:")
    print("    python test_model.py\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  PrismaTruth AI — Environment Setup")
    print("=" * 60)

    check_python_version()
    download_nltk_data()
    check_model_artifacts()
    check_data()
    print_summary()
