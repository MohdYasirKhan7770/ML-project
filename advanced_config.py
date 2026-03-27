import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_advanced_model")

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Validation configuration
SIMILARITY_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SEARCH_RESULTS = 3
SIMILARITY_THRESHOLD = 0.5 # If similarity > this, it's considered supported by real news

# Decision Weights
DECISION_WEIGHT_MODEL = 0.70
DECISION_WEIGHT_SIMILARITY = 0.30
