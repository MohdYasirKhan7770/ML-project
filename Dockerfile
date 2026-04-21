FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -m nltk.downloader stopwords wordnet punkt punkt_tab

# Copy application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the server
CMD ["python", "api.py"]
