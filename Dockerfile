# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK resources
RUN python -m nltk.downloader stopwords wordnet punkt punkt_tab

# Copy the rest of the application code (except what's in .dockerignore)
COPY . .

# Expose the ports the app runs on (FastAPI and Streamlit)
EXPOSE 8000 8501

# Run api.py by default
CMD ["python", "api.py"]
