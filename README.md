# Fake News Detector — AI Powered

A production-ready, AI-powered system to identify and classify fake news articles using Natural Language Processing (NLP) and Machine Learning.

## Features
- **Modern UI**: A sleek, dark-themed glassmorphic interface.
- **Real-time Analysis**: Instant classification with confidence scoring.
- **History Tracking**: Keep track of your recent analyses (stored locally).
- **Interactive Examples**: Pre-defined chips to quickly test the model.
- **Production Backend**: Built with FastAPI, featuring structured logging and configuration management.
- **Containerized**: Ready for deployment with Docker.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/JS/CSS (Premium Glassmorphism)
- **ML Engine**: Scikit-learn (TF-IDF Vectorization + Classifier)
- **DevOps**: Docker, python-dotenv

## Getting Started

### Prerequisites
- Python 3.10+
- [Optional] Docker

### Local Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ML-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python api.py
   ```
4. Access the UI at `http://127.0.0.1:8000`

### Docker Deployment
1. Build the image:
   ```bash
   docker build -t fake-news-detector .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 fake-news-detector
   ```

## Project Structure
- `api.py`: Main FastAPI application entry point.
- `config_manager.py`: Centralized configuration using `.env`.
- `predictor.py`: Core prediction logic and model loading.
- `static/`: Frontend assets (UI files).
- `models/`: Directory containing pre-trained model artifacts.

## License
MIT License
