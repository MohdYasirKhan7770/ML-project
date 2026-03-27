# Docker Deployment Walkthrough - Fake News Detector

The project has been successfully containerized and deployed using Docker Compose. Two services are now running: an API backend and a Streamlit frontend UI.

## Changes Made

### Docker Configuration
- **Dockerfile**: Updated to include system dependencies (`curl`), pre-download NLTK data (`stopwords`, `wordnet`, `punkt`), and expose ports `8000` (API) and `8501` (UI).
- **docker-compose.yml**: Created to manage the orchestration of the two services.
- **.dockerignore**: Added to exclude large raw data files and temporary artifacts, optimizing the image size.

## Verification Results

### Container Status
Both services are running and healthy:
- **fake-news-api**: Running on [http://localhost:8000](http://localhost:8000)
- **fake-news-ui**: Running on [http://localhost:8501](http://localhost:8501)

### Health Checks
- **API Health**: Verified via `curl http://localhost:8000/health`.
- **UI Availability**: Verified via logs showing the Streamlit server is active.

### Logs
The logs confirm that the ML models ([best_model.joblib](file:///c:/Users/YASIR%20KHAN/Desktop/ML-project/models/best_model.joblib) and [tfidf_vectorizer.joblib](file:///c:/Users/YASIR%20KHAN/Desktop/ML-project/models/tfidf_vectorizer.joblib)) are loaded successfully on startup for both the API and UI.

## How to Manage
- **Stop**: `docker compose down`
- **Restart**: `docker compose restart`
- **View Logs**: `docker compose logs -f`
