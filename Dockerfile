# Container image for the always-on Kalshi trading/learning worker.
FROM python:3.11-slim

# Faster, quieter Python; no .pyc clutter.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default model-artifact location; on Railway, mount a Volume here so trained
# models survive redeploys (DATABASE_URL handles the rest of the state).
ENV MODEL_DIR=/data/models
RUN mkdir -p /data/models

# Railway provides $PORT; the worker serves /health on it.
EXPOSE 8080

CMD ["python", "-m", "kalshi_algo.service.worker"]
