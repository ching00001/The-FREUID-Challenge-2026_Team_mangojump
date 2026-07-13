# Build from the repository root. All checkpoints are public and baked in.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv \
    HF_HOME=/app/docker/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY weights /app/weights
COPY docker/prepare_hf_cache.py /app/docker/prepare_hf_cache.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python docker/prepare_hf_cache.py

COPY src /app/src
COPY prepare_submission.py /app/

ENTRYPOINT ["python", "/app/prepare_submission.py"]
