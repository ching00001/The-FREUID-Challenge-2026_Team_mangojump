# Build from the repository root. All checkpoints are public; an optional
# Hugging Face token is not required.
FROM python:3.11-slim-bookworm

ARG HF_REPO=ching0206/freuid-2026-mangojump
ARG HF_REVISION=156f6e6ecf03e4a116ddf04a6a14be149a20fa9d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/docker/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY prepare_submission.py download_weights.py /app/
COPY docker/prepare_hf_cache.py /app/docker/prepare_hf_cache.py
RUN python download_weights.py --repo "$HF_REPO" \
      --revision "$HF_REVISION" --out /app/artifacts/system && \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
      python docker/prepare_hf_cache.py

ENTRYPOINT ["python", "/app/prepare_submission.py"]
