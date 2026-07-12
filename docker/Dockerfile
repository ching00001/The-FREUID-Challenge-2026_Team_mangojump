# FREUID 2026 — team mangojump offline inference image.
#
# Contract: docker run --network none \
#   -v <flat image dir>:/data:ro -v <out dir>:/submissions \
#   -e VARIANT=routed|plain  freuid-mangojump
# -> writes /submissions/submission.csv (id,label)
#
# Build (network needed once; see REPRODUCE.md):
#   python docker/prepare_hf_cache.py        # downloads backbone checkpoints
#   docker build -f docker/Dockerfile -t freuid-mangojump .
#
# Weights provenance: DoRA adapters + heads + router artifacts are baked in
# from artifacts/system/ (frozen commit); base backbones come from the HF
# cache prepared at build time and are never fetched at run time.

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python

# torch 2.11 nightly (cu128) == the environment the system was frozen with;
# if this dated wheel becomes unavailable, any torch 2.11+cu128 build works
# within the reproduction tolerance documented in REPRODUCE.md.
RUN python -m pip install --no-cache-dir --pre \
        torch==2.11.0.dev20251223+cu128 torchvision==0.25.0.dev20251223+cu128 \
        --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python -m pip install --no-cache-dir \
        timm==1.0.22 pandas==2.2.2 numpy==2.2.6 scikit-learn==1.7.2 \
        pillow==11.2.1

WORKDIR /app
COPY src /app/src
COPY artifacts/system /app/artifacts/system
COPY docker/hf_cache /app/hf

ENV HF_HOME=/app/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VARIANT=routed \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.predict_docker"]
