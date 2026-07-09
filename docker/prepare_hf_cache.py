"""Pre-download backbone checkpoints into docker/hf_cache (build-time, online).

Reads the backbone names straight from the frozen adapters so the cache always
matches artifacts/system/. DINOv3 checkpoints are gated on Hugging Face —
export HF_TOKEN (or `huggingface-cli login`) before running.

  python docker/prepare_hf_cache.py
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO / "docker" / "hf_cache")

import torch  # noqa: E402
import timm   # noqa: E402


def main():
    seen = set()
    for f in sorted((REPO / "artifacts" / "system" / "adapters_slim").glob("*.pt")):
        args = torch.load(f, map_location="cpu", weights_only=False)["args"]
        name = args["backbone"]
        if name in seen:
            continue
        seen.add(name)
        print(f"fetching {name} ...", flush=True)
        m = timm.create_model(name, pretrained=True, num_classes=0)
        del m
    print(f"done -> {os.environ['HF_HOME']} ({len(seen)} backbones)")


if __name__ == "__main__":
    main()
