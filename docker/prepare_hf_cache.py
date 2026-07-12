"""Pre-download backbone checkpoints into docker/hf_cache (build-time, online).

Reads the backbone names straight from the frozen adapters so the cache always
matches weights/. All four backbones are ungated timm mirrors — no Hugging
Face account or token is required (verified 2026-07-11).

  python docker/prepare_hf_cache.py
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO / "docker" / "hf_cache")

import torch  # noqa: E402
import timm   # noqa: E402


def local_hub_copy(name) -> bool:
    """Hydrate from the developer's existing HF cache when possible (offline,
    no gated-repo token needed). timm names map to hub repos 'timm/<name>'."""
    import shutil
    src_home = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".cache" / "huggingface"
    repo_dir = f"models--timm--{name}"
    src = src_home / "hub" / repo_dir
    if not src.is_dir():
        return False
    dst = Path(os.environ["HF_HOME"]) / "hub" / repo_dir
    if not dst.exists():
        shutil.copytree(src, dst)
    print(f"  copied from local cache: {repo_dir}")
    return True


ADAPTER_MEMBERS = {"dino", "dino_hplus", "siglip512", "dfn5b",
                   "dino_hplus_dlc", "dino_hplus_ds"}


def main():
    seen = set()
    for f in sorted((REPO / "weights").glob("*.pt")):
        if f.stem not in ADAPTER_MEMBERS:      # skip heads.pt (not an adapter)
            continue
        args = torch.load(f, map_location="cpu", weights_only=False)["args"]
        name = args["backbone"]
        if name in seen:
            continue
        seen.add(name)
        print(f"preparing {name} ...", flush=True)
        if not local_hub_copy(name):
            m = timm.create_model(name, pretrained=True, num_classes=0)
            del m
    print(f"done -> {os.environ['HF_HOME']} ({len(seen)} backbones)")


if __name__ == "__main__":
    main()
