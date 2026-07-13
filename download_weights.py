from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_REPO = "ching0206/freuid-2026-mangojump"
DEFAULT_REVISION = "8c145f9e0da49db26007f174d587d7d06b0d3d90"
SKIP = {".gitattributes", "README.md"}


def request(url: str, token: str = "") -> Request:
    headers = {"User-Agent": "freuid-docker-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return Request(url, headers=headers)


def repo_files(repo: str, revision: str, token: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{quote(repo, safe='/')}/revision/{quote(revision, safe='')}"
    with urlopen(request(url, token), timeout=60) as r:
        data = json.load(r)
    return [s["rfilename"] for s in data["siblings"] if s["rfilename"] not in SKIP]


def download_file(repo: str, revision: str, filename: str, out: Path, token: str) -> None:
    url = (
        f"https://huggingface.co/{quote(repo, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
    )
    dst = out / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urlopen(request(url, token), timeout=120) as r:
            shutil.copyfileobj(r, tmp)
    tmp_path.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    files = args.include or repo_files(args.repo, args.revision, token)
    out = Path(args.out)
    for filename in files:
        print(filename, flush=True)
        if not args.dry_run:
            download_file(args.repo, args.revision, filename, out, token)


if __name__ == "__main__":
    main()
