"""Verify the HF weight upload: compare remote LFS sha256 against local files.

  python docker/verify_hf_upload.py <repo_id>          # e.g. lilwu/freuid-2026-lilwu

Exit 0 = every local file in artifacts/system exists remotely with matching
size/sha256. No downloads needed (uses repo metadata only).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi

SYS = Path(__file__).resolve().parents[1] / "artifacts" / "system"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    repo = sys.argv[1]
    info = HfApi().model_info(repo, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    print(f"repo {repo} @ revision {info.sha}")

    ok = True
    for p in sorted(SYS.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(SYS).as_posix()
        r = remote.get(rel)
        if r is None:
            print(f"  MISSING remote: {rel}"); ok = False; continue
        if r.lfs is not None:                      # big file: exact sha256
            local = sha256(p)
            match = local == r.lfs.sha256
            print(f"  {'OK ' if match else 'SHA MISMATCH'} {rel} "
                  f"({p.stat().st_size/1e6:.1f} MB)")
            ok &= match
        else:                                      # small file: size check
            match = r.size in (None, p.stat().st_size)
            print(f"  {'OK ' if match else 'SIZE MISMATCH'} {rel}")
            ok &= match
    print("\nVERIFIED — remote matches local bit-for-bit" if ok
          else "\nFAILED — fix the items above and re-upload")
    print(f"pin this revision in the report: {info.sha}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
