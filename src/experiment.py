"""Experiment tracking — records EVERY run with its full hyperparameters.

Design goals (per user request: log all experiments + all hyperparameters):
  * One dataclass `ExperimentConfig` holds *every* knob. It is serialised
    verbatim to `experiments/<run_id>/config.json` so a run is fully
    reproducible from that single file.
  * `ExperimentLogger` writes per-epoch metrics to `metrics.jsonl`, free-form
    text to `log.txt`, and (at the end) a one-row summary into the global
    `experiments/registry.csv` so all runs are comparable at a glance.
  * Also captures the environment fingerprint (git-less): python/torch/timm
    versions, GPU name, and a hash of the resolved config.

No training logic here — pure bookkeeping.
"""
from __future__ import annotations

import json
import hashlib
import platform
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .data.paths import REPO_ROOT

EXP_ROOT = REPO_ROOT / "experiments"
REGISTRY_CSV = EXP_ROOT / "registry.csv"


@dataclass
class ExperimentConfig:
    # --- identity -----------------------------------------------------------
    name: str = "baseline"
    seed: int = 42
    notes: str = ""

    # --- data ---------------------------------------------------------------
    folds_csv: str = "artifacts/folds.csv"
    img_h: int = 384
    img_w: int = 608                  # ~1.585 aspect, matches the ID-1 crop
    cv_scheme: str = "skf"            # "skf" (stratified) or "loto" (type-LOO)
    val_fold: int = 0                 # which skf_fold is validation
    loto_type: str = ""               # held-out type when cv_scheme == "loto"
    subset: int = 0                   # >0: cap train rows (smoke tests)

    # --- model --------------------------------------------------------------
    arch: str = "single"             # "single" = full-image CNN; "patch" = patch-DINOv2
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    in_chans: int = 3

    # --- LoRA (for large backbones e.g. CLIP ViT-L/14) ----------------------
    lora_r: int = 0                  # 0 = no LoRA (normal/full training)
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # --- patch-DINOv2 (arch="patch") ----------------------------------------
    # Frozen DINOv2 over a grid of patches + attention-MIL pooling. Frozen so
    # the head uses DINOv2's generic features and cannot relearn the fragile
    # source-artifact; patches localise composite/pasted-photo frauds.
    patch_backbone: str = "vit_small_patch14_reg4_dinov2"
    patch_px: int = 224
    patch_grid_cols: int = 4
    patch_grid_rows: int = 3
    patch_freeze: bool = True
    patch_jitter: float = 0.15        # random per-cell crop jitter (train aug)

    # --- optimisation -------------------------------------------------------
    epochs: int = 6
    batch_size: int = 16
    grad_accum: int = 2               # effective batch = batch_size * grad_accum
    optimizer: str = "adamw"
    lr: float = 2e-4
    backbone_lr_mult: float = 1.0     # discriminative LR for backbone vs head
    weight_decay: float = 0.05
    scheduler: str = "cosine"
    warmup_epochs: float = 0.5
    min_lr_ratio: float = 0.02
    amp_dtype: str = "bf16"           # bf16 | fp16 | fp32
    grad_checkpointing: bool = True
    max_grad_norm: float = 1.0
    ema: bool = True
    ema_decay: float = 0.9995

    # --- loss ---------------------------------------------------------------
    loss: str = "bce"                 # bce | focal
    focal_gamma: float = 2.0
    pos_weight: float = 1.0
    label_smoothing: float = 0.0

    # --- augmentation (Phase 1 = light; Phase 2 adds recapture sim) ---------
    aug_level: str = "light"
    hflip_p: float = 0.0              # ID layout is not L/R symmetric -> off
    brightness: float = 0.1
    contrast: float = 0.1
    rotate_deg: float = 3.0
    scale_jitter: float = 0.05
    jpeg_p: float = 0.2               # mild re-encode even in Phase 1
    jpeg_quality_min: int = 50

    # --- Phase 2: realistic recapture (print-and-capture) augmentation ------
    # Train: apply src/aug/recapture at strength ~U[min,max] with prob p, to
    # BOTH classes equally, to destroy the fragile digital source-artifact and
    # force the model onto cues that survive capture.
    recapture_p: float = 0.0          # 0 = off (Phase-1 behaviour)
    recapture_strength_min: float = 0.3
    recapture_strength_max: float = 1.0
    # Val: evaluate on recapture-degraded val at this fixed strength (0 = clean).
    # rc0.9 ~= the public LB difficulty (calibrated), so it is our LB proxy.
    val_recapture: float = 0.0

    # --- inference ----------------------------------------------------------
    tta: str = "none"                 # none | hflip | multiscale
    use_ema_for_infer: bool = True

    # --- runtime ------------------------------------------------------------
    num_workers: int = 8
    eval_every: int = 1
    save_top_k: int = 1

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def _env_info() -> dict[str, Any]:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:
        pass
    return info


class ExperimentLogger:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{cfg.name}_{cfg.fingerprint()}"
        self.dir = EXP_ROOT / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.env = _env_info()
        # dump full config + environment immediately
        (self.dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        (self.dir / "env.json").write_text(
            json.dumps(self.env, indent=2), encoding="utf-8")
        self._metrics_fp = (self.dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.log(f"=== run {self.run_id} ===")
        self.log("config: " + json.dumps(asdict(cfg)))
        self.log("env: " + json.dumps(self.env))

    def log(self, msg: str) -> None:
        line = f"[{time.time()-self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with (self.dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        rec = {"step": step, "elapsed_s": round(time.time() - self.t0, 1),
               **metrics}
        self._metrics_fp.write(json.dumps(rec) + "\n")
        self._metrics_fp.flush()
        self.log("metrics " + json.dumps(rec))

    def save_path(self, fname: str) -> Path:
        return self.dir / fname

    def finalize(self, summary: dict[str, Any]) -> None:
        """Write final summary + append one comparable row to registry.csv."""
        summary = {"run_id": self.run_id, **summary,
                   "elapsed_min": round((time.time() - self.t0) / 60, 1)}
        (self.dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        self._append_registry(summary)
        self.log("FINAL " + json.dumps(summary))
        self._metrics_fp.close()

    def _append_registry(self, summary: dict[str, Any]) -> None:
        import csv
        c = asdict(self.cfg)
        row = {
            "run_id": self.run_id,
            "name": c["name"],
            "backbone": c["backbone"],
            "img": f"{c['img_h']}x{c['img_w']}",
            "cv": c["cv_scheme"],
            "val_fold": c["val_fold"],
            "loto_type": c["loto_type"],
            "epochs": c["epochs"],
            "eff_bs": c["batch_size"] * c["grad_accum"],
            "lr": c["lr"],
            "loss": c["loss"],
            "aug": c["aug_level"],
            "ema": c["ema"],
            "tta": c["tta"],
            **{k: summary.get(k) for k in
               ("freuid", "audet", "apcer_at_1pct_bpcer", "roc_auc",
                "best_epoch", "elapsed_min")},
            "notes": c["notes"],
        }
        EXP_ROOT.mkdir(parents=True, exist_ok=True)
        exists = REGISTRY_CSV.exists()
        with REGISTRY_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
