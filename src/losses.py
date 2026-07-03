"""Losses for binary fraud detection (single logit)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_loss(cfg):
    pos_weight = (torch.tensor([cfg.pos_weight], dtype=torch.float32)
                  if cfg.pos_weight and cfg.pos_weight != 1.0 else None)
    ls = float(cfg.label_smoothing)

    def _smooth(y: torch.Tensor) -> torch.Tensor:
        return y * (1 - ls) + 0.5 * ls if ls > 0 else y

    if cfg.loss == "bce":
        def loss_fn(logits, y):
            pw = pos_weight.to(logits.device) if pos_weight is not None else None
            return F.binary_cross_entropy_with_logits(
                logits, _smooth(y), pos_weight=pw)
        return loss_fn

    if cfg.loss == "focal":
        gamma = float(cfg.focal_gamma)

        def loss_fn(logits, y):
            ys = _smooth(y)
            p = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, ys, reduction="none")
            p_t = p * y + (1 - p) * (1 - y)
            mod = (1 - p_t).clamp(min=1e-6) ** gamma
            return (mod * ce).mean()
        return loss_fn

    raise ValueError(f"unknown loss {cfg.loss}")
