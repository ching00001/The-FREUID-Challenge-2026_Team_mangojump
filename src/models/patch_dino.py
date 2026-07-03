"""Patch-DINOv2 fraud detector with attention-MIL pooling.

Rationale (from Phase-2 negative result): a trainable CNN relearns the fragile
synthetic source-artifact that does not survive real capture. A FROZEN DINOv2
backbone exposes only generic, semantic features, so the small trainable head
must rely on cues that transfer (texture/structure of pasted regions, etc.).
Operating on a grid of patches localises composite/pasted-photo manipulations
and decouples document-template identity (helps unseen types).

Pipeline:
  image -> grid of K patches (each resized to patch_px) -> frozen DINOv2 ->
  per-patch features (B,K,d) -> gated attention-MIL pooling -> doc feature ->
  linear head -> doc logit. Attention weights say which patch looks fraudulent.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class AttnMILPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018): weights instances (patches)
    then sums. Lets the model focus on the few manipulated patches."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, f: torch.Tensor):           # f: (B, K, d)
        a = self.w(torch.tanh(self.V(f)) * torch.sigmoid(self.U(f)))  # (B,K,1)
        a = a.softmax(dim=1)
        pooled = (a * f).sum(dim=1)               # (B, d)
        return pooled, a.squeeze(-1)              # (B,d), (B,K)


class PatchDINO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.patch_backbone, pretrained=cfg.pretrained, num_classes=0,
            img_size=cfg.patch_px)
        self.frozen = cfg.patch_freeze
        if self.frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        d = self.backbone.num_features
        self.pool = AttnMILPool(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(cfg.drop_rate),
                                  nn.Linear(d, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:                            # keep frozen backbone in eval
            self.backbone.eval()
        return self

    def _features(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (N, 3, P, P)
        if self.frozen:
            with torch.no_grad():
                return self.backbone(patches)
        return self.backbone(patches)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: (B, K, 3, P, P)
        b, k = x.shape[:2]
        f = self._features(x.flatten(0, 1)).view(b, k, -1)   # (B,K,d)
        pooled, attn = self.pool(f)
        logit = self.head(pooled).squeeze(-1)                # (B,)
        return (logit, attn) if return_attn else logit


def patch_param_groups(model: PatchDINO, cfg):
    """Only the pooling + head are trainable when the backbone is frozen."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0},
    ]
