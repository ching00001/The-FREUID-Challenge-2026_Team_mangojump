"""Minimal LoRA (low-rank adaptation) for fine-tuning a frozen backbone.

Wraps target nn.Linear layers with a low-rank update: y = W0 x + (B A) x * (a/r),
W0 frozen, only A,B trained. Regularises fine-tuning of a large pretrained model
(e.g. CLIP ViT-L/14) on limited data -> better OOD generalisation than full
fine-tune. No external dependency (avoids peft).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays 0 -> no-op at init
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.drop(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + lora * self.scaling


def inject_lora(model: nn.Module, r: int, alpha: int, dropout: float,
                targets=("qkv", "proj", "fc1", "fc2")) -> int:
    """Replace matching nn.Linear modules with LoRALinear. Returns count."""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name,
                        LoRALinear(child, r, alpha, dropout))
                n += 1
    return n


def mark_trainable(model: nn.Module, also_train=("head",)) -> None:
    """Freeze everything except LoRA params and the named head(s)."""
    for nme, p in model.named_parameters():
        is_lora = "lora_A" in nme or "lora_B" in nme
        is_head = any(k in nme for k in also_train)
        p.requires_grad_(is_lora or is_head)
