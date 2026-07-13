"""Model factory — timm backbone with a single-logit fraud head."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class FraudNet(nn.Module):
    """timm backbone -> global pool -> 1 logit (fraud).

    Uses timm's built-in classifier head (num_classes=1) so global pooling and
    head init are handled correctly across architectures. Accepts non-square
    inputs (e.g. 384x608) because the head pools globally.
    """

    def __init__(self, cfg):
        super().__init__()
        kwargs = dict(
            pretrained=cfg.pretrained,
            num_classes=1,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            in_chans=cfg.in_chans,
        )
        # ViT-family backbones have a fixed patch grid -> must be told the input
        # size (img_h/img_w must be multiples of the patch size). CNNs (ConvNeXt)
        # don't take img_size, so only pass it for transformer backbones.
        if any(k in cfg.backbone for k in
               ("vit", "eva", "dinov2", "deit", "beit", "swin", "maxvit")):
            kwargs["img_size"] = (cfg.img_h, cfg.img_w)
        self.backbone = timm.create_model(cfg.backbone, **kwargs)
        if getattr(cfg, "lora_r", 0) > 0:
            from .lora import inject_lora, mark_trainable
            n = inject_lora(self.backbone, cfg.lora_r, cfg.lora_alpha,
                            cfg.lora_dropout)
            mark_trainable(self.backbone, also_train=("head",))
            trainable = sum(p.numel() for p in self.backbone.parameters()
                            if p.requires_grad) / 1e6
            print(f"[LoRA] injected into {n} linears; trainable={trainable:.2f}M")
        if cfg.grad_checkpointing and hasattr(self.backbone,
                                              "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)  # (B,)


def param_groups(model: nn.Module, cfg):
    """Discriminative LR: backbone vs head, with no weight decay on norm/bias."""
    head_keys = ("head", "fc", "classifier")
    decay, no_decay, head_decay, head_no_decay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = any(k in n for k in head_keys)
        no_wd = p.ndim <= 1 or n.endswith(".bias")
        if is_head:
            (head_no_decay if no_wd else head_decay).append(p)
        else:
            (no_decay if no_wd else decay).append(p)
    bb = cfg.lr * cfg.backbone_lr_mult
    return [
        {"params": decay, "lr": bb, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": bb, "weight_decay": 0.0},
        {"params": head_decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": head_no_decay, "lr": cfg.lr, "weight_decay": 0.0},
    ]
