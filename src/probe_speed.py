"""Directly time fwd+bwd s/iter for a backbone+resolution, to size a run.

Builds the same DoRA model as train_siglip512, runs a few warmup + timed
training iters at a given batch/resolution, reports s/iter and peak VRAM.
"""
import argparse
import time

import torch
import torch.nn as nn

from .train_DINOV3L_512 import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--img", type=int, default=0, help="override input size (interp)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--no_ckpt", action="store_true", help="disable grad checkpointing")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    args = ap.parse_args()

    class A:
        backbone = args.backbone; rank = 16; alpha = 32; attn_only = False
        img_size = args.img            # 0 = native; else interpolate pos-embed
    model, n, native, mean, std = build_model(A)
    img = args.img or native
    if args.no_ckpt and hasattr(model.bb, "set_grad_checkpointing"):
        model.bb.set_grad_checkpointing(False)
    model = model.cuda().train()
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    crit = nn.BCEWithLogitsLoss()
    torch.backends.cudnn.benchmark = True

    def step():
        x = torch.randn(args.batch, 3, img, img, device="cuda")
        y = torch.randint(0, 2, (args.batch,), device="cuda").float()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        loss.backward(); opt.step()

    warmup = 8 if args.compile else 3   # torch.compile's first calls are slow
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters
    peak = torch.cuda.max_memory_allocated() / 1e9
    flags = ("ckpt" if not args.no_ckpt else "no-ckpt") + (" compile" if args.compile else "")
    print(f"{args.backbone} img={img} batch={args.batch} [{flags}] "
          f"tokens={(img // (14 if 'patch14' in args.backbone else 16)) ** 2}")
    iters_total = (68985 // args.batch) * 3
    print(f"  s/iter={dt:.2f}  s/img={dt/args.batch:.3f}  peak_VRAM={peak:.1f}GB"
          f"  -> 3-epoch ~= {iters_total * dt / 3600:.1f}h")


if __name__ == "__main__":
    main()
