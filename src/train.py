"""Train one fold of a FREUID fraud detector, logging every hyperparameter.

Usage (any ExperimentConfig field is a CLI flag, auto-generated):
    python -m src.train --name baseline_cnv2t --val_fold 0 --epochs 6
    python -m src.train --cv_scheme loto --loto_type EGYPT/DL --name loto_egypt
    python -m src.train --subset 800 --epochs 1 --name smoke   # smoke test

Outputs go to experiments/<run_id>/ : config.json, metrics.jsonl, log.txt,
oof.csv, best.pt, summary.json  (+ a row in experiments/registry.csv).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.paths import REPO_ROOT, load_train
from .data.dataset import FreuidDataset
from .data.patch_dataset import PatchDataset
from .experiment import ExperimentConfig, ExperimentLogger
from .losses import build_loss
from .metric import freuid_score
from .models.factory import FraudNet, param_groups
from .models.patch_dino import PatchDINO, patch_param_groups


def _to_dev(x, device):
    """Move to device; channels_last only for 4D single-image tensors (patch
    tensors are 5D and ViT gains nothing from channels_last)."""
    x = x.to(device, non_blocking=True)
    return x.to(memory_format=torch.channels_last) if x.ndim == 4 else x


# --- CLI auto-generated from the dataclass so nothing is un-logged ----------
def _str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "t")


def parse_cfg() -> ExperimentConfig:
    # NOTE: `from __future__ import annotations` makes f.type a *string*, so we
    # infer the parser from the default VALUE's type. bool must be checked
    # before int (bool is an int subclass) and parsed explicitly, otherwise
    # argparse's type=bool turns "false" into True.
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(ExperimentConfig):
        d = f.default
        if isinstance(d, bool):
            ap.add_argument(f"--{f.name}", type=_str2bool, default=None)
        elif isinstance(d, int):
            ap.add_argument(f"--{f.name}", type=int, default=None)
        elif isinstance(d, float):
            ap.add_argument(f"--{f.name}", type=float, default=None)
        else:
            ap.add_argument(f"--{f.name}", type=str, default=None)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return ExperimentConfig(**overrides)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_fast_backends() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def make_splits(cfg: ExperimentConfig):
    # fold assignments only (avoid column collisions: type/label/is_digital come
    # from load_train as proper dtypes; folds.csv stores is_digital as a string).
    folds = pd.read_csv(REPO_ROOT / cfg.folds_csv,
                        dtype={"id": str})[["id", "skf_fold", "type_loo"]]
    df = load_train().df[["id", "abspath", "is_digital", "label", "type"]].copy()
    df = df.merge(folds, on="id", how="inner")
    # Always pull the ~20 real recaptured (is_digital=False) rows into a fixed
    # holdout: our only real-capture signal. Excluded from train AND val so it
    # is a clean (if tiny) recapture/private-test proxy reported every run.
    holdout = df[~df["is_digital"]].copy()
    df = df[df["is_digital"]].copy()
    if cfg.cv_scheme == "loto":
        assert cfg.loto_type, "loto requires --loto_type"
        tr = df[df["type"] != cfg.loto_type].copy()
        va = df[df["type"] == cfg.loto_type].copy()
    else:
        tr = df[df["skf_fold"] != cfg.val_fold].copy()
        va = df[df["skf_fold"] == cfg.val_fold].copy()
    if cfg.subset and cfg.subset > 0:
        tr = tr.sample(min(cfg.subset, len(tr)), random_state=cfg.seed)
        va = va.sample(min(max(cfg.subset // 4, 50), len(va)),
                       random_state=cfg.seed)
    return (tr.reset_index(drop=True), va.reset_index(drop=True),
            holdout.reset_index(drop=True))


def cosine_lr(step, total, warmup, base, min_ratio):
    if step < warmup:
        return base * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    ids, ys, ps = [], [], []
    for x, y, idb in loader:
        x = _to_dev(x, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logit = model(x)
        ps.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy()); ids.extend(idb)
    return np.array(ids), np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    cfg = parse_cfg()
    set_seed(cfg.seed)
    enable_fast_backends()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[cfg.amp_dtype]
    logger = ExperimentLogger(cfg)

    tr_df, va_df, ho_df = make_splits(cfg)
    logger.log(f"train={len(tr_df)}  val={len(va_df)}  "
               f"val_fraud_rate={va_df['label'].mean():.4f}  "
               f"real_holdout={len(ho_df)} (fraud {int(ho_df['label'].sum())}/"
               f"{len(ho_df)})")

    DS = PatchDataset if cfg.arch == "patch" else FreuidDataset
    tr_ds = DS(tr_df, cfg, train=True)
    va_ds = DS(va_df, cfg, train=False)
    ho_ds = DS(ho_df, cfg, train=False)
    # Windows + cu128 torch: each spawned worker re-imports the (large) torch
    # DLLs and commits memory. persistent_workers=True on BOTH loaders keeps
    # train+val workers alive together -> exceeds the commit/paging limit
    # (WinError 1455). So: NO persistent workers (train workers die before val
    # spawns), and keep the val loader's worker count small.
    va_workers = min(2, cfg.num_workers)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True,
                       drop_last=True, persistent_workers=False)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                       num_workers=va_workers, pin_memory=True,
                       persistent_workers=False)
    ho_ld = DataLoader(ho_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                       num_workers=min(2, cfg.num_workers), pin_memory=True)

    if cfg.arch == "patch":
        model = PatchDINO(cfg).to(device)
        opt = torch.optim.AdamW(patch_param_groups(model, cfg), lr=cfg.lr,
                                betas=(0.9, 0.999))
    else:
        model = FraudNet(cfg).to(device).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.lr,
                                betas=(0.9, 0.999))
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    ema = None
    if cfg.ema:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=cfg.ema_decay)

    loss_fn = build_loss(cfg)
    steps_per_epoch = max(1, len(tr_ld) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    base_lrs = [g["lr"] for g in opt.param_groups]

    best = {"freuid": 1e9, "epoch": -1}
    opt_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for it, (x, y, _) in enumerate(tr_ld):
            x = _to_dev(x, device)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype != torch.float32):
                logit = model(x)
                loss = loss_fn(logit, y) / cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum

            if (it + 1) % cfg.grad_accum == 0:
                lr = cosine_lr(opt_step, total_steps, warmup_steps, 1.0,
                               cfg.min_lr_ratio)
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * lr
                if cfg.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.max_grad_norm)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                opt_step += 1
                if opt_step % 50 == 0:
                    logger.log(f"  e{epoch} step {opt_step}/{total_steps} "
                               f"loss={running/(it+1):.4f} lr={base_lrs[0]*lr:.2e}")

        # --- validation (use EMA weights for eval if configured) ----------
        eval_model = ema.module if (ema is not None and cfg.use_ema_for_infer) else model
        ids, ys, ps = evaluate(eval_model, va_ld, device, amp_dtype)
        r = freuid_score(ys, ps)
        metrics = {
            "epoch": epoch, "train_loss": round(running / len(tr_ld), 5),
            "freuid": round(r.freuid, 5), "audet": round(r.audet, 5),
            "apcer@1bpcer": round(r.apcer_at_1pct_bpcer, 5),
            "roc_auc": round(r.roc_auc, 5)}
        # real recaptured holdout (tiny, noisy): AUC + score separation. This is
        # our only real-capture signal -> a rough recapture/private-test proxy.
        if len(ho_df) >= 4:
            _, hy, hp = evaluate(eval_model, ho_ld, device, amp_dtype)
            from sklearn.metrics import roc_auc_score
            try:
                metrics["ho_auc"] = round(float(roc_auc_score(hy, hp)), 4)
            except ValueError:
                metrics["ho_auc"] = None
            metrics["ho_gap"] = round(float(hp[hy == 1].mean() - hp[hy == 0].mean()), 4)
        logger.log_metrics(epoch, metrics)

        # Selection score: clean FREUID saturates (~0) for every model, so the
        # first epoch to hit ~0 would win arbitrarily. Prefer the LATEST (most
        # converged) saturated epoch via a tiny epoch bonus. ho_auc (n=20) is
        # too noisy to drive selection (it once picked an under-converged epoch
        # 0) -> keep it for monitoring only.
        sel = r.freuid - epoch * 1e-5
        if sel < best.get("sel", 1e9):
            best = {"sel": sel, "freuid": r.freuid, "audet": r.audet,
                    "apcer_at_1pct_bpcer": r.apcer_at_1pct_bpcer,
                    "roc_auc": r.roc_auc, "epoch": epoch}
            torch.save({"model": eval_model.state_dict(),
                        "cfg": dataclasses.asdict(cfg), "epoch": epoch},
                       logger.save_path("best.pt"))
            pd.DataFrame({"id": ids, "label": ys, "score": ps}).to_csv(
                logger.save_path("oof.csv"), index=False)
            logger.log(f"  * new best FREUID={r.freuid:.5f} @epoch {epoch}")

    logger.finalize({"best_epoch": best["epoch"], **{k: round(v, 5) for k, v in
                     best.items() if k != "epoch"}})


if __name__ == "__main__":
    main()
