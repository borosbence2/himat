"""CLI: fine-tune the DC-AE decoder on SVBRDF maps (Phase 1).

    python scripts/train_dc_ae.py [--max-steps N] [--batch-size B] [--lr LR]
                                  [--lambda-lpips W] [--finetune-encoder]

Run on the 4090 box after scripts/preprocess.py has populated the 1024² cache.
"""

from __future__ import annotations

import argparse
import dataclasses

from himat import config
from himat.train.vae_finetune import train_vae


def main() -> None:
    d = config.DEFAULT_VAE
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=d.max_steps)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--lambda-rec", type=float, default=d.lambda_rec)
    ap.add_argument("--lambda-lpips", type=float, default=d.lambda_lpips)
    ap.add_argument("--val-every", type=int, default=d.val_every)
    ap.add_argument("--finetune-encoder", action="store_true", default=d.finetune_encoder)
    ap.add_argument("--run-name", default="dc_ae")
    args = ap.parse_args()

    cfg = dataclasses.replace(
        d,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lambda_rec=args.lambda_rec,
        lambda_lpips=args.lambda_lpips,
        val_every=args.val_every,
        finetune_encoder=args.finetune_encoder,
    )
    train_vae(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
