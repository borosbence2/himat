"""CLI: HiMat fine-tune (Phase 3). Runs on the 4090 box.

Staged per MILESTONES P3:
  Stage A (smoke): tiny step count to confirm the loop runs end-to-end.
  Stage B (main):  ~50k steps, the real run.

    python scripts/train_himat.py --max-steps 1000   --run-name stageA
    python scripts/train_himat.py --max-steps 50000  --run-name stageB
"""

from __future__ import annotations

import argparse
import dataclasses

from himat import config
from himat.train.himat_finetune import train_himat


def main() -> None:
    d = config.DEFAULT_TRAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=d.max_steps)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--lora-rank", type=int, default=d.lora_rank)
    ap.add_argument("--lr-lora", type=float, default=d.lr_lora)
    ap.add_argument("--lr-decoder", type=float, default=d.lr_decoder)
    ap.add_argument("--val-every", type=int, default=d.val_every)
    ap.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    ap.add_argument("--dc-ae-ckpt", default=None, help="fine-tuned DC-AE (default: auto-detect)")
    ap.add_argument("--run-name", default="himat")
    args = ap.parse_args()

    cfg = dataclasses.replace(
        d,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        lr_lora=args.lr_lora,
        lr_decoder=args.lr_decoder,
        val_every=args.val_every,
        ckpt_every=args.ckpt_every,
    )
    train_himat(cfg, run_name=args.run_name, dc_ae_ckpt=args.dc_ae_ckpt)


if __name__ == "__main__":
    main()
