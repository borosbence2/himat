"""CLI: generate one SVBRDF from a prompt (Phase 4). Runs on the 4090 box.

    python scripts/sample.py --prompt "weathered red brick wall" --steps 20 --seed 0
"""

from __future__ import annotations

import argparse
import re

import torch

from himat import config
from himat.inference.pipeline import HiMatPipeline, save_svbrdf
from himat.models.himat import build_himat


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "material"


def _load_trainable(bundle, ckpt: str) -> None:
    from safetensors.torch import load_file

    state = load_file(ckpt)
    tr = {k[len("transformer."):]: v for k, v in state.items() if k.startswith("transformer.")}
    vae = {k[len("vae."):]: v for k, v in state.items() if k.startswith("vae.")}
    bundle.transformer.load_state_dict(tr, strict=False)
    bundle.vae.ae.load_state_dict(vae, strict=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", type=int, default=config.DEFAULT_TRAIN.inference_steps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_DIR / "himat_final.safetensors"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_himat()
    _load_trainable(bundle, args.ckpt)
    bundle.himat.to(device)
    bundle.transformer.to(device)
    bundle.vae.to(device)
    bundle.text_encoder.to(device)

    pipe = HiMatPipeline(bundle, device)
    maps = pipe.generate(args.prompt, steps=args.steps, seed=args.seed, cfg_scale=args.cfg_scale)
    out = save_svbrdf(maps, config.OUTPUTS_DIR / _slug(args.prompt))
    print(f"saved 5 maps -> {out}")


if __name__ == "__main__":
    main()
