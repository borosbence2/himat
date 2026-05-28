"""CLI: generate a gallery from a prompt list + report mean CLIPScore (Phase 4).

    python scripts/gallery.py --prompts prompts.txt --steps 20

Saves each material's 5 maps and an albedo contact sheet. Proper eval renders
under lighting (forfun-graphics) — albedo CLIPScore here is a tracking proxy.
Reuses scripts/sample.py's checkpoint loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from himat import config
from himat.inference.pipeline import HiMatPipeline, save_svbrdf
from himat.models.himat import build_himat

DEFAULT_PROMPTS = [
    "weathered red brick wall",
    "polished white marble",
    "rusty corrugated metal",
    "rough oak wood planks",
    "green moss on stone",
    "cracked dry mud",
    "woven blue fabric",
    "hammered copper",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="", help="text file, one prompt per line (default: built-in set)")
    ap.add_argument("--steps", type=int, default=config.DEFAULT_TRAIN.inference_steps)
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_DIR / "himat_final.safetensors"))
    ap.add_argument("--clipscore", action="store_true", help="also compute albedo CLIPScore")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [ln.strip() for ln in Path(args.prompts).read_text(encoding="utf-8").splitlines() if ln.strip()]

    from scripts.sample import _load_trainable, _slug  # reuse loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_himat()
    _load_trainable(bundle, args.ckpt)
    for m in (bundle.himat, bundle.transformer, bundle.vae, bundle.text_encoder):
        m.to(device)

    pipe = HiMatPipeline(bundle, device)
    scorer = None
    if args.clipscore:
        from himat.eval.clipscore import CLIPScorer

        scorer = CLIPScorer(device=device)

    from torchvision.utils import save_image

    previews, scores = [], []
    gallery_dir = config.OUTPUTS_DIR / "gallery"
    for p in prompts:
        maps = pipe.generate(p, steps=args.steps, seed=0)
        save_svbrdf(maps, gallery_dir / _slug(p))
        albedo = ((maps["albedo"] + 1) * 0.5).clamp(0, 1)
        previews.append(albedo.cpu())
        if scorer:
            scores.append(scorer.score(albedo, p))

    save_image(torch.stack(previews), str(gallery_dir / "contact_sheet.png"), nrow=4)
    print(f"gallery -> {gallery_dir}")
    if scores:
        print(f"albedo CLIPScore: mean={sum(scores)/len(scores):.2f}  n={len(scores)}")


if __name__ == "__main__":
    main()
