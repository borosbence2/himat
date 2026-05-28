"""Preprocess MatSynth → cached 1024² packed tensors.

    python scripts/preprocess.py [--limit N] [--split train]

Streams the HF dataset, resizes each material's maps to 1024², packs them into a
(3, 3, 1024, 1024) float32 tensor in [-1, 1], and writes one .safetensors per
material under config.CACHE_1024_DIR. Materials missing a required map are
skipped (logged).

Run on the 4090 box. Re-running skips already-cached materials.
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from himat import config
from himat.data import preproc


def _material_id(rec: dict, fallback: int) -> str:
    s = config.MATSYNTH
    name = rec.get(s.name)
    if isinstance(name, str) and name.strip():
        return name.strip().replace("/", "_").replace(" ", "_")
    return f"mat_{fallback:06d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N materials (0 = all)")
    ap.add_argument("--split", default="train", help="HF dataset split to read")
    ap.add_argument("--resolution", type=int, default=config.RESOLUTION)
    args = ap.parse_args()

    from datasets import load_dataset

    config.ensure_dirs()
    s = config.MATSYNTH
    out_dir = config.CACHE_1024_DIR

    ds = load_dataset(config.MATSYNTH_REPO_ID, split=args.split, streaming=True)

    n_done = n_skip = 0
    pbar = tqdm(ds, desc="preprocess")
    for i, rec in enumerate(pbar):
        if args.limit and n_done >= args.limit:
            break
        mid = _material_id(rec, i)
        out_path = out_dir / f"{mid}.safetensors"
        if out_path.exists():
            n_done += 1
            continue
        if not preproc.has_required(rec):
            n_skip += 1
            continue
        try:
            maps = {
                "basecolor": rec[s.basecolor],
                "normal": rec[s.normal],
                "roughness": rec[s.roughness],
                "metallic": rec[s.metallic],
                "height": rec[s.height],
            }
            packed = preproc.preprocess_material(maps, size=args.resolution)
            category = str(rec.get(s.category) or "")
            preproc.save_material(out_dir, mid, packed, category=category)
            n_done += 1
        except (KeyError, ValueError, OSError) as e:
            n_skip += 1
            pbar.write(f"skip {mid}: {e}")
        pbar.set_postfix(done=n_done, skip=n_skip)

    print(f"\ncached {n_done} materials, skipped {n_skip} → {out_dir}")
    print("next: python scripts/enrich_prompts.py")


if __name__ == "__main__":
    main()
