"""Convert raw MatSynth 4K maps → cached 1024² packed tensors.

Each material becomes one `<id>.safetensors` file under CACHE_1024_DIR holding a
single (M=3, C=3, 1024, 1024) float32 tensor in [-1, 1], plus the material id and
category as metadata. Caching the packed tensor avoids re-decoding 4K PNGs every
epoch — the dataset just memory-maps these.

Run on the box that has the dataset (the 4090 box); see scripts/preprocess.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from himat import config
from himat.data.matsynth import pack_maps

# Required maps; a material missing any of these is skipped.
REQUIRED = ("basecolor", "normal", "roughness", "metallic", "height")


def _to_float_chw(img: Image.Image, channels: int, size: int) -> torch.Tensor:
    """PIL image → (channels, size, size) float32 in [0, 1], resized.

    Height maps may be 16-bit ('I;16' / 'I'); handled by normalising by max range.
    """
    # Resize first (bicubic for colour, bilinear is fine; keep it simple/consistent).
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BICUBIC)

    mode = img.mode
    arr = np.asarray(img)

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype in (np.uint16, np.int32):
        # 16-bit height/displacement.
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    if arr.ndim == 2:
        arr = arr[..., None]  # (H, W, 1)
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)  # (C, H, W)

    # Force the channel count we want.
    if t.shape[0] < channels:
        t = t.repeat(channels, 1, 1)[:channels]
    elif t.shape[0] > channels:
        t = t[:channels]
    return t.contiguous()


def preprocess_material(maps: dict[str, Image.Image], size: int = config.RESOLUTION) -> torch.Tensor:
    """Take a dict of raw PIL maps → packed (3, 3, size, size) tensor in [-1, 1].

    `maps` keys use MatSynth field names: basecolor, normal, roughness,
    metallic, height. Raises KeyError if a required map is missing.
    """
    for k in REQUIRED:
        if k not in maps or maps[k] is None:
            raise KeyError(f"missing required map: {k}")

    albedo = _to_float_chw(maps["basecolor"], 3, size)        # [0,1]
    normal = _to_float_chw(maps["normal"], 3, size)           # [0,1] OpenGL encoded
    roughness = _to_float_chw(maps["roughness"], 1, size)     # [0,1]
    metallic = _to_float_chw(maps["metallic"], 1, size)       # [0,1]
    height = _to_float_chw(maps["height"], 1, size)           # [0,1]

    # To [-1, 1].
    to_signed = lambda t: t * 2.0 - 1.0  # noqa: E731
    return pack_maps(
        to_signed(albedo),
        to_signed(normal),
        to_signed(roughness),
        to_signed(metallic),
        to_signed(height),
    ).to(torch.float32)


def save_material(out_dir: Path, material_id: str, packed: torch.Tensor, category: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{material_id}.safetensors"
    save_file(
        {"maps": packed.contiguous()},
        str(path),
        metadata={"id": material_id, "category": category, "resolution": str(packed.shape[-1])},
    )
    return path


def has_required(record: dict) -> bool:
    """Whether a MatSynth record (HF dataset row) has all required maps."""
    s = config.MATSYNTH
    field_for = {
        "basecolor": s.basecolor,
        "normal": s.normal,
        "roughness": s.roughness,
        "metallic": s.metallic,
        "height": s.height,
    }
    return all(record.get(field_for[k]) is not None for k in REQUIRED)
