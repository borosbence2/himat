"""SVBRDF map specs + the pack/unpack between the 5 caller-facing maps and the
M=3 RGB-shaped tensor the model operates on.

The single source of truth for *how* roughness/metallic/height get folded into
one image and split back out. Both the preprocessing step and the inference
pipeline import these so they can never disagree.

Conventions (see config.py):
  * Everything is float32 in [-1, 1].
  * normal is OpenGL (Y+ up), as MatSynth ships it.
  * map tensors are (C, H, W); the stacked model tensor is (M, C, H, W).
"""

from __future__ import annotations

import torch

from himat import config

# Order of the three genuinely-3-channel and packed maps along the M axis.
# map 0 = albedo, map 1 = normal, map 2 = packed (roughness, metallic, height).
MAP_ORDER = ("albedo", "normal", "packed_rmh")


def pack_maps(
    albedo: torch.Tensor,
    normal: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    height: torch.Tensor,
) -> torch.Tensor:
    """Fold the 5 SVBRDF maps into the model's (M=3, C=3, H, W) tensor.

    albedo, normal: (3, H, W). roughness/metallic/height: (1, H, W) each.
    All expected in [-1, 1]. Returns (3, 3, H, W).
    """
    _check_chw(albedo, 3, "albedo")
    _check_chw(normal, 3, "normal")
    for t, name in ((roughness, "roughness"), (metallic, "metallic"), (height, "height")):
        _check_chw(t, 1, name)

    packed = torch.cat([roughness, metallic, height], dim=0)  # (3, H, W)
    stacked = torch.stack([albedo, normal, packed], dim=0)  # (3, 3, H, W)
    return stacked


def unpack_maps(stacked: torch.Tensor) -> dict[str, torch.Tensor]:
    """Inverse of pack_maps. Takes (M=3, C=3, H, W) → dict of the 5 maps.

    Returns float tensors in [-1, 1]: albedo/normal as (3, H, W);
    roughness/metallic/height as (1, H, W).
    """
    if stacked.shape[0] != config.NUM_MAPS or stacked.shape[1] != config.MAP_CHANNELS:
        raise ValueError(
            f"expected ({config.NUM_MAPS}, {config.MAP_CHANNELS}, H, W), got {tuple(stacked.shape)}"
        )
    albedo = stacked[0]
    normal = stacked[1]
    packed = stacked[2]
    return {
        "albedo": albedo,
        "normal": normal,
        "roughness": packed[0:1],
        "metallic": packed[1:2],
        "height": packed[2:3],
    }


def to_unit_range(stacked: torch.Tensor) -> dict[str, torch.Tensor]:
    """Like unpack_maps but maps each channel back to its [0, 1] storage range
    for saving as PNGs. Normal stays in [0, 1] OpenGL encoding (0.5 = 0)."""
    maps = unpack_maps(stacked)
    return {k: (v.clamp(-1, 1) + 1.0) * 0.5 for k, v in maps.items()}


def _check_chw(t: torch.Tensor, channels: int, name: str) -> None:
    if t.dim() != 3 or t.shape[0] != channels:
        raise ValueError(f"{name}: expected ({channels}, H, W), got {tuple(t.shape)}")
