"""MatSynthDataset: serves cached (M=3, C=3, 1024, 1024) packed tensors + prompts,
with normal-aware augmentation.

The subtle part is augmentation. The packed tensor's map 1 is a normal map: a
*vector field*, not a colour image. Under a spatial flip or rotation its in-plane
(Nx, Ny) components must transform too, or the normals stop matching the geometry.
Albedo, the packed roughness/metallic/height map, etc. are scalar-ish images and
only get the spatial transform.

  * horizontal flip (mirror left/right, flip W): Nx -> -Nx
  * vertical flip   (mirror top/bottom, flip H): Ny -> -Ny
  * 90 deg CCW rotation: (Nx, Ny) -> (-Ny, Nx), applied k times

Correctness of the rotation direction is pinned by tests/test_augment.py, which
builds a self-consistent height/normal pair and checks the relationship survives.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from himat import config

NORMAL_MAP_INDEX = 1  # map 1 in the stacked tensor is the normal


# --------------------------------------------------------------------------- #
# Augmentation primitives. All operate on a stacked (M, C, H, W) tensor.
# --------------------------------------------------------------------------- #
def flip_h_stacked(stacked: torch.Tensor) -> torch.Tensor:
    out = torch.flip(stacked, dims=[-1])
    out[NORMAL_MAP_INDEX, 0] = -out[NORMAL_MAP_INDEX, 0]  # Nx -> -Nx
    return out


def flip_v_stacked(stacked: torch.Tensor) -> torch.Tensor:
    out = torch.flip(stacked, dims=[-2])
    out[NORMAL_MAP_INDEX, 1] = -out[NORMAL_MAP_INDEX, 1]  # Ny -> -Ny
    return out


def rot90_stacked(stacked: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate the spatial plane by k*90 deg CCW and rotate the normal's in-plane
    vector to match."""
    k = k % 4
    if k == 0:
        return stacked.clone()
    out = torch.rot90(stacked, k, dims=[-2, -1])
    # Rotate the in-plane (Nx, Ny) to match torch.rot90's spatial direction.
    # The per-step map is (x, y) -> (y, -x): this is the handedness that keeps
    # the normal equal to -grad(height) after rotation (pinned by the covariance
    # test in tests/test_augment.py; the other handedness fails it).
    nx = out[NORMAL_MAP_INDEX, 0].clone()
    ny = out[NORMAL_MAP_INDEX, 1].clone()
    for _ in range(k):
        nx, ny = ny, -nx
    out[NORMAL_MAP_INDEX, 0] = nx
    out[NORMAL_MAP_INDEX, 1] = ny
    return out


def augment_stacked(
    stacked: torch.Tensor,
    *,
    flip: bool = True,
    rot90: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly compose flips + 90 deg rotations (the paper's section 5.1 aug set)."""
    def rand() -> float:
        return torch.rand((), generator=generator).item()

    out = stacked
    if flip and rand() < 0.5:
        out = flip_h_stacked(out)
    if flip and rand() < 0.5:
        out = flip_v_stacked(out)
    if rot90:
        out = rot90_stacked(out, int(rand() * 4))
    return out


# --------------------------------------------------------------------------- #
# Deterministic train/val split by material id (stable across machines/runs).
# --------------------------------------------------------------------------- #
def _hash_unit(material_id: str) -> float:
    h = hashlib.sha1(material_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def is_val(material_id: str, val_fraction: float) -> bool:
    return _hash_unit(material_id) < val_fraction


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MatSynthDataset(Dataset):
    """Serves (packed_maps, prompt) from the 1024² cache.

    Args:
        cache_dir: directory of <id>.safetensors files (default CACHE_1024_DIR).
        prompts_path: prompts.json mapping material id → prompt string.
        split: "train" | "val" | "all".
        augment: apply augmentation (forced off for val/all).
        val_fraction: holdout fraction for the deterministic split.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        prompts_path: Path | None = None,
        split: str = "train",
        augment: bool = True,
        val_fraction: float = config.DEFAULT_TRAIN.val_fraction,
        flip: bool = True,
        rot90: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir or config.CACHE_1024_DIR)
        self.split = split
        self.augment = augment and split == "train"
        self.flip = flip
        self.rot90 = rot90
        self.val_fraction = val_fraction

        all_files = sorted(self.cache_dir.glob("*.safetensors"))
        if not all_files:
            raise FileNotFoundError(
                f"no cached materials in {self.cache_dir} — run scripts/preprocess.py first"
            )

        def keep(p: Path) -> bool:
            mid = p.stem
            if split == "all":
                return True
            in_val = is_val(mid, val_fraction)
            return in_val if split == "val" else not in_val

        self.files = [p for p in all_files if keep(p)]
        if not self.files:
            raise ValueError(f"split '{split}' is empty (val_fraction={val_fraction})")

        self.prompts: dict[str, str] = {}
        pp = Path(prompts_path or config.PROMPTS_PATH)
        if pp.exists():
            self.prompts = json.loads(pp.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.files[idx]
        mid = path.stem
        maps = load_file(str(path))["maps"]  # (3, 3, 1024, 1024) float32 in [-1, 1]
        if self.augment:
            maps = augment_stacked(maps, flip=self.flip, rot90=self.rot90)
        prompt = self.prompts.get(mid, f"a {mid} material, tileable PBR surface")
        return maps, prompt
