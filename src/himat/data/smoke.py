"""Smoke test for the data pipeline: instantiate the dataset, pull one batch,
print shapes + a prompt, and assert no NaNs.

    python -m himat.data.smoke

Runs on the box that has the preprocessed cache (the 4090 box).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from himat.data.dataset import MatSynthDataset


def main() -> None:
    ds = MatSynthDataset(split="train")
    print(f"train materials: {len(ds)}")
    val = MatSynthDataset(split="val", augment=False)
    print(f"val materials:   {len(val)}")

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    maps, prompts = next(iter(loader))

    print(f"batch maps shape: {tuple(maps.shape)}  dtype={maps.dtype}")
    print(f"value range:      [{maps.min().item():.3f}, {maps.max().item():.3f}]")
    assert maps.shape[1:] == (3, 3, 1024, 1024), maps.shape
    assert not torch.isnan(maps).any(), "NaNs in maps!"
    print(f"example prompt:   {prompts[0]!r}")
    print("smoke OK")


if __name__ == "__main__":
    main()
