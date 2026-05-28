"""Normal-aware augmentation correctness. CPU-only.

Two layers of checks:
  1. Constant-field tests — pin the sign/direction of each transform by hand.
     A normal pointing +x must point +y after one CCW rotation, -x after a
     horizontal flip, etc. Bulletproof, easy to verify mentally.
  2. Height-consistency test — build a normal map as -grad(height) and confirm
     the relationship survives every augmentation (the vector field rotates with
     the spatial domain). Uses periodic central differences so it's exact.
"""

import torch

from himat.data.dataset import (
    NORMAL_MAP_INDEX,
    augment_stacked,
    flip_h_stacked,
    flip_v_stacked,
    is_val,
    rot90_stacked,
)

S = 16


def _stacked_with_normal(normal: torch.Tensor) -> torch.Tensor:
    """Build a (3,3,S,S) tensor with given normal in map 1, junk elsewhere."""
    albedo = torch.rand(3, S, S) * 2 - 1
    packed = torch.rand(3, S, S) * 2 - 1
    return torch.stack([albedo, normal, packed], dim=0)


def _const_normal(vec) -> torch.Tensor:
    n = torch.zeros(3, S, S)
    n[0], n[1], n[2] = vec
    return n


# --- layer 1: constant-field direction checks ----------------------------- #
def test_flip_h_negates_nx():
    out = flip_h_stacked(_stacked_with_normal(_const_normal((1.0, 0.0, 0.5))))
    n = out[NORMAL_MAP_INDEX]
    assert torch.allclose(n[0], torch.full_like(n[0], -1.0))
    assert torch.allclose(n[1], torch.zeros_like(n[1]))
    assert torch.allclose(n[2], torch.full_like(n[2], 0.5))  # nz untouched


def test_flip_v_negates_ny():
    out = flip_v_stacked(_stacked_with_normal(_const_normal((0.0, 1.0, 0.5))))
    n = out[NORMAL_MAP_INDEX]
    assert torch.allclose(n[1], torch.full_like(n[1], -1.0))
    assert torch.allclose(n[0], torch.zeros_like(n[0]))


# Direction convention is fixed by torch.rot90's spatial action; per step the
# in-plane vector maps (x, y) -> (y, -x). The covariance tests below are the
# real correctness check; these just pin the convention so a regression is loud.
def test_rot90_maps_x_to_negy():
    out = rot90_stacked(_stacked_with_normal(_const_normal((1.0, 0.0, 0.0))), 1)
    n = out[NORMAL_MAP_INDEX]
    assert torch.allclose(n[0], torch.zeros_like(n[0]), atol=1e-6)
    assert torch.allclose(n[1], -torch.ones_like(n[1]), atol=1e-6)


def test_rot90_maps_y_to_x():
    out = rot90_stacked(_stacked_with_normal(_const_normal((0.0, 1.0, 0.0))), 1)
    n = out[NORMAL_MAP_INDEX]
    assert torch.allclose(n[0], torch.ones_like(n[0]), atol=1e-6)
    assert torch.allclose(n[1], torch.zeros_like(n[1]), atol=1e-6)


def test_rot90_four_steps_identity():
    s = _stacked_with_normal(torch.rand(3, S, S) * 2 - 1)
    out = rot90_stacked(s, 4)
    assert torch.allclose(out, s, atol=1e-6)


# --- layer 2: height-consistency (covariance) check ------------------------ #
def _normal_from_height(h: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """n = (-scale*dh/dx, -scale*dh/dy, 1) with periodic central differences.

    Periodic diffs are exactly equivariant under rot90/flip, so the relationship
    is preserved to float precision after augmentation.
    """
    dx = (torch.roll(h, -1, dims=-1) - torch.roll(h, 1, dims=-1)) * 0.5
    dy = (torch.roll(h, -1, dims=-2) - torch.roll(h, 1, dims=-2)) * 0.5
    nx = -scale * dx
    ny = -scale * dy
    nz = torch.ones_like(h)
    return torch.cat([nx, ny, nz], dim=0)  # (3, S, S)


def _consistent_stacked() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (stacked, height) where the normal map = _normal_from_height(height)."""
    yy, xx = torch.meshgrid(torch.linspace(0, 2, S), torch.linspace(0, 2, S), indexing="ij")
    h = torch.sin(3.1 * xx) + torch.cos(2.3 * yy) + 0.5 * torch.sin(1.7 * xx + 0.9 * yy)
    h = h.unsqueeze(0)  # (1, S, S)
    normal = _normal_from_height(h)
    albedo = torch.rand(3, S, S) * 2 - 1
    # put height into the packed map channel 2 so we can re-derive it post-aug
    packed = torch.cat([torch.rand(1, S, S) * 2 - 1, torch.rand(1, S, S) * 2 - 1, h], dim=0)
    return torch.stack([albedo, normal, packed], dim=0), h


def _check_covariant(transform):
    stacked, _ = _consistent_stacked()
    out = transform(stacked)
    aug_height = out[2, 2:3]  # height channel after the same spatial transform
    expected_normal = _normal_from_height(aug_height)
    assert torch.allclose(out[NORMAL_MAP_INDEX], expected_normal, atol=1e-5), transform.__name__


def test_flip_h_covariant():
    _check_covariant(flip_h_stacked)


def test_flip_v_covariant():
    _check_covariant(flip_v_stacked)


def test_rot90_covariant():
    for k in (1, 2, 3):
        _check_covariant(lambda s, k=k: rot90_stacked(s, k))


# --- split determinism ----------------------------------------------------- #
def test_split_is_deterministic_and_stable():
    ids = [f"material_{i}" for i in range(2000)]
    a = {m: is_val(m, 0.1) for m in ids}
    b = {m: is_val(m, 0.1) for m in ids}
    assert a == b
    frac = sum(a.values()) / len(ids)
    assert 0.05 < frac < 0.15  # ~10% holdout, allow sampling slack


def test_augment_runs_and_preserves_shape():
    s = _stacked_with_normal(torch.rand(3, S, S) * 2 - 1)
    g = torch.Generator().manual_seed(0)
    out = augment_stacked(s, flip=True, rot90=True, generator=g)
    assert out.shape == s.shape
