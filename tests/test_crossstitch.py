"""CrossStitch correctness, CPU-only. The real validation (does it improve
inter-map alignment?) is an ablation on the 4090 box; these pin the invariants:
shape, identity-at-init, cross-map coupling, token-routing round-trip, gradients.
"""

import torch

from himat.models.crossstitch import (
    CrossStitch,
    apply_crossstitch_tokens,
    maps_to_tokens,
    tokens_to_maps,
)

B, M, C, H, W = 2, 3, 8, 4, 4


def _x():
    torch.manual_seed(0)
    return torch.randn(B, M, C, H, W)


def test_shape_preserved():
    cs = CrossStitch(C, num_maps=M)
    assert cs(_x()).shape == (B, M, C, H, W)


def test_identity_at_init():
    # zero-init output projections -> module is exactly identity before training
    cs = CrossStitch(C, num_maps=M)
    x = _x()
    assert torch.allclose(cs(x), x, atol=1e-6)


def test_wrong_map_count_raises():
    cs = CrossStitch(C, num_maps=M)
    try:
        cs(torch.randn(B, M + 1, C, H, W))
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong map count")


def test_cross_map_coupling_after_perturb():
    # once the output projections are non-zero, a change in one map's input must
    # affect another map's output — that's the whole point of CrossStitch.
    cs = CrossStitch(C, num_maps=M)
    torch.nn.init.normal_(cs.local_out.weight, std=0.1)
    torch.nn.init.normal_(cs.map_conv.weight, std=0.5)
    x = _x()
    out_a = cs(x)
    x2 = x.clone()
    x2[:, 0] += 1.0  # perturb only map 0's input
    out_b = cs(x2)
    # map 1's output should change because map 0 changed
    delta_map1 = (out_a[:, 1] - out_b[:, 1]).abs().max().item()
    assert delta_map1 > 1e-4, delta_map1


def test_token_routing_roundtrip():
    bm = B * M
    n = H * W
    hidden = torch.randn(bm, n, C)
    x = tokens_to_maps(hidden, M, H, W)
    assert x.shape == (B, M, C, H, W)
    back = maps_to_tokens(x)
    assert torch.allclose(back, hidden, atol=1e-6)


def test_apply_tokens_identity_at_init():
    cs = CrossStitch(C, num_maps=M)
    hidden = torch.randn(B * M, H * W, C)
    out = apply_crossstitch_tokens(hidden, cs, M, H, W)
    assert out.shape == hidden.shape
    assert torch.allclose(out, hidden, atol=1e-6)


def test_gradients_flow():
    cs = CrossStitch(C, num_maps=M)
    torch.nn.init.normal_(cs.local_out.weight, std=0.1)
    x = _x().requires_grad_(True)
    cs(x).pow(2).mean().backward()
    assert cs.local_out.weight.grad is not None
    assert cs.dw.weight.grad is not None  # upstream conv also receives gradient
