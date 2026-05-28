"""Flow-matching math. CPU-only."""

import torch

from himat.train.flow_matching import (
    add_noise,
    flow_matching_loss,
    sample_timesteps,
    velocity_target,
)


def test_timesteps_in_unit_interval():
    t = sample_timesteps(10000, torch.device("cpu"))
    assert t.shape == (10000,)
    assert t.min() > 0.0 and t.max() < 1.0
    assert 0.3 < t.mean().item() < 0.7  # logit-normal(0,1) is symmetric around 0.5


def test_add_noise_endpoints():
    z0 = torch.randn(4, 3, 8, 8)
    eps = torch.randn_like(z0)
    # t=0 -> z0, t=1 -> eps
    z_at_0 = add_noise(z0, eps, torch.zeros(4))
    z_at_1 = add_noise(z0, eps, torch.ones(4))
    assert torch.allclose(z_at_0, z0, atol=1e-6)
    assert torch.allclose(z_at_1, eps, atol=1e-6)


def test_add_noise_midpoint():
    z0 = torch.randn(2, 4)
    eps = torch.randn(2, 4)
    z_half = add_noise(z0, eps, torch.full((2,), 0.5))
    assert torch.allclose(z_half, 0.5 * z0 + 0.5 * eps, atol=1e-6)


def test_velocity_target():
    z0 = torch.randn(2, 4)
    eps = torch.randn(2, 4)
    assert torch.allclose(velocity_target(z0, eps), eps - z0)


def test_broadcast_over_5d_stacked_latent():
    # the real latents are (B, M, Cz, h, w); t must broadcast over all trailing dims
    z0 = torch.randn(2, 3, 4, 8, 8)
    eps = torch.randn_like(z0)
    z_half = add_noise(z0, eps, torch.full((2,), 0.5))
    assert z_half.shape == z0.shape
    assert torch.allclose(z_half, 0.5 * z0 + 0.5 * eps, atol=1e-6)


def test_perfect_model_zero_loss():
    # a model that returns exactly eps - z0 should drive loss to ~0; check the
    # plumbing by making the "model" recompute the target from its inputs.
    z0 = torch.randn(4, 3, 4, 4)

    def oracle(z_t, t, text_emb, text_mask):
        # invert: z_t = (1-t) z0 + t eps  => eps - z0 = (z_t - z0) / t
        tb = t.view(-1, 1, 1, 1)
        eps = (z_t - (1 - tb) * z0) / tb
        return eps - z0

    torch.manual_seed(0)
    loss, logs = flow_matching_loss(oracle, z0, text_emb=torch.randn(4, 5, 6))
    assert loss.item() < 1e-6
    assert "fm_loss" in logs


def test_loss_is_scalar_and_finite():
    z0 = torch.randn(3, 3, 4, 4)
    model = lambda z, t, e, m: torch.zeros_like(z)  # noqa: E731
    loss, _ = flow_matching_loss(model, z0, text_emb=torch.randn(3, 5, 6))
    assert loss.dim() == 0 and torch.isfinite(loss)
