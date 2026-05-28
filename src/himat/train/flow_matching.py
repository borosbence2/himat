"""Rectified-flow / flow-matching objective (paper sec 3, Eq. 1-2; Sana's loss).

Continuous time t in [0,1]. Linear interpolation between data z0 and noise eps:

    z_t = (1 - t) * z0 + t * eps          (rectified flow path)// noise at t=1
    v*  = eps - z0                         (constant target velocity)

The network predicts v(z_t, t, c); loss is MSE(v_pred, v*). Timesteps are sampled
from a logit-normal distribution (Sana/SD3 default) which concentrates samples in
the mid-noise regime where learning signal is richest.

All ops are shape-agnostic and broadcast t over the trailing dims, so they work on
the (B, M, Cz, h, w) stacked latents directly.
"""

from __future__ import annotations

import torch


def sample_timesteps(batch: int, device: torch.device, logit_mean: float = 0.0, logit_std: float = 1.0) -> torch.Tensor:
    """Logit-normal t in (0,1), shape (batch,)."""
    n = torch.randn(batch, device=device) * logit_std + logit_mean
    return torch.sigmoid(n)


def _expand_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    return t.view(t.shape[0], *([1] * (ndim - 1)))


def add_noise(z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """z_t = (1 - t) * z0 + t * eps. t is (B,), broadcast over z0's trailing dims."""
    tb = _expand_t(t, z0.dim())
    return (1.0 - tb) * z0 + tb * eps


def velocity_target(z0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Target velocity v* = eps - z0 (constant along the linear path)."""
    return eps - z0


def flow_matching_loss(
    model,
    z0: torch.Tensor,
    text_emb: torch.Tensor,
    text_mask: torch.Tensor | None = None,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One flow-matching step. `model(noisy, t, text_emb, text_mask) -> v_pred`.

    z0: clean latent (B, ...). Returns (loss, logdict).
    """
    b = z0.shape[0]
    eps = torch.randn_like(z0)
    t = sample_timesteps(b, z0.device, logit_mean, logit_std)
    z_t = add_noise(z0, eps, t)
    v_star = velocity_target(z0, eps)
    v_pred = model(z_t, t, text_emb, text_mask)
    loss = (v_pred - v_star).pow(2).mean()
    return loss, {"fm_loss": loss.item()}
