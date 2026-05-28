"""Loss functions.

P1 DC-AE fine-tune uses L1 + LPIPS (paper Eq. 4); no adversarial term (the paper
omits it, unstable at high resolution). LPIPS runs per map: each of the M maps is
a 3-channel image, exactly what LPIPS expects.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def l1_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (recon - target).abs().mean()


class MapLPIPS(nn.Module):
    """LPIPS over (B, M, C, H, W) tensors in [-1, 1] by folding maps into batch."""

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        import lpips

        self.net = lpips.LPIPS(net=net)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    @torch.no_grad()
    def _check(self, t: torch.Tensor) -> None:
        if t.dim() != 5 or t.shape[2] != 3:
            raise ValueError(f"expected (B, M, 3, H, W), got {tuple(t.shape)}")

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._check(recon)
        b, m, c, h, w = recon.shape
        r = recon.reshape(b * m, c, h, w)
        t = target.reshape(b * m, c, h, w)
        return self.net(r, t).mean()


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lpips_module: MapLPIPS | None,
    lambda_rec: float = 1.0,
    lambda_lpips: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined reconstruction loss. Returns (loss, scalar logdict)."""
    rec = l1_loss(recon, target)
    loss = lambda_rec * rec
    logs = {"l1": rec.item()}
    if lpips_module is not None and lambda_lpips > 0:
        lp = lpips_module(recon, target)
        loss = loss + lambda_lpips * lp
        logs["lpips"] = lp.item()
    logs["loss"] = loss.item()
    return loss, logs
