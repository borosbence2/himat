"""CrossStitch: the paper's lightweight cross-map consistency module (sec 4.3).

This is HiMat's novel contribution and the only part described purely in prose
(sec 4.3 + Eq. 7), so this implementation is a faithful *interpretation*, to be
validated by ablation on the 4090 box (does enabling it improve inter-map
alignment / val loss?).

Paper's stated design:
  * Operate on post-self-attention features f (M maps, Ĉ channels, ĥ x ŵ).
  * Rearrange so the map axis becomes a conv axis: 'm h w c -> (h w) c m', apply
    a 1D convolution across maps, restore layout (Eq. 7).
  * Dual branch:
      - local : depthwise-separable conv (spatial 3x3 then pointwise 1x1) for
                local mixing, plus the across-maps 1D conv.
      - global: average-pool over space, 1x1 conv across maps, GELU — shared
                semantic context, broadcast back.
  * Zero-initialised + residual so it is non-destructive at init (the pretrained
    DiT behaves identically until CrossStitch learns something).

Deviation from the letter of the paper: rather than zero-init *every* conv (which
starts all upstream convs with zero gradient), we zero-init only each branch's
output projection — the ControlNet "zero-conv" pattern. Same identity-at-init
behaviour, healthier gradient flow. Documented here so it's a deliberate choice.

Input/output layout for this module: (B, M, C, H, W). A helper converts to/from
the (B*M, N, C) token layout the DiT uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from himat import config


class CrossStitch(nn.Module):
    def __init__(self, channels: int, num_maps: int = config.NUM_MAPS, map_kernel: int = 3) -> None:
        super().__init__()
        self.num_maps = num_maps
        c = channels

        # --- local branch ---
        # depthwise-separable spatial mixing (per map, folded into batch)
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        self.pw = nn.Conv2d(c, c, kernel_size=1)
        # 1D conv across the map axis (the Eq. 7 core)
        self.map_conv = nn.Conv1d(c, c, kernel_size=map_kernel, padding=map_kernel // 2)
        self.local_out = nn.Conv2d(c, c, kernel_size=1)

        # --- global branch ---
        self.global_conv = nn.Conv1d(c, c, kernel_size=1)  # across maps, on pooled vectors
        self.act = nn.GELU()
        self.global_out = nn.Conv2d(c, c, kernel_size=1)

        # zero-init output projections -> identity at init, non-destructive
        for proj in (self.local_out, self.global_out):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, C, H, W) -> same shape (residual)."""
        b, m, c, h, w = x.shape
        if m != self.num_maps:
            raise ValueError(f"expected {self.num_maps} maps, got {m}")
        xf = x.reshape(b * m, c, h, w)

        # local: spatial depthwise-separable, then mix across maps
        local = self.pw(self.dw(xf))                                   # (B*M, C, H, W)
        lm = local.reshape(b, m, c, h * w).permute(0, 3, 2, 1)         # (B, HW, C, M)
        lm = lm.reshape(b * h * w, c, m)
        lm = self.map_conv(lm)                                         # mix across maps
        lm = lm.reshape(b, h * w, c, m).permute(0, 3, 2, 1)           # (B, M, C, HW)
        lm = lm.reshape(b * m, c, h, w)
        local = self.local_out(lm).reshape(b, m, c, h, w)

        # global: pooled per-map context shared across maps
        pooled = xf.mean(dim=(2, 3))                                   # (B*M, C)
        gm = pooled.reshape(b, m, c).permute(0, 2, 1)                  # (B, C, M)
        gm = self.act(self.global_conv(gm))                           # (B, C, M)
        gm = gm.permute(0, 2, 1).reshape(b * m, c, 1, 1)
        glob = self.global_out(gm).reshape(b, m, c, 1, 1)            # broadcast over H,W

        return x + local + glob


def tokens_to_maps(hidden: torch.Tensor, num_maps: int, h: int, w: int) -> torch.Tensor:
    """(B*M, N, C) token layout -> (B, M, C, H, W) for CrossStitch. N must be H*W."""
    bm, n, c = hidden.shape
    if n != h * w:
        raise ValueError(f"token count {n} != h*w={h * w}")
    b = bm // num_maps
    x = hidden.reshape(b, num_maps, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
    return x


def maps_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(B, M, C, H, W) -> (B*M, N, C) token layout."""
    b, m, c, h, w = x.shape
    return x.permute(0, 1, 3, 4, 2).reshape(b * m, h * w, c).contiguous()


def apply_crossstitch_tokens(
    hidden: torch.Tensor, module: CrossStitch, num_maps: int, h: int, w: int
) -> torch.Tensor:
    """Run CrossStitch on DiT token-layout hidden states, returning token layout."""
    x = tokens_to_maps(hidden, num_maps, h, w)
    x = module(x)
    return maps_to_tokens(x)
