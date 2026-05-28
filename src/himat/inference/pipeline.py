"""text -> SVBRDF sampling (Phase 4). UNTESTED here — needs the trained model.

Rectified-flow Euler sampler. The trained path is z_t = (1-t)*z0 + t*eps with
constant velocity v = eps - z0, so we start from noise at t=1 and integrate the
predicted velocity down to t=0:  z <- z + v_pred * (t_next - t_cur), with t
descending 1 -> 0. Optional classifier-free guidance against an empty prompt.

Output: the 5 SVBRDF maps as float tensors in [-1, 1] (use data.matsynth.to_unit_range
or save helpers to write PNGs).
"""

from __future__ import annotations

from pathlib import Path

import torch

from himat import config
from himat.data.matsynth import unpack_maps
from himat.models.himat import HiMatBundle


class HiMatPipeline:
    def __init__(self, bundle: HiMatBundle, device: torch.device) -> None:
        self.b = bundle
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        steps: int = config.DEFAULT_TRAIN.inference_steps,
        seed: int | None = None,
        cfg_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)

        text_emb, text_mask = self.b.text_encoder.encode([prompt], system=True)
        uncond_emb = uncond_mask = None
        if cfg_scale > 1.0:
            uncond_emb, uncond_mask = self.b.text_encoder.encode([""], system=True)

        m = config.NUM_MAPS
        cz = config.LATENT_CHANNELS
        h = w = config.LATENT_RESOLUTION
        z = torch.randn(1, m, cz, h, w, generator=gen, device=self.device)

        ts = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        for i in range(steps):
            t_cur, t_next = ts[i], ts[i + 1]
            tb = t_cur.expand(1)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                v = self.b.himat(z, tb, text_emb, text_mask)
                if cfg_scale > 1.0:
                    v_u = self.b.himat(z, tb, uncond_emb, uncond_mask)
                    v = v_u + cfg_scale * (v - v_u)
            z = z + v.float() * (t_next - t_cur)

        maps = self.b.vae.decode_scaled(z)  # (1, M, C, h_img, w_img)
        return unpack_maps(maps[0].clamp(-1, 1))


def save_svbrdf(maps: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Write the 5 maps as PNGs under out_dir."""
    from torchvision.utils import save_image

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in maps.items():
        img = ((t.clamp(-1, 1) + 1) * 0.5).cpu()
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        save_image(img, str(out_dir / f"{name}.png"))
    return out_dir
