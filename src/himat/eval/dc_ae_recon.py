"""DC-AE reconstruction metrics + a before/after visual grid.

Reports PSNR / SSIM / LPIPS on a validation subset (paper Tab. 2 reports these
plus rFID; rFID needs an Inception model over many samples and is left optional).
Metrics are computed on the maps mapped back to [0, 1], folding the M maps into
the batch so each map is scored as an image.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from himat.models.dc_ae import SVBRDFAutoencoder


def _to_unit(stacked: torch.Tensor) -> torch.Tensor:
    """(B, M, C, H, W) in [-1,1] -> [0,1], folded to (B*M, C, H, W)."""
    b, m, c, h, w = stacked.shape
    return ((stacked.clamp(-1, 1) + 1) * 0.5).reshape(b * m, c, h, w)


@torch.no_grad()
def compute_recon_metrics(
    vae: SVBRDFAutoencoder,
    val_ds: Dataset,
    device: torch.device,
    max_items: int = 64,
    lpips_net: str = "vgg",
) -> dict[str, float]:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    import lpips as lpips_lib

    lp = lpips_lib.LPIPS(net=lpips_net).to(device).eval()
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    vae.eval()
    psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0
    for maps, _ in loader:
        if n >= max_items:
            break
        maps = maps.to(device)
        recon = vae(maps)
        gt = _to_unit(maps)
        pr = _to_unit(recon)
        psnr_sum += psnr(pr, gt, data_range=1.0).item()
        ssim_sum += ssim(pr, gt, data_range=1.0).item()
        # LPIPS wants [-1,1]; fold maps to batch.
        b, m = maps.shape[:2]
        lpips_sum += lp(
            recon.reshape(b * m, *recon.shape[2:]).clamp(-1, 1),
            maps.reshape(b * m, *maps.shape[2:]).clamp(-1, 1),
        ).mean().item()
        n += 1

    n = max(n, 1)
    return {"psnr": psnr_sum / n, "ssim": ssim_sum / n, "lpips": lpips_sum / n, "count": n}


@torch.no_grad()
def save_recon_grid(
    vae: SVBRDFAutoencoder,
    val_ds: Dataset,
    device: torch.device,
    out_path: Path,
    n_materials: int = 4,
) -> Path:
    """Save a PNG comparing GT vs reconstruction for albedo + normal maps."""
    from torchvision.utils import save_image

    from himat.data.matsynth import unpack_maps

    vae.eval()
    rows = []
    for i in range(min(n_materials, len(val_ds))):
        maps, _ = val_ds[i]
        maps = maps.unsqueeze(0).to(device)
        recon = vae(maps)
        gt = unpack_maps(maps[0])
        pr = unpack_maps(recon[0])
        # albedo + normal, GT then recon, each mapped to [0,1]
        for key in ("albedo", "normal"):
            rows.append(((gt[key] + 1) * 0.5).clamp(0, 1).cpu())
            rows.append(((pr[key] + 1) * 0.5).clamp(0, 1).cpu())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(rows), str(out_path), nrow=4)
    return out_path
