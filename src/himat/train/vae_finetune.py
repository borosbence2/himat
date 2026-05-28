"""Fine-tune the DC-AE decoder on SVBRDF maps (Phase 1).

Decoder-only by default (encoder frozen): encode under no_grad, decode with grad,
optimise L1 + LPIPS. bf16 autocast, gradient accumulation, periodic validation +
checkpoint. Runs on the 4090 box; needs the 1024² cache from scripts/preprocess.py.

    python scripts/train_dc_ae.py --max-steps 15000
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from himat import config
from himat.data.dataset import MatSynthDataset
from himat.models.dc_ae import SVBRDFAutoencoder, load_dc_ae
from himat.train.losses import MapLPIPS, vae_loss


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_writer(run_name: str):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(config.RUNS_DIR / run_name))
    except Exception:  # tensorboard optional
        return None


def save_dc_ae(vae: SVBRDFAutoencoder, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in vae.ae.state_dict().items()}
    save_file(state, str(path), metadata={"scaling_factor": str(vae.scaling_factor)})
    return path


def train_vae(cfg: config.VAEFinetuneConfig | None = None, run_name: str = "dc_ae") -> Path:
    cfg = cfg or config.DEFAULT_VAE
    config.ensure_dirs()
    torch.manual_seed(cfg.seed)
    device = _device()

    train_ds = MatSynthDataset(split="train", augment=True, val_fraction=config.DEFAULT_TRAIN.val_fraction)
    val_ds = MatSynthDataset(split="val", augment=False, val_fraction=config.DEFAULT_TRAIN.val_fraction)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True
    )

    ae = load_dc_ae(dtype=torch.float32)
    vae = SVBRDFAutoencoder(ae).to(device)
    vae.configure_training(finetune_encoder=cfg.finetune_encoder)
    if hasattr(vae.ae, "enable_gradient_checkpointing"):
        try:
            vae.ae.enable_gradient_checkpointing()
        except Exception:
            pass

    lpips_module = MapLPIPS(net="vgg").to(device)
    opt = torch.optim.AdamW(vae.trainable_parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    writer = _maybe_writer(run_name)

    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"encoder_frozen={vae.encoder_frozen} trainable_tensors={len(vae.trainable_parameters())}"
    )

    best_psnr = -1.0
    step = 0
    accum = max(cfg.grad_accum, 1)
    opt.zero_grad(set_to_none=True)
    data_iter = iter(loader)

    while step < cfg.max_steps:
        for micro in range(accum):
            try:
                maps, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                maps, _ = next(data_iter)
            maps = maps.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if vae.encoder_frozen:
                    with torch.no_grad():
                        z = vae.encode(maps)
                    recon = vae.decode(z)
                else:
                    recon = vae(maps)
                loss, logs = vae_loss(
                    recon.float(), maps.float(), lpips_module, cfg.lambda_rec, cfg.lambda_lpips
                )
            (loss / accum).backward()

        torch.nn.utils.clip_grad_norm_(vae.trainable_parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            print(f"step {step}/{cfg.max_steps}  " + "  ".join(f"{k}={v:.4f}" for k, v in logs.items()))
            if writer:
                for k, v in logs.items():
                    writer.add_scalar(f"train/{k}", v, step)

        if step % cfg.val_every == 0 or step == cfg.max_steps:
            from himat.eval.dc_ae_recon import compute_recon_metrics, save_recon_grid

            metrics = compute_recon_metrics(vae, val_ds, device, max_items=64)
            print(f"[val] step {step}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, step)
            save_recon_grid(vae, val_ds, device, config.OUTPUTS_DIR / f"{run_name}_recon_{step}.png")
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_dc_ae(vae, config.CHECKPOINTS_DIR / "dc_ae_finetuned.safetensors")
                print(f"  saved best (psnr={best_psnr:.3f})")
            vae.train()

    if writer:
        writer.close()
    final = config.CHECKPOINTS_DIR / "dc_ae_finetuned_final.safetensors"
    save_dc_ae(vae, final)
    print(f"done. best val psnr={best_psnr:.3f}  final={final}")
    return final
