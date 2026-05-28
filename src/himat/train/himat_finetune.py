"""HiMat fine-tune loop (Phase 3). UNTESTED on the authoring box — needs Sana +
Gemma + the 1024² cache, so it runs on the 4090 box. The fold logic and loss it
relies on are unit-tested (test_dit_himat, test_flow_matching); this file wires
them to the real models, optimiser, EMA, validation sampling, and checkpoints.

Trainable: DiT LoRA + CrossStitch (both live inside the transformer after
injection) + DC-AE decoder. Frozen: Sana base weights, Gemma, DC-AE encoder.

    python scripts/train_himat.py --max-steps 50000
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from himat import config
from himat.data.dataset import MatSynthDataset
from himat.models.himat import HiMatBundle, build_himat
from himat.train.flow_matching import flow_matching_loss


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_trainable(bundle: HiMatBundle) -> tuple[list, list]:
    """Return (lora+crossstitch params, dc_ae decoder params) for separate LRs."""
    decoder_ids = {id(p) for p in bundle.vae.trainable_parameters()}
    main = [p for p in bundle.transformer.parameters() if p.requires_grad and id(p) not in decoder_ids]
    decoder = list(bundle.vae.trainable_parameters())
    return main, decoder


def save_trainable(bundle: HiMatBundle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {f"transformer.{k}": v for k, v in bundle.transformer.named_parameters() if v.requires_grad}
    for k, v in bundle.vae.ae.named_parameters():
        if v.requires_grad:
            state[f"vae.{k}"] = v
    save_file({k: v.detach().contiguous().cpu() for k, v in state.items()}, str(path))
    return path


class EMA:
    def __init__(self, params, decay: float):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]
        self.params = list(params)

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def train_himat(cfg: config.TrainConfig | None = None, run_name: str = "himat", dc_ae_ckpt: str | None = None):
    cfg = cfg or config.DEFAULT_TRAIN
    config.ensure_dirs()
    torch.manual_seed(cfg.seed)
    device = _device()

    if dc_ae_ckpt is None:
        cand = config.CHECKPOINTS_DIR / "dc_ae_finetuned.safetensors"
        dc_ae_ckpt = str(cand) if cand.exists() else None

    bundle = build_himat(lora_rank=cfg.lora_rank, finetuned_dc_ae=dc_ae_ckpt)
    bundle.himat.to(device)
    bundle.transformer.to(device)
    bundle.vae.to(device)
    bundle.text_encoder.to(device)

    train_ds = MatSynthDataset(split="train", augment=True, val_fraction=cfg.val_fraction)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    main_params, dec_params = collect_trainable(bundle)
    opt = torch.optim.AdamW(
        [{"params": main_params, "lr": cfg.lr_lora}, {"params": dec_params, "lr": cfg.lr_decoder}],
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_steps)
    ema = EMA(main_params + dec_params, cfg.ema_decay)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(config.RUNS_DIR / run_name))
    except Exception:
        writer = None

    n_main = sum(p.numel() for p in main_params)
    n_dec = sum(p.numel() for p in dec_params)
    print(f"device={device} train={len(train_ds)} trainable: lora+cs={n_main/1e6:.1f}M decoder={n_dec/1e6:.1f}M")

    step = 0
    accum = max(cfg.grad_accum, 1)
    opt.zero_grad(set_to_none=True)
    it = iter(loader)
    while step < cfg.max_steps:
        for _ in range(accum):
            try:
                maps, prompts = next(it)
            except StopIteration:
                it = iter(loader)
                maps, prompts = next(it)
            maps = maps.to(device, non_blocking=True)

            with torch.no_grad():
                text_emb, text_mask = bundle.text_encoder.encode(list(prompts), system=True)
                z0 = bundle.vae.encode_scaled(maps)  # encoder frozen
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, logs = flow_matching_loss(bundle.himat, z0.float(), text_emb, text_mask)
            (loss / accum).backward()

        torch.nn.utils.clip_grad_norm_(main_params + dec_params, 1.0)
        opt.step()
        sched.step()
        ema.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            print(f"step {step}/{cfg.max_steps}  fm_loss={logs['fm_loss']:.4f}  lr={sched.get_last_lr()[0]:.2e}")
            if writer:
                writer.add_scalar("train/fm_loss", logs["fm_loss"], step)

        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            save_trainable(bundle, config.CHECKPOINTS_DIR / f"{run_name}_step{step}.safetensors")
        if step % cfg.val_every == 0 or step == cfg.max_steps:
            _val_sample(bundle, device, config.OUTPUTS_DIR / f"{run_name}_val_{step}.png", cfg.inference_steps)

    if writer:
        writer.close()
    final = config.CHECKPOINTS_DIR / f"{run_name}_final.safetensors"
    save_trainable(bundle, final)
    print(f"done. final={final}")
    return final


@torch.no_grad()
def _val_sample(bundle: HiMatBundle, device, out_path: Path, steps: int):
    """Sample a few fixed prompts to eyeball progress."""
    from himat.inference.pipeline import HiMatPipeline

    prompts = ["weathered red brick wall", "polished marble", "rusty metal mesh", "rough oak planks"]
    pipe = HiMatPipeline(bundle, device)
    from torchvision.utils import save_image

    previews = []
    for p in prompts:
        maps = pipe.generate(p, steps=steps, seed=0)
        previews.append(((maps["albedo"] + 1) * 0.5).clamp(0, 1).cpu())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(previews), str(out_path), nrow=len(prompts))
