"""Top-level HiMat model: routes the M=3 maps through the linear DiT as an
enlarged batch, with shared text conditioning, and predicts the flow-matching
velocity for every map. CrossStitch (injected into the DiT) is the only cross-map
coupling.

The fold/unfold + text-repeat logic lives in `HiMat` and takes an injectable
`denoiser` callable, so it is unit-testable with a fake denoiser. `build_himat`
wires the real Sana DiT + Gemma + DC-AE (runs on the 4090 box).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from himat import config


class HiMat(nn.Module):
    """Denoiser over stacked SVBRDF latents.

    forward: (noisy (B,M,Cz,h,w), timestep (B,), text_emb (B,L,D), text_mask (B,L))
             -> predicted velocity (B,M,Cz,h,w)
    """

    def __init__(self, denoiser, num_maps: int = config.NUM_MAPS) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.num_maps = num_maps

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, m, cz, h, w = noisy.shape
        if m != self.num_maps:
            raise ValueError(f"expected {self.num_maps} maps, got {m}")
        zf = noisy.reshape(b * m, cz, h, w)
        t = timestep.repeat_interleave(m, dim=0)
        te = text_emb.repeat_interleave(m, dim=0)
        tm = text_mask.repeat_interleave(m, dim=0) if text_mask is not None else None
        pred = self.denoiser(zf, t, te, tm)
        return pred.reshape(b, m, cz, h, w)


@dataclass
class HiMatBundle:
    """Everything the trainer/sampler needs."""

    himat: HiMat
    transformer: nn.Module
    stitches: nn.ModuleList
    vae: nn.Module  # SVBRDFAutoencoder
    text_encoder: nn.Module


def build_himat(
    dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = config.DEFAULT_TRAIN.lora_rank,
    finetuned_dc_ae: str | None = None,
) -> HiMatBundle:
    """Wire the real submodules. Heavy (loads Sana + Gemma + DC-AE)."""
    from himat.models.dc_ae import SVBRDFAutoencoder, load_dc_ae
    from himat.models.dit import apply_lora, inject_crossstitch, load_sana_transformer
    from himat.models.text_encoder import GemmaTextEncoder

    transformer = load_sana_transformer(dtype=dtype)
    # channel dim of the DiT hidden state — CONFIRM against model config.
    channels = int(getattr(transformer.config, "num_attention_heads", 0)) * int(
        getattr(transformer.config, "attention_head_dim", 0)
    ) or int(getattr(transformer.config, "caption_channels", 0))
    stitches = inject_crossstitch(transformer, channels=channels)
    transformer = apply_lora(transformer, rank=lora_rank)

    ae = load_dc_ae(dtype=torch.float32)
    vae = SVBRDFAutoencoder(ae)
    if finetuned_dc_ae:
        from safetensors.torch import load_file

        vae.ae.load_state_dict(load_file(finetuned_dc_ae), strict=False)
    vae.configure_training(finetune_encoder=False)

    text_encoder = GemmaTextEncoder(dtype=dtype)

    def denoiser(latent, timestep, text_emb, text_mask):
        # CONFIRM Sana's forward kwarg names + output attr on the 4090 box.
        out = transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=text_emb,
            encoder_attention_mask=text_mask,
            return_dict=True,
        )
        return out.sample if hasattr(out, "sample") else out[0]

    return HiMatBundle(HiMat(denoiser), transformer, stitches, vae, text_encoder)
