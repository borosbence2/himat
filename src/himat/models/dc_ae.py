"""DC-AE wrapper for multi-map SVBRDF.

Sana's DC-AE (f32c32) is an RGB autoencoder: it compresses a 3-channel 1024²
image to a 32-channel 32² latent. Our material is M=3 such RGB-shaped maps
(albedo, normal, packed roughness/metallic/height), so we run the *same* AE on
each map independently and stack along a map axis. This is exactly the paper's
setup (section 3 + 4.2): one shared compressor, the cross-map coupling is left to
CrossStitch inside the DiT, not the AE.

P1 fine-tunes only the decoder (paper Eq. 4: L1 + LPIPS, no adversarial term),
because the off-the-shelf DC-AE — trained on natural images — reconstructs normal
maps with a colour bias (paper Fig. 3).

The wrapper is agnostic to the concrete AE: it only needs `.encode(x).latent`,
`.decode(z).sample`, and `.encoder` / `.decoder` submodules, so tests can inject
a tiny fake. `load_dc_ae()` returns the real diffusers `AutoencoderDC`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from himat import config


def load_dc_ae(model_id: str = config.DC_AE_MODEL_ID, dtype: torch.dtype = torch.float32):
    """Load the standalone diffusers DC-AE (downloads on first use)."""
    from diffusers import AutoencoderDC

    return AutoencoderDC.from_pretrained(model_id, torch_dtype=dtype)


def _latent_of(enc_out):
    return enc_out.latent if hasattr(enc_out, "latent") else enc_out[0]


def _sample_of(dec_out):
    return dec_out.sample if hasattr(dec_out, "sample") else dec_out[0]


class SVBRDFAutoencoder(nn.Module):
    """Runs a single RGB DC-AE over the M maps of an SVBRDF.

    Shapes: stacked maps are (B, M, C, H, W); latents are (B, M, Cz, h, w).
    The M maps are folded into the batch for the underlying AE, then unfolded.
    """

    def __init__(self, ae: nn.Module, scaling_factor: float | None = None) -> None:
        super().__init__()
        self.ae = ae
        if scaling_factor is None:
            scaling_factor = float(getattr(getattr(ae, "config", None), "scaling_factor", 1.0))
        self.scaling_factor = scaling_factor

    # --- core (un)folding ------------------------------------------------- #
    def encode(self, stacked: torch.Tensor) -> torch.Tensor:
        """(B, M, C, H, W) -> raw latent (B, M, Cz, h, w)."""
        b, m, c, h, w = stacked.shape
        x = stacked.reshape(b * m, c, h, w)
        z = _latent_of(self.ae.encode(x))
        return z.reshape(b, m, *z.shape[1:])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """raw latent (B, M, Cz, h, w) -> (B, M, C, H, W)."""
        b, m = z.shape[:2]
        zf = z.reshape(b * m, *z.shape[2:])
        x = _sample_of(self.ae.decode(zf))
        return x.reshape(b, m, *x.shape[1:])

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        """Reconstruction round-trip (used for the P1 decoder fine-tune)."""
        return self.decode(self.encode(stacked))

    # --- scaled latents for the DiT (P3) ---------------------------------- #
    def encode_scaled(self, stacked: torch.Tensor) -> torch.Tensor:
        return self.encode(stacked) * self.scaling_factor

    def decode_scaled(self, z_scaled: torch.Tensor) -> torch.Tensor:
        return self.decode(z_scaled / self.scaling_factor)

    # --- training configuration ------------------------------------------ #
    def configure_training(self, finetune_encoder: bool = False) -> None:
        """Freeze everything, then unfreeze the decoder (+ encoder if asked)."""
        for p in self.ae.parameters():
            p.requires_grad_(False)
        for p in self.ae.decoder.parameters():
            p.requires_grad_(True)
        if finetune_encoder:
            for p in self.ae.encoder.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.ae.parameters() if p.requires_grad]

    @property
    def encoder_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.ae.encoder.parameters())
