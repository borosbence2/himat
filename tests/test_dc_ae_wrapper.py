"""SVBRDFAutoencoder fold/unfold + freeze logic, via a tiny fake DC-AE.

No download, no big VRAM — verifies the multi-map plumbing the real fine-tune
relies on. CPU-only.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from himat.models.dc_ae import SVBRDFAutoencoder
from himat.train.losses import vae_loss


class FakeDCAE(nn.Module):
    """Mimics diffusers AutoencoderDC: encode().latent, decode().sample,
    encoder/decoder submodules, config.scaling_factor."""

    def __init__(self, stride: int = 8, cz: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, cz, kernel_size=stride, stride=stride)
        self.decoder = nn.ConvTranspose2d(cz, 3, kernel_size=stride, stride=stride)
        self.config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, x):
        return SimpleNamespace(latent=self.encoder(x))

    def decode(self, z):
        return SimpleNamespace(sample=self.decoder(z))


def _vae():
    torch.manual_seed(0)
    return SVBRDFAutoencoder(FakeDCAE(stride=8, cz=4))


def test_encode_shape():
    vae = _vae()
    x = torch.randn(2, 3, 3, 64, 64)  # (B, M, C, H, W)
    z = vae.encode(x)
    assert z.shape == (2, 3, 4, 8, 8), z.shape


def test_decode_shape_roundtrip():
    vae = _vae()
    x = torch.randn(2, 3, 3, 64, 64)
    recon = vae(x)
    assert recon.shape == x.shape, recon.shape


def test_scaling_factor_read_from_config():
    assert _vae().scaling_factor == 0.5


def test_scaled_roundtrip_matches_raw():
    vae = _vae()
    x = torch.randn(1, 3, 3, 64, 64)
    raw = vae.decode(vae.encode(x))
    scaled = vae.decode_scaled(vae.encode_scaled(x))
    assert torch.allclose(raw, scaled, atol=1e-5)


def test_configure_training_decoder_only():
    vae = _vae()
    vae.configure_training(finetune_encoder=False)
    assert vae.encoder_frozen
    assert all(not p.requires_grad for p in vae.ae.encoder.parameters())
    assert all(p.requires_grad for p in vae.ae.decoder.parameters())
    # decoder = weight + bias
    assert len(vae.trainable_parameters()) == 2


def test_configure_training_with_encoder():
    vae = _vae()
    vae.configure_training(finetune_encoder=True)
    assert not vae.encoder_frozen
    assert all(p.requires_grad for p in vae.ae.encoder.parameters())
    assert len(vae.trainable_parameters()) == 4  # enc + dec, weight+bias each


def test_decoder_gets_gradients():
    vae = _vae()
    vae.configure_training(finetune_encoder=False)
    x = torch.randn(1, 3, 3, 64, 64)
    with torch.no_grad():
        z = vae.encode(x)
    recon = vae.decode(z)
    loss, logs = vae_loss(recon, x, lpips_module=None, lambda_rec=1.0, lambda_lpips=0.0)
    loss.backward()
    assert vae.ae.decoder.weight.grad is not None
    assert vae.ae.encoder.weight.grad is None  # frozen + encoded under no_grad
    assert "l1" in logs and "loss" in logs


def test_vae_loss_without_lpips():
    recon = torch.zeros(1, 3, 3, 8, 8)
    target = torch.ones(1, 3, 3, 8, 8)
    loss, logs = vae_loss(recon, target, lpips_module=None, lambda_rec=1.0, lambda_lpips=0.5)
    assert abs(logs["l1"] - 1.0) < 1e-6  # |0-1| = 1
    assert "lpips" not in logs
    assert abs(loss.item() - 1.0) < 1e-6
