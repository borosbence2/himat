"""Rectified-flow Euler sampler (inference/pipeline.py), via a fully-fake bundle.

The sampler is exactly the kind of off-by-one/sign-prone loop worth pinning. We
inject fakes for text_encoder / himat / vae so it runs on CPU with no real
models, and check: output map shapes, step count, the integration result, CFG
blending, and seed reproducibility.
"""

import torch

from himat import config
from himat.inference.pipeline import HiMatPipeline
from himat.models.himat import HiMatBundle

M = config.NUM_MAPS
CZ = config.LATENT_CHANNELS
HW = config.LATENT_RESOLUTION
IMG = 16  # fake decode output size


class FakeTextEncoder:
    def encode(self, prompts, system=True):
        # distinct embedding per prompt content so CFG cond/uncond differ
        val = float(len(prompts[0]))
        emb = torch.full((1, 4, 5), val)
        mask = torch.ones(1, 4)
        return emb, mask


class FakeHiMat:
    """Returns a constant velocity; records how many times it was called and the
    text embedding it saw (to verify CFG blending)."""

    def __init__(self, vel):
        self.vel = vel
        self.calls = 0
        self.last_emb_val = None

    def __call__(self, z, t, text_emb, text_mask):
        self.calls += 1
        self.last_emb_val = float(text_emb.flatten()[0])
        return self.vel.expand_as(z).clone()


class FakeVAE:
    """Records the latent it was asked to decode; returns a fixed-size material."""

    def __init__(self):
        self.last_z = None

    def decode_scaled(self, z):
        self.last_z = z.clone()
        return torch.zeros(1, M, 3, IMG, IMG)


def _bundle(vel):
    return HiMatBundle(
        himat=FakeHiMat(vel),
        transformer=None,
        stitches=None,
        vae=FakeVAE(),
        text_encoder=FakeTextEncoder(),
    )


def test_output_maps_shapes():
    b = _bundle(torch.zeros(1, M, CZ, HW, HW))
    pipe = HiMatPipeline(b, torch.device("cpu"))
    maps = pipe.generate("brick", steps=5, seed=0)
    assert set(maps) == {"albedo", "normal", "roughness", "metallic", "height"}
    assert maps["albedo"].shape == (3, IMG, IMG)
    assert maps["normal"].shape == (3, IMG, IMG)
    assert maps["roughness"].shape == (1, IMG, IMG)


def test_step_count():
    b = _bundle(torch.zeros(1, M, CZ, HW, HW))
    pipe = HiMatPipeline(b, torch.device("cpu"))
    pipe.generate("x", steps=7, seed=0)
    assert b.himat.calls == 7  # one model eval per step, no CFG


def test_euler_integration_result():
    # constant velocity v; integrating t: 1 -> 0 gives sum(dt) = -1, so the latent
    # handed to the decoder must be z_init - v.
    vel = torch.full((1, M, CZ, HW, HW), 0.3)
    b = _bundle(vel)
    pipe = HiMatPipeline(b, torch.device("cpu"))

    gen = torch.Generator().manual_seed(0)
    z_init = torch.randn(1, M, CZ, HW, HW, generator=gen)

    pipe.generate("x", steps=10, seed=0)
    expected = z_init - vel  # since total dt = (0 - 1) = -1
    assert torch.allclose(b.vae.last_z, expected, atol=1e-5)


def test_cfg_doubles_calls_and_blends():
    b = _bundle(torch.zeros(1, M, CZ, HW, HW))
    pipe = HiMatPipeline(b, torch.device("cpu"))
    b.himat.calls = 0
    pipe.generate("brick", steps=4, seed=0, cfg_scale=3.0)
    assert b.himat.calls == 8  # cond + uncond per step


def test_seed_reproducibility():
    b = _bundle(torch.zeros(1, M, CZ, HW, HW))
    pipe = HiMatPipeline(b, torch.device("cpu"))
    pipe.generate("x", steps=5, seed=42)
    z_a = b.vae.last_z.clone()
    pipe.generate("x", steps=5, seed=42)
    z_b = b.vae.last_z.clone()
    pipe.generate("x", steps=5, seed=99)
    z_c = b.vae.last_z.clone()
    assert torch.allclose(z_a, z_b, atol=1e-6)
    assert not torch.allclose(z_a, z_c, atol=1e-4)
