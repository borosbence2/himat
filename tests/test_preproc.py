"""Preprocessing: schema check (runs everywhere) + PIL->tensor conversion (runs
where the torch/numpy bridge works — i.e. CI and the 4090 box, skipped on the
authoring box which has numpy 2.x against a numpy-1.x torch build)."""

import numpy as np
import pytest
import torch

from himat import config
from himat.data import preproc


# --- schema logic: no numpy/torch bridge needed ------------------------------ #
def _full_record():
    s = config.MATSYNTH
    return {s.basecolor: 1, s.normal: 1, s.roughness: 1, s.metallic: 1, s.height: 1}


def test_has_required_full():
    assert preproc.has_required(_full_record())


def test_has_required_missing_one():
    rec = _full_record()
    rec[config.MATSYNTH.height] = None
    assert not preproc.has_required(rec)


def test_has_required_empty():
    assert not preproc.has_required({})


# --- image conversion: needs the numpy bridge -------------------------------- #
def _bridge_ok():
    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32))
        return True
    except Exception:
        return False


bridge = pytest.mark.skipif(not _bridge_ok(), reason="torch/numpy bridge unavailable")


@bridge
def test_to_float_chw_rgb():
    from PIL import Image

    img = Image.new("RGB", (32, 32), (255, 128, 0))
    t = preproc._to_float_chw(img, 3, 32)
    assert t.shape == (3, 32, 32)
    assert 0.0 <= t.min().item() and t.max().item() <= 1.0
    assert abs(t[0].mean().item() - 1.0) < 1e-3  # R channel = 255 -> 1.0


@bridge
def test_to_float_chw_scalar_expands_to_requested_channels():
    from PIL import Image

    img = Image.new("L", (32, 32), 128)
    assert preproc._to_float_chw(img, 1, 32).shape == (1, 32, 32)
    assert preproc._to_float_chw(img, 3, 32).shape == (3, 32, 32)  # repeated


@bridge
def test_to_float_chw_resizes():
    from PIL import Image

    img = Image.new("RGB", (64, 64), (10, 20, 30))
    assert preproc._to_float_chw(img, 3, 16).shape == (3, 16, 16)


@bridge
def test_preprocess_material_packs_to_signed():
    from PIL import Image

    maps = {
        "basecolor": Image.new("RGB", (16, 16), (200, 100, 50)),
        "normal": Image.new("RGB", (16, 16), (128, 128, 255)),
        "roughness": Image.new("L", (16, 16), 64),
        "metallic": Image.new("L", (16, 16), 0),
        "height": Image.new("L", (16, 16), 200),
    }
    packed = preproc.preprocess_material(maps, size=16)
    assert packed.shape == (config.NUM_MAPS, config.MAP_CHANNELS, 16, 16)
    assert packed.min().item() >= -1.0 and packed.max().item() <= 1.0


@bridge
def test_preprocess_material_missing_raises():
    from PIL import Image

    maps = {
        "basecolor": Image.new("RGB", (16, 16)),
        "normal": Image.new("RGB", (16, 16)),
        "roughness": Image.new("L", (16, 16)),
        "metallic": Image.new("L", (16, 16)),
        # height missing
    }
    with pytest.raises(KeyError):
        preproc.preprocess_material(maps, size=16)
