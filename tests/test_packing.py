"""pack/unpack round-trip and range conversion. CPU-only (needs torch, no GPU)."""

import torch

from himat import config
from himat.data.matsynth import pack_maps, to_unit_range, unpack_maps


def _rand_maps(s=8):
    return (
        torch.rand(3, s, s) * 2 - 1,  # albedo
        torch.rand(3, s, s) * 2 - 1,  # normal
        torch.rand(1, s, s) * 2 - 1,  # roughness
        torch.rand(1, s, s) * 2 - 1,  # metallic
        torch.rand(1, s, s) * 2 - 1,  # height
    )


def test_pack_shape():
    packed = pack_maps(*_rand_maps())
    assert packed.shape == (config.NUM_MAPS, config.MAP_CHANNELS, 8, 8)


def test_pack_unpack_roundtrip():
    albedo, normal, rough, metal, height = _rand_maps()
    packed = pack_maps(albedo, normal, rough, metal, height)
    out = unpack_maps(packed)
    assert torch.equal(out["albedo"], albedo)
    assert torch.equal(out["normal"], normal)
    assert torch.equal(out["roughness"], rough)
    assert torch.equal(out["metallic"], metal)
    assert torch.equal(out["height"], height)


def test_to_unit_range():
    packed = pack_maps(*_rand_maps())
    unit = to_unit_range(packed)
    for v in unit.values():
        assert v.min() >= 0.0 and v.max() <= 1.0


def test_unpack_rejects_bad_shape():
    bad = torch.zeros(2, 3, 8, 8)
    try:
        unpack_maps(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong map count")
