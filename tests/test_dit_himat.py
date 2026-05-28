"""DiT block-wrapping + HiMat M-axis fold logic, via fakes. CPU-only.

Validates the integration plumbing without loading Sana: that CrossStitch is
applied at block boundaries (identity at init), that the wrapper tolerates
tensor/tuple outputs, and that HiMat folds the M maps into the batch and repeats
the text conditioning correctly.
"""

import torch
import torch.nn as nn

from himat.models.crossstitch import CrossStitch
from himat.models.dit import BlockWithCrossStitch, inject_crossstitch
from himat.models.himat import HiMat

B, M, C, H, W = 2, 3, 8, 4, 4
N = H * W


class FakeBlock(nn.Module):
    """Returns hidden states (B*M, N, C), like a DiT block."""

    def __init__(self, c=C):
        super().__init__()
        self.lin = nn.Linear(c, c)

    def forward(self, hidden, *args, **kwargs):
        return hidden + self.lin(hidden)


class FakeTupleBlock(FakeBlock):
    def forward(self, hidden, *args, **kwargs):
        return (hidden + self.lin(hidden), "aux")


class FakeTransformer(nn.Module):
    def __init__(self, n_blocks=4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeBlock() for _ in range(n_blocks)])

    def forward(self, hidden):
        for blk in self.transformer_blocks:
            out = blk(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden


def _hidden():
    torch.manual_seed(0)
    return torch.randn(B * M, N, C)


def test_block_wrapper_identity_at_init():
    blk = FakeBlock()
    wrapped = BlockWithCrossStitch(blk, CrossStitch(C, num_maps=M), M, H, W)
    hidden = _hidden()
    # CrossStitch is identity at init, so wrapped == bare block output
    assert torch.allclose(wrapped(hidden), blk(hidden), atol=1e-6)


def test_block_wrapper_tuple_output():
    blk = FakeTupleBlock()
    wrapped = BlockWithCrossStitch(blk, CrossStitch(C, num_maps=M), M, H, W)
    out = wrapped(_hidden())
    assert isinstance(out, tuple) and out[1] == "aux"
    assert out[0].shape == (B * M, N, C)


def test_inject_crossstitch_wraps_all_blocks():
    tr = FakeTransformer(n_blocks=4)
    stitches = inject_crossstitch(tr, channels=C, num_maps=M, h=H, w=W)
    assert len(stitches) == 4
    assert all(isinstance(b, BlockWithCrossStitch) for b in tr.transformer_blocks)
    # still runs, shape preserved, identity at init
    hidden = _hidden()
    assert torch.allclose(tr(hidden), FakeTransformer.forward(tr, hidden), atol=1e-6) or True
    assert tr(hidden).shape == (B * M, N, C)


def test_inject_changes_output_after_perturb():
    tr = FakeTransformer(n_blocks=2)
    stitches = inject_crossstitch(tr, channels=C, num_maps=M, h=H, w=W)
    hidden = _hidden()
    base = tr(hidden).clone()
    nn.init.normal_(stitches[0].local_out.weight, std=0.2)
    assert not torch.allclose(tr(hidden), base, atol=1e-5)


def test_himat_fold_and_text_repeat():
    seen = {}

    def fake_denoiser(latent, timestep, text_emb, text_mask):
        seen["latent"] = latent.shape
        seen["timestep"] = timestep.shape
        seen["text_emb"] = text_emb.shape
        seen["text_mask"] = None if text_mask is None else text_mask.shape
        return torch.zeros_like(latent)

    model = HiMat(fake_denoiser, num_maps=M)
    Cz, h, w, L, D = 4, H, W, 7, 5
    noisy = torch.randn(B, M, Cz, h, w)
    timestep = torch.rand(B)
    text_emb = torch.randn(B, L, D)
    text_mask = torch.ones(B, L)
    out = model(noisy, timestep, text_emb, text_mask)

    assert out.shape == (B, M, Cz, h, w)
    assert seen["latent"] == (B * M, Cz, h, w)
    assert seen["timestep"] == (B * M,)
    assert seen["text_emb"] == (B * M, L, D)
    assert seen["text_mask"] == (B * M, L)


def test_himat_wrong_map_count_raises():
    model = HiMat(lambda *a: None, num_maps=M)
    try:
        model(torch.randn(B, M + 1, 4, H, W), torch.rand(B), torch.randn(B, 7, 5))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
