"""Linear-attention DiT (Sana) + CrossStitch injection + LoRA.

We do NOT reimplement Sana's forward. Instead we wrap each transformer block so
that CrossStitch runs on the block's output hidden states (the paper inserts it
"after the self-attention layer"; placing it at the block boundary is the
least-invasive faithful approximation and keeps Sana's internals untouched).

The M=3 maps flow through the DiT as an enlarged batch (B*M): Sana denoises each
map with shared weights + shared text conditioning, and CrossStitch is the only
place the maps exchange information — exactly the paper's design intent.

Sana-specifics to confirm on the 4090 box (marked CONFIRM below):
  * the block-list attribute name (`transformer_blocks` in current diffusers),
  * that a block returns hidden states as a tensor or tuple-with-tensor-first,
  * the LoRA target module names.
The block wrapper is written to tolerate tensor-or-tuple outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from himat import config
from himat.models.crossstitch import CrossStitch, apply_crossstitch_tokens


class BlockWithCrossStitch(nn.Module):
    """Wrap a DiT block; apply CrossStitch to its output hidden states."""

    def __init__(self, block: nn.Module, crossstitch: CrossStitch, num_maps: int, h: int, w: int) -> None:
        super().__init__()
        self.block = block
        self.crossstitch = crossstitch
        self.num_maps = num_maps
        self.h = h
        self.w = w

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            hidden, rest = out[0], out[1:]
            hidden = apply_crossstitch_tokens(hidden, self.crossstitch, self.num_maps, self.h, self.w)
            return (hidden, *rest)
        return apply_crossstitch_tokens(out, self.crossstitch, self.num_maps, self.h, self.w)


def _get_block_list(transformer: nn.Module, block_attr: str) -> nn.ModuleList:
    blocks = getattr(transformer, block_attr, None)
    if blocks is None:
        raise AttributeError(
            f"transformer has no '{block_attr}' — inspect the model and pass block_attr "
            f"(current diffusers Sana uses 'transformer_blocks')"
        )
    return blocks


def inject_crossstitch(
    transformer: nn.Module,
    channels: int,
    num_maps: int = config.NUM_MAPS,
    h: int = config.LATENT_RESOLUTION,
    w: int = config.LATENT_RESOLUTION,
    block_attr: str = "transformer_blocks",
) -> nn.ModuleList:
    """Wrap every block in-place with a CrossStitch. Returns the CrossStitch
    ModuleList so the trainer can optimise/checkpoint just these."""
    blocks = _get_block_list(transformer, block_attr)
    stitches = nn.ModuleList()
    for i, blk in enumerate(blocks):
        cs = CrossStitch(channels, num_maps=num_maps)
        blocks[i] = BlockWithCrossStitch(blk, cs, num_maps, h, w)
        stitches.append(cs)
    return stitches


# --- loaders ------------------------------------------------------------- #
def load_sana_transformer(model_id: str = config.SANA_MODEL_ID, dtype: torch.dtype = torch.bfloat16):
    """Load Sana's linear-attention DiT (the `transformer` subfolder)."""
    from diffusers import SanaTransformer2DModel

    return SanaTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)


# Default LoRA targets for Sana's attention + MLP projections. CONFIRM names
# against the loaded model (print module names) before the real run.
DEFAULT_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "linear_1", "linear_2"]


def apply_lora(
    transformer: nn.Module,
    rank: int = config.DEFAULT_TRAIN.lora_rank,
    alpha: int = config.DEFAULT_TRAIN.lora_alpha,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Attach a LoRA adapter to the DiT (diffusers PEFT integration)."""
    from peft import LoraConfig

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules or DEFAULT_LORA_TARGETS,
    )
    # diffusers models expose add_adapter; fall back to peft.get_peft_model.
    if hasattr(transformer, "add_adapter"):
        transformer.add_adapter(cfg)
        return transformer
    from peft import get_peft_model

    return get_peft_model(transformer, cfg)
