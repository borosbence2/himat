"""Frozen Gemma-2-2B-IT text encoder (Sana's text encoder).

Returns last-hidden-state embeddings + attention mask for the DiT's cross
attention. Gemma is decoder-only, so we take hidden states (not logits). The
paper (sec 4.4) also wraps prompts in a system-level template at inference —
applied here via prompts.wrap_for_inference before encoding when system=True.

~5 GB in bf16; runs on the 4090 box. Frozen (no grad).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from himat import config
from himat.data import prompts as P


class GemmaTextEncoder(nn.Module):
    def __init__(self, model_id: str = config.GEMMA_MODEL_ID, dtype: torch.dtype = torch.bfloat16, max_len: int = 300):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_len = max_len

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, prompts: list[str], system: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """list[str] -> (embeddings (B, L, D), attention_mask (B, L))."""
        if system:
            prompts = [P.wrap_for_inference(p) for p in prompts]
        device = next(self.model.parameters()).device
        tok = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)
        out = self.model(**tok)
        return out.last_hidden_state, tok.attention_mask
