"""CLIPScore for generated materials (Phase 4). UNTESTED here — needs open_clip
+ the trained model.

Proper evaluation (paper sec 5.2) renders the SVBRDF under environment lighting
and scores the *render* against the prompt. That rendering belongs to the
forfun-graphics integration; until then, scoring the albedo map is a cheap proxy
that still tracks text-material alignment. compute_clipscore works on any image
tensor, so swap in rendered images once available.
"""

from __future__ import annotations

import torch


class CLIPScorer:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: torch.device | None = None):
        import open_clip

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def score(self, image: torch.Tensor, prompt: str) -> float:
        """image: (3, H, W) in [0,1]. Returns cosine-sim * 100 (CLIPScore convention)."""
        from torchvision.transforms.functional import resize

        img = resize(image, [224, 224]).unsqueeze(0).to(self.device)
        # CLIP normalisation
        mean = torch.tensor([0.4815, 0.4578, 0.4082], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2686, 0.2613, 0.2758], device=self.device).view(1, 3, 1, 1)
        img = (img - mean) / std
        text = self.tokenizer([prompt]).to(self.device)

        img_feat = self.model.encode_image(img)
        txt_feat = self.model.encode_text(text)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return (100.0 * (img_feat * txt_feat).sum(dim=-1)).item()
