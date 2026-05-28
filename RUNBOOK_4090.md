# RUNBOOK — what to run on the RTX 4090 box

Everything in the repo that needs a real GPU, the MatSynth dataset, or the
Sana/Gemma weights is deferred to here. The code is written and the machine-
independent logic is unit-tested (50/50 on the authoring box). This is the
ordered checklist to take it from "code" to "trained model that turns text into
textures".

Work top to bottom. Each `CONFIRM` is a Sana/diffusers-specific assumption that
the authoring box couldn't verify — check it once, fix the one indicated spot if
it's wrong, move on.

---

## 0. Environment

- [ ] `git clone https://github.com/borosbence2/himat && cd himat`
- [ ] `python -m venv .venv && . .venv/Scripts/activate` (or `source .venv/bin/activate`)
- [ ] Install torch with the box's CUDA wheel, e.g. cu124:
      `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- [ ] `pip install -e ".[eval,dev]"`
- [ ] `huggingface-cli login` (MatSynth + Sana + Gemma access; accept Gemma + Sana licenses on the Hub first)
- [ ] `pytest -q` — expect 50 passing (sanity that the env is wired)

## 1. Verify the MatSynth schema  ← do before downloading 80 GB

- [ ] `python scripts/download_matsynth.py --inspect`
- [ ] **CONFIRM** the printed field names against `src/himat/config.py::MatSynthSchema`:
      especially `height` vs `displacement`, and where tags/description live
      (top-level vs nested `metadata`). Fix `MatSynthSchema` if they differ.

## 2. Data pipeline

- [ ] `python scripts/download_matsynth.py`  (~80 GB → `data/matsynth/`)
- [ ] `python scripts/preprocess.py`  (→ `data/cache/matsynth_1024/`, one `.safetensors` per material)
      - Start with `--limit 50` to verify it writes valid tensors, then run full.
      - Expect ~10–20% skipped (missing maps). ≥3,000 cached is the bar.
- [ ] `python scripts/enrich_prompts.py --mode template`  (fast, no model)
      - Optional richer prompts: `--mode llm` (downloads Gemma-2-2B, paper sec 4.4).
- [ ] `python -m himat.data.smoke`  → prints batch shape `(B,3,3,1024,1024)`, a prompt, `smoke OK`

## 3. Phase 1 — DC-AE decoder fine-tune

- [ ] First run will download `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`.
- [ ] **CONFIRM** diffusers `AutoencoderDC.encode().latent` / `.decode().sample`
      attribute names (the wrapper in `models/dc_ae.py` already falls back to
      tuple indexing, but verify the latent shape is `(B,32,32,32)` for a 1024 input).
- [ ] `python scripts/train_dc_ae.py --max-steps 15000`
      - Watch `runs/` in tensorboard; val PSNR should climb toward ~30 (paper Tab. 2: 30.28).
      - Eyeball `outputs/dc_ae_recon_*.png`: normal maps should lose the colour bias.
- [ ] Output: `checkpoints/dc_ae_finetuned.safetensors`
- [ ] If normal-map reconstruction stays poor, re-run with `--finetune-encoder`.

## 4. Phase 2 — confirm the DiT integration  ← the highest-risk phase

Run this throwaway inspection once to lock the Sana-specific bits:

```python
import torch
from himat.models.dit import load_sana_transformer
m = load_sana_transformer(dtype=torch.bfloat16)
print(type(m).__name__)
print("config:", m.config)                       # CONFIRM channel dim derivation in build_himat
print("block attr 'transformer_blocks':", hasattr(m, "transformer_blocks"))
names = {n.split('.')[-1] for n,_ in m.named_modules()}
print("for LoRA targets, look for:", sorted(n for n in names if any(k in n for k in ("to_q","to_k","to_v","linear","proj","ff"))))
```

- [ ] **CONFIRM** block-list attribute is `transformer_blocks` (else pass `block_attr=` to `inject_crossstitch`).
- [ ] **CONFIRM** the hidden channel dim used in `build_himat` (`models/himat.py`)
      matches the DiT's token channel dimension — CrossStitch is built with it.
- [ ] **CONFIRM** `DEFAULT_LORA_TARGETS` in `models/dit.py` are real module names; adjust.
- [ ] **CONFIRM** Sana's forward kwargs + output attr in `build_himat`'s `denoiser`
      (`hidden_states`, `timestep`, `encoder_hidden_states`, `encoder_attention_mask`,
      `.sample`). Sana also may need a spatial/grid arg — check its `forward` signature.
- [ ] Tiny forward-parity smoke (build the bundle, run one batch through
      `bundle.himat(noisy, t, text_emb, mask)`; confirm output shape
      `(B,3,32,32,32)` and no error). With CrossStitch zero-init it should match
      vanilla Sana on per-map inputs.

## 5. Phase 3 — HiMat training

- [ ] **Stage A (smoke):** `python scripts/train_himat.py --max-steps 1000 --run-name stageA`
      - Goal: loop runs end-to-end, loss is finite and trends down, a val grid is written.
- [ ] **Stage B (main):** `python scripts/train_himat.py --max-steps 50000 --run-name stageB`
      - ~1–1.5 days on the 4090. Watch `runs/stageB` loss + `outputs/stageB_val_*.png`.
      - Decision gate (see below): only go to Stage C (more steps) if Stage B shows clear progress.
- [ ] Output: `checkpoints/stageB_final.safetensors` (LoRA + CrossStitch + DC-AE decoder; small)
- [ ] If OOM at batch 1: lower `--lora-rank`, confirm grad checkpointing, or drop CrossStitch channel dim.

## 6. Phase 4 — sample + evaluate

- [ ] `python scripts/sample.py --prompt "weathered red brick wall" --ckpt checkpoints/stageB_final.safetensors`
      → 5 PNGs in `outputs/weathered-red-brick-wall/`
- [ ] `python scripts/gallery.py --ckpt checkpoints/stageB_final.safetensors --clipscore`
      → contact sheet + mean albedo CLIPScore (≥25 means text→material was learned)
- [ ] Load a generated material into forfun-graphics' M8 material system for a real shaded look.

**Prototype is "done"** when a typed prompt yields 5 maps that load as a
recognisably on-topic material in the engine.

---

## Open decisions to make here (none block starting)

- **Compute budget:** how many days for Stage B? 50k (~1 day) vs 100–200k (~2–4 days).
- **Experiment tracker:** default tensorboard (local). Swap to wandb/aim if wanted.
- **DTDMat:** scrape it for richer text descriptions, or stick with template/Gemma prompts?
- **Add Deschaintre 2018** (~2k more materials) if MatSynth-only diversity is weak.

## All the CONFIRM items in one place

| # | Assumption | File to fix |
|---|---|---|
| 1 | MatSynth field names (height/displacement, tags location) | `config.py::MatSynthSchema` |
| 2 | `AutoencoderDC` encode/decode attrs + latent shape (B,32,32,32) | `models/dc_ae.py` (has fallback) |
| 3 | DiT block-list attr = `transformer_blocks` | pass `block_attr=` to `inject_crossstitch` |
| 4 | DiT hidden channel dim | `models/himat.py::build_himat` |
| 5 | LoRA target module names | `models/dit.py::DEFAULT_LORA_TARGETS` |
| 6 | Sana forward kwargs + output attr (+ any grid arg) | `models/himat.py::build_himat` denoiser |
| 7 | Latent scaling (encode_scaled/decode_scaled vs raw) is consistent train↔sample | already consistent; verify visually |

## What's already verified (authoring box, 50/50 tests)

- Map pack/unpack, normal-aware augmentation (incl. rot90 handedness via covariance test)
- Prompt enrichment + tag extraction
- DC-AE multi-map fold/unfold, decoder-only freeze, scaled round-trip, gradient flow
- CrossStitch: shape, identity-at-init, cross-map coupling, token routing, gradients
- DiT block-wrapping (CrossStitch injection) + HiMat M-axis fold + text repeat
- Flow-matching: endpoints, midpoint, velocity target, 5D broadcast, oracle→zero-loss
