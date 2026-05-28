# HiMat-lite Prototype — Implementation Plan

## Goal

End state of the prototype: **a trained model that takes a text prompt and produces a 1024×1024 SVBRDF (albedo, normal, roughness, metallic, height) on a local GPU.** Output quality good enough to drop into `forfun-graphics`' material system for visual evaluation; not yet shipped to end users, not yet distilled, not yet wrapped in C++.

## Non-goals for this prototype

Pushed to follow-up work after the trained model exists:

- ONNX export, C++ runtime, engine integration (separate phase)
- 4-step distillation for weak end-user devices (separate phase)
- 2K / 4K progressive resolution (paper's main complexity; not needed at proto)
- Wavelet (SWT) loss (mainly buys high-freq detail at 4K)
- Noise-rolling for tileability (workaround for 4K streaks — irrelevant at 1K)
- ControlNet / image-prompt conditioning (text-only for now)
- Commercial-license clean room (Sana NSCL + Gemma terms are fine for personal/research use)

## Architecture (target)

```
prompt ──► Gemma-2-2B (frozen)
              │ text embedding
              ▼
        Linear-attention DiT (Sana 1.6B, LoRA-adapted)
        ┌──── CrossStitch after each block ────┐  (zero-init, dual-branch 1D conv)
        └──────────────────────────────────────┘
              │ latent z  (M × 32 × 32 × C)
              │   M = 3 "maps": albedo, normal, (roughness ⊕ metallic ⊕ height)
              ▼
        DC-AE decoder (Sana's, fine-tuned on SVBRDF)
              │
              ▼
        5 maps at 1024² (after splitting packed scalar maps)
```

The three packed images (M=3) follow the paper §3: scalar r, m, h are concatenated into one 3-channel image, so encoder/decoder always see RGB-shaped tensors.

## Prerequisites

**Two-machine setup** (git is the transport between them):

| Machine | Role | Runs |
|---|---|---|
| Authoring box (RTX 2060, 6 GB, Win10, Py 3.10) | Write code | CPU/small-GPU unit tests, lint, syntax |
| Training box (RTX 4090, 24 GB) | Heavy compute | MatSynth download + preprocess, smoke tests, training, sampling |

- The 6 GB box **cannot train or run full inference** — it's for authoring + the
  CPU-only test suite. All GPU-heavy work (data preprocess, training, sampling)
  happens on the 4090 box.
- **Training box:** RTX 4090 (24 GB). LoRA fine-tune at 1024² with bf16 + grad
  checkpointing, batch 1 + grad accumulation.
- **CUDA:** match the installed PyTorch wheel (authoring box currently has
  torch 2.1.2+cu118; training box should use a recent cu124 build).
- **Disk (training box):** ≥150 GB. MatSynth 4K ≈ 80 GB; Sana ≈ 10 GB; checkpoints ≈ 20 GB.
- **HuggingFace account** (MatSynth + Sana weights).
- **Env caveat (authoring box):** numpy 2.2.6 is installed against a torch built
  for numpy 1.x → a harmless `_ARRAY_API not found` warning. Tensor ops work;
  `torch.from_numpy` (used in preproc) may need `pip install "numpy<2"` if run
  locally. Irrelevant on the training box with a matched env.

## Decisions baked in

| Decision | Choice | Reason |
|---|---|---|
| Base model | Sana 1.6B 1024px (`Efficient-Large-Model/Sana_1600M_1024px`) | HiMat's prior; weights open; 1024-native |
| Text encoder | Gemma-2-2B-IT (Sana default) | Matches Sana training; ~5 GB bf16 manageable |
| Fine-tune strategy | LoRA on DiT (rank 32) + full-train CrossStitch + full-train DC-AE decoder | Fits 24 GB; reversible if quality short |
| Dataset | MatSynth alone for proto (~4,069 mats) | Deschaintre 2018 adds ~2K but is fiddlier to obtain; defer |
| Resolution | 1024² fixed | No progressive ramp at proto |
| Loss | Flow-matching velocity prediction (Sana's loss) + LPIPS for DC-AE | Paper §3 Eq. 2 + Eq. 4 |
| Prompt enrichment | Local Gemma-2-2B with paper's template (paraphrase MatSynth tags) | Cheap, offline, no Gemini API dep |

## Open decisions (need a call before P3)

- **Compute budget envelope:** how many wall-clock days are we willing to burn on training? Affects whether we target 50 k steps (~1 day on 4090) or 200 k steps (~4 days).
- **Experiment tracking:** wandb (cloud) vs aim (local) vs tensorboard (local). Default: tensorboard, swap later.
- **Whether to scrape DTDMat** (Chen et al. 2024) for the richer text descriptions instead of paraphrasing MatSynth tags ourselves.

---

## Phase 0 — Project skeleton + data pipeline

**Status (authoring box): code complete, 22/22 unit tests pass.** Remaining steps
need the dataset + GPU and run on the 4090 box: actually download MatSynth,
preprocess to the 1024² cache, generate prompts, run `himat.data.smoke`. Before
the full preprocess, run `python scripts/download_matsynth.py --inspect` to
confirm the real MatSynth field names against `config.MatSynthSchema` (esp.
height-vs-displacement and where tags live).

**Deliverable:** repo builds, dataset downloaded and cached at 1024², text prompts available per material, `python -m himat.data.smoke` prints one tensor batch.

### Tasks

1. Initialise `himat/` as a Python package alongside the paper PDF.
   - `pyproject.toml` (or `requirements.txt`), `uv venv`, `.gitignore` (data/, checkpoints/, outputs/, .venv/).
   - Dependencies (pin major versions, prefer small set):
     - `torch` (CUDA 12.4 wheel), `torchvision`, `accelerate`
     - `diffusers` (has Sana support since 0.32)
     - `transformers`, `peft` (LoRA)
     - `einops`, `pillow`, `numpy`, `tqdm`, `pyyaml`
     - `lpips` (decoder loss), `torchmetrics` (CLIPScore)
     - dev: `pytest`, `ruff`, `tensorboard`
2. Source layout:
   ```
   himat/
     src/himat/
       config.py
       data/{matsynth.py, preproc.py, prompts.py, dataset.py}
       models/{dc_ae.py, crossstitch.py, dit.py, text_encoder.py, himat.py}
       train/{flow_matching.py, schedules.py, vae_finetune.py, himat_finetune.py}
       eval/{clipscore.py, gallery.py, smoke.py}
       inference/pipeline.py
     scripts/{download_matsynth.py, enrich_prompts.py, train_dc_ae.py, train_himat.py, sample.py}
     tests/
     MILESTONES.md
   ```
3. **Download MatSynth** (`scripts/download_matsynth.py`):
   - `huggingface_hub.snapshot_download("gvecchio/MatSynth", repo_type="dataset")`.
   - Store under `data/matsynth/` (gitignored).
   - Filter to CC0 split first to keep licensing simple; CC-BY is fine but separate flag.
4. **Preprocess at 1024²** (`scripts/preproc.py`):
   - Read raw 4K PNGs/EXRs per material.
   - Resize bicubic to 1024² (height map: nearest or bilinear, decide and document).
   - Pack into per-material tensor file: `{albedo: (3,1024,1024), normal: (3,1024,1024), roughness: (1,1024,1024), metallic: (1,1024,1024), height: (1,1024,1024)}`.
   - Save as `.safetensors` or memory-mapped `.npy` files; cache under `data/cache/matsynth_1024/`.
   - Skip materials missing required maps; log skips.
5. **Prompt generation** (`scripts/enrich_prompts.py`):
   - Pull MatSynth's tag-style labels from its JSON metadata.
   - Run local Gemma-2-2B-IT with a fixed template (paper §4.4 mentions templates for color/texture/roughness/imperfections) to produce one enriched prompt per material.
   - Save `data/cache/prompts.json: {material_id: prompt}`.
   - **Fallback** if Gemma local is slow: hand-craft a simpler template that just concatenates tags into a sentence. Quality lower but unblocks P3.
6. **`MatSynthDataset`** (`data/dataset.py`):
   - PyTorch `Dataset` returning `(maps_dict, prompt_str)`.
   - Augmentation: random horizontal/vertical flip, 90° rotations (paper §5.1). Normal map needs sign flip on flipped axes — *document this carefully, easy to break*.
   - Normalisation: albedo to [-1, 1]; normal kept in [-1, 1] convention; roughness/metallic/height to [-1, 1]. Match what the DC-AE expects.
7. Smoke test: `python -m himat.data.smoke` instantiates dataset, prints one batch shapes + a prompt.

### Validation

- ≥3,000 materials cached at 1024² with all 5 maps present.
- One enriched prompt per cached material.
- Smoke test produces correctly shaped tensors and no NaNs.

### Effort: 3–5 evenings.

### Risks
- MatSynth maps are not perfectly uniform across submitters — some materials are missing height or metallic. Plan: detect and skip; expect ~10–20% loss.
- Gemma-2-2B local inference for ~4 k prompts ≈ a few hours on a 4090 — acceptable.

---

## Phase 1 — DC-AE decoder adaptation for SVBRDF

**Status (authoring box): code complete, 30/30 unit tests pass.** Built:
`models/dc_ae.py` (SVBRDFAutoencoder — folds the M=3 maps into the AE's batch,
decoder-only freeze, scaled-latent helpers for P3), `train/losses.py` (L1+LPIPS,
no GAN), `train/vae_finetune.py` (bf16 + grad-accum loop, periodic val + best-PSNR
checkpoint), `eval/dc_ae_recon.py` (PSNR/SSIM/LPIPS + before/after grid),
`scripts/train_dc_ae.py`. The fold/unfold + freeze plumbing is verified here
against a tiny fake DC-AE. **Remaining (4090 box):** `pip install -e ".[eval]"`,
then `python scripts/train_dc_ae.py` against the real `mit-han-lab/dc-ae-f32c32`
weights once the 1024² cache exists. First real run should confirm the diffusers
`AutoencoderDC` encode/decode return-attr names (`.latent` / `.sample`) — the
wrapper already falls back to tuple indexing if they differ.

**Deliverable:** a fine-tuned DC-AE decoder that reconstructs SVBRDF maps from latents with `rFID` ≤ paper's reported 1.29 (Tab. 2), or at least visibly better than the off-the-shelf Sana DC-AE on normal maps (paper Fig. 3 motivation).

### Tasks

1. **Load Sana's DC-AE** (`models/dc_ae.py`):
   - Pull via `diffusers` or directly from Sana repo (compression F=32, 4K→128, so 1024→32).
   - Freeze encoder for now (paper does too in Eq. 4 phrasing).
2. **Wrap I/O for multi-map**:
   - At training time, run encoder per "map" (M=3: albedo, normal, packed-RMH) independently → 3 latents stacked along the batch dimension.
   - Decoder reconstructs each independently then we split RMH back out.
   - **Important:** the encoder is unchanged from RGB pre-training; the work is on the decoder only.
3. **Loss** (`train/vae_finetune.py`):
   - `L_vae = λ_rec * L1(x̂, x) + λ_lpips * LPIPS(x̂, x)` (paper Eq. 4).
   - LPIPS uses VGG backbone; apply per-channel-group rather than to raw normal/RMH (LPIPS expects RGB-like input — feed each map separately).
   - λ_rec ≈ 1.0, λ_lpips ≈ 0.5 starting point.
   - **No adversarial loss** (paper §3 explicitly omits).
4. **Training loop**:
   - AdamW, lr 1e-5, bf16, batch size 4 per step × grad accum 4 = effective 16.
   - 10–20 k steps. Eval every 1 k steps on a held-out 200-material split.
   - Save best by combined val metric.
5. **Evaluation** (`eval/dc_ae_recon.py`):
   - Compute rFID, PSNR, SSIM, LPIPS on val split.
   - Save before/after grid of normal map reconstructions (paper Fig. 10).
6. Checkpoint: `checkpoints/dc_ae_finetuned.safetensors`.

### Validation

- Reconstruction PSNR on val ≥ 28 dB (paper baseline DC-AE = 28.71; fine-tuned = 30.28).
- Visual check: normal maps no longer show the color-bias the paper highlights in Fig. 3.

### Effort: 4–6 evenings + a half-day training run.

### Risks
- Sana's DC-AE encoder might not handle non-RGB inputs (normal maps with negative components) well even though the decoder gets fine-tuned. Mitigation: if quality is poor, fine-tune the encoder too.
- VAE adversarial-free training can be slow to converge on rich textures — accept and move on if rFID ≈ paper baseline.

---

## Phase 2 — Linear-DiT + CrossStitch wiring

**Status (authoring box): code complete, plumbing unit-tested (13 tests).** Built:
`models/crossstitch.py` (dual-branch module + token routing — shape, identity-at-init,
cross-map coupling, gradients all verified against fakes), `models/dit.py`
(`BlockWithCrossStitch` wrapper + `inject_crossstitch` + Sana loader + LoRA helper —
block-wrapping verified with a fake transformer), `models/text_encoder.py` (frozen
Gemma wrapper), `models/himat.py` (M-axis fold + text-repeat verified via a fake
denoiser; `build_himat` factory). **Remaining (4090 box): the §4 CONFIRM block in
RUNBOOK_4090.md** — Sana's real block-attr, channel dim, LoRA target names, forward
signature, and a zero-init forward-parity smoke. This is the project's highest-risk
step; CrossStitch is a prose-only interpretation to validate by ablation.

**Deliverable:** a HiMat model object that produces correctly shaped output and, with CrossStitch outputs zeroed (initial state), behaves identically to vanilla Sana on RGB inputs. Forward + backward pass on a synthetic batch succeeds end-to-end.

### Tasks

1. **Pull Sana 1.6B 1024px DiT** (`models/dit.py`):
   - Via `diffusers.SanaTransformer2DModel` (uses `Efficient-Large-Model/Sana_1600M_1024px`).
   - Inspect block structure — paper §5.1 says 20 blocks; confirm.
2. **Implement `CrossStitch`** (`models/crossstitch.py`):
   - Input: `(M, H, W, C)` latent features. Paper's einops sequence:
     - `m h w c -> (h w) c m`
     - 1D conv across the `m` axis
     - `(h w) c m -> m h w c`
   - Two parallel branches (paper §4.3):
     - **Local branch:** depthwise-separable: spatial 3×3 then pointwise 1×1.
       - "Spatial" here = over the `(h w)` tokens for each (c, m) pair. Implement as a depthwise 1D conv after a permute that puts spatial as the conv axis. Re-read paper §4.3 carefully — the local branch mixes neighbouring map-pixels at the same `(h, w)` *and* nearby `(h, w)` for a given `m`.
     - **Global branch:** avg-pool over spatial → 1×1 conv → GELU → broadcast back.
   - Output: residual add. **Zero-init the final projection layer of each branch** so initial output is exactly the residual (paper §4.3).
3. **Inject CrossStitch into the DiT** (`models/himat.py`):
   - After each linear-attention layer, before the MLP. Wrap each Sana block; route the M-axis through CrossStitch then restore.
   - The M-axis is **not** present in vanilla Sana — we add it: input to the HiMat DiT is `(B, M, C, h, w)` where `M=3` is fixed and the batch+M get flattened for attention but un-flattened for CrossStitch. Document this precisely; it's the trickiest implementation detail.
4. **LoRA wrap** the DiT's attention QKV + MLP gate/up projections (`peft.LoraConfig(r=32, lora_alpha=32, target_modules=...)`). Verify trainable param count drops to ~50–80 M (vs 1.6 B total).
5. **Text-encoder integration** (`models/text_encoder.py`):
   - Wrap Gemma-2-2B-IT (frozen, bf16).
   - System-prompt template from paper §4.4 wrapped around the runtime prompt.
6. **End-to-end forward sanity** (`tests/test_himat_forward.py`):
   - With CrossStitch zero-init: HiMat output ≈ Sana output on equivalent single-map input.
   - With LoRA at init (zero): HiMat output unchanged.
   - Backward pass produces gradients only on LoRA params + CrossStitch + DC-AE decoder.

### Validation

- Forward latency at 1024² on the 4090: under 1.5 s per step (paper Tab. 3: 0.30 s/step at 1024². We're slower because of CrossStitch overhead; should not be 5× worse).
- Forward + backward fits in 24 GB at batch 1 with grad checkpointing.
- Trainable param count printed and sanity-checked.

### Effort: 5–7 evenings.

### Risks (highest in the project)
- **CrossStitch is the novel module and the paper's description is prose-level.** Section §4.3 + Eq. 7 are the only spec. There will be a couple of guess-and-iterate rounds. Plan for two end-to-end forward sanity sessions, separated by careful re-reading.
- **diffusers Sana wrapping** might not expose per-block hooks cleanly. Fallback: copy Sana's block class verbatim and modify in-place.

---

## Phase 3 — Training loop

**Status (authoring box): code complete; flow-matching math unit-tested (7 tests,
incl. oracle→zero-loss).** Built: `train/flow_matching.py` (rectified-flow velocity
loss + logit-normal t sampling + add_noise — verified), `train/himat_finetune.py`
(staged loop: LoRA+CrossStitch+decoder param groups, EMA, cosine LR, val sampling,
checkpoint), `scripts/train_himat.py`. The loop itself is **untested** (needs the
models) — runs on the 4090 box per RUNBOOK §5 (Stage A smoke → Stage B main).

**Deliverable:** a HiMat checkpoint that, when sampled from a held-out prompt, produces recognisable, on-topic SVBRDFs ("dark cracked leather", "polished marble", "rusty metal mesh"). Quality target: comparable to ReflectanceFusion or MatFuse — *not* full paper-quality, which used multi-GPU training.

### Tasks

1. **Diffusion / flow-matching loss** (`train/flow_matching.py`):
   - Velocity prediction per paper §3 Eq. 2: `v = ε − z_0`.
   - Sample `t ~ U(0,1)` per batch element; logit-normal sampling is the Sana default and known to help.
   - MSE on velocity.
2. **Training loop** (`train/himat_finetune.py`):
   - Trainable params: DiT-LoRA + CrossStitch + DC-AE decoder.
   - Frozen: DC-AE encoder, Gemma-2-2B text encoder, vanilla DiT base weights.
   - Optimiser: AdamW, lr 5e-6 (LoRA + CrossStitch), 1e-5 (DC-AE dec, slightly higher because trained from scratch effectively).
   - Cosine schedule with 1 k warmup steps.
   - bf16 mixed precision, gradient checkpointing on DiT blocks.
   - EMA of trainable params (decay 0.999) — helps a lot at low compute.
   - Save checkpoint every 2 k steps; keep last 5 + best-by-val.
   - Resume-on-restart.
3. **Validation during training**:
   - Held-out set of 64 prompts (paper-style descriptions).
   - Every 2 k steps: sample 8 fixed prompt+seed combinations at 20 inference steps; save as image grid to tensorboard.
   - Compute val loss on a 200-material held-out split.
4. **Run schedule**:
   - **Stage A (smoke):** 1 k steps, batch 1, no LoRA, all params frozen except CrossStitch. Just verify the loop runs end-to-end and the model isn't outputting noise after a tiny tune.
   - **Stage B (main):** 50 k steps with full setup. ~1–1.5 days on the 4090.
   - **Stage C (optional, decide after B):** another 50–100 k if quality looks promising.
5. Final ckpt: `checkpoints/himat_lite_1024_stageB.safetensors` (LoRA + CrossStitch + DC-AE dec only — small file, ~300 MB).

### Validation

- Sampled outputs from 8 fixed held-out prompts at end of Stage B are **clearly material-like** (not noise, not collapsed to a single texture). Human eye check.
- Val loss strictly decreases over training (sanity).
- Inter-map alignment visible: a crack in albedo is also a crease in normal, etc. (This is what CrossStitch is supposed to give.)

### Effort: 1 week scaffolding + 1–4 days of GPU train (Stage A < 1h, Stage B ~36h, Stage C optional).

### Risks
- **Training instability with linear attention + new module.** Mitigations: keep lr low, warmup long, EMA, snapshot frequently, manual loss-curve watch.
- **Mode collapse on a 4k-material dataset.** Likely if quality is poor; consider adding Deschaintre 2018 as a stretch goal.
- **OOM at batch >1.** Already mitigated by checkpointing; if still hits, drop CrossStitch dim or LoRA rank.

---

## Phase 4 — Eval + sample gallery

**Status (authoring box): code complete (untested — needs the trained model).**
Built: `inference/pipeline.py` (rectified-flow Euler sampler + CFG, decode_scaled,
save PNGs), `scripts/sample.py`, `scripts/gallery.py`, `eval/clipscore.py`
(open_clip; albedo proxy until forfun-graphics rendering is wired). Runs on the
4090 box per RUNBOOK §6.

**Deliverable:** `scripts/sample.py "your prompt"` produces a 5-map SVBRDF saved as PNGs; a generated gallery of ~50 prompts for review; metric numbers logged.

### Tasks

1. **End-to-end pipeline** (`inference/pipeline.py`):
   - Tokenise prompt → Gemma embedding → DiT denoise loop (20 steps default, EulerDiscrete or paper's matching scheduler) → DC-AE decode → 5 maps.
   - Save outputs as `albedo.png`, `normal.png`, `roughness.png`, `metallic.png`, `height.png` under `outputs/<prompt-slug>/`.
2. **Sample script** (`scripts/sample.py`): CLI with `--prompt`, `--seed`, `--steps`.
3. **Gallery generation** (`scripts/gallery.py`):
   - 50 held-out prompts.
   - Render each material under a fixed environment lighting (use existing `forfun-graphics`' IBL setup or a quick offline mitsuba/blender bake; could also be done by hand-loading in helmet_demo).
   - Compose into a contact-sheet image.
4. **Metrics** (`eval/clipscore.py`):
   - CLIPScore on rendered images vs prompt text. Use OpenCLIP ViT-L.
   - Report mean + stdev across gallery.

### Validation

- 50/50 gallery samples produce valid non-noise outputs.
- CLIPScore mean ≥ 25 (paper reports 30.27; we won't match that with limited compute but anything ≥ 25 means the model learnt the text→material mapping).
- Visual sanity: at least 30/50 prompts produce a material that a human would call "on topic".

### Effort: 3–5 evenings.

---

## What "prototype done" looks like

A short demo recording: text prompt typed in CLI → 5 PNGs land on disk in ~10 s on the 4090 → load them as a material in `helmet_demo` → see a recognisably-on-topic shaded sphere.

That's the gate to the *next* phase (ONNX export + distillation + cross-platform deploy + engine integration), which gets its own MILESTONES update.

---

## Risk register (top-level)

| Risk | Severity | Mitigation |
|---|---|---|
| CrossStitch implementation mismatches the paper's intent | High | Two careful re-read passes; sanity test with zero-init; ablation runs to confirm it actually helps |
| Compute budget overruns | Medium | Set 50k-step ceiling for Stage B; promote to C only if Stage B shows clear progress |
| MatSynth alone is too small to learn diverse materials | Medium | Pull Deschaintre 2018 as a stretch dataset; fall back to fewer material categories |
| Sana DC-AE encoder doesn't handle normal/RMH well even with decoder fine-tune | Medium | Fine-tune encoder too (more compute, less surprising) |
| diffusers' Sana wrapping doesn't expose per-block injection points | Low | Copy-paste Sana's block class and edit |
| License (NSCL + Gemma) blocks future commercial use | Low for prototype | Document; revisit if commercial path opens |

## Effort summary

| Phase | Wall-clock (evening cadence) |
|---|---|
| P0 — repo + data | 3–5 evenings |
| P1 — DC-AE decoder | 4–6 evenings + ½ day train |
| P2 — DiT + CrossStitch | 5–7 evenings |
| P3 — training loop | 1 week + 1–4 days train |
| P4 — eval + gallery | 3–5 evenings |
| **Total** | **5–7 weeks** to a usable prototype model |

## After the prototype (not in this plan, but signposted)

- **Distillation** (Sana-Sprint recipe → 4-step student).
- **ONNX export** for both DiT and DC-AE decoder; per-platform packaging.
- **`forfun_himat` C++ library** with ORT CUDA EP (Windows/Nvidia) + CoreML EP (Apple Silicon).
- **Engine integration**: prompt-driven material slot in `helmet_demo`, then in `alpine-sun` as a stretch.
- **Optional**: re-add wavelet loss + noise rolling + 2K/4K progressive for a higher-quality v2.

## References

- Paper: `cgf70343.pdf` (HiMat, EG 2026) — primary spec.
- Sana 1.6B 1024px: <https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px>
- Sana code: <https://github.com/NVlabs/Sana>
- MatSynth: <https://huggingface.co/datasets/gvecchio/MatSynth>
- Deep-research file: `deep-research-report.md` (use with caution — has some inaccuracies, e.g. the Mermaid diagram reverses encoder/decoder order)
