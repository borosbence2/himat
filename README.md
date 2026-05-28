# HiMat-lite

![CI](https://github.com/borosbence2/himat/actions/workflows/ci.yml/badge.svg)

Prototype reimplementation of **HiMat: DiT-based Ultra-High Resolution SVBRDF
Generation** (Wang et al., Eurographics 2026), scoped down to a **1024×1024,
text→SVBRDF** generator. See [MILESTONES.md](MILESTONES.md) for the full plan.
When you have an RTX 4090 box, follow [RUNBOOK_4090.md](RUNBOOK_4090.md) to take
the code from "written" to "trained model".

Goal of the prototype: a trained model that turns a text prompt into a 5-map
SVBRDF (albedo, normal, roughness, metallic, height) on a local GPU, good
enough to load into the `forfun-graphics` material system for visual review.

## Two-machine workflow

This repo is developed across two machines; git is the transport.

| Machine | Role | What runs here |
|---|---|---|
| Dev box (RTX 2060, 6 GB) | Authoring | Write code, run CPU-only unit tests, lint |
| Training box (RTX 4090, 24 GB) | Heavy compute | MatSynth download + preprocess, smoke tests, training, sampling |

The package itself is plain portable Python — nothing here is dev-box specific.
Anything that needs a real GPU or the ~80 GB dataset runs on the 4090 box.

## Setup (on the 4090 box)

```bash
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate
# Install torch with the CUDA wheel matching the box first, e.g. CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[eval,dev]"
huggingface-cli login    # needed for MatSynth + Sana weights
```

On the dev box you can `pip install -e ".[dev]"` with a CPU torch build to run
the shape/logic unit tests without a GPU.

## Data pipeline

```bash
python scripts/download_matsynth.py        # → data/matsynth/  (~80 GB, 4K)
python scripts/preprocess.py               # → data/cache/matsynth_1024/  (1024², packed)
python scripts/enrich_prompts.py           # → data/cache/prompts.json
python -m himat.data.smoke                 # sanity: prints one batch
```

## Layout

```
src/himat/
  config.py            # paths, map specs, hyperparameters
  data/                # dataset, preprocessing, prompt enrichment
  models/              # dc_ae, crossstitch, dit, text_encoder, himat (P1-P2)
  train/               # flow-matching + fine-tune loops (P3)
  eval/                # clipscore, gallery (P4)
  inference/           # text→SVBRDF pipeline (P4)
scripts/               # CLI entrypoints
tests/                 # CPU-only unit tests
```

## Testing

CPU-only, no GPU or dataset needed. `conftest.py` puts `src/` on the path, so
after `pip install pytest torch numpy safetensors`:

```bash
pytest -q          # 58 pass; 5 preproc image tests skip unless the torch/numpy
                   # bridge is healthy (they run in CI and on the 4090 box)
ruff check .
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs ruff + pytest on
every push with a CPU torch build — it deliberately skips the heavy model stack,
which only the 4090 box installs.
