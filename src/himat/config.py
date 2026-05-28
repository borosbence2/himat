"""Central configuration: paths, SVBRDF map specs, and hyperparameters.

Everything that the rest of the package needs to agree on lives here so the data
pipeline, models, and training loop can't drift apart on channel counts or
normalisation conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths. Resolved relative to the repo root so the code is machine-agnostic.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_MATSYNTH_DIR = DATA_DIR / "matsynth"
CACHE_DIR = DATA_DIR / "cache"
CACHE_1024_DIR = CACHE_DIR / "matsynth_1024"
PROMPTS_PATH = CACHE_DIR / "prompts.json"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
OUTPUTS_DIR = REPO_ROOT / "outputs"
RUNS_DIR = REPO_ROOT / "runs"


# --------------------------------------------------------------------------- #
# SVBRDF map layout.
#
# The model works on M=3 "maps", each a 3-channel image (paper section 3): the
# three scalar maps (roughness, metallic, height) are packed into one RGB-shaped
# tensor so the DC-AE, pretrained on RGB, always sees 3-channel inputs.
#
#   map 0: albedo        (R, G, B)
#   map 1: normal        (Nx, Ny, Nz)   OpenGL convention (Y+ up), MatSynth native
#   map 2: packed-RMH    (roughness, metallic, height)
#
# All maps are stored in the cache as float32 in [-1, 1].
# --------------------------------------------------------------------------- #
NUM_MAPS = 3  # M, the CrossStitch map axis
MAP_CHANNELS = 3  # every map is a 3-channel image
RESOLUTION = 1024

# Canonical map names → how many real scalar channels each contributes.
# (albedo and normal are genuinely 3-channel; the packed map carries 3 scalars.)
SCALAR_MAPS = ("roughness", "metallic", "height")  # packed, in this order, into map 2
VECTOR_MAPS = ("albedo", "normal")  # genuinely 3-channel maps (maps 0 and 1)

# The five output maps a caller ultimately wants, with channel counts.
OUTPUT_MAPS: dict[str, int] = {
    "albedo": 3,
    "normal": 3,
    "roughness": 1,
    "metallic": 1,
    "height": 1,
}


# --------------------------------------------------------------------------- #
# MatSynth field names. These are the column names we expect from the HF dataset
# `gvecchio/MatSynth`. VERIFY against the real schema on first download (see
# scripts/download_matsynth.py) and fix here if they differ — this is the single
# place that knows the dataset's vocabulary.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatSynthSchema:
    # Map columns (PIL images in the HF dataset).
    basecolor: str = "basecolor"
    normal: str = "normal"
    roughness: str = "roughness"
    metallic: str = "metallic"
    height: str = "height"  # MatSynth calls displacement "height"
    # Metadata used for prompts / filtering.
    name: str = "name"
    category: str = "category"
    metadata: str = "metadata"  # nested dict with tags, description, license, ...
    # Field inside `metadata` holding the human description / tags, if present.
    description_key: str = "description"
    tags_key: str = "tags"
    license_key: str = "license"


MATSYNTH = MatSynthSchema()


# --------------------------------------------------------------------------- #
# Latent geometry. DC-AE compresses by F=32 (paper section 3): 1024 -> 32.
# --------------------------------------------------------------------------- #
DC_AE_COMPRESSION = 32
LATENT_RESOLUTION = RESOLUTION // DC_AE_COMPRESSION  # 32
LATENT_CHANNELS = 32  # DC-AE f32c32: 32x spatial compression, 32 latent channels


# --------------------------------------------------------------------------- #
# Base-model identifiers on the Hub.
# --------------------------------------------------------------------------- #
SANA_MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
# Standalone DC-AE (f32c32) used by Sana — same encoder/decoder, easier to load
# in isolation than fishing it out of the full pipeline.
DC_AE_MODEL_ID = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
GEMMA_MODEL_ID = "google/gemma-2-2b-it"
MATSYNTH_REPO_ID = "gvecchio/MatSynth"


# --------------------------------------------------------------------------- #
# Training hyperparameters (P3 defaults; override per run via CLI/yaml).
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    resolution: int = RESOLUTION
    batch_size: int = 1            # per-step; 24 GB at 1024 with grad checkpointing
    grad_accum: int = 16           # effective batch 16
    lr_lora: float = 5e-6          # DiT LoRA + CrossStitch
    lr_decoder: float = 1e-5       # DC-AE decoder (effectively trained from scratch)
    warmup_steps: int = 1000
    max_steps: int = 50_000        # Stage B ceiling
    ema_decay: float = 0.999
    lora_rank: int = 32
    lora_alpha: int = 32
    val_every: int = 2000
    ckpt_every: int = 2000
    val_prompts: int = 64
    inference_steps: int = 20      # paper default
    seed: int = 0
    # Augmentation toggles.
    aug_flip: bool = True
    aug_rot90: bool = True
    # Holdout split for validation (by material, deterministic).
    val_fraction: float = 0.03


@dataclass
class VAEFinetuneConfig:
    resolution: int = RESOLUTION
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-5
    max_steps: int = 15_000
    lambda_rec: float = 1.0
    lambda_lpips: float = 0.5
    val_every: int = 1000
    ckpt_every: int = 1000
    seed: int = 0
    finetune_encoder: bool = False  # decoder-only by default (paper Eq. 4)


DEFAULT_TRAIN = TrainConfig()
DEFAULT_VAE = VAEFinetuneConfig()


def ensure_dirs() -> None:
    """Create the output/cache directories if missing (safe to call repeatedly)."""
    for d in (DATA_DIR, CACHE_DIR, CACHE_1024_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
