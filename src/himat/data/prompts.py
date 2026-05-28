"""Prompt enrichment: MatSynth ships terse tag-style labels; the paper (section
4.4) expands these into richer descriptions with an LLM using templates that call
out colour, texture, roughness, and surface imperfections.

Two paths here:
  * enrich_with_llm(...)  — uses a local Gemma-2-2B-IT to paraphrase tags into a
    fluent description (matches the paper; needs a GPU, runs on the 4090 box).
  * enrich_template(...)  — pure-Python fallback that stitches tags into a
    sentence. Lower quality but needs no model; unblocks the data pipeline.

Both produce one string per material, written to PROMPTS_PATH by
scripts/enrich_prompts.py.
"""

from __future__ import annotations

# Instruction template handed to the LLM. The paper keeps the exact templates in
# its supplementary material (not public), so this is our reconstruction of the
# described intent: describe a tileable PBR material for a generative model,
# emphasising the four attributes the paper names.
LLM_TEMPLATE = (
    "You are describing a tileable PBR surface material for a text-to-material "
    "generator. Given the material category and tags, write ONE concise sentence "
    "(max 30 words) describing its appearance. Explicitly mention: dominant "
    "colour, surface texture/pattern, roughness (matte vs glossy), and any "
    "imperfections (scratches, cracks, wear). Do not mention lighting or cameras.\n"
    "Category: {category}\n"
    "Tags: {tags}\n"
    "Description:"
)

# System-level prompt prepended at inference (paper section 4.4 / Lumina-Image
# 2.0 idea: a fixed system prompt lifts quality without architecture changes).
SYSTEM_PROMPT = (
    "Generate a high-resolution, tileable, physically-based surface material "
    "with consistent albedo, normal, roughness, metallic, and height maps."
)


def enrich_template(category: str, tags: list[str] | None) -> str:
    """Pure-Python fallback prompt. Deterministic, no model required."""
    tags = tags or []
    cat = (category or "material").strip().lower()
    tag_str = ", ".join(t.strip().lower() for t in tags if t and t.strip())
    if tag_str:
        return f"a {cat} material, {tag_str}, tileable PBR surface"
    return f"a {cat} material, tileable PBR surface"


def build_llm_prompt(category: str, tags: list[str] | None) -> str:
    """Render the instruction sent to the LLM for one material."""
    tag_str = ", ".join(t for t in (tags or []) if t) or "(none)"
    return LLM_TEMPLATE.format(category=category or "unknown", tags=tag_str)


def wrap_for_inference(prompt: str) -> str:
    """Prepend the system prompt at generation time."""
    return f"{SYSTEM_PROMPT}\n{prompt.strip()}"


def extract_tags(record: dict) -> tuple[str, list[str]]:
    """Pull (category, tags) from a MatSynth HF record, tolerant of schema drift.

    Tags may live directly on the record or nested under `metadata`. Returns a
    best-effort (category, tags) — empty values are fine, the template handles it.
    """
    from himat import config

    s = config.MATSYNTH
    category = str(record.get(s.category) or "").strip()

    tags: list[str] = []
    meta = record.get(s.metadata)
    if isinstance(meta, dict):
        raw = meta.get(s.tags_key) or meta.get("keywords") or []
        if isinstance(raw, str):
            tags = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
        elif isinstance(raw, (list, tuple)):
            tags = [str(t).strip() for t in raw if str(t).strip()]
        # A free-text description, if present, is the richest signal — use it as a tag.
        desc = meta.get(s.description_key)
        if isinstance(desc, str) and desc.strip():
            tags.insert(0, desc.strip())
    # Top-level tags as a fallback.
    if not tags:
        raw = record.get(s.tags_key)
        if isinstance(raw, (list, tuple)):
            tags = [str(t).strip() for t in raw if str(t).strip()]
    return category, tags
