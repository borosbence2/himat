"""Generate one text prompt per cached material → config.PROMPTS_PATH.

    python scripts/enrich_prompts.py [--mode llm|template] [--limit N]

  template : pure-Python, stitches MatSynth tags into a sentence. No model.
  llm      : paraphrases tags with a local Gemma-2-2B-IT (paper section 4.4).
             Needs a GPU; run on the 4090 box.

Streams the HF dataset for metadata (tags/category) keyed by the same material
id the preprocessor used, so prompts line up with the cached tensors.
"""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from himat import config
from himat.data import prompts as P


def _material_id(rec: dict, fallback: int) -> str:
    name = rec.get(config.MATSYNTH.name)
    if isinstance(name, str) and name.strip():
        return name.strip().replace("/", "_").replace(" ", "_")
    return f"mat_{fallback:06d}"


def _load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.GEMMA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.GEMMA_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tok, model


def _llm_generate(tok, model, instruction: str) -> str:
    import torch

    msgs = [{"role": "user", "content": instruction}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return " ".join(text.strip().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm", "template"], default="template")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from datasets import load_dataset

    config.ensure_dirs()
    ds = load_dataset(config.MATSYNTH_REPO_ID, split=args.split, streaming=True)

    tok = model = None
    if args.mode == "llm":
        print("loading Gemma-2-2B-IT...")
        tok, model = _load_llm()

    out: dict[str, str] = {}
    if config.PROMPTS_PATH.exists():
        out = json.loads(config.PROMPTS_PATH.read_text(encoding="utf-8"))

    for i, rec in enumerate(tqdm(ds, desc=f"prompts({args.mode})")):
        if args.limit and len(out) >= args.limit:
            break
        mid = _material_id(rec, i)
        if mid in out:
            continue
        category, tags = P.extract_tags(rec)
        if args.mode == "template":
            out[mid] = P.enrich_template(category, tags)
        else:
            out[mid] = _llm_generate(tok, model, P.build_llm_prompt(category, tags))

    config.PROMPTS_PATH.write_text(json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} prompts → {config.PROMPTS_PATH}")


if __name__ == "__main__":
    main()
