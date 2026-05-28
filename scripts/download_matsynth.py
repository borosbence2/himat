"""Download the MatSynth dataset from the Hugging Face Hub.

    python scripts/download_matsynth.py [--cc0-only] [--inspect]

MatSynth is large (~80 GB at 4K). It is a streaming-friendly HF dataset; we use
the `datasets` library so preprocessing can iterate without holding it all in RAM.

--inspect prints the schema of the first record so config.MatSynthSchema can be
verified/corrected against the real field names before a full preprocess run.

Run on the 4090 box (disk + bandwidth). Requires `huggingface-cli login`.
"""

from __future__ import annotations

import argparse

from himat import config


def inspect_schema() -> None:
    from datasets import load_dataset

    print("loading one record (streaming) to inspect schema...")
    ds = load_dataset(config.MATSYNTH_REPO_ID, split="train", streaming=True)
    rec = next(iter(ds))
    print("\nTop-level fields:")
    for k, v in rec.items():
        t = type(v).__name__
        extra = ""
        if isinstance(v, dict):
            extra = f" keys={list(v.keys())}"
        print(f"  {k}: {t}{extra}")
    print(
        "\nCompare against config.MatSynthSchema and fix names there if they "
        "differ (esp. height vs displacement, tags location)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print schema of first record and exit")
    ap.add_argument("--cc0-only", action="store_true", help="(reserved) filter to CC0 license split")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.inspect:
        inspect_schema()
        return

    from huggingface_hub import snapshot_download

    print(f"downloading {config.MATSYNTH_REPO_ID} → {config.RAW_MATSYNTH_DIR}")
    print("this is large (~80 GB) and will take a while...")
    snapshot_download(
        repo_id=config.MATSYNTH_REPO_ID,
        repo_type="dataset",
        local_dir=str(config.RAW_MATSYNTH_DIR),
    )
    print("done. next: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
