from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemical_embedding_pipeline.generate_transformer_embeddings import MODEL_IDS
from chemical_embedding_pipeline.progress import progress


DEFAULT_MODELS = sorted(set(MODEL_IDS.values()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache Hugging Face checkpoints used by transformer embedding generators."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Hugging Face model ids to cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory. Defaults to the standard HF cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact the network; only verify that files are already cached.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("generated_lfc_embeddings/hf_checkpoint_cache.json"),
    )
    return parser.parse_args()


def cache_model(model_id: str, cache_dir: Path | None, local_files_only: bool) -> dict[str, object]:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    kwargs = {
        "repo_id": model_id,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    snapshot_path = snapshot_download(**kwargs)
    AutoConfig.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=True,
    )
    AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=True,
    )
    return {
        "model_id": model_id,
        "status": "cached",
        "snapshot_path": snapshot_path,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for model_id in progress(args.models, desc="Caching HF checkpoints", unit="model"):
        try:
            rows.append(cache_model(model_id, args.cache_dir, args.local_files_only))
        except Exception as exc:
            rows.append(
                {
                    "model_id": model_id,
                    "status": "missing" if args.local_files_only else "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        if row["status"] == "cached":
            print(f"OK      {row['model_id']} -> {row['snapshot_path']}")
        else:
            print(f"{row['status'].upper():7} {row['model_id']} {row.get('error', '')}")
    print(f"Wrote {args.output_json}")

    if any(row["status"] != "cached" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
