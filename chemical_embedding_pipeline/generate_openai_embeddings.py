from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from openai import OpenAI

from chemical_embedding_pipeline.env import load_dotenv_key
from chemical_embedding_pipeline.progress import progress_range


DEFAULT_FIELDS = {
    "sciplex": {
        "name": "{index}",
        "smiles": "{canonical_smiles}",
        "name_smiles": "{index}\nSMILES: {canonical_smiles}",
    },
    "tahoe": {
        "name": "{drug}",
        "smiles": "{canonical_smiles}",
        "main_smiles": "{main_molecule_smiles}",
        "name_smiles": "{drug}\nSMILES: {canonical_smiles}",
        "metadata": (
            "Drug: {drug}\nTargets: {targets}\nMOA broad: {moa-broad}\n"
            "MOA fine: {moa-fine}\nHuman approved: {human-approved}\n"
            "Clinical trials: {clinical-trials}\nSMILES: {canonical_smiles}"
        ),
    },
}

OPENAI_MODEL_PROVENANCE = {
    "text-embedding-3-small": {
        "role": "paper_candidate",
        "note": "Paper-aligned candidate for released 1536-dimensional chatgpt matrices.",
        "source_url": "https://www.biorxiv.org/content/10.64898/2026.02.18.706454v1",
    },
    "text-embedding-3-large": {
        "role": "sota_candidate",
        "note": "OpenAI model docs describe this as the most capable embedding model.",
        "source_url": "https://platform.openai.com/docs/models/text-embedding-3-large",
    },
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenAI embedding candidates for released `chatgpt` matrices."
    )
    parser.add_argument("--dataset", choices=["sciplex", "tahoe"], required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model. The paper names text-embedding-3-small for textual metadata.",
    )
    parser.add_argument(
        "--embedding-name",
        default="chatgpt",
        help="Name to use for the output obsm key.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional output dimension for text-embedding-3 models.",
    )
    parser.add_argument(
        "--template-name",
        default="smiles",
        help="Named template from DEFAULT_FIELDS, unless --template is supplied.",
    )
    parser.add_argument("--template", default=None)
    parser.add_argument(
        "--cache-json",
        type=Path,
        default=Path("generated_chemical_embeddings/openai_embedding_cache.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--inputs-json",
        type=Path,
        default=None,
        help="Optional path to write rendered input strings for audit/debugging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render inputs and manifest metadata without calling the OpenAI API.",
    )
    return parser.parse_args()


def normalize_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def render_inputs(adata: ad.AnnData, dataset: str, template_name: str, template: str | None) -> list[str]:
    if template is None:
        try:
            template = DEFAULT_FIELDS[dataset][template_name]
        except KeyError as exc:
            raise KeyError(f"Unknown template {template_name!r} for {dataset}") from exc

    rows = []
    for idx, (_, obs_row) in enumerate(adata.obs.iterrows()):
        values = {column: normalize_value(obs_row[column]) for column in adata.obs.columns}
        values["index"] = str(adata.obs_names[idx])
        rows.append(template.format(**values))
    return rows


def load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def cache_key(model: str, text: str, dimensions: int | None) -> str:
    return json.dumps({"model": model, "dimensions": dimensions, "input": text}, sort_keys=True)


def write_inputs(path: Path, obs_names: list[str], texts: list[str]) -> None:
    rows = [{"index": index, "input": text} for index, text in zip(obs_names, texts, strict=True)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    load_dotenv_key("OPENAI_API_KEY", Path(".env"))
    args = parse_args()
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate OpenAI embeddings.")

    adata = ad.read_h5ad(args.input_h5ad)
    texts = render_inputs(adata, args.dataset, args.template_name, args.template)
    if args.inputs_json is not None:
        write_inputs(args.inputs_json, [str(value) for value in adata.obs_names], texts)

    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "model": args.model,
        "model_provenance": OPENAI_MODEL_PROVENANCE.get(
            args.model,
            {
                "role": "custom",
                "note": "User-specified OpenAI embedding model.",
                "source_url": None,
            },
        ),
        "dimensions": args.dimensions,
        "embedding_name": args.embedding_name,
        "template_name": args.template_name,
        "template": args.template or DEFAULT_FIELDS[args.dataset][args.template_name],
        "cache_json": str(args.cache_json),
        "batch_size": args.batch_size,
        "sleep": args.sleep,
        "inputs_json": str(args.inputs_json) if args.inputs_json is not None else None,
        "dry_run": args.dry_run,
        "packages": {
            "openai": package_version("openai"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "anndata": package_version("anndata"),
        },
        "embeddings": {},
    }
    if args.dry_run:
        manifest["rendered_input_count"] = len(texts)
        manifest["rendered_input_preview"] = texts[: min(3, len(texts))]
        args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_h5ad.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Rendered {len(texts)} OpenAI inputs")
        if args.inputs_json is not None:
            print(f"Wrote {args.inputs_json}")
        print(f"Wrote {manifest_path}")
        return

    cache = load_cache(args.cache_json)
    client = OpenAI()

    for start in progress_range(0, len(texts), args.batch_size, desc="OpenAI batches", unit="batch"):
        batch = texts[start : start + args.batch_size]
        missing = [text for text in batch if cache_key(args.model, text, args.dimensions) not in cache]
        if missing:
            kwargs = {"model": args.model, "input": missing}
            if args.dimensions is not None:
                kwargs["dimensions"] = args.dimensions
            response = client.embeddings.create(**kwargs)
            for text, item in zip(missing, response.data, strict=True):
                cache[cache_key(args.model, text, args.dimensions)] = item.embedding
            save_cache(args.cache_json, cache)
            if args.sleep:
                time.sleep(args.sleep)

    matrix = np.asarray(
        [cache[cache_key(args.model, text, args.dimensions)] for text in texts],
        dtype=np.float32,
    )
    out = ad.AnnData(obs=adata.obs.copy())
    out.obsm[args.embedding_name] = matrix

    manifest["embeddings"] = {
        args.embedding_name: {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
        }
    }
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
