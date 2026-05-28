from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("DGLBACKEND", "pytorch")

import anndata as ad
import numpy as np
import pandas as pd

from chemical_embedding_pipeline.progress import progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tahoe pretrained molfeat GIN embeddings.")
    parser.add_argument("--dataset", choices=["tahoe"], default="tahoe")
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--smiles-column", default="main_molecule_smiles")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model-store-dir",
        type=Path,
        default=Path("generated_lfc_embeddings/molfeat_model_store"),
        help="Local molfeat model-store cache for pretrained GIN artifacts.",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Use an existing local model store without downloading from the public HTTP mirror.",
    )
    return parser.parse_args()


def smiles_series(adata: ad.AnnData, column: str) -> list[str]:
    if column not in adata.obs:
        raise KeyError(f"{column!r} is not present in input H5AD obs.")
    return pd.Series(adata.obs[column].astype(object)).fillna("").astype(str).tolist()


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request) as response:
        path.write_bytes(response.read())


def prepare_model_store(model_store_dir: Path, skip_download: bool) -> None:
    base_url = "https://fs.molfeat.datamol.io/artifacts/dgllife"
    for name in progress(
        ["gin_supervised_contextpred", "gin_supervised_edgepred"],
        desc="Preparing GIN models",
        unit="model",
    ):
        model_dir = model_store_dir / "dgllife" / name / "0"
        if not skip_download:
            download_file(f"{base_url}/{name}/0/metadata.json", model_dir / "metadata.json")
            download_file(f"{base_url}/{name}/0/model.save", model_dir / "model.save")
        if not (model_dir / "metadata.json").exists():
            raise FileNotFoundError(f"Missing molfeat metadata: {model_dir / 'metadata.json'}")
        if not (model_dir / "model.save").exists():
            raise FileNotFoundError(f"Missing molfeat model artifact: {model_dir / 'model.save'}")
    os.environ["MOLFEAT_MODEL_STORE_BUCKET"] = str(model_store_dir)


def main() -> None:
    args = parse_args()
    prepare_model_store(args.model_store_dir, args.skip_model_download)
    try:
        from molfeat.trans.pretrained import PretrainedDGLTransformer
    except Exception as exc:
        raise RuntimeError(
            "Pretrained molfeat embeddings require compatible dgl/dgllife/molfeat installs."
        ) from exc

    adata = ad.read_h5ad(args.input_h5ad)
    smiles = smiles_series(adata, args.smiles_column)
    out = ad.AnnData(obs=adata.obs.copy())
    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "smiles_column": args.smiles_column,
        "batch_size": args.batch_size,
        "model_store_dir": str(args.model_store_dir),
        "embeddings": {},
    }
    for kind, key in progress(
        [
        ("gin_supervised_contextpred", "gin_supervised_contextpred"),
        ("gin_supervised_edgepred", "gin_supervised_edgepred"),
        ],
        desc="Pretrained molfeat embeddings",
        unit="model",
    ):
        transformer = PretrainedDGLTransformer(kind=kind, dtype=np.float32)
        matrix = transformer(smiles, batch_size=args.batch_size)
        matrix = np.asarray(matrix, dtype=np.float32)
        out.obsm[key] = matrix
        manifest["embeddings"][key] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "transformer": "PretrainedDGLTransformer",
            "kind": kind,
        }

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
