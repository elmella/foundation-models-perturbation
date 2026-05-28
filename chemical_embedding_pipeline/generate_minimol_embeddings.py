from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MiniMol embeddings for benchmark compounds."
    )
    parser.add_argument("--dataset", choices=["sciplex", "tahoe"], required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Defaults to canonical_smiles for SciPlex and main_molecule_smiles for Tahoe.",
    )
    parser.add_argument("--embedding-name", default="MiniMol")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--featurization-n-jobs",
        type=int,
        default=1,
        help="MiniMol/Graphium featurization workers. Use 1 in sandboxed environments.",
    )
    parser.add_argument(
        "--featurization-backend",
        default="threading",
        help="MiniMol/Graphium joblib backend. 'threading' avoids loky semaphore checks.",
    )
    return parser.parse_args()


def smiles_series(adata: ad.AnnData, column: str) -> list[str]:
    if column not in adata.obs:
        raise KeyError(f"{column!r} is not present in input H5AD obs.")
    return pd.Series(adata.obs[column].astype(object)).fillna("").astype(str).tolist()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    try:
        from minimol import Minimol
        import pkg_resources
    except ImportError as exc:
        raise RuntimeError(
            "MiniMol is not installed. Install the public package in a compatible "
            "environment, then rerun this script: pip install minimol"
        ) from exc

    smiles_column = args.smiles_column
    if smiles_column is None:
        smiles_column = "main_molecule_smiles" if args.dataset == "tahoe" else "canonical_smiles"

    adata = ad.read_h5ad(args.input_h5ad)
    smiles = smiles_series(adata, smiles_column)
    model = Minimol(batch_size=args.batch_size)
    if hasattr(model, "datamodule"):
        model.datamodule.featurization_n_jobs = args.featurization_n_jobs
        model.datamodule.featurization_backend = args.featurization_backend
    embeddings = model(smiles)
    matrix = np.vstack(
        [
            value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            for value in embeddings
        ]
    ).astype(np.float32)

    out = ad.AnnData(obs=adata.obs.copy())
    out.obsm[args.embedding_name] = matrix
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)

    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "smiles_column": smiles_column,
        "embedding_name": args.embedding_name,
        "batch_size": args.batch_size,
        "featurization_n_jobs": args.featurization_n_jobs,
        "featurization_backend": args.featurization_backend,
        "package": "minimol",
        "package_versions": {
            "minimol": package_version("minimol"),
            "graphium": package_version("graphium"),
            "torch": package_version("torch"),
            "torch-geometric": package_version("torch-geometric"),
            "torch-scatter": package_version("torch-scatter"),
            "torch-sparse": package_version("torch-sparse"),
            "torch-cluster": package_version("torch-cluster"),
        },
        "checkpoint": {
            "name": "minimol_v1",
            "state_dict_path": pkg_resources.resource_filename(
                "minimol.ckpts.minimol_v1", "state_dict.pth"
            ),
            "config_path": pkg_resources.resource_filename(
                "minimol.ckpts.minimol_v1", "config.yaml"
            ),
            "fingerprint_layer": "gnn:15",
            "graph_pooling": "global_max_pool",
        },
        "embeddings": {
            args.embedding_name: {
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
            }
        },
    }
    manifest["checkpoint"]["state_dict_sha256"] = sha256_file(
        Path(manifest["checkpoint"]["state_dict_path"])
    )
    manifest["checkpoint"]["config_sha256"] = sha256_file(
        Path(manifest["checkpoint"]["config_path"])
    )
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
