from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic random baseline embeddings.")
    parser.add_argument("--dataset", choices=["sciplex", "tahoe"], required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--embedding-name", default="random")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input_h5ad)
    rng = np.random.default_rng(args.seed)
    matrix = rng.random((adata.n_obs, args.dim), dtype=np.float64)
    out = ad.AnnData(obs=adata.obs.copy())
    out.obsm[args.embedding_name] = matrix
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "embedding_name": args.embedding_name,
        "dim": args.dim,
        "seed": args.seed,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
    }
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
