from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple Tahoe metadata one-hot embeddings.")
    parser.add_argument("--dataset", choices=["tahoe"], default="tahoe")
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--moa-column", default="moa-fine")
    parser.add_argument("--targets-column", default="targets")
    return parser.parse_args()


def split_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def labels_from_column(values: pd.Series) -> list[str]:
    labels = set()
    for value in values:
        labels.update(split_values(value))
    return sorted(labels)


def one_hot(values: pd.Series, labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(values), len(labels)), dtype=np.int64)
    for row_idx, value in enumerate(values):
        for label in split_values(value):
            if label in label_to_idx:
                matrix[row_idx, label_to_idx[label]] = 1
    return matrix


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input_h5ad)
    for column in [args.moa_column, args.targets_column]:
        if column not in adata.obs:
            raise KeyError(f"{column!r} is not present in input H5AD obs.")

    moa_labels = labels_from_column(adata.obs[args.moa_column])
    target_labels = labels_from_column(adata.obs[args.targets_column])
    out = ad.AnnData(obs=adata.obs.copy())
    out.obsm["moa_onehot"] = one_hot(adata.obs[args.moa_column], moa_labels)
    out.obsm["target_onehot"] = one_hot(adata.obs[args.targets_column], target_labels)

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "embeddings": {
            "moa_onehot": {
                "shape": list(out.obsm["moa_onehot"].shape),
                "dtype": str(out.obsm["moa_onehot"].dtype),
                "labels": moa_labels,
            },
            "target_onehot": {
                "shape": list(out.obsm["target_onehot"].shape),
                "dtype": str(out.obsm["target_onehot"].dtype),
                "labels": target_labels,
            },
        },
    }
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
