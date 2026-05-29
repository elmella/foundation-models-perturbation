from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from chemical_embedding_pipeline.progress import progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine HF differential-expression H5AD files into one LFC-ready H5AD. "
            "The selected layer is copied into X so downstream evaluators can use it as "
            "the expression target."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--layer", default="logFC")
    parser.add_argument(
        "--context-from-filename",
        action="store_true",
        help="Add a context column from each input filename stem, stripping a trailing _de.",
    )
    parser.add_argument("--context-col", default="cell_type")
    parser.add_argument("--pattern", default="*.h5ad")
    parser.add_argument(
        "--obs-name-cols",
        nargs="+",
        default=["cell_type", "pubchem_cid", "perturbagen", "drug"],
        help="Columns to combine into stable obs names when present.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help=(
            "Keep rows whose obs COLUMN matches VALUE. May be passed multiple times. "
            "Matching is string-based so values like 10, 10.0, and 10000.0 can be selected exactly."
        ),
    )
    return parser.parse_args()


def filename_context(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_de"):
        return stem[:-3]
    return stem


def as_target_matrix(matrix):
    if sparse.issparse(matrix):
        return matrix.copy()
    return np.asarray(matrix).copy()


def stable_obs_names(obs: pd.DataFrame, columns: list[str], fallback_prefix: str) -> list[str]:
    present = [col for col in columns if col in obs.columns]
    if not present:
        return [f"{fallback_prefix}:{i}" for i in range(len(obs))]
    values = obs[present].astype(str).agg("|".join, axis=1).tolist()
    counts: dict[str, int] = {}
    names = []
    for value in values:
        count = counts.get(value, 0)
        counts[value] = count + 1
        names.append(value if count == 0 else f"{value}|{count}")
    return names


def parse_where(filters: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for item in filters:
        if "=" not in item:
            raise ValueError(f"Invalid --where {item!r}; expected COLUMN=VALUE")
        column, value = item.split("=", 1)
        column = column.strip()
        value = value.strip()
        if not column:
            raise ValueError(f"Invalid --where {item!r}; column is empty")
        parsed.append((column, value))
    return parsed


def apply_filters(dge: ad.AnnData, filters: list[tuple[str, str]], path: Path) -> ad.AnnData:
    if not filters:
        return dge
    mask = pd.Series(True, index=dge.obs.index)
    for column, value in filters:
        if column not in dge.obs.columns:
            raise ValueError(f"{path} is missing --where column {column!r}")
        mask &= dge.obs[column].astype(str).eq(value)
    if not mask.any():
        details = ", ".join(f"{column}={value}" for column, value in filters)
        raise ValueError(f"{path} has no rows matching {details}")
    return dge[mask.to_numpy()].copy()


def read_dge(path: Path, args: argparse.Namespace) -> ad.AnnData:
    dge = ad.read_h5ad(path)
    dge = apply_filters(dge, parse_where(args.where), path)
    if args.layer not in dge.layers:
        raise ValueError(f"{path} is missing layer {args.layer!r}; available layers: {list(dge.layers.keys())}")

    out = ad.AnnData(
        X=as_target_matrix(dge.layers[args.layer]),
        obs=dge.obs.copy(),
        var=dge.var.copy(),
        uns=dict(dge.uns),
    )
    if args.context_from_filename:
        out.obs[args.context_col] = filename_context(path)
    elif args.context_col not in out.obs.columns:
        raise ValueError(
            f"{path} is missing context column {args.context_col!r}; "
            "pass --context-from-filename to derive it from filenames."
        )
    out.obs["_source_h5ad"] = str(path)
    out.obs_names = stable_obs_names(out.obs, args.obs_name_cols, path.stem)
    out.uns["lfc_target_layer"] = args.layer
    return out


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.pattern}")

    adatas = [read_dge(path, args) for path in progress(paths, desc="Reading DGE H5ADs", unit="file")]
    combined = ad.concat(adatas, join="inner", merge="same")
    combined.uns["lfc_target_layer"] = args.layer
    combined.uns["lfc_source_files"] = [str(path) for path in paths]

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(args.output_h5ad)
    print(f"Wrote {args.output_h5ad}")
    print(f"Rows: {combined.n_obs}; genes: {combined.n_vars}; contexts: {combined.obs[args.context_col].nunique()}")


if __name__ == "__main__":
    main()
