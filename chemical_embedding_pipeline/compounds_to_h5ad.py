from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a compound metadata table to the H5AD shape expected by "
            "the chemical embedding generators."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV/TSV compound table.")
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument(
        "--index-column",
        default=None,
        help="Column to use as obs_names. Defaults to pubchem_cid, chembl_id, drug, or row ids.",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Input delimiter. Defaults to file extension: tab for .tsv, comma otherwise.",
    )
    return parser.parse_args()


def choose_index_column(df: pd.DataFrame, requested: str | None) -> str | None:
    if requested is not None:
        if requested not in df:
            raise KeyError(f"{requested!r} is not present in the input table.")
        return requested
    for candidate in ["pubchem_cid", "cid", "chembl_id", "drug", "compound", "name"]:
        if candidate in df and df[candidate].notna().all() and df[candidate].astype(str).is_unique:
            return candidate
    return None


def main() -> None:
    args = parse_args()
    sep = args.sep
    if sep is None:
        sep = "\t" if args.input.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(args.input, sep=sep)
    if df.empty:
        raise ValueError("Input compound table is empty.")

    index_column = choose_index_column(df, args.index_column)
    if index_column is None:
        obs_names = [f"compound_{idx}" for idx in range(df.shape[0])]
    else:
        obs_names = df[index_column].astype(str).tolist()

    obs = df.copy()
    obs.index = pd.Index(obs_names, name="compound_id")
    out = ad.AnnData(obs=obs)
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)

    manifest = {
        "input": str(args.input),
        "output_h5ad": str(args.output_h5ad),
        "index_column": index_column,
        "rows": int(obs.shape[0]),
        "columns": obs.columns.tolist(),
    }
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
