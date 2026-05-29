from __future__ import annotations

import argparse
from pathlib import Path

from chemical_embedding_pipeline.general_lfc_eval import (
    DEFAULT_EMBEDDINGS,
    DatasetSpec,
    append_rows,
    evaluate_dataset,
    existing_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed molecular-holdout LFC eval for any HF-style perturbation H5AD "
            "and matching compound embedding H5AD."
        )
    )
    parser.add_argument("--dataset-name", required=True, help="Name to write in the output CSV.")
    parser.add_argument(
        "--heldout-dataset",
        required=True,
        help="Dataset label in heldout_molecules.tsv, e.g. srivatsan20_sciplex3.",
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--embedding-h5ad", type=Path, required=True)
    parser.add_argument(
        "--lpm-dir",
        type=Path,
        default=None,
        help="Optional LPM export directory with molecule_metadata.tsv and molecule_embeddings.npy.",
    )
    parser.add_argument("--lpm-name", default="lpm_embedding")
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=["all"],
        help="Embedding keys to evaluate. Use 'all' for the generated default set.",
    )
    parser.add_argument("--estimators", nargs="+", choices=["knn", "lasso"], default=["knn", "lasso"])
    parser.add_argument("--context-cols", nargs="+", default=["cell_type"])
    parser.add_argument("--compound-col", default="perturbagen")
    parser.add_argument(
        "--map-sciplex-cids-to-drugs",
        action="store_true",
        help=(
            "Map heldout PubChem CIDs and LPM export rows to sci-Plex drug names. "
            "Use this when evaluating benchmark/data/sciplex/sciplex3_pseudobulk_filtered.h5ad "
            "with --compound-col drug."
        ),
    )
    parser.add_argument(
        "--sciplex-cid-map-pkl",
        type=Path,
        default=Path("molecule_embeddings/tahoe_sci_op3_updated.pkl"),
        help="PKL with sciplex3 original_pert_name/pubchem_cid columns for CID-to-drug mapping.",
    )
    parser.add_argument("--control-col", default="is_control")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--heldout-molecules-path",
        type=Path,
        default=Path("lpm_paper10_ft_morgan_learned_fixmol_best_embeddings/heldout_molecules.tsv"),
    )
    parser.add_argument("--test-split-labels", nargs="+", default=["test"])
    parser.add_argument("--split-mode", choices=["fixed", "kfold"], default="fixed")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-compounds", type=int, default=20)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--store-test-compounds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_names = DEFAULT_EMBEDDINGS if args.embeddings == ["all"] else args.embeddings
    spec = DatasetSpec(
        name=args.dataset_name,
        heldout_dataset=args.heldout_dataset,
        lpm_dir=args.lpm_dir,
        lpm_name=args.lpm_name,
        input_h5ad=args.input_h5ad,
        embedding_h5ad=args.embedding_h5ad,
        embedding_nested_dir=args.embedding_h5ad.parent.name,
    )
    completed = existing_keys(args.output_csv) if args.resume else set()
    rows = evaluate_dataset(spec, args, embedding_names, completed)
    if rows:
        append_rows(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
