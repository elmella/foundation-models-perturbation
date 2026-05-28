from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from chemical_embedding_pipeline.progress import progress


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DATASETS = {
    "sciplex": {
        "reference": Path("benchmark/data/sciplex/sciplex_drug_embeddings.h5ad"),
        "structure": [
            "avalon",
            "cats2D",
            "cats3D",
            "ecfp",
            "erg",
            "fcfp",
            "maccs",
            "secfp",
            "topological",
        ],
        "transformers": [
            ("chemberta_mlm", "mean_no_special", ["ChemBERTa-77M-MLM"]),
            ("chemberta_mtr", "mean", ["ChemBERTa-77M-MTR"]),
        ],
        "openai_template": "smiles",
    },
    "tahoe": {
        "reference": Path("benchmark/data/tahoe/tahoe_drug_embeddings.h5ad"),
        "structure": ["avalon", "ecfp:2", "erg", "maccs", "secfp", "topological"],
        "transformers": [
            ("chemberta_mlm", "mean_no_special", ["ChemBERTa-77M-MLM"]),
            ("chemberta_mtr", "mean", ["ChemBERTa-77M-MTR"]),
            ("molt5", "cls", ["MolT5"]),
        ],
        "openai_template": "metadata",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the practical chemical embeddings used in SciPlex/Tahoe LFC evals."
    )
    parser.add_argument("--dataset", choices=["sciplex", "tahoe", "both"], default="both")
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=None,
        help="Input compounds H5AD. Defaults to the released H5AD for single-dataset runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated_lfc_embeddings"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--smiles-column",
        default=None,
        help=(
            "Optional SMILES column for structure/transformer/MiniMol embeddings. "
            "Defaults to canonical_smiles for SciPlex and main_molecule_smiles for Tahoe."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "structure",
            "all_molfeat",
            "transformers",
            "minimol",
            "extras",
            "metadata",
            "pubchem",
            "pretrained_molfeat",
            "random",
            "openai_paper",
            "openai_sota",
        ],
        default=["structure", "transformers", "minimol"],
        help=(
            "Embedding groups to generate. all_molfeat emits the broader feasible "
            "molfeat structure set. MiniMol/extras are Tahoe-only. "
            "extras expands to metadata, pubchem, pretrained_molfeat, and random."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use already-downloaded Hugging Face checkpoints for transformer models.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network-backed extras such as PubChem.",
    )
    parser.add_argument(
        "--pubchem-cache-json",
        type=Path,
        default=None,
        help="Optional PubChem Fingerprint2D cache JSON for cactvs generation.",
    )
    parser.add_argument(
        "--minimol-python",
        type=Path,
        default=Path(".chemembed-minimol-py311/bin/python"),
        help="Python executable for the optional MiniMol environment.",
    )
    parser.add_argument(
        "--pretrained-molfeat-python",
        type=Path,
        default=None,
        help=(
            "Optional Python executable for pretrained molfeat GIN embeddings. "
            "Use this when the main env has incompatible DGL/torch binaries."
        ),
    )
    parser.add_argument(
        "--legacy-secfp-python",
        type=Path,
        default=None,
        help="Optional legacy RDKit/molfeat Python executable used only to generate secfp.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare generated embeddings to the released H5AD when using released inputs.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also write one merged H5AD per dataset containing all generated obsm keys.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run([str(part) for part in command], cwd=REPO_ROOT, check=True)


def verify(reference: Path, candidate: Path, output_dir: Path, token: str, dry_run: bool) -> Path:
    output_csv = output_dir / f"{token}_verification.csv"
    run(
        [
            PYTHON,
            "-m",
            "chemical_embedding_pipeline.verify_embeddings",
            "--reference-h5ad",
            reference,
            "--candidate-h5ad",
            candidate,
            "--output-csv",
            output_csv,
            "--output-json",
            output_dir / f"{token}_verification.json",
        ],
        dry_run,
    )
    return output_csv


def dataset_names(choice: str) -> list[str]:
    return ["sciplex", "tahoe"] if choice == "both" else [choice]


def smiles_args(column: str | None) -> list[str]:
    return ["--smiles-column", column] if column else []


def write_verification_summary(paths: list[Path], output_csv: Path) -> None:
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            rows.append(
                {
                    "verification_csv": str(path),
                    "embedding": row.get("embedding", ""),
                    "status": row.get("status", ""),
                    "exact": row.get("exact", ""),
                    "allclose_1e-6": row.get("allclose_1e-6", ""),
                    "allclose_1e-5": row.get("allclose_1e-5", ""),
                    "mean_abs_diff": row.get("mean_abs_diff", ""),
                    "binary_jaccard_mean": row.get("binary_jaccard_mean", ""),
                    "row_exact_fraction": row.get("row_exact_fraction", ""),
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


def main() -> None:
    args = parse_args()
    if args.input_h5ad is not None and args.dataset == "both":
        raise ValueError("--input-h5ad can only be used with a single --dataset.")
    steps = set(args.steps)
    if "extras" in steps:
        steps.update({"metadata", "pubchem", "pretrained_molfeat", "random"})

    datasets = dataset_names(args.dataset)
    for dataset in progress(datasets, desc="Datasets", unit="dataset"):
        config = DATASETS[dataset]
        input_h5ad = args.input_h5ad or config["reference"]
        prefix = args.prefix or dataset
        out_dir = args.output_dir / dataset if args.dataset == "both" else args.output_dir
        generated: list[Path] = []
        verification_csvs: list[Path] = []

        if "structure" in steps or "all_molfeat" in steps:
            if "all_molfeat" in steps:
                structure_names = []
            else:
                structure_names = list(config["structure"])
            if args.legacy_secfp_python is not None and "secfp" in structure_names:
                structure_names.remove("secfp")
            output = out_dir / f"{prefix}_structure_embeddings.h5ad"
            command = [
                    PYTHON,
                    "-m",
                    "chemical_embedding_pipeline.generate_molfeat_embeddings",
                    "--dataset",
                    dataset,
                    "--input-h5ad",
                    input_h5ad,
                    "--output-h5ad",
                    output,
                    *smiles_args(args.smiles_column),
            ]
            if structure_names:
                command.extend(["--embeddings", *structure_names])
            run(command, args.dry_run)
            generated.append(output)
            if args.verify and input_h5ad == config["reference"]:
                verification_csvs.append(
                    verify(config["reference"], output, out_dir, f"{prefix}_structure", args.dry_run)
                )
            if args.legacy_secfp_python is not None:
                legacy_output = out_dir / f"{prefix}_legacy_secfp_embeddings.h5ad"
                run(
                    [
                        args.legacy_secfp_python,
                        "-m",
                        "chemical_embedding_pipeline.generate_molfeat_embeddings",
                        "--dataset",
                        dataset,
                        "--input-h5ad",
                        input_h5ad,
                        "--output-h5ad",
                        legacy_output,
                        "--embeddings",
                        "secfp",
                        *smiles_args(args.smiles_column),
                    ],
                    args.dry_run,
                )
                generated.append(legacy_output)
                if args.verify and input_h5ad == config["reference"]:
                    verification_csvs.append(
                        verify(
                            config["reference"],
                            legacy_output,
                            out_dir,
                            f"{prefix}_legacy_secfp",
                            args.dry_run,
                        )
                    )

        if "transformers" in steps:
            for token, pooling, names in progress(
                config["transformers"],
                desc=f"{dataset} transformer models",
                unit="model",
            ):
                output = out_dir / f"{prefix}_{token}_embeddings.h5ad"
                command = [
                    PYTHON,
                    "-m",
                    "chemical_embedding_pipeline.generate_transformer_embeddings",
                    "--dataset",
                    dataset,
                    "--input-h5ad",
                    input_h5ad,
                    "--output-h5ad",
                    output,
                    "--pooling",
                    pooling,
                    "--embeddings",
                    *names,
                    *smiles_args(args.smiles_column),
                ]
                if args.local_files_only:
                    command.append("--local-files-only")
                run(command, args.dry_run)
                generated.append(output)
                if args.verify and input_h5ad == config["reference"]:
                    verification_csvs.append(
                        verify(config["reference"], output, out_dir, f"{prefix}_{token}", args.dry_run)
                    )

        if "minimol" in steps and dataset == "tahoe":
            output = out_dir / f"{prefix}_minimol_embeddings.h5ad"
            run(
                [
                    args.minimol_python,
                    "-m",
                    "chemical_embedding_pipeline.generate_minimol_embeddings",
                    "--dataset",
                    dataset,
                    "--input-h5ad",
                    input_h5ad,
                    "--output-h5ad",
                    output,
                    *smiles_args(args.smiles_column),
                ],
                args.dry_run,
            )
            generated.append(output)
            if args.verify and input_h5ad == config["reference"]:
                verification_csvs.append(
                    verify(config["reference"], output, out_dir, f"{prefix}_minimol", args.dry_run)
                )

        if "metadata" in steps and dataset == "tahoe":
            output = out_dir / f"{prefix}_metadata_embeddings.h5ad"
            run(
                [
                    PYTHON,
                    "-m",
                    "chemical_embedding_pipeline.generate_metadata_embeddings",
                    "--dataset",
                    "tahoe",
                    "--input-h5ad",
                    input_h5ad,
                    "--output-h5ad",
                    output,
                ],
                args.dry_run,
            )
            generated.append(output)
            if args.verify and input_h5ad == config["reference"]:
                verification_csvs.append(
                    verify(config["reference"], output, out_dir, f"{prefix}_metadata", args.dry_run)
                )

        if "pubchem" in steps and dataset == "tahoe" and not args.skip_network:
            output = out_dir / f"{prefix}_pubchem_embeddings.h5ad"
            command = [
                    PYTHON,
                    "-m",
                    "chemical_embedding_pipeline.generate_pubchem_embeddings",
                    "--dataset",
                    "tahoe",
                    "--input-h5ad",
                    input_h5ad,
                    "--output-h5ad",
                    output,
            ]
            if args.pubchem_cache_json is not None:
                command.extend(["--cache-json", args.pubchem_cache_json])
            run(command, args.dry_run)
            generated.append(output)
            if args.verify and input_h5ad == config["reference"]:
                verification_csvs.append(
                    verify(config["reference"], output, out_dir, f"{prefix}_pubchem", args.dry_run)
                )

        if "pretrained_molfeat" in steps and dataset == "tahoe":
            output = out_dir / f"{prefix}_pretrained_molfeat_embeddings.h5ad"
            pretrained_python = args.pretrained_molfeat_python or PYTHON
            run(
                [
                    pretrained_python,
                    "-m",
                    "chemical_embedding_pipeline.generate_pretrained_molfeat_embeddings",
                    "--dataset",
                    "tahoe",
                    "--input-h5ad",
                    input_h5ad,
                    "--output-h5ad",
                    output,
                    *smiles_args(args.smiles_column),
                ],
                args.dry_run,
            )
            generated.append(output)
            if args.verify and input_h5ad == config["reference"]:
                verification_csvs.append(
                    verify(
                        config["reference"],
                        output,
                        out_dir,
                        f"{prefix}_pretrained_molfeat",
                        args.dry_run,
                    )
                )

        if "random" in steps:
            output = out_dir / f"{prefix}_random_embeddings.h5ad"
            run(
                [
                    PYTHON,
                    "-m",
                    "chemical_embedding_pipeline.generate_random_embeddings",
                    "--dataset",
                    dataset,
                    "--input-h5ad",
                    input_h5ad,
                    "--output-h5ad",
                    output,
                ],
                args.dry_run,
            )
            generated.append(output)
            if args.verify and dataset == "tahoe" and input_h5ad == config["reference"]:
                verification_csvs.append(
                    verify(config["reference"], output, out_dir, f"{prefix}_random", args.dry_run)
                )

        openai_steps = [
            ("openai_paper", "text-embedding-3-small", "chatgpt", "openai_3_small"),
            (
                "openai_sota",
                "text-embedding-3-large",
                "openai_text_embedding_3_large",
                "openai_3_large",
            ),
        ]
        openai_steps = [item for item in openai_steps if item[0] in steps]
        if openai_steps:
            for step, model, embedding_name, suffix in progress(
                openai_steps,
                desc=f"{dataset} OpenAI embeddings",
                unit="embedding",
            ):
                output = out_dir / f"{prefix}_{suffix}_embeddings.h5ad"
                run(
                    [
                        PYTHON,
                        "-m",
                        "chemical_embedding_pipeline.generate_openai_embeddings",
                        "--dataset",
                        dataset,
                        "--input-h5ad",
                        input_h5ad,
                        "--output-h5ad",
                        output,
                        "--model",
                        model,
                        "--embedding-name",
                        embedding_name,
                        "--template-name",
                        config["openai_template"],
                    ],
                    args.dry_run,
                )
                generated.append(output)
                if args.verify and embedding_name == "chatgpt" and input_h5ad == config["reference"]:
                    verification_csvs.append(
                        verify(config["reference"], output, out_dir, f"{prefix}_{suffix}", args.dry_run)
                    )

        if args.merge and generated:
            run(
                [
                    PYTHON,
                    "-m",
                    "chemical_embedding_pipeline.merge_embedding_h5ads",
                    "--output-h5ad",
                    out_dir / f"{prefix}_lfc_embeddings.h5ad",
                    "--collision-policy",
                    "skip_identical",
                    *generated,
                ],
                args.dry_run,
            )

        if args.verify and verification_csvs and not args.dry_run:
            write_verification_summary(
                verification_csvs,
                out_dir / f"{prefix}_verification_summary.csv",
            )


if __name__ == "__main__":
    main()
