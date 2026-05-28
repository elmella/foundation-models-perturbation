from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused reproducibility checks for the simplified LFC embedding pipeline."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated_lfc_embeddings/reproducibility_suite"))
    parser.add_argument(
        "--include-sciplex-structure",
        action="store_true",
        help="Also run the expanded SciPlex A8 structure verification.",
    )
    parser.add_argument(
        "--include-pubchem",
        action="store_true",
        help="Also verify Tahoe PubChem cactvs if a cache is available or network is allowed.",
    )
    parser.add_argument(
        "--include-transformers",
        action="store_true",
        help="Also run cached SciPlex/Tahoe ChemBERTa/MolT5 verification.",
    )
    parser.add_argument(
        "--include-minimol",
        action="store_true",
        help="Also run Tahoe MiniMol verification in its side environment.",
    )
    parser.add_argument(
        "--include-pretrained-molfeat",
        action="store_true",
        help="Also run Tahoe pretrained GIN verification in its side environment.",
    )
    parser.add_argument(
        "--include-legacy-secfp",
        action="store_true",
        help="Also verify exact SECFP using the legacy RDKit/molfeat side environment.",
    )
    parser.add_argument(
        "--all-feasible",
        action="store_true",
        help="Run all implemented non-OpenAI checks, including legacy SECFP.",
    )
    parser.add_argument(
        "--pubchem-cache-json",
        type=Path,
        default=Path("generated_chemical_embeddings/pubchem_fingerprint2d_cache.json"),
    )
    parser.add_argument(
        "--minimol-python",
        type=Path,
        default=Path(".chemembed-minimol-py311/bin/python"),
    )
    parser.add_argument(
        "--pretrained-molfeat-python",
        type=Path,
        default=Path(".chemembed-pretrained-molfeat-py311/bin/python"),
    )
    parser.add_argument(
        "--legacy-secfp-python",
        type=Path,
        default=Path(".chemembed-legacy-py311/bin/python"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[object], dry_run: bool) -> None:
    print("$ " + " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run([str(part) for part in command], cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.all_feasible:
        args.include_sciplex_structure = True
        args.include_pubchem = True
        args.include_transformers = True
        args.include_minimol = True
        args.include_pretrained_molfeat = True
        args.include_legacy_secfp = True

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run([PYTHON, "-m", "chemical_embedding_pipeline.check_coverage"], args.dry_run)
    run(
        [
            PYTHON,
            "-m",
            "chemical_embedding_pipeline.audit_lfc_score_coverage",
            "--output-csv",
            output_dir / "lfc_score_coverage.csv",
            "--output-md",
            output_dir / "lfc_score_coverage.md",
        ],
        args.dry_run,
    )
    run(
        [
            PYTHON,
            "-m",
            "chemical_embedding_pipeline.generate_lfc_embeddings",
            "--dataset",
            "tahoe",
            "--steps",
            "metadata",
            "random",
            "--output-dir",
            output_dir / "tahoe_exact",
            "--verify",
            "--merge",
        ],
        args.dry_run,
    )

    if args.include_pubchem:
        command: list[object] = [
            PYTHON,
            "-m",
            "chemical_embedding_pipeline.generate_lfc_embeddings",
            "--dataset",
            "tahoe",
            "--steps",
            "pubchem",
            "--output-dir",
            output_dir / "tahoe_pubchem",
            "--verify",
            "--merge",
        ]
        if args.pubchem_cache_json.exists():
            command.extend(["--pubchem-cache-json", args.pubchem_cache_json])
        run(command, args.dry_run)

    if args.include_sciplex_structure:
        run(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_lfc_embeddings",
                "--dataset",
                "sciplex",
                "--steps",
                "structure",
                "random",
                "--output-dir",
                output_dir / "sciplex_structure",
                "--verify",
                "--merge",
            ],
            args.dry_run,
        )

    if args.include_transformers:
        run(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_lfc_embeddings",
                "--dataset",
                "both",
                "--steps",
                "transformers",
                "--output-dir",
                output_dir / "transformers",
                "--local-files-only",
                "--verify",
                "--merge",
            ],
            args.dry_run,
        )

    if args.include_minimol:
        run(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_lfc_embeddings",
                "--dataset",
                "tahoe",
                "--steps",
                "minimol",
                "--minimol-python",
                args.minimol_python,
                "--output-dir",
                output_dir / "tahoe_minimol",
                "--verify",
                "--merge",
            ],
            args.dry_run,
        )

    if args.include_pretrained_molfeat:
        run(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_lfc_embeddings",
                "--dataset",
                "tahoe",
                "--steps",
                "pretrained_molfeat",
                "--pretrained-molfeat-python",
                args.pretrained_molfeat_python,
                "--output-dir",
                output_dir / "tahoe_pretrained_molfeat",
                "--verify",
                "--merge",
            ],
            args.dry_run,
        )

    if args.include_legacy_secfp:
        run(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_lfc_embeddings",
                "--dataset",
                "both",
                "--steps",
                "structure",
                "--legacy-secfp-python",
                args.legacy_secfp_python,
                "--output-dir",
                output_dir / "legacy_secfp",
                "--verify",
                "--merge",
            ],
            args.dry_run,
        )

    print(f"Reproducibility suite complete: {output_dir}")


if __name__ == "__main__":
    main()
