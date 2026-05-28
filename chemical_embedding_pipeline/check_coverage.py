from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_FILES = {
    "generate_lfc_embeddings.py",
    "audit_lfc_score_coverage.py",
    "run_reproducibility_suite.py",
    "generate_dataset_compound_embeddings.py",
    "download_temp_hf_datasets.py",
    "env.py",
    "generate_molfeat_embeddings.py",
    "generate_transformer_embeddings.py",
    "generate_minimol_embeddings.py",
    "generate_metadata_embeddings.py",
    "generate_pubchem_embeddings.py",
    "generate_pretrained_molfeat_embeddings.py",
    "generate_random_embeddings.py",
    "generate_openai_embeddings.py",
    "progress.py",
    "verify_embeddings.py",
    "merge_embedding_h5ads.py",
    "HANDOFF.md",
}

EXPECTED_README_TERMS = {
    "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR",
    "MiniMol",
    "MolT5",
    "chatgpt",
    "openai_text_embedding_3_large",
    "maccs",
    "cats2D",
    "cats3D",
    "ecfp",
    "fcfp",
    "topological",
    "secfp",
    "avalon",
    "erg",
    "ecfp:2",
    "moa_onehot",
    "target_onehot",
    "cactvs",
    "gin_supervised_contextpred",
    "gin_supervised_edgepred",
    "random",
    "all_molfeat",
    "audit_lfc_score_coverage",
    "run_reproducibility_suite",
    "generate_dataset_compound_embeddings",
    "download_temp_hf_datasets",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight guard that the simplified embedding package still covers expected generators."
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir
    missing_files = sorted(name for name in EXPECTED_FILES if not (package_dir / name).exists())
    readme_text = (package_dir / "README.md").read_text()
    coverage_text = (package_dir / "COVERAGE.md").read_text()
    text = readme_text + "\n" + coverage_text
    missing_terms = sorted(term for term in EXPECTED_README_TERMS if term not in text)

    if missing_files or missing_terms:
        if missing_files:
            print("Missing expected files:")
            for name in missing_files:
                print(f"- {name}")
        if missing_terms:
            print("Missing expected coverage terms:")
            for term in missing_terms:
                print(f"- {term}")
        raise SystemExit(1)

    print("Coverage self-check passed.")


if __name__ == "__main__":
    main()
