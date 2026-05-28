from __future__ import annotations

import argparse
from pathlib import Path

from chemical_embedding_pipeline.progress import progress


DEFAULT_REPO_ID = "theislab/temp"
DATASET_NAMES = [
    "l1000_phase1",
    "l1000_phase2",
    "novartis_batch_2500",
    "op3",
    "sciplex",
    "tahoe",
    "vcpi_0001",
    "vcpi_0002",
]
PROCESSED_FILES = {
    "l1000_phase1": "l1000_phase1/l1000_phase1_level3_deg_ready_landmark_processed.h5ad",
    "l1000_phase2": "l1000_phase2/l1000_phase2_level3_deg_ready_landmark_processed.h5ad",
    "novartis_batch_2500": "novartis_batch_2500/novartis_standardized_processed.h5ad",
    "op3": "op3/op3_standardized_processed.h5ad",
    "sciplex": "sciplex/srivatsan20_sciplex3_processed.h5ad",
    "tahoe": "tahoe/tahoe_processed.h5ad",
    "vcpi_0001": "vcpi_0001/vcpi_0001_standardized_processed.h5ad",
    "vcpi_0002": "vcpi_0002/vcpi_0002_standardized_processed.h5ad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected datasets from the theislab/temp Hugging Face repo. "
            "By default this downloads only the canonical processed dataset H5AD."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=DATASET_NAMES,
        help="Dataset names to download. Defaults to none unless --all is passed.",
    )
    parser.add_argument("--all", action="store_true", help="Download all known datasets.")
    parser.add_argument(
        "--include-sep-rep",
        action="store_true",
        help="Also download sep_rep differential-expression H5ADs for each selected dataset.",
    )
    parser.add_argument(
        "--include-group-rep",
        action="store_true",
        help="Also download group_rep differential-expression H5ADs for each selected dataset.",
    )
    parser.add_argument(
        "--include-all-filetypes",
        action="store_true",
        help="Download every file under each selected dataset folder.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/theislab_temp"))
    parser.add_argument(
        "--include-root-files",
        action="store_true",
        help="Also download root metadata files such as README and manifests.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List files that would be downloaded without downloading them.",
    )
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return DATASET_NAMES
    return list(args.datasets)


def root_files(files: list[str]) -> list[str]:
    return [
        path
        for path in files
        if "/" not in path
        and (
            path.endswith(".md")
            or path.endswith(".json")
            or path.endswith(".tsv")
            or path == ".gitattributes"
        )
    ]


def dataset_files(dataset: str, files: list[str], args: argparse.Namespace) -> list[str]:
    prefix = f"{dataset}/"
    if args.include_all_filetypes:
        return [path for path in files if path.startswith(prefix)]

    selected = []
    processed = PROCESSED_FILES[dataset]
    if processed in files:
        selected.append(processed)
    else:
        print(f"Warning: expected processed file is missing from repo listing: {processed}")

    if args.include_sep_rep:
        selected.extend(path for path in files if path.startswith(f"{prefix}sep_rep/"))
    if args.include_group_rep:
        selected.extend(path for path in files if path.startswith(f"{prefix}group_rep/"))
    return selected


def main() -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to download from Hugging Face.") from exc

    args = parse_args()
    datasets = selected_datasets(args)
    if not datasets and not args.include_root_files:
        print("No datasets requested. Pass dataset names or --all.")
        return

    api = HfApi()
    files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset", revision=args.revision)
    selected = []
    for dataset in datasets:
        selected.extend(dataset_files(dataset, files, args))
    if args.include_root_files:
        selected.extend(root_files(files))
    selected = sorted(dict.fromkeys(selected))

    if args.list_only:
        for path in selected:
            print(path)
        print(f"{len(selected)} files matched.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in progress(selected, desc="Downloading files", unit="file"):
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            filename=path,
            local_dir=args.output_dir,
        )
        print(local_path)
    print(f"Downloaded {len(selected)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
