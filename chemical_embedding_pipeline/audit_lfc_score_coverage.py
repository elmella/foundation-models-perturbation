from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SCORE_FILES = {
    "sciplex": Path("results/scores/sciplex_lfc_heldout_molecules_fresh.csv"),
    "tahoe": Path("results/scores/tahoe_lfc_heldout_molecules_fresh.csv"),
}


COVERAGE = {
    "sciplex": {
        "implemented": {
            "avalon": "current RDKit/molfeat candidate; released bits differ",
            "cats2D": "current RDKit/molfeat; exact against released",
            "cats3D": "current RDKit/molfeat with deterministic ETKDG conformers; not exact",
            "ecfp": "current RDKit/molfeat; exact against released",
            "erg": "current RDKit/molfeat; allclose against released",
            "fcfp": "current RDKit/molfeat; exact against released",
            "maccs": "current RDKit/molfeat; exact against released",
            "secfp": "current RDKit/molfeat by default; optional legacy path is exact",
            "topological": "current RDKit/molfeat; exact against released",
            "ChemBERTa-77M-MLM": "public Hugging Face checkpoint candidate; not exact",
            "ChemBERTa-77M-MTR": "public Hugging Face checkpoint candidate; not exact",
            "random": "deterministic negative-control generator; no released SciPlex H5AD key",
        },
        "optional_api": {
            "chatgpt": "OpenAI text-embedding-3-small candidate; requires OPENAI_API_KEY",
        },
        "out_of_scope": {
            "morgan_initialized_lpm": "user asked not to inspect/use LPM embeddings",
            "pca": "expression-derived baseline, not a reusable chemical compound embedding",
        },
    },
    "tahoe": {
        "implemented": {
            "avalon": "current RDKit/molfeat candidate; released bits differ",
            "ecfp:2": "current RDKit/molfeat; exact against released",
            "erg": "current RDKit/molfeat; allclose against released",
            "maccs": "current RDKit/molfeat; exact against released",
            "secfp": "current RDKit/molfeat by default; optional legacy path is exact",
            "topological": "current RDKit/molfeat; exact against released",
            "ChemBERTa-77M-MLM": "public Hugging Face checkpoint candidate; not exact",
            "ChemBERTa-77M-MTR": "public Hugging Face checkpoint candidate; not exact",
            "MolT5": "public Hugging Face checkpoint candidate; not exact",
            "MiniMol": "public MiniMol package candidate; not exact",
            "random": "deterministic generator; exact against released",
        },
        "optional_api": {
            "chatgpt": "OpenAI text-embedding-3-small candidate; requires OPENAI_API_KEY",
        },
        "excluded": {
            "boltz_affinity_pred_value_fragment": "Boltz path intentionally skipped as too heavy",
            "boltz_affinity_pred_value_protein": "Boltz path intentionally skipped as too heavy",
            "unimol2-570m-H": "UniMol2 heavy model path intentionally skipped for now",
        },
        "out_of_scope": {
            "morgan_initialized_lpm": "user asked not to inspect/use LPM embeddings",
            "pca": "expression-derived baseline, not a reusable chemical compound embedding",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit simplified generator coverage against LFC score-file embeddings."
    )
    parser.add_argument("--output-csv", type=Path, default=Path("generated_lfc_embeddings/lfc_score_coverage.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("generated_lfc_embeddings/lfc_score_coverage.md"))
    return parser.parse_args()


def classify(dataset: str, embedding: str) -> tuple[str, str]:
    coverage = COVERAGE[dataset]
    for status in ["implemented", "optional_api", "excluded", "out_of_scope"]:
        if embedding in coverage.get(status, {}):
            return status, coverage[status][embedding]
    return "missing", "not classified by the simplified LFC embedding pipeline"


def collect_rows() -> list[dict[str, str]]:
    rows = []
    for dataset, path in DEFAULT_SCORE_FILES.items():
        frame = pd.read_csv(path)
        for embedding in sorted(frame["embedding"].dropna().unique()):
            status, note = classify(dataset, embedding)
            estimators = ",".join(sorted(frame.loc[frame["embedding"] == embedding, "estimator"].unique()))
            rows.append(
                {
                    "dataset": dataset,
                    "embedding": embedding,
                    "status": status,
                    "estimators": estimators,
                    "note": note,
                }
            )
    return rows


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines = ["# LFC Score Coverage", ""]
    for dataset in ["sciplex", "tahoe"]:
        lines.extend([f"## {dataset}", ""])
        subset = frame[frame["dataset"] == dataset]
        for status in ["implemented", "optional_api", "excluded", "out_of_scope", "missing"]:
            part = subset[subset["status"] == status]
            if part.empty:
                continue
            lines.extend([f"### {status}", ""])
            for row in part.to_dict(orient="records"):
                lines.append(f"- `{row['embedding']}`: {row['note']}")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    args = parse_args()
    frame = pd.DataFrame(collect_rows())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    write_markdown(frame, args.output_md)
    print(frame.groupby(["dataset", "status"]).size().to_string())
    missing = frame[frame["status"] == "missing"]
    if not missing.empty:
        print(missing.to_string(index=False))
        raise SystemExit(1)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
