from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare regenerated chemical embeddings against benchmark H5AD embeddings."
    )
    parser.add_argument("--reference-h5ad", type=Path, required=True)
    parser.add_argument("--candidate-h5ad", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Embedding keys to compare. Defaults to intersection of obsm keys.",
    )
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=[1e-6, 1e-5],
        help="Absolute/relative allclose tolerances to report.",
    )
    return parser.parse_args()


def _binary_jaccard(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference > 0
    cand = candidate > 0
    union = np.logical_or(ref, cand).sum(axis=1)
    intersect = np.logical_and(ref, cand).sum(axis=1)
    scores = np.ones(ref.shape[0], dtype=np.float64)
    mask = union > 0
    scores[mask] = intersect[mask] / union[mask]
    return float(np.mean(scores))


def tolerance_label(value: float) -> str:
    base, exponent = f"{value:.0e}".split("e")
    return f"{base}e{int(exponent)}"


def compare_key(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerances: list[float],
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        result = {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "exact": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "binary_jaccard_mean": None,
        }
        for tolerance in tolerances:
            result[f"allclose_{tolerance_label(tolerance)}"] = False
        return result
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    diff = np.abs(ref.astype(np.float64) - cand.astype(np.float64))
    finite = diff[np.isfinite(diff)]
    value_equal = (ref == cand) | (np.isnan(ref) & np.isnan(cand))
    row_exact = np.all(value_equal, axis=1) if ref.ndim == 2 else np.asarray([np.array_equal(ref, cand, equal_nan=True)])
    result = {
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "exact": bool(np.array_equal(ref, cand, equal_nan=True)),
        "row_exact_fraction": float(row_exact.mean()),
        "row_mismatch_count": int((~row_exact).sum()),
        "max_abs_diff": float(finite.max()) if finite.size else None,
        "mean_abs_diff": float(finite.mean()) if finite.size else None,
        "nan_mismatch_count": int(np.logical_xor(np.isnan(ref), np.isnan(cand)).sum()),
        "binary_jaccard_mean": _binary_jaccard(ref, cand),
    }
    for tolerance in tolerances:
        result[f"allclose_{tolerance_label(tolerance)}"] = bool(
            np.allclose(ref, cand, atol=tolerance, rtol=tolerance, equal_nan=True)
        )
    return result


def main() -> None:
    args = parse_args()
    reference = ad.read_h5ad(args.reference_h5ad)
    candidate = ad.read_h5ad(args.candidate_h5ad)

    if not reference.obs_names.equals(candidate.obs_names):
        raise ValueError("Reference and candidate obs_names differ; align rows first.")

    keys = args.embeddings or sorted(set(reference.obsm.keys()) & set(candidate.obsm.keys()))
    rows = []
    for key in keys:
        if key not in reference.obsm:
            rows.append({"embedding": key, "status": "missing_reference"})
            continue
        if key not in candidate.obsm:
            rows.append({"embedding": key, "status": "missing_candidate"})
            continue
        result = compare_key(reference.obsm[key], candidate.obsm[key], args.tolerances)
        result["embedding"] = key
        result["status"] = "compared"
        rows.append(result)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")
    print(df.to_string(index=False))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
