from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple generated chemical embedding H5ADs into one H5AD."
    )
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument(
        "--collision-policy",
        choices=["error", "skip_identical", "overwrite", "prefix"],
        default="skip_identical",
        help=(
            "How to handle duplicate obsm keys. skip_identical keeps the first "
            "copy only when arrays are exactly equal."
        ),
    )
    parser.add_argument(
        "--prefix-from-filename",
        action="store_true",
        help="When collision-policy=prefix, prefix duplicate keys with the input stem.",
    )
    parser.add_argument("input_h5ads", nargs="+", type=Path)
    return parser.parse_args()


def prefixed_key(path: Path, key: str, use_stem: bool) -> str:
    if not use_stem:
        return key
    stem = path.name
    for suffix in ["_embeddings.h5ad", ".h5ad"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}__{key}"


def main() -> None:
    args = parse_args()
    if not args.input_h5ads:
        raise ValueError("At least one input H5AD is required.")

    base = ad.read_h5ad(args.input_h5ads[0])
    out = ad.AnnData(obs=base.obs.copy(), uns=base.uns.copy())
    manifest = {
        "inputs": [str(path) for path in args.input_h5ads],
        "collision_policy": args.collision_policy,
        "embeddings": {},
        "skipped_identical": [],
    }

    for path in args.input_h5ads:
        current = ad.read_h5ad(path)
        if not base.obs_names.equals(current.obs_names):
            raise ValueError(f"obs_names in {path} do not match the first input.")
        for key in current.obsm.keys():
            target_key = key
            if target_key in out.obsm:
                if args.collision_policy == "skip_identical":
                    if np.array_equal(np.asarray(out.obsm[target_key]), np.asarray(current.obsm[key])):
                        manifest["skipped_identical"].append({"input": str(path), "embedding": key})
                        continue
                    raise ValueError(
                        f"Duplicate embedding {key!r} in {path} differs from existing value. "
                        "Use --collision-policy prefix or overwrite."
                    )
                if args.collision_policy == "error":
                    raise ValueError(f"Duplicate embedding {key!r} in {path}.")
                if args.collision_policy == "prefix":
                    target_key = prefixed_key(path, key, args.prefix_from_filename)
                    if target_key in out.obsm:
                        raise ValueError(f"Prefixed embedding key {target_key!r} already exists.")
                if args.collision_policy == "overwrite":
                    target_key = key

            out.obsm[target_key] = np.asarray(current.obsm[key])
            manifest["embeddings"][target_key] = {
                "input": str(path),
                "source_key": key,
                "shape": list(out.obsm[target_key].shape),
                "dtype": str(out.obsm[target_key].dtype),
            }

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
