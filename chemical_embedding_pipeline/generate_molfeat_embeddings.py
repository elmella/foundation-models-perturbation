from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import anndata as ad
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from molfeat.calc import (
    CATS,
    ElectroShapeDescriptors,
    FPCalculator,
    MordredDescriptors,
    Pharmacophore2D,
    RDKitDescriptors2D,
    ScaffoldKeyCalculator,
    USRDescriptors,
)
from molfeat.trans import MoleculeTransformer

from chemical_embedding_pipeline.progress import progress


FP_NAMES = {
    "atompair": ("atompair", {"length": 2000}),
    "atompair-count": ("atompair", {"length": 2000, "counting": True}),
    "avalon": ("avalon", {"length": 2000}),
    "ecfp": ("ecfp", {"length": 2000}),
    "ecfp-count": ("ecfp", {"length": 2000, "counting": True}),
    "ecfp:2": ("ecfp", {"length": 2000, "radius": 1}),
    "ecfp:4": ("ecfp", {"length": 2000, "radius": 2}),
    "erg": ("erg", {}),
    "estate": ("estate", {}),
    "fcfp": ("fcfp", {"length": 2000}),
    "fcfp-count": ("fcfp", {"length": 2000, "counting": True}),
    "layered": ("layered", {"length": 2000}),
    "maccs": ("maccs", {}),
    "pattern": ("pattern", {"length": 2000}),
    "rdkit": ("rdkit", {"length": 2000}),
    "rdkit-count": ("rdkit", {"length": 2000, "counting": True}),
    "secfp": ("secfp", {"length": 2000}),
    "topological": ("topological", {"length": 2000}),
    "topological-count": ("topological", {"length": 2000, "counting": True}),
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


DEFAULT_EMBEDDINGS = {
    "sciplex": [
        "atompair",
        "atompair-count",
        "avalon",
        "cats2D",
        "cats3D",
        "desc2D",
        "ecfp",
        "ecfp-count",
        "erg",
        "estate",
        "fcfp",
        "fcfp-count",
        "layered",
        "maccs",
        "mordred",
        "pattern",
        "pharm2D",
        "rdkit",
        "rdkit-count",
        "scaffoldkeys",
        "secfp",
        "skeys",
        "topological",
        "topological-count",
    ],
    "tahoe": [
        "avalon",
        "cats2d",
        "desc2D",
        "ecfp:2",
        "ecfp:4",
        "erg",
        "estate",
        "fcfp",
        "maccs",
        "secfp",
        "topological",
    ],
}


def build_calculator(name: str, replace_nan: bool = True):
    if name in FP_NAMES:
        method, params = FP_NAMES[name]
        return FPCalculator(method, **params)
    if name in {"cats2D", "cats2d"}:
        return CATS(use_3d_distances=False)
    if name == "cats3D":
        return CATS(use_3d_distances=True)
    if name == "desc2D":
        return RDKitDescriptors2D(replace_nan=replace_nan)
    if name == "mordred":
        return MordredDescriptors(ignore_3D=True, replace_nan=replace_nan)
    if name == "pharm2D":
        return Pharmacophore2D(length=2000)
    if name in {"skeys", "scaffoldkeys"}:
        return ScaffoldKeyCalculator(normalize=False, use_scaffold=False)
    if name == "usr":
        return USRDescriptors("USR", replace_nan=True)
    if name == "usrcat":
        return USRDescriptors("USRCAT", replace_nan=True)
    if name == "electroshape":
        return ElectroShapeDescriptors(replace_nan=True)
    raise KeyError(f"No molfeat calculator registered for {name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate molfeat chemical embeddings for benchmark compounds."
    )
    parser.add_argument("--dataset", choices=["sciplex", "tahoe"], required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Defaults to canonical_smiles for SciPlex and main_molecule_smiles for Tahoe.",
    )
    parser.add_argument("--embeddings", nargs="+", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Output dtype used by MoleculeTransformer.",
    )
    parser.add_argument(
        "--preserve-nan",
        action="store_true",
        help="Do not replace NaNs in descriptor calculators such as desc2D and mordred.",
    )
    return parser.parse_args()


def smiles_series(adata: ad.AnnData, column: str) -> list[str]:
    if column not in adata.obs:
        raise KeyError(f"{column!r} is not present in input H5AD obs.")
    return pd.Series(adata.obs[column].astype(object)).fillna("").astype(str).tolist()


def embed_3d_molecules(smiles: list[str], seed: int = 0) -> list[Chem.Mol | None]:
    mols: list[Chem.Mol | None] = []
    for value in progress(smiles, desc="Embedding 3D conformers", unit="mol"):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            mols.append(None)
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            mols.append(None)
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass
        mols.append(mol)
    return mols


def restore_failed_rows(result: object, expected_rows: int, dtype: np.dtype) -> np.ndarray:
    if isinstance(result, tuple):
        matrix, kept_indices = result
    else:
        matrix, kept_indices = result, None
    matrix = np.nan_to_num(np.asarray(matrix, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape[0] == expected_rows:
        return matrix
    if kept_indices is None:
        raise ValueError(f"Expected {expected_rows} rows, got {matrix.shape[0]}.")
    restored = np.zeros((expected_rows, matrix.shape[1]), dtype=dtype)
    restored[np.asarray(kept_indices, dtype=int)] = matrix
    return restored


def main() -> None:
    args = parse_args()
    smiles_column = args.smiles_column
    if smiles_column is None:
        smiles_column = "main_molecule_smiles" if args.dataset == "tahoe" else "canonical_smiles"
    adata = ad.read_h5ad(args.input_h5ad)
    smiles = smiles_series(adata, smiles_column)

    out = ad.AnnData(obs=adata.obs.copy())
    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "smiles_column": smiles_column,
        "dtype": args.dtype,
        "preserve_nan": args.preserve_nan,
        "packages": {
            "rdkit": package_version("rdkit"),
            "molfeat": package_version("molfeat"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
        "embeddings": {},
    }

    dtype = np.float64 if args.dtype == "float64" else np.float32
    replace_nan = not args.preserve_nan
    mols_3d: list[Chem.Mol | None] | None = None
    embedding_names = args.embeddings or DEFAULT_EMBEDDINGS[args.dataset]
    for name in progress(embedding_names, desc="Molfeat embeddings", unit="embedding"):
        calc = build_calculator(name, replace_nan=replace_nan)
        transformer = MoleculeTransformer(calc, n_jobs=args.n_jobs, dtype=dtype)
        inputs = smiles
        if name == "cats3D":
            if mols_3d is None:
                mols_3d = embed_3d_molecules(smiles)
            inputs = mols_3d
        result = transformer(inputs, ignore_errors=True)
        matrix = restore_failed_rows(result, len(smiles), dtype)
        out.obsm[name] = matrix
        manifest["embeddings"][name] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "calculator": type(calc).__name__,
        }

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
