from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemical_embedding_pipeline.progress import progress


DATA_EMBEDDING_ROOT = Path("data/generated_lfc_embeddings/l1000")

DEFAULT_EMBEDDINGS = [
    "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR",
    "MiniMol",
    "MolT5",
    "avalon",
    "cactvs",
    "cats2D",
    "cats3D",
    "chatgpt",
    "ecfp",
    "ecfp:2",
    "erg",
    "fcfp",
    "gin_supervised_contextpred",
    "gin_supervised_edgepred",
    "maccs",
    "openai_text_embedding_3_large",
    "random",
    "secfp",
    "topological",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_h5ad: Path
    embedding_h5ad: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run molecule-heldout LFC regression on L1000 phase datasets using generated "
            "compound embeddings plus an optional LPM export."
        )
    )
    parser.add_argument(
        "--phase1-h5ad",
        type=Path,
        default=Path("data/theislab_temp/l1000_phase1/l1000_phase1_level3_deg_ready_landmark_processed.h5ad"),
    )
    parser.add_argument(
        "--phase1-embeddings",
        type=Path,
        default=Path(
            "data/generated_lfc_embeddings/l1000/"
            "l1000_phase1_level3_deg_ready_landmark_processed_compound_embeddings.h5ad"
        ),
    )
    parser.add_argument(
        "--phase2-h5ad",
        type=Path,
        default=Path("data/theislab_temp/l1000_phase2/l1000_phase2_level3_deg_ready_landmark_processed.h5ad"),
    )
    parser.add_argument(
        "--phase2-embeddings",
        type=Path,
        default=Path(
            "data/generated_lfc_embeddings/l1000/"
            "l1000_phase2_level3_deg_ready_landmark_processed_compound_embeddings.h5ad"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["phase1", "phase2"],
        default=["phase1", "phase2"],
        help="Which L1000 phase datasets to evaluate.",
    )
    parser.add_argument(
        "--lpm-dir",
        type=Path,
        default=Path("lpm_paper10_ft_morgan_learned_fixmol_best_embeddings/lincs_phase1"),
        help="LPM export directory containing molecule_metadata.tsv and molecule_embeddings.npy.",
    )
    parser.add_argument("--lpm-name", default="lpm_ft_morgan_learned_fixmol_lincs_phase1")
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=["all"],
        help="Embedding keys to evaluate. Use 'all' for the 20 generated embeddings.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=["knn", "lasso"],
        default=["knn", "lasso"],
    )
    parser.add_argument(
        "--context-cols",
        nargs="+",
        default=["cell_type"],
        help="Columns that define separate LFC tasks. Use --context-cols all to pool all rows.",
    )
    parser.add_argument("--compound-col", default="perturbagen")
    parser.add_argument("--control-col", default="is_control")
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Include control rows such as DMSO in the regression task.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--min-compounds", type=int, default=50)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/scores/l1000_lfc_embedding_eval.csv"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in --output-csv.",
    )
    parser.add_argument(
        "--store-test-compounds",
        action="store_true",
        help="Store semicolon-separated heldout compound ids for each fold. This can make the CSV large.",
    )
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def context_label(obs: pd.DataFrame, context_cols: list[str]) -> pd.Series:
    if context_cols == ["all"]:
        return pd.Series("all", index=obs.index)
    missing = [col for col in context_cols if col not in obs.columns]
    if missing:
        raise ValueError(f"Missing context columns: {missing}")
    return obs[context_cols].astype(str).agg("|".join, axis=1)


def dense_rows(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def aggregate_context(
    adata: ad.AnnData,
    row_indices: np.ndarray,
    compounds: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    compound_order = pd.Index(pd.unique(compounds.astype(str)))
    compound_codes = compound_order.get_indexer(compounds.astype(str))
    sums = np.zeros((len(compound_order), adata.n_vars), dtype=np.float64)
    counts = np.zeros(len(compound_order), dtype=np.int64)

    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    sorted_codes = compound_codes[order]
    for start in range(0, len(sorted_rows), chunk_size):
        rows = sorted_rows[start : start + chunk_size]
        codes = sorted_codes[start : start + chunk_size]
        values = dense_rows(adata.X[rows, :]).astype(np.float64, copy=False)
        np.add.at(sums, codes, values)
        np.add.at(counts, codes, 1)

    keep = counts > 0
    means = sums[keep] / counts[keep, None]
    return compound_order[keep].astype(str).tolist(), means, counts[keep]


def load_generated_embeddings(path: Path, names: list[str]) -> tuple[pd.Index, dict[str, np.ndarray]]:
    emb = ad.read_h5ad(path, backed="r")
    try:
        available = set(emb.obsm.keys())
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"{path} is missing embeddings: {missing}")
        obs_names = pd.Index(emb.obs_names.astype(str))
        matrices = {name: np.asarray(emb.obsm[name], dtype=np.float64) for name in names}
    finally:
        emb.file.close()
    return obs_names, matrices


def load_lpm_embeddings(path: Path, name: str) -> tuple[pd.Index, dict[str, np.ndarray]]:
    metadata_path = path / "molecule_metadata.tsv"
    embeddings_path = path / "molecule_embeddings.npy"
    metadata = pd.read_csv(metadata_path, sep="\t")
    values = np.load(embeddings_path)
    if len(metadata) != values.shape[0]:
        raise ValueError(
            f"{metadata_path} has {len(metadata)} rows but {embeddings_path} has {values.shape[0]} embeddings"
        )
    return pd.Index(metadata["symbol"].astype(str)), {name: values.astype(np.float64, copy=False)}


def align_embeddings(
    compounds: list[str],
    embedding_index: pd.Index,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = embedding_index.get_indexer(pd.Index(compounds).astype(str))
    keep = positions >= 0
    return keep, matrix[positions[keep]]


def l2(y, y_pred):
    return -np.linalg.norm(y - y_pred, axis=1).mean()


def build_estimator(name: str, n_train: int, n_features: int, inner_splits: int):
    n_inner_train = max(1, int(n_train * (inner_splits - 1) / inner_splits))
    pca_n_components = min(100, n_inner_train - 1, n_features)
    if name == "knn":
        steps = [("scaler", StandardScaler())]
        if pca_n_components >= 1:
            steps.append(("pca", PCA(n_components=pca_n_components)))
        steps.append(("pseudobulk", KNeighborsRegressor()))
        valid_ks = [k for k in [20, 40, 60, 80, 100] if k < n_inner_train]
        if not valid_ks:
            valid_ks = [max(1, n_inner_train - 1)]
        grid = {"pseudobulk__n_neighbors": sorted(set(valid_ks))}
    elif name == "lasso":
        steps = [("scaler", StandardScaler())]
        if pca_n_components >= 1:
            steps.append(("pca", PCA(n_components=pca_n_components)))
        steps.append(("pseudobulk", Lasso(max_iter=5000)))
        grid = {"pseudobulk__alpha": [1e-3, 1e-2, 1e-1, 1]}
    else:
        raise ValueError(f"Unknown estimator: {name}")
    return GridSearchCV(
        Pipeline(steps),
        grid,
        cv=KFold(n_splits=inner_splits, shuffle=True, random_state=42),
        scoring=make_scorer(l2),
    )


def existing_keys(path: Path) -> set[tuple[str, str, str, str, int]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(
        zip(
            df["dataset"].astype(str),
            df["context"].astype(str),
            df["estimator"].astype(str),
            df["embedding"].astype(str),
            df["fold"].astype(int),
        )
    )


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def resolve_existing_path(path: Path, fallbacks: list[Path], label: str) -> Path:
    if path.exists():
        return path
    for fallback in fallbacks:
        if fallback.exists():
            return fallback
    options = "\n".join(f"  - {candidate}" for candidate in [path, *fallbacks])
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{options}")


def embedding_fallbacks(filename: str, nested_dir: str) -> list[Path]:
    return [
        Path("generated_lfc_embeddings/l1000") / nested_dir / filename,
        DATA_EMBEDDING_ROOT / nested_dir / filename,
        Path("generated_lfc_embeddings/l1000") / filename,
    ]


def evaluate_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
    embedding_names: list[str],
    completed: set[tuple[str, str, str, str, int]],
) -> list[dict]:
    input_h5ad = resolve_existing_path(
        spec.input_h5ad,
        [],
        (
            f"{spec.name} expression H5AD. This is the original L1000 data, not the "
            "compound embedding H5AD"
        ),
    )
    embedding_h5ad = resolve_existing_path(
        spec.embedding_h5ad,
        embedding_fallbacks(spec.embedding_h5ad.name, spec.embedding_h5ad.parent.name),
        f"{spec.name} compound embedding H5AD",
    )
    print(f"Loading {spec.name}: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad, backed="r")
    obs = adata.obs.copy()
    if args.compound_col not in obs.columns:
        raise ValueError(f"{spec.input_h5ad} is missing compound column {args.compound_col!r}")

    mask = obs[args.compound_col].notna()
    if not args.include_controls and args.control_col in obs.columns:
        mask &= ~bool_series(obs[args.control_col])
    obs = obs.loc[mask].copy()
    obs["_context"] = context_label(obs, args.context_cols)
    row_positions = np.flatnonzero(mask.to_numpy())

    generated_index, generated = load_generated_embeddings(embedding_h5ad, embedding_names)
    lpm_index, lpm = load_lpm_embeddings(args.lpm_dir, args.lpm_name)
    embeddings = {**generated, **lpm}
    indices = {name: generated_index for name in generated}
    indices.update({name: lpm_index for name in lpm})

    contexts = obs["_context"].value_counts()
    contexts = contexts.index.tolist()
    if args.max_contexts is not None:
        contexts = contexts[: args.max_contexts]

    rows: list[dict] = []
    context_iter = progress(contexts, desc=f"{spec.name} contexts", unit="context")
    for context in context_iter:
        context_mask = obs["_context"].to_numpy() == context
        context_rows = row_positions[context_mask]
        context_compounds = obs.loc[context_mask, args.compound_col].astype(str).to_numpy()
        compounds, y, replicate_counts = aggregate_context(
            adata, context_rows, context_compounds, chunk_size=args.chunk_size
        )
        if len(compounds) < args.min_compounds:
            continue

        for embedding_name, matrix in progress(
            embeddings.items(),
            desc=f"{spec.name} {context} embeddings",
            unit="embedding",
        ):
            keep, x = align_embeddings(compounds, indices[embedding_name], matrix)
            if keep.sum() < args.min_compounds:
                print(
                    f"Skipping {spec.name} {context} {embedding_name}: "
                    f"only {int(keep.sum())} compounds with embeddings"
                )
                continue
            y_keep = y[keep]
            compounds_keep = np.asarray(compounds, dtype=object)[keep]
            counts_keep = replicate_counts[keep]
            outer_splits = min(args.n_splits, len(compounds_keep))
            if outer_splits < 2:
                continue
            splitter = KFold(n_splits=outer_splits, shuffle=True, random_state=args.random_seed)

            for fold, (train_idx, test_idx) in enumerate(splitter.split(x)):
                for estimator_name in args.estimators:
                    key = (spec.name, str(context), estimator_name, embedding_name, int(fold))
                    if args.resume and key in completed:
                        continue
                    model = build_estimator(
                        estimator_name,
                        n_train=len(train_idx),
                        n_features=x.shape[1],
                        inner_splits=min(args.inner_splits, len(train_idx)),
                    )
                    model.fit(x[train_idx], y_keep[train_idx])
                    pred = model.predict(x[test_idx])
                    per_sample_l2 = np.linalg.norm(y_keep[test_idx] - pred, axis=1)
                    row = {
                        "dataset": spec.name,
                        "context": context,
                        "estimator": estimator_name,
                        "embedding": embedding_name,
                        "fold": fold,
                        "l2": float(per_sample_l2.mean()),
                        "n_train_compounds": int(len(train_idx)),
                        "n_test_compounds": int(len(test_idx)),
                        "n_total_compounds": int(len(compounds_keep)),
                        "n_context_rows": int(len(context_rows)),
                        "mean_test_replicates": float(counts_keep[test_idx].mean()),
                        "best_params": json.dumps(model.best_params_, sort_keys=True),
                    }
                    if args.store_test_compounds:
                        row["test_compounds"] = ";".join(compounds_keep[test_idx].astype(str).tolist())
                    rows.append(row)
                if len(rows) >= 20:
                    append_rows(args.output_csv, rows)
                    completed.update(
                        (r["dataset"], r["context"], r["estimator"], r["embedding"], int(r["fold"]))
                        for r in rows
                    )
                    rows = []
    adata.file.close()
    return rows


def main() -> None:
    args = parse_args()
    embedding_names = DEFAULT_EMBEDDINGS if args.embeddings == ["all"] else args.embeddings
    specs = {
        "phase1": DatasetSpec("l1000_phase1", args.phase1_h5ad, args.phase1_embeddings),
        "phase2": DatasetSpec("l1000_phase2", args.phase2_h5ad, args.phase2_embeddings),
    }
    completed = existing_keys(args.output_csv) if args.resume else set()
    final_rows: list[dict] = []
    for key in args.datasets:
        final_rows.extend(evaluate_dataset(specs[key], args, embedding_names, completed))
    if final_rows:
        append_rows(args.output_csv, final_rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
