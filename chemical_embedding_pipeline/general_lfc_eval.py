from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemical_embedding_pipeline.progress import progress


DATA_EMBEDDING_ROOT = Path("data/generated_lfc_embeddings/l1000")
TEMP_DATA_ROOT = Path("data/theislab_temp")

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

SPECIAL_EMBEDDINGS = {"random", "pca"}
BASELINE_ESTIMATORS = {"no change", "context mean"}
BASELINE_OUTPUT_ESTIMATOR = "baseline"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    heldout_dataset: str
    lpm_dir: Path | None
    lpm_name: str
    input_h5ad: Path
    embedding_h5ad: Path
    embedding_nested_dir: str
    input_fallbacks: tuple[Path, ...] = ()
    embedding_fallbacks: tuple[Path, ...] = ()


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
        missing = [name for name in names if name not in available and name not in SPECIAL_EMBEDDINGS]
        if missing:
            raise ValueError(f"{path} is missing embeddings: {missing}")
        obs_names = pd.Index(emb.obs_names.astype(str))
        matrices = {
            name: np.asarray(emb.obsm[name], dtype=np.float64)
            for name in names
            if name not in SPECIAL_EMBEDDINGS
        }
    finally:
        emb.file.close()
    return obs_names, matrices


SCIPLEX_DATASET_ALIASES = {"srivatsan20_sciplex3", "sciplex", "sciplex3"}


def should_map_sciplex_cids(args: argparse.Namespace, spec: DatasetSpec) -> bool:
    return bool(
        getattr(args, "map_sciplex_cids_to_drugs", False)
        and spec.heldout_dataset.lower() in SCIPLEX_DATASET_ALIASES
    )


def load_sciplex_cid_to_drug(path: Path) -> dict[str, str]:
    pkl_df = pd.read_pickle(path)
    required = {"dataset", "pubchem_cid", "original_pert_name"}
    missing = required - set(pkl_df.columns)
    if missing:
        raise ValueError(f"{path} is missing required sci-Plex mapping columns: {sorted(missing)}")
    sciplex = pkl_df[pkl_df["dataset"].astype(str).str.lower().eq("sciplex3")]
    sciplex = sciplex.dropna(subset=["pubchem_cid", "original_pert_name"])
    cid = sciplex["pubchem_cid"].astype(float).astype(int).astype(str)
    drug = sciplex["original_pert_name"].astype(str)
    return dict(zip(cid, drug))


def load_lpm_embeddings(
    path: Path,
    name: str,
    *,
    index_map: dict[str, str] | None = None,
) -> tuple[pd.Index, dict[str, np.ndarray]]:
    metadata_path = path / "molecule_metadata.tsv"
    embeddings_path = path / "molecule_embeddings.npy"
    metadata = pd.read_csv(metadata_path, sep="\t")
    values = np.load(embeddings_path)
    if len(metadata) != values.shape[0]:
        raise ValueError(
            f"{metadata_path} has {len(metadata)} rows but {embeddings_path} has {values.shape[0]} embeddings"
        )
    index = metadata["symbol"].astype(str)
    if index_map is not None:
        mapped = index.map(index_map)
        keep = mapped.notna().to_numpy()
        index = mapped.loc[keep].astype(str)
        values = values[keep]
    return pd.Index(index), {name: values.astype(np.float64, copy=False)}


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


def build_estimator(name: str, n_train: int, n_features: int, inner_splits: int, n_jobs: int, *, use_pca: bool = True):
    n_inner_train = max(1, int(n_train * (inner_splits - 1) / inner_splits))
    pca_n_components = min(100, n_inner_train - 1, n_features)
    if name == "no change":
        return DummyRegressor(strategy="constant", constant=np.zeros(n_features))
    if name == "context mean":
        return DummyRegressor()
    if name == "knn":
        steps = [("scaler", StandardScaler())]
        if use_pca and pca_n_components >= 1:
            steps.append(("pca", PCA(n_components=pca_n_components)))
        steps.append(("pseudobulk", KNeighborsRegressor()))
        max_k = n_inner_train - 1
        valid_ks = [k for k in [20, 40, 60, 80, 100] if k < max_k]
        if not valid_ks:
            valid_ks = [max(1, max_k)]
        grid = {"pseudobulk__n_neighbors": sorted(set(valid_ks))}
    elif name == "lasso":
        steps = [("scaler", StandardScaler())]
        if use_pca and pca_n_components >= 1:
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
        n_jobs=n_jobs,
    )


def predict_baseline(name: str, y_train: np.ndarray, n_test: int) -> np.ndarray:
    if name == "no change":
        return np.zeros((n_test, y_train.shape[1]), dtype=np.float64)
    if name == "context mean":
        return np.repeat(y_train.mean(axis=0, keepdims=True), n_test, axis=0)
    raise ValueError(f"Unknown baseline: {name}")


def existing_keys(path: Path) -> set[tuple[str, str, str, str, int, str, str]]:
    if not path.exists():
        return set()
    fixed_header = [
        "dataset",
        "context",
        "estimator",
        "embedding",
        "fold",
        "split",
        "test_split_labels",
        "l2",
        "n_train_compounds",
        "n_test_compounds",
        "n_total_compounds",
        "n_context_rows",
        "mean_test_replicates",
        "best_params",
    ]
    keys = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            if not record or not record.get("dataset"):
                continue
            if not all(column in record and record[column] not in (None, "") for column in fixed_header[:7]):
                continue
            keys.add(
                (
                    record["dataset"],
                    record["context"],
                    record["estimator"],
                    record["embedding"],
                    int(record["fold"]),
                    record["split"],
                    record["test_split_labels"],
                )
            )
    return keys


def update_completed(
    completed: set[tuple[str, str, str, str, int, str, str]],
    rows: list[dict],
) -> None:
    completed.update(
        (
            r["dataset"],
            r["context"],
            r["estimator"],
            r["embedding"],
            int(r["fold"]),
            r["split"],
            r["test_split_labels"],
        )
        for r in rows
    )


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_test_compounds(args: argparse.Namespace, spec: DatasetSpec) -> set[str]:
    heldout = pd.read_csv(args.heldout_molecules_path, sep="\t")
    split_labels = {str(label).lower() for label in args.test_split_labels}
    mask = (
        heldout["dataset"].astype(str).str.lower().eq(spec.heldout_dataset.lower())
        & heldout["split"].astype(str).str.lower().isin(split_labels)
    )
    compounds = heldout.loc[mask, "molecule"].astype(str)
    if should_map_sciplex_cids(args, spec):
        cid_to_drug = load_sciplex_cid_to_drug(args.sciplex_cid_map_pkl)
        compounds = compounds.map(cid_to_drug).dropna().astype(str)
    return set(compounds)


def split_indices(
    compounds: np.ndarray,
    test_compounds: set[str],
    args: argparse.Namespace,
) -> list[tuple[int, np.ndarray, np.ndarray, str]]:
    if args.split_mode == "fixed":
        is_test = np.isin(compounds.astype(str), list(test_compounds))
        train_idx = np.flatnonzero(~is_test)
        test_idx = np.flatnonzero(is_test)
        return [(0, train_idx, test_idx, "fixed_test")]

    outer_splits = min(args.n_splits, len(compounds))
    if outer_splits < 2:
        return []
    splitter = KFold(n_splits=outer_splits, shuffle=True, random_state=args.random_seed)
    return [
        (fold, train_idx, test_idx, "kfold")
        for fold, (train_idx, test_idx) in enumerate(splitter.split(compounds))
    ]


def finite_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values = []
    for i in range(y_pred.shape[0]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = fn(y_true[i, :], y_pred[i, :])[0]
        except Exception:
            value = np.nan
        values.append(value)
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "primary_metric": "L2",
        "L2": float(np.mean(np.linalg.norm(y_pred - y_true, ord=2, axis=1))),
        "MSE": float(np.mean(np.square(y_pred - y_true))),
        "MAE": float(np.mean(np.abs(y_pred - y_true))),
        "Spearman": finite_corr(spearmanr, y_true, y_pred),
        "Pearson": finite_corr(pearsonr, y_true, y_pred),
    }


def special_embedding_matrix(name: str, y: np.ndarray, random_seed: int) -> np.ndarray | None:
    if name == "random":
        rng = np.random.default_rng(random_seed)
        return rng.random((y.shape[0], 100))
    if name == "pca":
        n_components = min(100, y.shape[0], y.shape[1])
        return PCA(n_components=n_components).fit_transform(y)
    return None


def resolve_existing_path(path: Path, fallbacks: list[Path], label: str) -> Path:
    if path.exists():
        return path
    for fallback in fallbacks:
        if fallback.exists():
            return fallback
    options = "\n".join(f"  - {candidate}" for candidate in [path, *fallbacks])
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{options}")


def l1000_embedding_fallbacks(filename: str, nested_dir: str) -> tuple[Path, ...]:
    return (
        DATA_EMBEDDING_ROOT / filename,
        DATA_EMBEDDING_ROOT / nested_dir / filename,
        Path("generated_lfc_embeddings/l1000") / nested_dir / filename,
        Path("generated_lfc_embeddings/l1000") / filename,
    )


def l1000_expression_fallbacks(filename: str, nested_dir: str) -> tuple[Path, ...]:
    return (
        TEMP_DATA_ROOT / filename,
        TEMP_DATA_ROOT / nested_dir / filename,
    )


def evaluate_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
    embedding_names: list[str],
    completed: set[tuple[str, str, str, str, int, str, str]],
) -> list[dict]:
    regression_estimators = [name for name in args.estimators if name not in BASELINE_ESTIMATORS]
    baseline_estimators = [name for name in args.estimators if name in BASELINE_ESTIMATORS]
    input_h5ad = resolve_existing_path(
        spec.input_h5ad,
        list(spec.input_fallbacks),
        (
            f"{spec.name} expression H5AD. This must contain the regression target in X, "
            "not the compound embedding H5AD"
        ),
    )
    embedding_h5ad = resolve_existing_path(
        spec.embedding_h5ad,
        list(spec.embedding_fallbacks),
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

    sciplex_cid_to_drug = (
        load_sciplex_cid_to_drug(args.sciplex_cid_map_pkl) if should_map_sciplex_cids(args, spec) else None
    )

    generated_index, generated = load_generated_embeddings(embedding_h5ad, embedding_names)
    embeddings = dict(generated)
    indices = {name: generated_index for name in generated}
    lpm_index: pd.Index | None = None
    if spec.lpm_dir is not None:
        lpm_index, lpm = load_lpm_embeddings(spec.lpm_dir, spec.lpm_name, index_map=sciplex_cid_to_drug)
        embeddings.update(lpm)
        indices.update({name: lpm_index for name in lpm})
    special_names = [name for name in embedding_names if name in SPECIAL_EMBEDDINGS]
    test_compounds = load_test_compounds(args, spec)
    print(
        f"{spec.name}: using {len(test_compounds)} predefined test compounds "
        f"from {args.heldout_molecules_path}"
    )

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
        compounds_array = np.asarray(compounds, dtype=object)
        if getattr(args, "restrict_to_lpm", True):
            if lpm_index is None:
                raise ValueError("--restrict-to-lpm requires --lpm-dir.")
            lpm_keep = compounds_array.astype(str)
            lpm_keep = np.isin(lpm_keep, lpm_index.astype(str))
            if int(lpm_keep.sum()) < args.min_compounds:
                print(
                    f"Skipping {spec.name} {context}: only {int(lpm_keep.sum())} compounds "
                    "after LPM restriction"
                )
                continue
            compounds = compounds_array[lpm_keep].astype(str).tolist()
            y = y[lpm_keep]
            replicate_counts = replicate_counts[lpm_keep]

        context_embeddings = dict(embeddings)
        context_indices = dict(indices)
        for special_name in special_names:
            special_matrix = special_embedding_matrix(special_name, y, args.random_seed)
            if special_matrix is not None:
                context_embeddings[special_name] = special_matrix
                context_indices[special_name] = pd.Index(pd.Index(compounds).astype(str))

        context_splits = split_indices(np.asarray(compounds, dtype=str), test_compounds, args)
        for fold, train_idx, test_idx, split_name in context_splits:
            if len(train_idx) < args.min_compounds or len(test_idx) == 0:
                for baseline_name in baseline_estimators:
                    print(
                        f"Skipping {spec.name} {context} {baseline_name} {split_name}: "
                        f"train={len(train_idx)} test={len(test_idx)}"
                    )
                continue
            for baseline_name in baseline_estimators:
                split_labels = ",".join(args.test_split_labels)
                key = (
                    spec.name,
                    str(context),
                    BASELINE_OUTPUT_ESTIMATOR,
                    baseline_name,
                    int(fold),
                    split_name,
                    split_labels,
                )
                if args.resume and key in completed:
                    continue
                pred = predict_baseline(baseline_name, y[train_idx], len(test_idx))
                metrics = regression_metrics(y[test_idx], pred)
                row = {
                    "dataset": spec.name,
                    "context": context,
                    "estimator": BASELINE_OUTPUT_ESTIMATOR,
                    "embedding": baseline_name,
                    "fold": fold,
                    "split": split_name,
                    "test_split_labels": split_labels,
                    "l2": metrics["L2"],
                    **metrics,
                    "n_train_compounds": int(len(train_idx)),
                    "n_test_compounds": int(len(test_idx)),
                    "n_total_compounds": int(len(compounds)),
                    "n_context_rows": int(len(context_rows)),
                    "mean_test_replicates": float(replicate_counts[test_idx].mean()),
                    "best_params": "{}",
                }
                if args.store_test_compounds:
                    row["test_compounds"] = ";".join(np.asarray(compounds, dtype=str)[test_idx].tolist())
                rows.append(row)
            if len(rows) >= 20:
                append_rows(args.output_csv, rows)
                update_completed(completed, rows)
                rows = []

        for embedding_name, matrix in progress(
            context_embeddings.items(),
            desc=f"{spec.name} {context} embeddings",
            unit="embedding",
        ):
            keep, x = align_embeddings(compounds, context_indices[embedding_name], matrix)
            if keep.sum() < args.min_compounds:
                print(
                    f"Skipping {spec.name} {context} {embedding_name}: "
                    f"only {int(keep.sum())} compounds with embeddings"
                )
                continue
            y_keep = y[keep]
            compounds_keep = np.asarray(compounds, dtype=object)[keep]
            counts_keep = replicate_counts[keep]
            splits = split_indices(compounds_keep.astype(str), test_compounds, args)
            for fold, train_idx, test_idx, split_name in splits:
                if len(train_idx) < args.min_compounds or len(test_idx) == 0:
                    print(
                        f"Skipping {spec.name} {context} {embedding_name} {split_name}: "
                        f"train={len(train_idx)} test={len(test_idx)}"
                    )
                    continue
                for estimator_name in regression_estimators:
                    split_labels = ",".join(args.test_split_labels)
                    key = (
                        spec.name,
                        str(context),
                        estimator_name,
                        embedding_name,
                        int(fold),
                        split_name,
                        split_labels,
                    )
                    if args.resume and key in completed:
                        continue
                    model = build_estimator(
                        estimator_name,
                        n_train=len(train_idx),
                        n_features=x.shape[1],
                        inner_splits=min(args.inner_splits, len(train_idx)),
                        n_jobs=args.n_jobs,
                        use_pca=embedding_name != "pca",
                    )
                    model.fit(x[train_idx], y_keep[train_idx])
                    pred = model.predict(x[test_idx])
                    metrics = regression_metrics(y_keep[test_idx], pred)
                    row = {
                        "dataset": spec.name,
                        "context": context,
                        "estimator": estimator_name,
                        "embedding": embedding_name,
                        "fold": fold,
                        "split": split_name,
                        "test_split_labels": split_labels,
                        "l2": metrics["L2"],
                        **metrics,
                        "n_train_compounds": int(len(train_idx)),
                        "n_test_compounds": int(len(test_idx)),
                        "n_total_compounds": int(len(compounds_keep)),
                        "n_context_rows": int(len(context_rows)),
                        "mean_test_replicates": float(counts_keep[test_idx].mean()),
                        "best_params": json.dumps(getattr(model, "best_params_", {}), sort_keys=True),
                    }
                    if args.store_test_compounds:
                        row["test_compounds"] = ";".join(compounds_keep[test_idx].astype(str).tolist())
                    rows.append(row)
                if len(rows) >= 20:
                    append_rows(args.output_csv, rows)
                    update_completed(completed, rows)
                    rows = []
    adata.file.close()
    return rows


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=["all"],
        help="Embedding keys to evaluate. Use 'all' for the generated default set.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=["knn", "lasso", "no change", "context mean"],
        default=["knn", "lasso"],
    )
    parser.add_argument("--context-cols", nargs="+", default=["cell_type"])
    parser.add_argument("--compound-col", default="perturbagen")
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
    parser.add_argument(
        "--restrict-to-lpm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict every embedding/estimator to the shared compounds present in --lpm-dir.",
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run molecule-heldout LFC regression for a target H5AD and compound embedding H5AD."
    )
    parser.add_argument(
        "--l1000",
        choices=["phase1", "phase2", "both"],
        default=None,
        help="Use built-in L1000 phase paths, heldout labels, and LPM defaults.",
    )
    parser.add_argument("--dataset-name", help="Name to write in the output CSV for a single generic dataset.")
    parser.add_argument("--heldout-dataset", help="Dataset label in heldout_molecules.tsv for a generic dataset.")
    parser.add_argument("--input-h5ad", type=Path, help="Target H5AD with regression values in X.")
    parser.add_argument("--embedding-h5ad", type=Path, help="Compound embedding H5AD.")
    parser.add_argument("--lpm-dir", type=Path, default=None)
    parser.add_argument("--lpm-name", default="lpm_embedding")
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
        "--phase1-lpm-dir",
        type=Path,
        default=Path("lpm_paper10_ft_morgan_learned_fixmol_best_embeddings/lincs_phase1"),
    )
    parser.add_argument(
        "--phase2-lpm-dir",
        type=Path,
        default=Path("lpm_paper10_ft_morgan_learned_fixmol_best_embeddings/lincs_phase2"),
    )
    parser.add_argument("--phase1-lpm-name", default="lpm_ft_morgan_learned_fixmol_lincs_phase1")
    parser.add_argument("--phase2-lpm-name", default="lpm_ft_morgan_learned_fixmol_lincs_phase2")
    add_common_args(parser)
    return parser.parse_args()


def generic_spec(args: argparse.Namespace) -> DatasetSpec:
    required = {
        "--dataset-name": args.dataset_name,
        "--heldout-dataset": args.heldout_dataset,
        "--input-h5ad": args.input_h5ad,
        "--embedding-h5ad": args.embedding_h5ad,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required generic dataset arguments: {', '.join(missing)}")
    return DatasetSpec(
        name=args.dataset_name,
        heldout_dataset=args.heldout_dataset,
        lpm_dir=args.lpm_dir,
        lpm_name=args.lpm_name,
        input_h5ad=args.input_h5ad,
        embedding_h5ad=args.embedding_h5ad,
        embedding_nested_dir=args.embedding_h5ad.parent.name,
    )


def l1000_specs(args: argparse.Namespace) -> dict[str, DatasetSpec]:
    phase1_nested = "l1000_phase1_level3_deg_ready_landmark_processed"
    phase2_nested = "l1000_phase2_level3_deg_ready_landmark_processed"
    return {
        "phase1": DatasetSpec(
            "l1000_phase1",
            "LINCS_phase1_level3_epsilon",
            args.phase1_lpm_dir,
            args.phase1_lpm_name,
            args.phase1_h5ad,
            args.phase1_embeddings,
            phase1_nested,
            input_fallbacks=l1000_expression_fallbacks(args.phase1_h5ad.name, args.phase1_h5ad.parent.name),
            embedding_fallbacks=l1000_embedding_fallbacks(args.phase1_embeddings.name, phase1_nested),
        ),
        "phase2": DatasetSpec(
            "l1000_phase2",
            "LINCS_phase2_level3",
            args.phase2_lpm_dir,
            args.phase2_lpm_name,
            args.phase2_h5ad,
            args.phase2_embeddings,
            phase2_nested,
            input_fallbacks=l1000_expression_fallbacks(args.phase2_h5ad.name, args.phase2_h5ad.parent.name),
            embedding_fallbacks=l1000_embedding_fallbacks(args.phase2_embeddings.name, phase2_nested),
        ),
    }


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.l1000 is None:
        return [generic_spec(args)]
    specs = l1000_specs(args)
    if args.l1000 == "both":
        return [specs["phase1"], specs["phase2"]]
    return [specs[args.l1000]]


def main() -> None:
    args = parse_args()
    embedding_names = DEFAULT_EMBEDDINGS if args.embeddings == ["all"] else args.embeddings
    completed = existing_keys(args.output_csv) if args.resume else set()
    final_rows: list[dict] = []
    for spec in selected_specs(args):
        final_rows.extend(evaluate_dataset(spec, args, embedding_names, completed))
    if final_rows:
        append_rows(args.output_csv, final_rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
