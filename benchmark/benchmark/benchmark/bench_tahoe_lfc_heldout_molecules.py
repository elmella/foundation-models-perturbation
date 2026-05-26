"""
Benchmark Tahoe LFC on the fixed LPM molecular holdout split.

This runner does not use the original Tahoe cross-validation folds. It uses:
  - train: embedded Tahoe compounds not labeled by cfg.test_split_labels
  - test: embedded Tahoe compounds labeled by cfg.test_split_labels
    in heldout_molecules.tsv

Usage:
    uv run python -m benchmark.benchmark.bench_tahoe_lfc_heldout_molecules \
      --config-name config_tahoe_lfc_heldout_molecules --multirun
"""

import json
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmark import BenchmarkTask
from benchmark import paths
from benchmark.task.task import SUBMISSION_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LPM_EXPORT_ROOT = REPO_ROOT / "lpm_paper10_ft_morgan_learned_fixmol_best_embeddings"
SPLIT_ID = "heldout_molecules"
TAHOE_DATASET_ALIASES = {"tahoe100", "tahoe"}


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def get_lpm_path(cfg):
    return resolve_repo_path(cfg.get("lpm_source_path", LPM_EXPORT_ROOT / "tahoe100"))


def get_restriction_path(cfg):
    return resolve_repo_path(cfg.get("lpm_restriction_path", get_lpm_path(cfg)))


def get_heldout_path(cfg):
    return resolve_repo_path(
        cfg.get("heldout_molecules_path", LPM_EXPORT_ROOT / "heldout_molecules.tsv")
    )


def get_submission_root(cfg):
    return resolve_repo_path(cfg.get("submission_root", SUBMISSION_DIR))


def get_lpm_export_paths(cfg, *, restriction=False):
    source_path = get_restriction_path(cfg) if restriction else get_lpm_path(cfg)
    source_path = Path(source_path)
    export_dir = source_path if source_path.is_dir() else source_path.parent
    return export_dir / "molecule_metadata.tsv", export_dir / "molecule_embeddings.npy"


def get_lpm_export_cid_to_embedding(cfg):
    metadata_path, embeddings_path = get_lpm_export_paths(cfg)
    metadata = pd.read_csv(metadata_path, sep="\t")
    embeddings = np.load(embeddings_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"{metadata_path} has {len(metadata)} rows but {embeddings_path} has "
            f"{embeddings.shape[0]} embeddings"
        )

    return {
        int(row["symbol"]): np.asarray(embeddings[row_idx], dtype=np.float64)
        for row_idx, row in metadata.iterrows()
    }


def get_valid_cids(cfg):
    metadata_path, embeddings_path = get_lpm_export_paths(cfg, restriction=True)
    metadata = pd.read_csv(metadata_path, sep="\t")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"{metadata_path} has {len(metadata)} rows but {embeddings_path} has "
            f"{embeddings.shape[0]} embeddings"
        )
    return set(metadata["symbol"].astype(int).values)


def get_test_split_labels(cfg):
    labels = cfg.get("test_split_labels", ["test", "val"])
    if isinstance(labels, str):
        labels = [labels]
    return tuple(str(label).lower() for label in labels)


def format_test_split_labels(cfg):
    return ",".join(get_test_split_labels(cfg))


def split_id(cfg):
    labels = set(get_test_split_labels(cfg))
    if labels == {"test", "val"}:
        return SPLIT_ID
    return SPLIT_ID + "_" + "_".join(get_test_split_labels(cfg))


def submission_split(cfg):
    return f"{cfg.cell_line}.{split_id(cfg)}"


def get_heldout_cids(cfg):
    heldout = pd.read_csv(get_heldout_path(cfg), sep="\t")
    dataset = heldout["dataset"].astype(str).str.lower()
    split = heldout["split"].astype(str).str.lower()
    mask = dataset.isin(TAHOE_DATASET_ALIASES) & split.isin(get_test_split_labels(cfg))
    return set(heldout.loc[mask, "molecule"].astype(int).values)


def submission_name(cfg):
    if cfg.estimator_name in ["context mean", "no change"]:
        return cfg.estimator_name + " baseline"
    return cfg.estimator_name + "_baseline_" + cfg.emb_name


def filter_to_cids(adata, cids):
    mask = adata.obs["pert_id"].astype(int).isin(cids)
    return adata[mask].copy()


def already_submitted(cfg, n_valid_cids):
    dataset_normalized = cfg.task_name.replace("-", "_")
    name = submission_name(cfg)
    submission_dir = (
        get_submission_root(cfg) / dataset_normalized / submission_split(cfg)
    )
    if not submission_dir.exists():
        return False

    required_markers = [
        f"Restricted to {n_valid_cids} compounds with LPM embeddings",
        f"Embedding source: {get_lpm_path(cfg)}",
        f"Restriction source: {get_restriction_path(cfg)}",
        "Split: fixed molecular holdout",
        f"Test split labels: {format_test_split_labels(cfg)}",
    ]
    for path in submission_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        description = data.get("description", "") or ""
        if data.get("name") == name and all(
            marker in description for marker in required_markers
        ):
            return True
    return False


def load_split(cfg, valid_cids):
    adata = ad.read_h5ad(paths.TAHOE_PSEUDOBULK_WITH_SPLITS)
    cell_lines = sorted(adata.obs["cell_line"].unique())
    if cfg.cell_line not in cell_lines:
        raise ValueError(f"Tahoe only supports cell lines {cell_lines}, got {cfg.cell_line}")

    adata = adata[adata.obs["cell_line"] == cfg.cell_line].copy()
    adata.obs = adata.obs.rename(columns={"pubchem_cid": "pert_id"})
    test_cids = valid_cids & get_heldout_cids(cfg)
    train = filter_to_cids(adata, valid_cids - test_cids)
    test = filter_to_cids(adata, test_cids)
    return train, test, test_cids


def load_embeddings(cfg, train, test):
    if cfg.emb_name == "random":
        return (
            "random",
            np.random.random((train.X.shape[0], 100)),
            np.random.random((test.X.shape[0], 100)),
        )
    if cfg.emb_name == "pca":
        emb_name = "pca"
        pca_emb = PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))
        return emb_name, pca_emb[: train.shape[0]], pca_emb[train.shape[0] :]
    if cfg.emb_name in ("LPM_emb", "morgan_initialized_lpm"):
        emb_lookup = get_lpm_export_cid_to_embedding(cfg)
        train_emb = np.stack([emb_lookup[int(cid)] for cid in train.obs["pert_id"].astype(int)])
        test_emb = np.stack([emb_lookup[int(cid)] for cid in test.obs["pert_id"].astype(int)])
        return cfg.emb_name, train_emb, test_emb

    emb = ad.read_h5ad(paths.TAHOE_DRUG_EMBEDDINGS)
    train_emb = emb[train.obs["pert_id"].astype(int).astype(str).tolist()].obsm[cfg.emb_name]
    test_emb = emb[test.obs["pert_id"].astype(int).astype(str).tolist()].obsm[cfg.emb_name]
    return cfg.emb_name, train_emb, test_emb


def build_estimator(cfg, train, train_emb):
    n_inner_train = int(train.n_obs * 4 / 5)
    pca_n_components = min(100, n_inner_train - 1, train_emb.shape[1])

    assert cfg.estimator_name in ["no change", "context mean", "knn", "lasso"]
    if cfg.estimator_name == "no change":
        pipeline = DummyRegressor(strategy="constant", constant=np.zeros(train.n_vars))
        hparam_grid = {}
    elif cfg.estimator_name == "context mean":
        pipeline = DummyRegressor()
        hparam_grid = {}
    elif cfg.estimator_name == "knn":
        if cfg.emb_name == "pca":
            pipeline = Pipeline([("pseudobulk", KNeighborsRegressor())])
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n_components)),
                ("pseudobulk", KNeighborsRegressor()),
            ])
        max_k = n_inner_train - 1
        valid_ks = [k for k in [20, 40, 60, 80, 100] if k < max_k]
        if not valid_ks:
            valid_ks = [max(1, max_k)]
        hparam_grid = {"pseudobulk__n_neighbors": valid_ks}
    elif cfg.estimator_name == "lasso":
        if cfg.emb_name == "pca":
            pipeline = Pipeline([("pseudobulk", Lasso())])
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n_components)),
                ("pseudobulk", Lasso()),
            ])
        hparam_grid = {"pseudobulk__alpha": [1e-3, 1e-2, 1e-1, 1]}

    def l2(y, y_pred):
        return -np.linalg.norm(y - y_pred, axis=1).mean()

    return GridSearchCV(
        pipeline,
        hparam_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring=make_scorer(l2),
    )


def embedding_description(cfg, emb_name):
    if emb_name == "random":
        return "Embedding: np.random.random((..., 100))\n"
    if emb_name == "pca":
        return "Embedding: PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))\n"
    if emb_name in ("LPM_emb", "morgan_initialized_lpm"):
        return f"Embedding: {emb_name} from {get_lpm_path(cfg)}\n"
    return "Embedding: " + emb_name + " from " + str(paths.TAHOE_DRUG_EMBEDDINGS) + "\n"


@hydra.main(
    version_base=None,
    config_path="config/tahoe",
    config_name="config_tahoe_lfc_heldout_molecules",
)
def main(cfg: DictConfig) -> None:
    valid_cids = get_valid_cids(cfg)
    if already_submitted(cfg, len(valid_cids)):
        print(
            f"Skipping {cfg.cell_line}.{SPLIT_ID} {cfg.estimator_name} "
            f"{cfg.emb_name}: already submitted for fixed molecular holdout"
        )
        return None

    train, test, test_cids = load_split(cfg, valid_cids)
    if train.n_obs == 0 or test.n_obs == 0:
        print(f"Skipping {cfg.cell_line}.{SPLIT_ID}: train={train.n_obs}, test={test.n_obs}")
        return None

    emb_name, train_emb, test_emb = load_embeddings(cfg, train, test)
    assert train_emb.shape[0] == train.n_obs
    assert test_emb.shape[0] == test.n_obs

    estimator = build_estimator(cfg, train, train_emb)
    estimator.fit(train_emb, train.X)
    preds = estimator.predict(test_emb)
    test_pred = ad.AnnData(preds, obs=test.obs.copy(), var=test.var.copy())

    task = BenchmarkTask(cfg.task_name, submission_split(cfg))
    task.test = test

    description = embedding_description(cfg, emb_name)
    description += "Sklearn pipeline: " + str(estimator) + "\n"
    description += "Best params: " + str(estimator.best_params_) + "\n"
    description += "Split: fixed molecular holdout\n"
    description += f"Train compounds: {len(valid_cids - test_cids)}\n"
    description += f"Test compounds: {len(test_cids)}\n"
    description += f"Restricted to {len(valid_cids)} compounds with LPM embeddings\n"
    description += f"Embedding source: {get_lpm_path(cfg)}\n"
    description += "Embedding source format: lpm_export\n"
    description += f"Restriction source: {get_restriction_path(cfg)}\n"
    description += "Restriction source format: lpm_export\n"
    description += f"Test split labels: {format_test_split_labels(cfg)}\n"

    return task.submit(
        test_pred,
        name=submission_name(cfg),
        description=description,
        submission_dir=get_submission_root(cfg),
    )


if __name__ == "__main__":
    main()
