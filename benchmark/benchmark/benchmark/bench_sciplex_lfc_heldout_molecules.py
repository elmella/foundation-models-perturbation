"""
Benchmark sci-Plex LFC on the fixed LPM molecular holdout split.

This runner does not use the original sci-Plex cross-validation folds. It uses:
  - train: embedded sci-Plex compounds not labeled val/test in heldout_molecules.tsv
  - test: embedded sci-Plex compounds labeled val or test in heldout_molecules.tsv

Usage:
    uv run python -m benchmark.benchmark.bench_sciplex_lfc_heldout_molecules \
      --config-name config_sciplex_lfc_heldout_molecules --multirun
"""

import json

import anndata as ad
import hydra
import numpy as np
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
from benchmark.benchmark.bench_sciplex_lfc_restricted import (
    filter_to_valid,
    get_heldout_drugs,
    get_lpm_format,
    get_pkl_path,
    get_restriction_path,
    get_submission_root,
    get_valid_drugs,
    load_pkl_embeddings,
    split_by_heldout,
)

SPLIT_ID = "heldout_molecules"


def submission_name(cfg):
    if cfg.estimator_name in ["context mean", "no change"]:
        return cfg.estimator_name + " baseline"
    suffix = cfg.get("result_label", cfg.emb_name)
    return cfg.estimator_name + "_baseline_" + suffix


def submission_split(cell_line):
    return f"{cell_line}.{SPLIT_ID}"


def already_submitted(cfg, n_valid_drugs):
    dataset_normalized = cfg.task_name.replace("-", "_")
    name = submission_name(cfg)
    submission_dir = (
        get_submission_root(cfg) / dataset_normalized / submission_split(cfg.cell_line)
    )
    if not submission_dir.exists():
        return False

    required_markers = [
        f"Restricted to {n_valid_drugs} drugs with LPM embeddings",
        f"Embedding source: {get_pkl_path(cfg)}",
        f"Restriction source: {get_restriction_path(cfg)}",
        "Split: fixed molecular holdout",
        "Test split labels: test,val",
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


def load_split(cfg, valid_drugs):
    adata = ad.read_h5ad(paths.SCIPLEX_PSEUDOBULK_FILTERED)
    cell_lines = sorted(adata.obs["cell_line"].unique())
    if cfg.cell_line not in cell_lines:
        raise ValueError(f"sci-Plex only supports cell lines {cell_lines}, got {cfg.cell_line}")

    adata = adata[adata.obs["cell_line"] == cfg.cell_line].copy()
    adata.obs["pert_id"] = adata.obs["drug"]
    heldout_drugs = get_heldout_drugs(cfg)
    test_drugs = valid_drugs & heldout_drugs
    train, test = split_by_heldout(adata, valid_drugs, test_drugs)
    return train, test, test_drugs


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
    if cfg.emb_name in ("ECFP:2_pkl", "LPM_emb"):
        pkl_col = "ECFP:2" if cfg.emb_name == "ECFP:2_pkl" else "LPM_emb"
        emb_lookup = load_pkl_embeddings(cfg, pkl_col)
        train_emb = np.stack([emb_lookup[str(drug)] for drug in train.obs["drug"].astype(str)])
        test_emb = np.stack([emb_lookup[str(drug)] for drug in test.obs["drug"].astype(str)])
        return cfg.emb_name, train_emb, test_emb

    emb = ad.read_h5ad(paths.SCIPLEX_DRUG_EMBEDDINGS)
    train_emb = emb[train.obs["drug"].astype(str).tolist()].obsm[cfg.emb_name]
    test_emb = emb[test.obs["drug"].astype(str).tolist()].obsm[cfg.emb_name]
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
    if emb_name in ("ECFP:2_pkl", "LPM_emb"):
        return f"Embedding: {emb_name} from {get_pkl_path(cfg)}\n"
    return "Embedding: " + emb_name + " from " + str(paths.SCIPLEX_DRUG_EMBEDDINGS) + "\n"


@hydra.main(
    version_base=None,
    config_path="config/sciplex",
    config_name="config_sciplex_lfc_heldout_molecules",
)
def main(cfg: DictConfig) -> None:
    valid_drugs = get_valid_drugs(cfg)
    if already_submitted(cfg, len(valid_drugs)):
        print(
            f"Skipping {cfg.cell_line}.{SPLIT_ID} {cfg.estimator_name} "
            f"{cfg.emb_name}: already submitted for fixed molecular holdout"
        )
        return None

    train, test, test_drugs = load_split(cfg, valid_drugs)
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

    task = BenchmarkTask(cfg.task_name, submission_split(cfg.cell_line))
    task.test = test

    description = embedding_description(cfg, emb_name)
    description += "Sklearn pipeline: " + str(estimator) + "\n"
    description += "Best params: " + str(estimator.best_params_) + "\n"
    description += "Split: fixed molecular holdout\n"
    description += f"Train compounds: {len(valid_drugs - test_drugs)}\n"
    description += f"Test compounds: {len(test_drugs)}\n"
    description += f"Restricted to {len(valid_drugs)} drugs with LPM embeddings\n"
    description += f"Embedding source: {get_pkl_path(cfg)}\n"
    description += f"Embedding source format: {get_lpm_format(cfg)}\n"
    description += f"Restriction source: {get_restriction_path(cfg)}\n"
    description += f"Restriction source format: {get_lpm_format(cfg, restriction=True)}\n"
    description += "Test split labels: test,val\n"

    return task.submit(
        test_pred,
        name=submission_name(cfg),
        description=description,
        submission_dir=get_submission_root(cfg),
    )


if __name__ == "__main__":
    main()
