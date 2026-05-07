"""
Benchmark script for sci-Plex LFC with drug restriction.

This mirrors bench_sciplex_lfc.py, but:
1. Restricts drugs to those with LPM embeddings in tahoe_sci_op3_updated.pkl.
2. Supports loading ECFP:2 and LPM_emb from the pkl file.
3. Supports existing h5ad embeddings on the same restricted drug set.
4. Skips exact restricted submissions that have already been computed.

Usage:
    python -m benchmark.benchmark.bench_sciplex_lfc_restricted --config-name config_sciplex_lfc_restricted --multirun
    python -m benchmark.benchmark.bench_sciplex_lfc_restricted --config-name config_sciplex_lfc_restricted_baseline --multirun
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

PKL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "molecule_embeddings"
    / "tahoe_sci_op3_updated.pkl"
)


def get_valid_drugs():
    """Return sci-Plex drug names with available LPM embeddings."""
    pkl_df = pd.read_pickle(PKL_PATH)
    sciplex_pkl = pkl_df[pkl_df["dataset"] == "sciplex3"]
    sciplex_lpm = sciplex_pkl[sciplex_pkl["LPM_emb"].notna()].drop_duplicates(
        subset="original_pert_name", keep="first"
    )
    return set(sciplex_lpm["original_pert_name"].astype(str).values)


def load_pkl_embeddings(emb_name):
    """Load embedding lookup from pkl. Returns dict: sci-Plex drug name -> np.array."""
    pkl_df = pd.read_pickle(PKL_PATH)
    sciplex_pkl = pkl_df[pkl_df["dataset"] == "sciplex3"]
    sciplex_lpm = sciplex_pkl[sciplex_pkl["LPM_emb"].notna()].drop_duplicates(
        subset="original_pert_name", keep="first"
    )
    lookup = {}
    for _, row in sciplex_lpm.iterrows():
        drug = str(row["original_pert_name"])
        lookup[drug] = np.asarray(row[emb_name], dtype=np.float64)
    return lookup


def filter_to_valid(adata, valid_drugs):
    """Filter an AnnData object to only drugs in valid_drugs."""
    mask = adata.obs["drug"].astype(str).isin(valid_drugs)
    return adata[mask].copy()


def submission_name(cfg):
    """Return the submitted model name for this config."""
    if cfg.estimator_name in ["context mean", "no change"]:
        return cfg.estimator_name + " baseline"
    return cfg.estimator_name + "_baseline_" + cfg.emb_name


def already_submitted(cfg, n_valid_drugs):
    """Check if this exact restricted sci-Plex LFC run already has a submission."""
    dataset_normalized = cfg.task_name.replace("-", "_")
    fold = f"{cfg.cell_line}.{cfg.fold_id}"
    name = submission_name(cfg)
    restriction_marker = f"Restricted to {n_valid_drugs} drugs with LPM embeddings"
    submission_dir = SUBMISSION_DIR / dataset_normalized / fold
    if not submission_dir.exists():
        return False

    for path in submission_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        description = data.get("description", "") or ""
        if data.get("name") == name and restriction_marker in description:
            return True
    return False


@hydra.main(version_base=None, config_path="config/sciplex", config_name="config_sciplex_lfc_restricted")
def main(cfg: DictConfig) -> None:

    valid_drugs = get_valid_drugs()
    if already_submitted(cfg, len(valid_drugs)):
        print(
            f"Skipping {cfg.cell_line}.{cfg.fold_id} {cfg.estimator_name} "
            f"{cfg.emb_name}: already submitted for {len(valid_drugs)} drugs"
        )
        return None

    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    train = filter_to_valid(train, valid_drugs)
    test = filter_to_valid(test, valid_drugs)
    task.test = test

    if train.n_obs == 0 or test.n_obs == 0:
        print(f"Skipping {cfg.cell_line}.{cfg.fold_id}: train={train.n_obs}, test={test.n_obs}")
        return None

    if cfg.emb_name == "random":
        emb_name = "random"
        train_emb = np.random.random((train.X.shape[0], 100))
        test_emb = np.random.random((test.X.shape[0], 100))
    elif cfg.emb_name == "pca":
        emb_name = "pca"
        pca_emb = PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))
        train_emb = pca_emb[: train.shape[0]]
        test_emb = pca_emb[train.shape[0] :]
    elif cfg.emb_name in ("ECFP:2_pkl", "LPM_emb"):
        pkl_col = "ECFP:2" if cfg.emb_name == "ECFP:2_pkl" else "LPM_emb"
        emb_name = cfg.emb_name
        emb_lookup = load_pkl_embeddings(pkl_col)
        train_emb = np.stack([emb_lookup[str(drug)] for drug in train.obs["drug"].astype(str)])
        test_emb = np.stack([emb_lookup[str(drug)] for drug in test.obs["drug"].astype(str)])
    else:
        emb_name = cfg.emb_name
        emb = ad.read_h5ad(paths.SCIPLEX_DRUG_EMBEDDINGS)
        train_emb = emb[train.obs["drug"].astype(str).tolist()].obsm[cfg.emb_name]
        test_emb = emb[test.obs["drug"].astype(str).tolist()].obsm[cfg.emb_name]

    assert train_emb.shape[0] == train.n_obs
    assert test_emb.shape[0] == test.n_obs

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

    estimator = GridSearchCV(
        pipeline,
        hparam_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring=make_scorer(l2),
    )

    estimator.fit(train_emb, train.X)
    preds = estimator.predict(test_emb)
    test_pred = ad.AnnData(preds, obs=test.obs.copy(), var=test.var.copy())

    name = submission_name(cfg)

    if emb_name == "random":
        description = "Embedding: np.random.random((..., 100))\n"
    elif emb_name == "pca":
        description = "Embedding: PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))\n"
    elif emb_name in ("ECFP:2_pkl", "LPM_emb"):
        description = f"Embedding: {emb_name} from {PKL_PATH}\n"
    else:
        description = "Embedding: " + emb_name + " from " + str(paths.SCIPLEX_DRUG_EMBEDDINGS) + "\n"
    description += "Sklearn pipeline: " + str(estimator) + "\n"
    description += "Best params: " + str(estimator.best_params_) + "\n"
    description += f"Restricted to {len(valid_drugs)} drugs with LPM embeddings\n"

    return task.submit(test_pred, name=name, description=description)


if __name__ == "__main__":
    main()
