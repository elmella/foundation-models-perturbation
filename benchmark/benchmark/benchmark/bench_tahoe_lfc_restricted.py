"""
Benchmark script for tahoe LFC with compound restriction.

Identical approach to bench_tahoe_lfc.py, but:
1. Restricts compounds to those with LPM embeddings in tahoe_sci_op3_updated.pkl
2. Supports loading ECFP:2 and LPM_emb from the pkl file
3. Also supports all existing h5ad embeddings (on restricted compounds)

Usage:
    python bench_tahoe_lfc_restricted.py --config-name config_tahoe_lfc_restricted --multirun
    python bench_tahoe_lfc_restricted.py --config-name config_tahoe_lfc_restricted_baseline --multirun
"""

import hydra
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from omegaconf import DictConfig

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold

from benchmark import BenchmarkTask
from benchmark import paths

# Path to the pkl file with new embeddings
PKL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "molecule_embeddings" / "tahoe_sci_op3_updated.pkl"


def get_valid_cids():
    """Load pkl and return set of valid pubchem_cids (compounds with LPM_emb)."""
    pkl_df = pd.read_pickle(PKL_PATH)
    tahoe_pkl = pkl_df[pkl_df["dataset"] == "tahoe"]
    tahoe_pkl_lpm = tahoe_pkl[tahoe_pkl["LPM_emb"].notna()].drop_duplicates(
        subset="pubchem_cid", keep="first"
    )
    return set(tahoe_pkl_lpm["pubchem_cid"].astype(int).values)


def load_pkl_embeddings(emb_name):
    """Load embedding lookup from pkl. Returns dict: pubchem_cid -> np.array."""
    pkl_df = pd.read_pickle(PKL_PATH)
    tahoe_pkl = pkl_df[pkl_df["dataset"] == "tahoe"]
    tahoe_pkl_lpm = tahoe_pkl[tahoe_pkl["LPM_emb"].notna()].drop_duplicates(
        subset="pubchem_cid", keep="first"
    )
    lookup = {}
    for _, row in tahoe_pkl_lpm.iterrows():
        cid = int(row["pubchem_cid"])
        lookup[cid] = np.asarray(row[emb_name], dtype=np.float64)
    return lookup


def filter_to_valid(adata, valid_cids):
    """Filter an AnnData object to only compounds in valid_cids."""
    mask = adata.obs["pert_id"].astype(int).isin(valid_cids)
    return adata[mask].copy()


@hydra.main(version_base=None, config_path="config/tahoe", config_name="config_tahoe_lfc_restricted")
def main(cfg: DictConfig) -> None:

    # Get the set of valid compounds (those with LPM embeddings in pkl)
    valid_cids = get_valid_cids()

    # Load the training and test data
    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    # Restrict to valid compounds
    train = filter_to_valid(train, valid_cids)
    test = filter_to_valid(test, valid_cids)

    # Update task.test so submit() evaluates on restricted set
    task.test = test

    if train.n_obs == 0 or test.n_obs == 0:
        print(f"Skipping {cfg.cell_line}.{cfg.fold_id}: train={train.n_obs}, test={test.n_obs}")
        return None

    # Load embedding
    if cfg.emb_name == "random":
        emb_name = "random"
        train_emb = np.random.random((train.X.shape[0], 100))
        test_emb = np.random.random((test.X.shape[0], 100))
    elif cfg.emb_name == "pca":
        emb_name = "pca"
        pca_emb = PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))
        train_emb = pca_emb[:train.shape[0]]
        test_emb = pca_emb[train.shape[0]:]
    elif cfg.emb_name in ("ECFP:2_pkl", "LPM_emb"):
        # Load from pkl file
        pkl_col = "ECFP:2" if cfg.emb_name == "ECFP:2_pkl" else "LPM_emb"
        emb_name = cfg.emb_name
        emb_lookup = load_pkl_embeddings(pkl_col)
        train_emb = np.stack([emb_lookup[int(cid)] for cid in train.obs["pert_id"].astype(int)])
        test_emb = np.stack([emb_lookup[int(cid)] for cid in test.obs["pert_id"].astype(int)])
    else:
        # Load from h5ad (existing embeddings)
        emb = ad.read_h5ad(paths.TAHOE_DRUG_EMBEDDINGS)
        emb_name = cfg.emb_name
        train_emb = emb[train.obs["pert_id"].astype(int).astype(str).tolist()].obsm[cfg.emb_name]
        test_emb = emb[test.obs["pert_id"].astype(int).astype(str).tolist()].obsm[cfg.emb_name]

    assert train_emb.shape[0] == train.n_obs
    assert test_emb.shape[0] == test.n_obs

    # Define estimator pipeline (same as bench_tahoe_lfc.py, but with PCA n_components
    # adapted for the smaller restricted compound set)
    # With ~139 compounds and 5-fold outer CV (~106 train), 5-fold inner CV gives ~85
    # inner train samples. PCA n_components must be <= min(n_inner_train, n_features).
    n_inner_train = int(train.n_obs * 4 / 5)  # approximate inner CV train size
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

    # Build estimator
    def l2(y, y_pred):
        return -np.linalg.norm(y - y_pred, axis=1).mean()

    estimator = GridSearchCV(
        pipeline,
        hparam_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring=make_scorer(l2),
    )

    # Fit the model and make predictions
    estimator.fit(train_emb, train.X)
    preds = estimator.predict(test_emb)
    test_pred = ad.AnnData(preds, obs=test.obs.copy(), var=test.var.copy())

    # Name the model
    if cfg.estimator_name in ["context mean", "no change"]:
        name = cfg.estimator_name + " baseline"
    else:
        name = cfg.estimator_name + "_baseline_" + emb_name

    # Describe the model
    if emb_name == "random":
        description = "Embedding: np.random.random((..., 100))\n"
    elif emb_name == "pca":
        description = "Embedding: PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))\n"
    elif emb_name in ("ECFP:2_pkl", "LPM_emb"):
        description = f"Embedding: {emb_name} from {PKL_PATH}\n"
    else:
        description = "Embedding: " + emb_name + " from " + str(paths.TAHOE_DRUG_EMBEDDINGS) + "\n"
    description += "Sklearn pipeline: " + str(estimator) + "\n"
    description += "Best params: " + str(estimator.best_params_) + "\n"
    description += f"Restricted to {len(valid_cids)} compounds with LPM embeddings\n"

    # Evaluate the predictions
    return task.submit(test_pred, name=name, description=description)

if __name__ == "__main__":
    main()
