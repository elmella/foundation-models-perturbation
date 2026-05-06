"""
Benchmark script for tahoe DEG with compound restriction.

Identical approach to bench_tahoe_deg.py, but:
1. Restricts compounds to those with LPM embeddings in tahoe_sci_op3_updated.pkl
2. Supports loading ECFP:2 and LPM_emb from the pkl file
3. Also supports all existing h5ad embeddings (on restricted compounds)

Usage:
    python bench_tahoe_deg_restricted.py --config-name config_tahoe_deg_restricted --multirun
    python bench_tahoe_deg_restricted.py --config-name config_tahoe_deg_restricted_baseline --multirun
"""

from pathlib import Path
import json
import pandas as pd

import hydra
import torch
import numpy as np
import anndata as ad
from omegaconf import DictConfig


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from benchmark.benchmark import logistic_regression
from benchmark import BenchmarkTask
from benchmark import paths
from benchmark.task.task import SUBMISSION_DIR

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


def submission_name(cfg):
    """Return the submitted model name for this config."""
    return f"{cfg.estimator_name}_{cfg.emb_name}"


def already_submitted(cfg, n_valid_cids):
    """Check if this exact restricted DEG run already has a submission."""
    dataset_normalized = cfg.task_name.replace("-", "_")
    fold = f"{cfg.cell_line}.{cfg.fold_id}"
    name = submission_name(cfg)
    restriction_marker = f"Restricted to {n_valid_cids} compounds with LPM embeddings"
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


@hydra.main(version_base=None, config_path="config/tahoe", config_name="config_tahoe_deg_restricted")
def main(cfg: DictConfig):

    # Get the set of valid compounds (those with LPM embeddings in pkl)
    valid_cids = get_valid_cids()
    if already_submitted(cfg, len(valid_cids)):
        print(
            f"Skipping {cfg.cell_line}.{cfg.fold_id} {cfg.estimator_name} "
            f"{cfg.emb_name}: already submitted for {len(valid_cids)} compounds"
        )
        return None

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

    # Load embeddings
    emb_name = cfg.emb_name
    if emb_name == "pca":
        pca_emb = PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))
        train_emb = pca_emb[: train.shape[0]]
        test_emb = pca_emb[train.shape[0]:]
    elif emb_name == "random":
        train_emb = np.random.random((train.X.shape[0], 100))
        test_emb = np.random.random((test.X.shape[0], 100))
    elif cfg.emb_name in ("ECFP:2_pkl", "LPM_emb"):
        # Load from pkl file
        pkl_col = "ECFP:2" if cfg.emb_name == "ECFP:2_pkl" else "LPM_emb"
        emb_name = cfg.emb_name
        emb_lookup = load_pkl_embeddings(pkl_col)
        train_emb = np.stack([emb_lookup[int(cid)] for cid in train.obs["pert_id"].astype(int)])
        test_emb = np.stack([emb_lookup[int(cid)] for cid in test.obs["pert_id"].astype(int)])
    else:
        compound_embeddings = ad.read_h5ad(paths.TAHOE_DRUG_EMBEDDINGS)
        compound_embeddings.obs["pubchem_cid"] = compound_embeddings.obs_names.astype(np.int64)
        train_emb = compound_embeddings[train.obs["pert_id"].astype(int).astype(str).tolist()].obsm[emb_name]
        test_emb = compound_embeddings[test.obs["pert_id"].astype(int).astype(str).tolist()].obsm[emb_name]

    assert train_emb.shape[0] == train.n_obs
    assert test_emb.shape[0] == test.n_obs

    # Adapt PCA n_components for smaller restricted compound set
    n_inner_train = int(train.n_obs * 4 / 5)  # approximate inner CV train size
    pca_n_components = min(100, n_inner_train - 1, train_emb.shape[1])

    # Define estimator pipeline
    assert cfg.estimator_name in ["logistic_regression", "no_change", "prior", "most_frequent"]
    if cfg.estimator_name == "no_change":
        estimator = DummyClassifier(strategy="constant", constant=np.ones(test.n_vars))
    elif cfg.estimator_name == "prior":
        estimator = DummyClassifier(strategy="prior")
    elif cfg.estimator_name == "most_frequent":
        estimator = DummyClassifier(strategy="most_frequent")
    elif cfg.estimator_name == "logistic_regression":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_n_components)),
            ("classifier", logistic_regression.LogisticRegression(C=cfg.model.C, balance_loss=cfg.model.balance_loss)),
        ])
        no_scaling_model = Pipeline([
            ("classifier", logistic_regression.LogisticRegression(C=cfg.model.C, balance_loss=cfg.model.balance_loss)),
        ])
        estimator = no_scaling_model if emb_name == "pca" else model

    if cfg.estimator_name in {"prior", "most_frequent", "no_change"}:
        # Pad the prediction with most frequent,
        # and add one example of each class so predictions always have 3 classes
        dummy_train_examples = train.X + 1
        dummy_train_examples = np.concatenate((
            dummy_train_examples,
            np.repeat(np.arange(3)[:, None], dummy_train_examples.shape[1], axis=1),
        ))
        backup_train_emb = np.random.random((dummy_train_examples.shape[0], 100))
        backup_test_emb = np.random.random((test.X.shape[0], 100))

        # there is one sample of each class added on to make sure the dummy classifiers
        # predicts the right number of classes. This doesn't effect most_frequent and
        # no_change but does slightly regularize prior
        estimator.fit(backup_train_emb, dummy_train_examples)
        predictions = torch.Tensor(np.stack(estimator.predict_proba(backup_test_emb), axis=1)).transpose(2, 1)
    else:
        estimator.fit(train_emb, train.X + 1)
        predictions = estimator.predict_proba(test_emb).transpose(2, 1)

    # Name the model
    if cfg.estimator_name in {"prior", "most_frequent", "no_change"}:
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
    description += "Model: " + str(estimator) + "\n"
    description += f"Restricted to {len(valid_cids)} compounds with LPM embeddings\n"

    # Evaluate the predictions
    return task.submit(predictions, name=submission_name(cfg), description=description)


if __name__ == "__main__":
    main()
