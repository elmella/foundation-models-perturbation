"""
Benchmark script for tahoe DEG with PCA embeddings from tahoe_PCA_emb.pkl
and additional embeddings (MiniMol, AIDOcell, chatgpt, LPM, ECFP:2).

Restricts compounds to those present in the pkl for a given dosage.
Each dosage has a different cell line and compound subset:
  - dose 3.33333 uM: CVCL_0320, 101 compounds
  - dose 10 uM: CVCL_0023, 75 compounds

PCA embeddings are identified by {emb_type}_{version}_{dim}, e.g. PCA.logFC_v1_64.
Additional embeddings: MiniMol, AIDOcell_100M_Norman_Aligned_(D=640)_concat, chatgpt
  (from h5ad), LPM_emb, ECFP:2_pkl (from tahoe_sci_op3_updated.pkl).

Usage:
    cd ~/valinor/foundation-models-perturbation/benchmark
    source .venv/bin/activate

    # Dose 3.33333 (CVCL_0320)
    python benchmark/benchmark/bench_tahoe_deg_pca_emb.py --config-name config_tahoe_deg_pca_emb_dose3 --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4

    # Dose 10 (CVCL_0023)
    python benchmark/benchmark/bench_tahoe_deg_pca_emb.py --config-name config_tahoe_deg_pca_emb_dose10 --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4
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

PKL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "molecule_embeddings" / "tahoe_PCA_emb.pkl"
SCI_PKL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "molecule_embeddings" / "tahoe_sci_op3_updated.pkl"

# Embeddings loaded from tahoe_sci_op3_updated.pkl (cfg name -> pkl column name)
SCI_PKL_EMBEDDINGS = {
    "ECFP:2_pkl": "ECFP:2",
}

# Embeddings loaded from h5ad (obsm keys)
H5AD_EMBEDDINGS = {
    "MiniMol",
    "AIDOcell_100M_Norman_Aligned_(D=640)_concat",
    "chatgpt",
}


def get_valid_cids(dose):
    """Return set of pubchem_cids for the given dosage."""
    pkl_df = pd.read_pickle(PKL_PATH)
    subset = pkl_df[pkl_df["l1000_pert_dose_uM"] == dose]
    return set(subset["pubchem_cid"].astype(int).unique())


def load_pca_embeddings(emb_type, version, dim, dose):
    """Load embedding lookup from pkl. Returns dict: pubchem_cid -> np.array."""
    pkl_df = pd.read_pickle(PKL_PATH)
    subset = pkl_df[
        (pkl_df["l1000_pert_dose_uM"] == dose)
        & (pkl_df["version"] == version)
        & (pkl_df["dim"] == dim)
    ]
    lookup = {}
    for _, row in subset.iterrows():
        cid = int(row["pubchem_cid"])
        lookup[cid] = np.asarray(row[emb_type], dtype=np.float64)
    return lookup


def load_sci_pkl_embeddings(col_name):
    """Load embedding lookup from tahoe_sci_op3_updated.pkl. Returns dict: pubchem_cid -> np.array."""
    pkl_df = pd.read_pickle(SCI_PKL_PATH)
    tahoe_pkl = pkl_df[pkl_df["dataset"] == "tahoe"]
    tahoe_pkl = tahoe_pkl[tahoe_pkl[col_name].notna()].drop_duplicates(
        subset="pubchem_cid", keep="first"
    )
    lookup = {}
    for _, row in tahoe_pkl.iterrows():
        cid = int(row["pubchem_cid"])
        lookup[cid] = np.asarray(row[col_name], dtype=np.float64)
    return lookup


def filter_to_valid(adata, valid_cids):
    """Filter an AnnData object to only compounds in valid_cids."""
    mask = adata.obs["pert_id"].astype(int).isin(valid_cids)
    return adata[mask].copy()


def get_submission_dir(dose):
    """Custom submission directory per dose."""
    dose_str = str(dose).replace(".", "_")
    return SUBMISSION_DIR / f"expression/pert_prediction_tahoe_deg_pca_emb_dose_{dose_str}"


def already_submitted(cfg):
    """Check if this combo already has a submission."""
    fold = f"{cfg.cell_line}.{cfg.fold_id}"
    submission_name = f"{cfg.estimator_name}_{cfg.emb_name}"
    submission_dir = get_submission_dir(cfg.dose) / fold
    if not submission_dir.exists():
        return False
    for f in submission_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            if data.get("name") == submission_name:
                return True
    return False


@hydra.main(version_base=None, config_path="config/tahoe", config_name="config_tahoe_deg_pca_emb_dose3")
def main(cfg: DictConfig):

    if already_submitted(cfg):
        print(f"Skipping {cfg.cell_line}.{cfg.fold_id} {cfg.emb_name}: already submitted")
        return None

    dose = cfg.dose
    valid_cids = get_valid_cids(dose)

    # Load training and test data
    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    # Restrict to valid compounds for this dose
    train = filter_to_valid(train, valid_cids)
    test = filter_to_valid(test, valid_cids)
    task.test = test

    if train.n_obs == 0 or test.n_obs == 0:
        print(f"Skipping {cfg.cell_line}.{cfg.fold_id}: train={train.n_obs}, test={test.n_obs}")
        return None

    # Load embeddings
    emb_name = cfg.emb_name
    if emb_name == "pca":
        n_components = min(100, train.n_obs + test.n_obs - 1, train.n_vars)
        pca_emb = PCA(n_components=n_components).fit_transform(
            np.concatenate((train.X, test.X))
        )
        train_emb = pca_emb[: train.n_obs]
        test_emb = pca_emb[train.n_obs :]
    elif emb_name == "random":
        train_emb = np.random.random((train.n_obs, 100))
        test_emb = np.random.random((test.n_obs, 100))
    elif emb_name in SCI_PKL_EMBEDDINGS:
        pkl_col = SCI_PKL_EMBEDDINGS[emb_name]
        emb_lookup = load_sci_pkl_embeddings(pkl_col)
        # Filter to compounds that exist in this embedding
        emb_cids = set(emb_lookup.keys())
        train = filter_to_valid(train, emb_cids)
        test = filter_to_valid(test, emb_cids)
        task.test = test
        if train.n_obs == 0 or test.n_obs == 0:
            print(f"Skipping {cfg.cell_line}.{cfg.fold_id} {emb_name}: no overlapping compounds")
            return None
        train_cids = train.obs["pert_id"].astype(int).values
        test_cids = test.obs["pert_id"].astype(int).values
        train_emb = np.stack([emb_lookup[int(c)] for c in train_cids])
        test_emb = np.stack([emb_lookup[int(c)] for c in test_cids])
    elif emb_name in H5AD_EMBEDDINGS:
        compound_embeddings = ad.read_h5ad(paths.TAHOE_DRUG_EMBEDDINGS)
        # Filter to compounds that exist in the h5ad
        h5ad_cids = set(compound_embeddings.obs_names.astype(int))
        train = filter_to_valid(train, h5ad_cids)
        test = filter_to_valid(test, h5ad_cids)
        task.test = test
        if train.n_obs == 0 or test.n_obs == 0:
            print(f"Skipping {cfg.cell_line}.{cfg.fold_id} {emb_name}: no overlapping compounds")
            return None
        train_emb = compound_embeddings[train.obs["pert_id"].astype(int).astype(str).tolist()].obsm[emb_name]
        test_emb = compound_embeddings[test.obs["pert_id"].astype(int).astype(str).tolist()].obsm[emb_name]
    else:
        # Parse "PCA.logFC_v1_64" -> emb_type, version, dim
        parts = emb_name.rsplit("_", 2)
        emb_type, version, dim = parts[0], parts[1], int(parts[2])
        emb_lookup = load_pca_embeddings(emb_type, version, dim, dose)
        train_cids = train.obs["pert_id"].astype(int).values
        test_cids = test.obs["pert_id"].astype(int).values
        train_emb = np.stack([emb_lookup[int(c)] for c in train_cids])
        test_emb = np.stack([emb_lookup[int(c)] for c in test_cids])

    assert train_emb.shape[0] == train.n_obs
    assert test_emb.shape[0] == test.n_obs

    # Adapt PCA n_components for small compound set
    n_inner_train = int(train.n_obs * 4 / 5)
    pca_n_components = min(100, n_inner_train - 1, train_emb.shape[1])

    # Define estimator
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
        dummy_train_examples = train.X + 1
        dummy_train_examples = np.concatenate((
            dummy_train_examples,
            np.repeat(np.arange(3)[:, None], dummy_train_examples.shape[1], axis=1),
        ))
        backup_train_emb = np.random.random((dummy_train_examples.shape[0], 100))
        backup_test_emb = np.random.random((test.X.shape[0], 100))

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

    # Description
    if emb_name == "random":
        description = "Embedding: np.random.random((..., 100))\n"
    elif emb_name == "pca":
        description = "Embedding: PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))\n"
    elif emb_name in SCI_PKL_EMBEDDINGS:
        description = f"Embedding: {emb_name} from {SCI_PKL_PATH}\n"
    elif emb_name in H5AD_EMBEDDINGS:
        description = f"Embedding: {emb_name} from {paths.TAHOE_DRUG_EMBEDDINGS}\n"
    else:
        description = f"Embedding: {emb_name} from {PKL_PATH}\n"
    description += "Model: " + str(estimator) + "\n"
    description += f"Restricted to {len(valid_cids)} compounds at dose {dose} uM\n"

    return task.submit(
        predictions,
        name=f"{cfg.estimator_name}_{cfg.emb_name}",
        description=description,
        submission_dir=get_submission_dir(dose),
    )


if __name__ == "__main__":
    main()
