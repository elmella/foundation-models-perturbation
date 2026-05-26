"""
Benchmark script for sci-Plex LFC with drug restriction.

This mirrors bench_sciplex_lfc.py, but:
1. Restricts drugs to those with LPM embeddings.
2. Supports loading ECFP:2 and LPM_emb from the pkl/export files.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PKL_PATH = REPO_ROOT / "molecule_embeddings" / "tahoe_sci_op3_updated.pkl"
LPM_EXPORT_ROOT = REPO_ROOT / "lpm_paper10_ft_morgan_learned_fixmol_best_embeddings"
SCIPLEX_DATASET_ALIASES = {"srivatsan20_sciplex3", "sciplex", "sciplex3"}


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def get_pkl_path(cfg):
    return resolve_repo_path(cfg.get("lpm_source_path", PKL_PATH))


def get_restriction_path(cfg):
    return resolve_repo_path(cfg.get("lpm_restriction_path", get_pkl_path(cfg)))


def get_submission_root(cfg):
    return resolve_repo_path(cfg.get("submission_root", SUBMISSION_DIR))


def get_lpm_format(cfg, *, restriction=False):
    key = "lpm_restriction_format" if restriction else "lpm_source_format"
    return cfg.get(key, "tahoe_sci_op3")


def get_heldout_path(cfg):
    return resolve_repo_path(
        cfg.get("heldout_molecules_path", LPM_EXPORT_ROOT / "heldout_molecules.tsv")
    )


def get_sciplex_drug_to_cid():
    pkl_df = pd.read_pickle(PKL_PATH)
    sciplex_pkl = pkl_df[pkl_df["dataset"] == "sciplex3"]
    sciplex_pkl = sciplex_pkl.dropna(subset=["pubchem_cid", "original_pert_name"])
    return dict(
        zip(
            sciplex_pkl["original_pert_name"].astype(str),
            sciplex_pkl["pubchem_cid"].astype(int).astype(str),
        )
    )


def get_lpm_export_paths(cfg, *, restriction=False):
    source_path = get_restriction_path(cfg) if restriction else get_pkl_path(cfg)
    source_path = Path(source_path)
    if source_path.is_dir():
        export_dir = source_path
    else:
        export_dir = source_path.parent
    return export_dir / "molecule_metadata.tsv", export_dir / "molecule_embeddings.npy"


def get_lpm_export_drug_to_embedding(cfg):
    metadata_path, embeddings_path = get_lpm_export_paths(cfg)
    metadata = pd.read_csv(metadata_path, sep="\t")
    embeddings = np.load(embeddings_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"{metadata_path} has {len(metadata)} rows but {embeddings_path} has "
            f"{embeddings.shape[0]} embeddings"
        )

    cid_to_drug = {cid: drug for drug, cid in get_sciplex_drug_to_cid().items()}
    lookup = {}
    for row_idx, row in metadata.iterrows():
        cid = str(row["symbol"])
        if cid in cid_to_drug:
            lookup[cid_to_drug[cid]] = np.asarray(embeddings[row_idx], dtype=np.float64)
    return lookup


def get_heldout_drugs(cfg):
    heldout = pd.read_csv(get_heldout_path(cfg), sep="\t")
    dataset = heldout["dataset"].astype(str).str.lower()
    split = heldout["split"].astype(str).str.lower()
    mask = dataset.isin(SCIPLEX_DATASET_ALIASES) & split.isin({"test", "val"})
    heldout_cids = set(heldout.loc[mask, "molecule"].astype(str))
    cid_to_drug = {cid: drug for drug, cid in get_sciplex_drug_to_cid().items()}
    return {cid_to_drug[cid] for cid in heldout_cids & set(cid_to_drug)}


def get_valid_drugs(cfg):
    """Return sci-Plex drug names with available LPM embeddings."""
    lpm_format = get_lpm_format(cfg, restriction=True)
    if lpm_format == "tahoe_sci_op3":
        pkl_df = pd.read_pickle(get_restriction_path(cfg))
        sciplex_pkl = pkl_df[pkl_df["dataset"] == "sciplex3"]
        sciplex_lpm = sciplex_pkl[sciplex_pkl["LPM_emb"].notna()].drop_duplicates(
            subset="original_pert_name", keep="first"
        )
        return set(sciplex_lpm["original_pert_name"].astype(str).values)
    if lpm_format == "pubchem_symbol_lpm_style":
        pkl_df = pd.read_pickle(get_restriction_path(cfg))
        cid_to_drug = {cid: drug for drug, cid in get_sciplex_drug_to_cid().items()}
        valid_cids = set(
            pkl_df[pkl_df["lpm_style_embeddings"].notna()]["symbol"].astype(str).values
        )
        return {cid_to_drug[cid] for cid in valid_cids & set(cid_to_drug)}
    if lpm_format == "lpm_export":
        return set(get_lpm_export_drug_to_embedding(cfg).keys())
    raise ValueError(f"Unsupported lpm format: {lpm_format}")


def load_pkl_embeddings(cfg, emb_name):
    """Load embedding lookup from pkl. Returns dict: sci-Plex drug name -> np.array."""
    lpm_format = get_lpm_format(cfg)
    if lpm_format == "tahoe_sci_op3":
        pkl_df = pd.read_pickle(get_pkl_path(cfg))
        sciplex_pkl = pkl_df[pkl_df["dataset"] == "sciplex3"]
        sciplex_lpm = sciplex_pkl[sciplex_pkl["LPM_emb"].notna()].drop_duplicates(
            subset="original_pert_name", keep="first"
        )
        lookup = {}
        for _, row in sciplex_lpm.iterrows():
            drug = str(row["original_pert_name"])
            lookup[drug] = np.asarray(row[emb_name], dtype=np.float64)
        return lookup
    if lpm_format == "pubchem_symbol_lpm_style":
        pkl_df = pd.read_pickle(get_pkl_path(cfg))
        cid_to_drug = {cid: drug for drug, cid in get_sciplex_drug_to_cid().items()}
        lookup = {}
        for _, row in pkl_df[pkl_df["lpm_style_embeddings"].notna()].iterrows():
            cid = str(row["symbol"])
            if cid in cid_to_drug:
                lookup[cid_to_drug[cid]] = np.asarray(
                    row["lpm_style_embeddings"], dtype=np.float64
                )
        return lookup
    if lpm_format == "lpm_export":
        return get_lpm_export_drug_to_embedding(cfg)
    raise ValueError(f"Unsupported lpm format: {lpm_format}")


def filter_to_valid(adata, valid_drugs):
    """Filter an AnnData object to only drugs in valid_drugs."""
    mask = adata.obs["drug"].astype(str).isin(valid_drugs)
    return adata[mask].copy()


def split_by_heldout(adata, valid_drugs, test_drugs):
    train_drugs = set(valid_drugs) - set(test_drugs)
    train = filter_to_valid(adata, train_drugs)
    test = filter_to_valid(adata, test_drugs)
    return train, test


def submission_name(cfg):
    """Return the submitted model name for this config."""
    if cfg.estimator_name in ["context mean", "no change"]:
        return cfg.estimator_name + " baseline"
    suffix = cfg.get("result_label", cfg.emb_name)
    return cfg.estimator_name + "_baseline_" + suffix


def already_submitted(cfg, n_valid_drugs):
    """Check if this exact restricted sci-Plex LFC run already has a submission."""
    dataset_normalized = cfg.task_name.replace("-", "_")
    fold = f"{cfg.cell_line}.{cfg.fold_id}"
    name = submission_name(cfg)
    restriction_marker = f"Restricted to {n_valid_drugs} drugs with LPM embeddings"
    source_marker = f"Embedding source: {get_pkl_path(cfg)}"
    restriction_source_marker = f"Restriction source: {get_restriction_path(cfg)}"
    custom_source = "lpm_source_path" in cfg or "lpm_restriction_path" in cfg
    submission_dir = get_submission_root(cfg) / dataset_normalized / fold
    if not submission_dir.exists():
        return False

    for path in submission_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        description = data.get("description", "") or ""
        if data.get("name") != name or restriction_marker not in description:
            continue
        if custom_source and (
            source_marker not in description or restriction_source_marker not in description
        ):
            continue
        return True
    return False


@hydra.main(version_base=None, config_path="config/sciplex", config_name="config_sciplex_lfc_restricted")
def main(cfg: DictConfig) -> None:

    valid_drugs = get_valid_drugs(cfg)
    if already_submitted(cfg, len(valid_drugs)):
        print(
            f"Skipping {cfg.cell_line}.{cfg.fold_id} {cfg.estimator_name} "
            f"{cfg.emb_name}: already submitted for {len(valid_drugs)} drugs"
        )
        return None

    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    if cfg.get("test_split_from_heldout", False):
        adata = ad.concat([train, test], axis=0)
        heldout_drugs = get_heldout_drugs(cfg)
        test_drugs = valid_drugs & heldout_drugs
        train, test = split_by_heldout(adata, valid_drugs, test_drugs)
    else:
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
        emb_lookup = load_pkl_embeddings(cfg, pkl_col)
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
        description = f"Embedding: {emb_name} from {get_pkl_path(cfg)}\n"
    else:
        description = "Embedding: " + emb_name + " from " + str(paths.SCIPLEX_DRUG_EMBEDDINGS) + "\n"
    description += "Sklearn pipeline: " + str(estimator) + "\n"
    description += "Best params: " + str(estimator.best_params_) + "\n"
    description += f"Restricted to {len(valid_drugs)} drugs with LPM embeddings\n"
    description += f"Embedding source: {get_pkl_path(cfg)}\n"
    description += f"Embedding source format: {get_lpm_format(cfg)}\n"
    description += f"Restriction source: {get_restriction_path(cfg)}\n"
    description += f"Restriction source format: {get_lpm_format(cfg, restriction=True)}\n"
    if cfg.get("test_split_from_heldout", False):
        description += f"Test split source: {get_heldout_path(cfg)}\n"
        description += "Test split labels: test,val\n"

    return task.submit(
        test_pred,
        name=name,
        description=description,
        submission_dir=get_submission_root(cfg),
    )


if __name__ == "__main__":
    main()
