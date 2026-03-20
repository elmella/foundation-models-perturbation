"""
Collect restricted DEG JSON submission files into a single CSV score file.

Reads all JSON files from the submissions directory produced by
bench_tahoe_deg_restricted.py and aggregates them into a CSV matching
the format of results/scores/tahoe_deg.csv.

Usage:
    python collect_scores_deg_restricted.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMISSION_DIR = Path(__file__).resolve().parent.parent.parent / "submissions"
OUTPUT_CSV = REPO_ROOT / "results" / "scores" / "tahoe_deg_restricted.csv"
PKL_PATH = REPO_ROOT / "molecule_embeddings" / "tahoe_sci_op3_updated.pkl"
EXP_ERROR_CSV = REPO_ROOT / "results" / "exp_error" / "exp_err_taheo_deg_n_trials_inner_3_outer_3.csv"

BASELINE_ESTIMATORS = {"no_change", "prior", "most_frequent"}


def normalize_name(raw_name):
    """Normalize submission name to match tahoe_deg.csv convention.

    logistic_regression_chatgpt -> chatgpt
    no_change_random -> no_change
    prior_random -> prior
    most_frequent_random -> most_frequent
    """
    for base in BASELINE_ESTIMATORS:
        if raw_name.startswith(base + "_"):
            return base
    if raw_name.startswith("logistic_regression_"):
        return raw_name[len("logistic_regression_"):]
    return raw_name


def compute_restricted_exp_error():
    """Compute experimental error (q=0.1) on restricted compounds per (cell_line, fold)."""
    import anndata as ad
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark import paths

    # Valid CIDs
    pkl_df = pd.read_pickle(PKL_PATH)
    tahoe_pkl = pkl_df[pkl_df["dataset"] == "tahoe"]
    tahoe_lpm = tahoe_pkl[tahoe_pkl["LPM_emb"].notna()].drop_duplicates(
        subset="pubchem_cid", keep="first"
    )
    valid_cids = set(tahoe_lpm["pubchem_cid"].astype(int).values)

    # Per-perturbation exp error (q=0.1)
    exp = pd.read_csv(EXP_ERROR_CSV)
    exp_q01 = exp[exp["quantile"] == 0.1].copy()

    # Load adata for fold splits and cid mapping
    adata = ad.read_h5ad(paths.TAHOE_PSEUDOBULK_WITH_SPLITS)

    cell_lines = sorted(adata.obs["cell_line"].unique())
    rows = []
    for cl in cell_lines:
        cl_mask = adata.obs["cell_line"] == cl
        for fold_id in range(5):
            fold_col = f"fold_{fold_id}"
            test_mask = cl_mask & (adata.obs[fold_col] == "test")
            test_cids = set(adata.obs.loc[test_mask, "pubchem_cid"].astype(int).unique())
            test_cids_restricted = test_cids & valid_cids

            # Match exp error rows for this cell line
            exp_cl = exp_q01[exp_q01["cell_line"] == cl]
            if len(exp_cl) == 0:
                # Try with Tahoe100M_ prefix
                exp_cl = exp_q01[exp_q01["cell_line"] == f"Tahoe100M_{cl}"]

            # Filter to fold
            exp_cl_fold = exp_cl[exp_cl["fold"] == fold_id]

            # Use the mean f1_score from the experimental error
            mean_f1 = exp_cl_fold["f1_score"].mean() if len(exp_cl_fold) > 0 else np.nan

            rows.append({
                "dataset": "",
                "fold": f"{cl}.{fold_id}",
                "metrics": "",
                "name": "experimental_error",
                "roc_auc": np.nan,
                "average_precision": np.nan,
                "f1_score": mean_f1,
                "recall": np.nan,
                "precision": np.nan,
                "f1_classwise": np.nan,
                "f1_score_nans_are_zeros": np.nan,
                "primary_metric": "f1_score",
                "cell_line": cl,
                "quantile": 0.1,
            })

    print(f"Computed {len(rows)} experimental error rows")
    return pd.DataFrame(rows)


def collect():
    task_dir = SUBMISSION_DIR / "expression" / "pert_prediction_tahoe_deg"

    if not task_dir.exists():
        print(f"No submissions found at {task_dir}")
        return

    rows = []
    json_files = list(task_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {task_dir}")

    for json_path in sorted(json_files):
        with open(json_path) as f:
            data = json.load(f)

        # Only include restricted runs
        description = data.get("description", "") or ""
        if "Restricted to" not in description:
            continue

        metrics = data.get("metrics", {})
        fold = data.get("fold", "")
        cell_line = fold.split(".")[0] if "." in fold else ""
        raw_name = data.get("name", "")

        rows.append({
            "timestamp": data.get("timestamp", ""),
            "dataset": data.get("dataset", ""),
            "fold": fold,
            "metrics": str(metrics),
            "name": normalize_name(raw_name),
            "roc_auc": metrics.get("roc_auc"),
            "average_precision": metrics.get("average_precision"),
            "f1_score": metrics.get("f1_score"),
            "recall": metrics.get("recall"),
            "precision": metrics.get("precision"),
            "f1_classwise": metrics.get("f1_classwise"),
            "f1_score_nans_are_zeros": metrics.get("f1_score_nans_are_zeros"),
            "primary_metric": metrics.get("primary_metric", "f1_score"),
            "cell_line": cell_line,
            "quantile": np.nan,
        })

    if not rows:
        print("No restricted submissions found")
        return

    df = pd.DataFrame(rows)

    # Deduplicate: keep latest submission per (name, fold)
    n_before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=["name", "fold"], keep="last")
    df = df.drop(columns=["timestamp"]).reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"Removed {n_dupes} duplicate submissions (kept latest)")

    # Add experimental error rows
    exp_df = compute_restricted_exp_error()
    df = pd.concat([df, exp_df], ignore_index=True)

    df.to_csv(OUTPUT_CSV, index=True)

    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Unique models: {df['name'].nunique()}")
    print(f"Unique cell lines: {df['cell_line'].nunique()}")
    print(f"Unique folds: {df['fold'].nunique()}")
    print(f"\nMean f1_score by model:")
    print(df.groupby("name")["f1_score"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    collect()
