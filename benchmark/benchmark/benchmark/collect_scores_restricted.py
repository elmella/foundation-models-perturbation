"""
Collect JSON submission files into a single CSV score file.

Reads all JSON files from the submissions directory produced by
bench_tahoe_lfc_restricted.py and aggregates them into a CSV matching
the format of results/scores/tahoe_lfc.csv.

Usage:
    python collect_scores_restricted.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMISSION_DIR = Path(__file__).resolve().parent.parent.parent / "submissions"
OUTPUT_CSV = REPO_ROOT / "results" / "scores" / "tahoe_lfc_restricted.csv"
PKL_PATH = REPO_ROOT / "molecule_embeddings" / "tahoe_sci_op3_updated.pkl"
EXP_ERROR_CSV = REPO_ROOT / "results" / "exp_error" / "exp_err_Tahoe_outer_10_inner_20_seed_5050.csv"


def compute_restricted_exp_error():
    """Compute experimental error (q=0.9) on restricted compounds per (cell_line, fold)."""
    import anndata as ad
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark import paths

    # Valid CIDs
    pkl_df = pd.read_pickle(PKL_PATH)
    tahoe_pkl = pkl_df[pkl_df["dataset"] == "tahoe"]
    tahoe_lpm = tahoe_pkl[tahoe_pkl["LPM_emb"].notna()].drop_duplicates(subset="pubchem_cid", keep="first")
    valid_cids = set(tahoe_lpm["pubchem_cid"].astype(int).values)

    # Per-perturbation exp error (q=0.9)
    exp = pd.read_csv(EXP_ERROR_CSV)
    exp_q09 = exp[exp["quantile"] == 0.9].copy()

    # cid -> drug name mapping
    adata = ad.read_h5ad(paths.TAHOE_PSEUDOBULK_WITH_SPLITS)
    cid_drug = adata.obs[["pubchem_cid", "drug"]].drop_duplicates()
    cid_to_drug = dict(zip(cid_drug["pubchem_cid"].astype(int), cid_drug["drug"]))

    # For each (cell_line, fold), compute mean L2 over restricted test drugs
    cell_lines = sorted(adata.obs["cell_line"].unique())
    rows = []
    for cl in cell_lines:
        cl_mask = adata.obs["cell_line"] == cl
        for fold_id in range(5):
            fold_col = f"fold_{fold_id}"
            test_mask = cl_mask & (adata.obs[fold_col] == "test")
            test_cids = set(adata.obs.loc[test_mask, "pubchem_cid"].astype(int).unique())
            test_cids_restricted = test_cids & valid_cids
            test_drugs = {cid_to_drug[c] for c in test_cids_restricted if c in cid_to_drug}

            exp_cl = exp_q09[exp_q09["cell_line"] == f"Tahoe100M_{cl}"]
            exp_cl_test = exp_cl[exp_cl["pert_id"].isin(test_drugs)]

            mean_l2 = exp_cl_test["value"].mean() if len(exp_cl_test) > 0 else np.nan

            rows.append({
                "dataset": "",
                "fold": f"{cl}.{fold_id}",
                "metrics": "",
                "name": "Experimental error (quantile 0.9)",
                "primary_metric": "",
                "L2": mean_l2,
                "MSE": np.nan,
                "MAE": np.nan,
                "Spearman": np.nan,
                "Pearson": np.nan,
                "cell_line": cl,
            })

    print(f"Computed {len(rows)} experimental error rows on restricted compounds")
    return pd.DataFrame(rows)


def collect():
    # Look for tahoe regression submissions
    task_dir = SUBMISSION_DIR / "expression" / "pert_prediction_tahoe_regression"

    if not task_dir.exists():
        print(f"No submissions found at {task_dir}")
        return

    rows = []
    json_files = list(task_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {task_dir}")

    for json_path in sorted(json_files):
        with open(json_path) as f:
            data = json.load(f)

        # Only include restricted runs (check description)
        description = data.get("description", "") or ""
        if "Restricted to" not in description:
            continue

        metrics = data.get("metrics", {})
        fold = data.get("fold", "")
        cell_line = fold.split(".")[0] if "." in fold else ""

        rows.append({
            "timestamp": data.get("timestamp", ""),
            "dataset": data.get("dataset", ""),
            "fold": fold,
            "metrics": str(metrics),
            "name": data.get("name", ""),
            "primary_metric": metrics.get("primary_metric", "L2"),
            "L2": metrics.get("L2"),
            "MSE": metrics.get("MSE"),
            "MAE": metrics.get("MAE"),
            "Spearman": metrics.get("Spearman"),
            "Pearson": metrics.get("Pearson"),
            "cell_line": cell_line,
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

    # Add experimental error rows (restricted to same compounds)
    exp_df = compute_restricted_exp_error()
    df = pd.concat([df, exp_df], ignore_index=True)

    df.to_csv(OUTPUT_CSV, index=True)

    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Unique models: {df['name'].nunique()}")
    print(f"Unique cell lines: {df['cell_line'].nunique()}")
    print(f"Unique folds: {df['fold'].nunique()}")
    print(f"\nMean L2 by model:")
    print(df.groupby("name")["L2"].mean().sort_values().to_string())


if __name__ == "__main__":
    collect()
