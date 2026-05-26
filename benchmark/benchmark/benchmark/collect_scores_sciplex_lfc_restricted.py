"""
Collect restricted sci-Plex LFC JSON submissions into a score CSV.

Reads restricted submissions produced by bench_sciplex_lfc_restricted.py and
writes results/scores/sciplex_lfc_restricted.csv.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SUBMISSION_DIR = Path(__file__).resolve().parent.parent.parent / "submissions"
SUBMISSION_DIR = Path(os.environ.get("SCIPLEX_SUBMISSION_DIR", DEFAULT_SUBMISSION_DIR))
OUTPUT_CSV = Path(
    os.environ.get(
        "SCIPLEX_OUTPUT_CSV",
        REPO_ROOT / "results" / "scores" / "sciplex_lfc_restricted.csv",
    )
)
RESTRICTION_MARKER = os.environ.get("SCIPLEX_RESTRICTION_MARKER", "Restricted to")
PKL_PATH = REPO_ROOT / "molecule_embeddings" / "tahoe_sci_op3_updated.pkl"
EXP_ERROR_CSV = REPO_ROOT / "results" / "exp_error" / "exp_err_sciplex_outer_10_inner_20_seed_5050.csv"


def get_valid_drugs_from_submissions():
    valid_drugs = None
    task_dir = SUBMISSION_DIR / "expression" / "pert_prediction_sciplex3_regression"
    for json_path in sorted(task_dir.rglob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        description = data.get("description", "") or ""
        if RESTRICTION_MARKER not in description:
            continue
        restriction_path = None
        restriction_format = "tahoe_sci_op3"
        for line in description.splitlines():
            if line.startswith("Restriction source:"):
                restriction_path = Path(line.split(":", 1)[1].strip())
            if line.startswith("Restriction source format:"):
                restriction_format = line.split(":", 1)[1].strip()
        if restriction_path is None:
            continue
        pkl_df = pd.read_pickle(restriction_path)
        if restriction_format == "pubchem_symbol_lpm_style":
            old_df = pd.read_pickle(PKL_PATH)
            sciplex_pkl = old_df[old_df["dataset"] == "sciplex3"].dropna(
                subset=["pubchem_cid", "original_pert_name"]
            )
            cid_to_drug = dict(
                zip(
                    sciplex_pkl["pubchem_cid"].astype(int).astype(str),
                    sciplex_pkl["original_pert_name"].astype(str),
                )
            )
            valid_cids = set(
                pkl_df[pkl_df["lpm_style_embeddings"].notna()]["symbol"].astype(str).values
            )
            valid_drugs = {cid_to_drug[cid] for cid in valid_cids & set(cid_to_drug)}
        else:
            sciplex_pkl = pkl_df[pkl_df["dataset"] == "sciplex3"]
            sciplex_lpm = sciplex_pkl[sciplex_pkl["LPM_emb"].notna()].drop_duplicates(
                subset="original_pert_name", keep="first"
            )
            valid_drugs = set(sciplex_lpm["original_pert_name"].astype(str).values)
        break
    if valid_drugs is not None:
        return valid_drugs

    pkl_df = pd.read_pickle(PKL_PATH)
    sciplex_pkl = pkl_df[pkl_df["dataset"] == "sciplex3"]
    sciplex_lpm = sciplex_pkl[sciplex_pkl["LPM_emb"].notna()].drop_duplicates(
        subset="original_pert_name", keep="first"
    )
    return set(sciplex_lpm["original_pert_name"].astype(str).values)


def compute_restricted_exp_error():
    """Compute experimental error (q=0.9) on restricted drugs per (cell_line, fold)."""
    import anndata as ad
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark import paths

    valid_drugs = get_valid_drugs_from_submissions()

    exp = pd.read_csv(EXP_ERROR_CSV)
    exp_q09 = exp[exp["quantile"] == 0.9].copy()

    adata = ad.read_h5ad(paths.SCIPLEX_PSEUDOBULK_FILTERED)
    cell_lines = sorted(adata.obs["cell_line"].unique())
    rows = []
    for cl in cell_lines:
        cl_mask = adata.obs["cell_line"] == cl
        for fold_id in range(5):
            fold_col = f"fold_{fold_id}"
            test_mask = cl_mask & (adata.obs[fold_col] == "test")
            test_drugs = set(adata.obs.loc[test_mask, "drug"].astype(str).unique())
            test_drugs_restricted = test_drugs & valid_drugs

            exp_cl = exp_q09[exp_q09["cell_line"] == f"sciplex_{cl}"]
            exp_cl_test = exp_cl[exp_cl["pert_id"].isin(test_drugs_restricted)]
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

    print(f"Computed {len(rows)} experimental error rows on restricted drugs")
    return pd.DataFrame(rows)


def collect():
    task_dir = SUBMISSION_DIR / "expression" / "pert_prediction_sciplex3_regression"

    if not task_dir.exists():
        print(f"No submissions found at {task_dir}")
        return

    rows = []
    json_files = list(task_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {task_dir}")

    for json_path in sorted(json_files):
        with open(json_path) as f:
            data = json.load(f)

        description = data.get("description", "") or ""
        if RESTRICTION_MARKER not in description:
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

    n_before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=["name", "fold"], keep="last")
    df = df.drop(columns=["timestamp"]).reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"Removed {n_dupes} duplicate submissions (kept latest)")

    exp_df = compute_restricted_exp_error()
    df = pd.concat([df, exp_df], ignore_index=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=True)

    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Unique models: {df['name'].nunique()}")
    print(f"Unique cell lines: {df['cell_line'].nunique()}")
    print(f"Unique folds: {df['fold'].nunique()}")
    print("\nMean L2 by model:")
    print(df.groupby("name")["L2"].mean().sort_values().to_string())


if __name__ == "__main__":
    collect()
