"""
Collect LPM-restricted PCA embedding JSON submission files into CSV score files.

Produces 4 output CSVs (one per task/dose combination):
  - results/scores/tahoe_lfc_pca_emb_lpm_dose3.csv
  - results/scores/tahoe_lfc_pca_emb_lpm_dose10.csv
  - results/scores/tahoe_deg_pca_emb_lpm_dose3.csv
  - results/scores/tahoe_deg_pca_emb_lpm_dose10.csv

Usage:
    python collect_scores_pca_emb_lpm.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMISSION_DIR = Path(__file__).resolve().parent.parent.parent / "submissions"

BASELINE_ESTIMATORS_DEG = {"no_change", "prior", "most_frequent"}
BASELINE_ESTIMATORS_LFC = {"context mean", "no change"}

# LFC metrics columns
LFC_METRIC_COLS = ["L2", "MSE", "MAE", "Spearman", "Pearson"]

# DEG metrics columns
DEG_METRIC_COLS = [
    "roc_auc", "average_precision", "f1_score", "recall", "precision",
    "f1_classwise", "f1_score_nans_are_zeros",
]


def normalize_name_lfc(raw_name):
    """Normalize LFC submission name."""
    return raw_name


def normalize_name_deg(raw_name):
    """Normalize DEG submission name.

    logistic_regression_PCA.logFC_v1_64 -> PCA.logFC_v1_64
    no_change_random -> no_change
    prior_random -> prior
    most_frequent_random -> most_frequent
    logistic_regression_random -> random
    logistic_regression_pca -> pca
    """
    for base in BASELINE_ESTIMATORS_DEG:
        if raw_name.startswith(base + "_"):
            return base
    if raw_name.startswith("logistic_regression_"):
        return raw_name[len("logistic_regression_"):]
    return raw_name


def collect_lfc(task_dir, output_csv):
    """Collect LFC (regression) submissions."""
    if not task_dir.exists():
        print(f"No submissions found at {task_dir}")
        return

    rows = []
    json_files = list(task_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {task_dir}")

    for json_path in sorted(json_files):
        with open(json_path) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        fold = data.get("fold", "")
        cell_line = fold.split(".")[0] if "." in fold else ""

        rows.append({
            "timestamp": data.get("timestamp", ""),
            "dataset": data.get("dataset", ""),
            "fold": fold,
            "metrics": str(metrics),
            "name": normalize_name_lfc(data.get("name", "")),
            "primary_metric": metrics.get("primary_metric", "L2"),
            "L2": metrics.get("L2"),
            "MSE": metrics.get("MSE"),
            "MAE": metrics.get("MAE"),
            "Spearman": metrics.get("Spearman"),
            "Pearson": metrics.get("Pearson"),
            "cell_line": cell_line,
        })

    if not rows:
        print("No submissions found")
        return

    df = pd.DataFrame(rows)

    # Deduplicate: keep latest submission per (name, fold)
    n_before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=["name", "fold"], keep="last")
    df = df.drop(columns=["timestamp"]).reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"Removed {n_dupes} duplicate submissions (kept latest)")

    df.to_csv(output_csv, index=True)

    print(f"\nSaved {len(df)} rows to {output_csv}")
    print(f"Unique models: {df['name'].nunique()}")
    print(f"Unique cell lines: {df['cell_line'].nunique()}")
    print(f"Unique folds: {df['fold'].nunique()}")
    print(f"\nMean L2 by model:")
    print(df.groupby("name")["L2"].mean().sort_values().to_string())


def collect_deg(task_dir, output_csv):
    """Collect DEG (classification) submissions."""
    if not task_dir.exists():
        print(f"No submissions found at {task_dir}")
        return

    rows = []
    json_files = list(task_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {task_dir}")

    for json_path in sorted(json_files):
        with open(json_path) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        fold = data.get("fold", "")
        cell_line = fold.split(".")[0] if "." in fold else ""
        raw_name = data.get("name", "")

        rows.append({
            "timestamp": data.get("timestamp", ""),
            "dataset": data.get("dataset", ""),
            "fold": fold,
            "metrics": str(metrics),
            "name": normalize_name_deg(raw_name),
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
        print("No submissions found")
        return

    df = pd.DataFrame(rows)

    # Deduplicate: keep latest submission per (name, fold)
    n_before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=["name", "fold"], keep="last")
    df = df.drop(columns=["timestamp"]).reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"Removed {n_dupes} duplicate submissions (kept latest)")

    df.to_csv(output_csv, index=True)

    print(f"\nSaved {len(df)} rows to {output_csv}")
    print(f"Unique models: {df['name'].nunique()}")
    print(f"Unique cell lines: {df['cell_line'].nunique()}")
    print(f"Unique folds: {df['fold'].nunique()}")
    print(f"\nMean f1_score by model:")
    print(df.groupby("name")["f1_score"].mean().sort_values(ascending=False).to_string())


def main():
    print("=" * 60)
    print("Collecting LFC dose 3.33333 (LPM-restricted)")
    print("=" * 60)
    collect_lfc(
        SUBMISSION_DIR / "expression" / "pert_prediction_tahoe_regression_pca_emb_lpm_dose_3_33333",
        REPO_ROOT / "results" / "scores" / "tahoe_lfc_pca_emb_lpm_dose3.csv",
    )

    print("\n" + "=" * 60)
    print("Collecting LFC dose 10 (LPM-restricted)")
    print("=" * 60)
    collect_lfc(
        SUBMISSION_DIR / "expression" / "pert_prediction_tahoe_regression_pca_emb_lpm_dose_10_0",
        REPO_ROOT / "results" / "scores" / "tahoe_lfc_pca_emb_lpm_dose10.csv",
    )

    print("\n" + "=" * 60)
    print("Collecting DEG dose 3.33333 (LPM-restricted)")
    print("=" * 60)
    collect_deg(
        SUBMISSION_DIR / "expression" / "pert_prediction_tahoe_deg_pca_emb_lpm_dose_3_33333",
        REPO_ROOT / "results" / "scores" / "tahoe_deg_pca_emb_lpm_dose3.csv",
    )

    print("\n" + "=" * 60)
    print("Collecting DEG dose 10 (LPM-restricted)")
    print("=" * 60)
    collect_deg(
        SUBMISSION_DIR / "expression" / "pert_prediction_tahoe_deg_pca_emb_lpm_dose_10_0",
        REPO_ROOT / "results" / "scores" / "tahoe_deg_pca_emb_lpm_dose10.csv",
    )


if __name__ == "__main__":
    main()
