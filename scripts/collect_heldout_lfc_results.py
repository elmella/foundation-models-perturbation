#!/usr/bin/env python3
"""Collect fixed molecular holdout LFC submissions into score CSVs."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DATASETS = {
    "sciplex": "expression/pert_prediction_sciplex3_regression",
    "tahoe": "expression/pert_prediction_tahoe_regression",
}


def parse_name(name):
    if name.endswith(" baseline"):
        return name.removesuffix(" baseline"), ""
    match = re.match(r"^(?P<estimator>.+?)_baseline_(?P<embedding>.+)$", name)
    if match:
        return match.group("estimator"), match.group("embedding")
    return "", ""


def description_value(description, key):
    prefix = f"{key}:"
    for line in description.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def collect(submission_root, dataset_key, output_csv, split_labels=None):
    task_dir = submission_root / DATASETS[dataset_key]
    if not task_dir.exists():
        raise SystemExit(f"No submissions found at {task_dir}")

    rows = []
    for json_path in sorted(task_dir.rglob("*.json")):
        with json_path.open() as f:
            data = json.load(f)

        description = data.get("description", "") or ""
        if "Split: fixed molecular holdout" not in description:
            continue

        observed_split_labels = description_value(description, "Test split labels")
        if split_labels and observed_split_labels != split_labels:
            continue

        metrics = data.get("metrics", {})
        fold = data.get("fold", "")
        estimator, embedding = parse_name(data.get("name", ""))
        cell_line = fold.split(".", 1)[0] if "." in fold else ""

        rows.append({
            "timestamp": data.get("timestamp", ""),
            "dataset": data.get("dataset", ""),
            "fold": fold,
            "cell_line": cell_line,
            "split": fold.split(".", 1)[1] if "." in fold else "",
            "test_split_labels": observed_split_labels,
            "name": data.get("name", ""),
            "estimator": estimator,
            "embedding": embedding,
            "primary_metric": metrics.get("primary_metric", "L2"),
            "L2": metrics.get("L2"),
            "MSE": metrics.get("MSE"),
            "MAE": metrics.get("MAE"),
            "Spearman": metrics.get("Spearman"),
            "Pearson": metrics.get("Pearson"),
            "json_path": str(json_path),
        })

    if not rows:
        raise SystemExit(f"No matching fixed molecular holdout submissions found in {task_dir}")

    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["name", "fold", "test_split_labels"],
        keep="last",
    )
    n_dupes = n_before - len(df)
    df = df.sort_values(["cell_line", "estimator", "embedding"]).reset_index(drop=True)
    df = df.drop(columns=["timestamp"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} rows to {output_csv}")
    print(f"Removed {n_dupes} duplicate rows (kept latest)")
    print(f"Unique cell lines: {df['cell_line'].nunique()}")
    print(f"Unique embeddings: {df['embedding'].nunique()}")
    print(f"Unique estimators: {df['estimator'].nunique()}")
    print("\nBest mean L2 by embedding/estimator:")
    print(
        df.groupby(["embedding", "estimator"])["L2"]
        .mean()
        .sort_values()
        .head(20)
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--test-split-labels",
        default=None,
        help="Optional exact label marker to keep, for example 'test' or 'test,val'.",
    )
    args = parser.parse_args()

    collect(
        submission_root=args.submission_root,
        dataset_key=args.dataset,
        output_csv=args.output_csv,
        split_labels=args.test_split_labels,
    )


if __name__ == "__main__":
    main()
