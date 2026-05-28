#!/usr/bin/env python3
"""Create fixed molecular holdout LFC result figures from score CSVs."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCIPLEX_LABELS = {
    "morgan_initialized_lpm": "Morgan-initialized LPM",
    "random": "Random Embeddings",
    "pca": "PCA / idealized baseline",
    "ChemBERTa-77M-MLM": "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR": "ChemBERTa-77M-MTR",
    "chatgpt": "ChatGPT",
    "maccs": "MACCS",
    "topological": "Topological",
    "secfp": "SECFP",
    "avalon": "Avalon",
    "erg": "ErG",
    "ecfp": "ECFP",
    "fcfp": "FCFP",
    "cats2D": "CATS2D",
    "cats3D": "CATS3D",
}


TAHOE_LABELS = {
    "morgan_initialized_lpm": ("Morgan-initialized LPM", "Response Embedding"),
    "random": ("Random Embeddings", "Negative Control"),
    "pca": ("PCA / idealized baseline", "Positive Control"),
    "ChemBERTa-77M-MLM": ("ChemBERTa-77M-MLM", "SMILES Transformer"),
    "ChemBERTa-77M-MTR": ("ChemBERTa-77M-MTR", "SMILES Transformer"),
    "chatgpt": ("ChatGPT", "LLM"),
    "maccs": ("MACCS", "Molecule Structure"),
    "topological": ("Topological", "Molecule Structure"),
    "secfp": ("SECFP", "Molecule Structure"),
    "avalon": ("Avalon", "Molecule Structure"),
    "erg": ("ErG", "Molecule Structure"),
    "ecfp:2": ("ECFP:2", "Molecule Structure"),
    "MiniMol": ("MiniMol", "Molecule Structure"),
    "MolT5": ("MolT5", "SMILES Transformer"),
    "unimol2-570m-H": ("Uni-Mol2 570M-H", "Molecule Structure"),
    "boltz_affinity_pred_value_fragment": ("Boltz affinity fragment", "Protein Affinity"),
    "boltz_affinity_pred_value_protein": ("Boltz affinity protein", "Protein Affinity"),
}


def load_figure_index(repo_root):
    path = repo_root / "results" / "metadata" / "fig_index.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_figure(fig, fig_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sciplex(score_csv, fig_dir, include_pca):
    df = pd.read_csv(score_csv)
    if not include_pca:
        df = df[df["embedding"] != "pca"].copy()
    df = df[df["embedding"] != "ECFP:2_pkl"].copy()
    df["Display name"] = df["embedding"].map(lambda x: SCIPLEX_LABELS.get(x, x))

    fig_dir.mkdir(parents=True, exist_ok=True)
    cell_palette = {"A549": "#4C78A8", "K562": "#F58518", "MCF7": "#54A24B"}
    cell_order = ["A549", "K562", "MCF7"]
    suffix = "with_pca" if include_pca else "no_pca"

    for estimator, title in [("knn", "KNN"), ("lasso", "LASSO")]:
        d = df[df["estimator"] == estimator].copy()
        if d.empty:
            continue
        order = d.groupby("Display name", observed=True)["L2"].mean().sort_values().index.tolist()
        fig_h = max(6.5, 0.36 * len(order) + 1.6)
        fig, ax = plt.subplots(figsize=(8.8, fig_h), constrained_layout=True)
        sns.stripplot(
            data=d,
            y="Display name",
            order=order,
            x="L2",
            hue="cell_line",
            hue_order=[c for c in cell_order if c in set(d["cell_line"])],
            palette=cell_palette,
            dodge=False,
            jitter=0.18,
            alpha=0.9,
            s=7,
            ax=ax,
        )
        means = d.groupby("Display name", observed=True)["L2"].mean().reindex(order)
        y_positions = {name: i for i, name in enumerate(order)}
        ax.scatter(
            means.values,
            [y_positions[name] for name in order],
            marker="D",
            s=42,
            color="black",
            label="Mean",
            zorder=10,
        )
        ax.axvline(means.min(), color="black", linestyle="--", linewidth=1, alpha=0.55)
        ax.grid(axis="y")
        ax.set_title(f"SciPlex LFC heldout molecules test-only ({title})")
        ax.set_xlabel("L2")
        ax.set_ylabel(None)
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique.append((handle, label))
        ax.legend(
            [handle for handle, _ in unique],
            [label for _, label in unique],
            title=None,
            frameon=False,
            loc="lower right",
        )
        save_figure(fig, fig_dir, f"sciplex_lfc_heldout_test_only_{suffix}_{estimator}")


def plot_tahoe(score_csv, fig_dir, include_pca, fig_index):
    df = pd.read_csv(score_csv)
    if not include_pca:
        df = df[df["embedding"] != "pca"].copy()
    df["Display name"] = df["embedding"].map(lambda x: TAHOE_LABELS.get(x, (x, "Molecule Structure"))[0])
    df["Model type"] = df["embedding"].map(lambda x: TAHOE_LABELS.get(x, (x, "Molecule Structure"))[1])

    palette = {
        **fig_index.get("drugs_model_type_palette", {}),
        "Response Embedding": "#4C78A8",
        "Negative Control": "#9E9E9E",
        "Positive Control": "#000000",
        "Molecule Structure": "#F58518",
        "SMILES Transformer": "#54A24B",
        "LLM": "#B279A2",
        "Protein Affinity": "#E45756",
    }

    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "with_pca" if include_pca else "no_pca"

    for estimator, title in [("knn", "KNN"), ("lasso", "LASSO")]:
        d = df[df["estimator"] == estimator].copy()
        if d.empty:
            continue
        order = d.groupby("Display name", observed=True)["L2"].mean().sort_values().index.tolist()
        fig_h = max(6.8, 0.38 * len(order) + 1.8)
        fig, ax = plt.subplots(figsize=(9.2, fig_h), constrained_layout=True)
        sns.pointplot(
            data=d,
            y="Display name",
            order=order,
            x="L2",
            hue="Model type",
            palette=palette,
            join=False,
            dodge=False,
            errorbar=("ci", 95),
            n_boot=5000,
            seed=42,
            markers="D",
            scale=0.85,
            errwidth=1.4,
            capsize=0.18,
            ax=ax,
        )
        ax.axvline(
            d.groupby("Display name", observed=True)["L2"].mean().min(),
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.55,
        )
        ax.grid(axis="y")
        ax.set_title(f"Tahoe LFC heldout molecules test-only ({title})")
        ax.set_xlabel("Mean L2 with 95% CI across cell lines")
        ax.set_ylabel(None)
        ax.legend(title=None, frameon=False, loc="lower right")
        save_figure(fig, fig_dir, f"tahoe_lfc_heldout_test_only_{suffix}_{estimator}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sciplex-csv", type=Path, required=True)
    parser.add_argument("--tahoe-csv", type=Path, required=True)
    parser.add_argument(
        "--sciplex-fig-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tahoe-fig-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--include-pca", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    fig_index = load_figure_index(repo_root)
    mpl.rcParams.update(fig_index.get("mpl_params", {}))

    sciplex_fig_dir = args.sciplex_fig_dir or (
        repo_root / "results" / "figures" / "fig_sciplex_lfc_heldout_molecules_test_only"
    )
    tahoe_fig_dir = args.tahoe_fig_dir or (
        repo_root / "results" / "figures" / "fig_tahoe_lfc_heldout_molecules_test_only"
    )

    plot_sciplex(args.sciplex_csv, sciplex_fig_dir, include_pca=False)
    plot_tahoe(args.tahoe_csv, tahoe_fig_dir, include_pca=False, fig_index=fig_index)
    if args.include_pca:
        plot_sciplex(args.sciplex_csv, sciplex_fig_dir, include_pca=True)
        plot_tahoe(args.tahoe_csv, tahoe_fig_dir, include_pca=True, fig_index=fig_index)

    print(f"Saved SciPlex figures to {sciplex_fig_dir}")
    print(f"Saved Tahoe figures to {tahoe_fig_dir}")


if __name__ == "__main__":
    main()
