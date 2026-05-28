from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from chemical_embedding_pipeline.progress import progress, progress_range


MODEL_IDS = {
    "ChemBERTa-77M-MLM": "DeepChem/ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR": "DeepChem/ChemBERTa-77M-MTR",
    "MolT5": "laituan245/molt5-large",
}

ENCODER_ONLY_MODEL_NAMES = {"MolT5"}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolved_snapshot_path(model_or_tokenizer: object) -> str | None:
    name_or_path = getattr(model_or_tokenizer, "name_or_path", None)
    if name_or_path:
        return str(name_or_path)
    config = getattr(model_or_tokenizer, "config", None)
    config_name = getattr(config, "_name_or_path", None)
    if config_name:
        return str(config_name)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate transformer SMILES embeddings for benchmark compounds."
    )
    parser.add_argument("--dataset", choices=["sciplex", "tahoe"], required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Column containing SMILES. Defaults to main_molecule_smiles for Tahoe and canonical_smiles for SciPlex.",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=list(MODEL_IDS),
        choices=sorted(MODEL_IDS),
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "mean_no_special", "cls", "first_non_special", "eos", "max"],
        default="mean",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already-downloaded Hugging Face checkpoints.",
    )
    parser.add_argument(
        "--prefer-safetensors",
        action="store_true",
        help=(
            "Ask transformers to prefer safetensors checkpoints. By default, local "
            "offline runs use the cached PyTorch weights and avoid the background "
            "Hub safetensors conversion check."
        ),
    )
    return parser.parse_args()


def smiles_series(adata: ad.AnnData, column: str) -> list[str]:
    if column not in adata.obs:
        raise KeyError(f"{column!r} is not present in input H5AD obs.")
    return pd.Series(adata.obs[column].astype(object)).fillna("").astype(str).tolist()


def pool_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    special_token_ids: list[int],
    pooling: str,
) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    if pooling == "first_non_special":
        valid = attention_mask.bool()
        special = torch.zeros_like(valid)
        for token_id in special_token_ids:
            special |= input_ids.eq(token_id)
        candidate = valid & ~special
        first_index = candidate.float().argmax(dim=1)
        return last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            first_index,
        ]
    if pooling == "eos":
        return last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            attention_mask.sum(dim=1) - 1,
        ]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    if pooling == "mean_no_special":
        valid = attention_mask.bool()
        special = torch.zeros_like(valid)
        for token_id in special_token_ids:
            special |= input_ids.eq(token_id)
        mask = (valid & ~special).unsqueeze(-1).to(last_hidden_state.dtype)
    if pooling == "max":
        hidden = last_hidden_state.masked_fill(mask.eq(0), torch.finfo(last_hidden_state.dtype).min)
        return hidden.max(dim=1).values
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def encode_smiles(
    smiles: list[str],
    name: str,
    model_id: str,
    pooling: str,
    batch_size: int,
    device: str,
    local_files_only: bool,
    prefer_safetensors: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    model_cls = T5EncoderModel if name in ENCODER_ONLY_MODEL_NAMES else AutoModel
    model = model_cls.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        use_safetensors=prefer_safetensors,
    )
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in progress_range(0, len(smiles), batch_size, desc=f"{name} batches", unit="batch"):
            batch_smiles = smiles[start : start + batch_size]
            batch = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            pooled = pool_hidden_state(
                hidden,
                batch["attention_mask"],
                batch["input_ids"],
                tokenizer.all_special_ids,
                pooling,
            )
            chunks.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    metadata = {
        "tokenizer_path": resolved_snapshot_path(tokenizer),
        "model_path": resolved_snapshot_path(model),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "hidden_size": int(getattr(model.config, "hidden_size", getattr(model.config, "d_model", -1))),
        "special_token_ids": list(tokenizer.all_special_ids),
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
    }
    return np.concatenate(chunks, axis=0), metadata


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    smiles_column = args.smiles_column or (
        "main_molecule_smiles" if args.dataset == "tahoe" else "canonical_smiles"
    )
    adata = ad.read_h5ad(args.input_h5ad)
    smiles = smiles_series(adata, smiles_column)

    out = ad.AnnData(obs=adata.obs.copy())
    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "smiles_column": smiles_column,
        "pooling": args.pooling,
        "batch_size": args.batch_size,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "prefer_safetensors": args.prefer_safetensors,
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
        "embeddings": {},
    }
    for name in progress(args.embeddings, desc="Transformer embeddings", unit="model"):
        matrix, model_metadata = encode_smiles(
            smiles=smiles,
            name=name,
            model_id=MODEL_IDS[name],
            pooling=args.pooling,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=args.local_files_only,
            prefer_safetensors=args.prefer_safetensors,
        )
        out.obsm[name] = matrix
        manifest["embeddings"][name] = {
            "model_id": MODEL_IDS[name],
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            **model_metadata,
        }

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
