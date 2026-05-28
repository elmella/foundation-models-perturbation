from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import anndata as ad
import pandas as pd

from chemical_embedding_pipeline.env import load_dotenv_key
from chemical_embedding_pipeline.progress import progress


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

SMILES_CANDIDATES = [
    "canonical_smiles",
    "main_molecule_smiles",
    "smiles",
    "SMILES",
    "canonical_SMILES",
    "molecule_smiles",
    "compound_smiles",
]
COMPOUND_ID_CANDIDATES = [
    "pubchem_cid",
    "cid",
    "drug",
    "compound",
    "compound_name",
    "perturbagen",
    "pert_iname",
    "pert_name",
    "pert_id",
    "perturbation",
    "perturbation_name",
    "condition",
    "name",
]
CID_CANDIDATES = ["pubchem_cid", "cid", "PubChem CID", "pubchem"]

STRUCTURE_EMBEDDINGS = [
    "avalon",
    "cats2D",
    "cats3D",
    "ecfp",
    "ecfp:2",
    "erg",
    "fcfp",
    "maccs",
    "secfp",
    "topological",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all configured compound embeddings for arbitrary H5AD datasets."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-h5ad", action="append", type=Path, help="Local input H5AD. Repeatable.")
    source.add_argument("--input-dir", type=Path, help="Directory containing input H5AD files.")
    source.add_argument("--hf-dataset", help="Hugging Face dataset repo id, e.g. theislab/temp.")
    parser.add_argument(
        "--hf-file",
        action="append",
        default=None,
        help="H5AD file path inside --hf-dataset. Repeatable. Defaults to --hf-pattern matches.",
    )
    parser.add_argument("--hf-pattern", default="*.h5ad")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument(
        "--list-hf-files",
        action="store_true",
        help="List matching files in --hf-dataset and exit without downloading.",
    )
    parser.add_argument(
        "--max-hf-files",
        type=int,
        default=1,
        help=(
            "Maximum matched Hugging Face files to download when --hf-file is not provided. "
            "Default is 1 to avoid accidentally downloading a whole dataset repository."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated_lfc_embeddings/new_datasets"))
    parser.add_argument("--compound-id-column", default=None)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--cid-column", default=None)
    parser.add_argument(
        "--smiles-cache-json",
        type=Path,
        default=Path("generated_lfc_embeddings/pubchem_canonical_smiles_cache.json"),
    )
    parser.add_argument(
        "--pubchem-sleep",
        type=float,
        default=0.0,
        help="Sleep between PubChem metadata requests when filling missing SMILES.",
    )
    parser.add_argument(
        "--pubchem-workers",
        type=int,
        default=2,
        help="Concurrent PubChem chunk requests when SMILES are missing.",
    )
    parser.add_argument(
        "--pubchem-chunk-size",
        type=int,
        default=100,
        help="Number of CIDs per PubChem property request.",
    )
    parser.add_argument(
        "--pubchem-retries",
        type=int,
        default=5,
        help="Retries per PubChem chunk request when PUG REST is busy.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "structure",
            "transformers",
            "minimol",
            "pretrained_molfeat",
            "pubchem",
            "random",
            "openai_paper",
            "openai_sota",
        ],
    )
    parser.add_argument("--include-openai", action="store_true")
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-3d", action="store_true", help="Skip conformer-dependent 3D structure embeddings.")
    parser.add_argument("--minimol-python", type=Path, default=Path(".chemembed-minimol-py311/bin/python"))
    parser.add_argument(
        "--pretrained-molfeat-python",
        type=Path,
        default=Path(".chemembed-pretrained-molfeat-py311/bin/python"),
    )
    parser.add_argument("--legacy-secfp-python", type=Path, default=None)
    parser.add_argument(
        "--pubchem-cache-json",
        type=Path,
        default=Path("generated_lfc_embeddings/pubchem_fingerprint2d_cache.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[object], dry_run: bool) -> None:
    is_merge = "chemical_embedding_pipeline.merge_embedding_h5ads" in {str(part) for part in command}
    if "--output-h5ad" in command and not is_merge:
        output_index = command.index("--output-h5ad") + 1
        output_h5ad = Path(command[output_index])
        if output_h5ad.exists() and not dry_run:
            print(f"Skipping existing {output_h5ad}")
            return
    print("$ " + " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run([str(part) for part in command], cwd=REPO_ROOT, check=True)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "dataset"


def first_existing(columns: pd.Index, candidates: list[str], requested: str | None = None) -> str | None:
    if requested is not None:
        if requested not in columns:
            raise KeyError(f"{requested!r} is not present in obs columns.")
        return requested
    lower = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def local_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_h5ad:
        return args.input_h5ad
    if args.input_dir:
        return sorted(args.input_dir.glob("*.h5ad"))
    return download_hf_inputs(args)


def download_hf_inputs(args: argparse.Namespace) -> list[Path]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --hf-dataset. Install it or download H5ADs locally."
        ) from exc

    repo_id = args.hf_dataset
    files = args.hf_file
    if files is None:
        api = HfApi()
        matches = [
            path
            for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=args.hf_revision)
            if fnmatch.fnmatch(path, args.hf_pattern)
        ]
        if args.list_hf_files:
            for path in matches:
                print(path)
            raise SystemExit(0)
        if args.max_hf_files is not None and len(matches) > args.max_hf_files:
            preview = "\n".join(matches[: min(20, len(matches))])
            raise ValueError(
                f"{len(matches)} files matched {args.hf_pattern!r} in {repo_id!r}, which exceeds "
                f"--max-hf-files={args.max_hf_files}. Pass --hf-file for specific files, "
                "--list-hf-files to inspect matches, or raise --max-hf-files.\n"
                f"First matches:\n{preview}"
            )
        files = matches
    elif args.list_hf_files:
        for path in progress(files, desc="Downloading HF H5ADs", unit="file"):
            print(path)
        raise SystemExit(0)
    if not files:
        raise FileNotFoundError(f"No H5AD files matched in {repo_id!r}.")

    download_dir = args.output_dir / "_hf_downloads" / slugify(repo_id)
    paths = []
    for file_name in files:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type="dataset",
            revision=args.hf_revision,
            local_dir=download_dir,
        )
        paths.append(Path(path))
    return paths


def extract_compounds(
    input_h5ad: Path,
    output_h5ad: Path,
    compound_id_column: str | None,
    smiles_column: str | None,
    cid_column: str | None,
    smiles_cache_json: Path,
    pubchem_sleep: float,
    pubchem_workers: int,
    pubchem_chunk_size: int,
    pubchem_retries: int,
    skip_network: bool,
) -> dict[str, object]:
    adata = ad.read_h5ad(input_h5ad, backed="r")
    obs = adata.obs.copy()
    if obs.empty:
        raise ValueError(f"{input_h5ad} has empty obs metadata.")

    resolved_smiles = first_existing(obs.columns, SMILES_CANDIDATES, smiles_column)
    resolved_id = first_existing(obs.columns, COMPOUND_ID_CANDIDATES, compound_id_column)
    resolved_cid = first_existing(obs.columns, CID_CANDIDATES, cid_column)
    if resolved_id is None and resolved_cid is not None:
        resolved_id = resolved_cid
    elif resolved_id is None:
        resolved_id = resolved_smiles
    if resolved_id is None:
        raise KeyError(
            f"Could not infer a compound ID column in {input_h5ad}. "
            f"Tried: {', '.join(COMPOUND_ID_CANDIDATES)}. Pass --compound-id-column."
        )

    frame = obs.copy()
    frame["_compound_id"] = frame[resolved_id].astype(object).fillna("").astype(str)
    if resolved_smiles is None:
        if resolved_cid is None:
            raise KeyError(
                f"Could not infer a SMILES or PubChem CID column in {input_h5ad}. "
                f"Tried SMILES: {', '.join(SMILES_CANDIDATES)}. Pass --smiles-column."
            )
        frame["_smiles"] = fill_smiles_from_pubchem(
            frame[resolved_cid].astype(object).fillna("").astype(str).tolist(),
            smiles_cache_json,
            pubchem_sleep,
            pubchem_workers,
            pubchem_chunk_size,
            pubchem_retries,
            skip_network,
        )
        smiles_source = "pubchem_cid"
    else:
        frame["_smiles"] = frame[resolved_smiles].astype(object).fillna("").astype(str)
        smiles_source = resolved_smiles
    frame = frame[(frame["_compound_id"] != "") & (frame["_smiles"] != "")]
    if frame.empty:
        raise ValueError(f"No rows in {input_h5ad} have both compound id and SMILES.")
    compounds = frame.drop_duplicates("_compound_id", keep="first").copy()
    compounds.index = pd.Index(compounds["_compound_id"].astype(str), name="compound_id")

    compounds["canonical_smiles"] = compounds["_smiles"].astype(str)
    compounds["main_molecule_smiles"] = compounds["_smiles"].astype(str)
    if "drug" not in compounds.columns:
        compounds["drug"] = compounds.index.astype(str)
    if resolved_cid is not None and "pubchem_cid" not in compounds.columns:
        compounds["pubchem_cid"] = compounds[resolved_cid].astype(object).fillna("").astype(str)
    compounds = compounds.drop(columns=["_compound_id", "_smiles"])

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(obs=compounds).write_h5ad(output_h5ad)
    manifest = {
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "source_rows": int(obs.shape[0]),
        "compound_rows": int(compounds.shape[0]),
        "compound_id_column": resolved_id,
        "smiles_column": resolved_smiles,
        "smiles_source": smiles_source,
        "smiles_cache_json": str(smiles_cache_json) if resolved_smiles is None else None,
        "cid_column": resolved_cid,
        "obs_columns": compounds.columns.tolist(),
    }
    output_h5ad.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_json_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def canonical_smiles_for_cid(cid: str, cache: dict[str, str], sleep: float, skip_network: bool) -> str:
    cid = str(cid).strip()
    if not cid or cid.lower() == "nan":
        return ""
    if cid in cache:
        return cache[cid]
    if skip_network:
        return ""
    encoded = urllib.parse.quote(cid)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{encoded}/property/CanonicalSMILES/JSON"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    props = payload["PropertyTable"]["Properties"][0]
    value = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES") or props.get("SMILES") or ""
    cache[cid] = value
    if sleep:
        time.sleep(sleep)
    return value


def chunked(values: list[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [values[start : start + size] for start in range(0, len(values), size)]


def smiles_from_props(props: dict[str, object]) -> str:
    for key in ["CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "SMILES"]:
        value = props.get(key)
        if value:
            return str(value)
    return ""


def fetch_pubchem_smiles_chunk(cids: list[str], retries: int, sleep: float) -> dict[str, str]:
    encoded = urllib.parse.quote(",".join(cids))
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{encoded}/property/CanonicalSMILES,IsomericSMILES/JSON"
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = payload.get("PropertyTable", {}).get("Properties", [])
            result = {}
            for props in rows:
                cid = str(props.get("CID", "")).strip()
                smiles = smiles_from_props(props)
                if cid and smiles:
                    result[cid] = smiles
            if sleep:
                time.sleep(sleep)
            return result
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = exc
        time.sleep(min(30.0, (2**attempt) + 0.25 * attempt))
    assert last_error is not None
    raise last_error


def fill_smiles_from_pubchem(
    cids: list[str],
    cache_json: Path,
    sleep: float,
    workers: int,
    chunk_size: int,
    retries: int,
    skip_network: bool,
) -> list[str]:
    cache = load_json_cache(cache_json)
    normalized = [str(cid).strip() for cid in cids]
    unique_cids = sorted(
        {
            cid
            for cid in normalized
            if cid and cid.lower() != "nan" and (cid not in cache or not cache[cid])
        }
    )
    if unique_cids and not skip_network:
        chunks = chunked(unique_cids, chunk_size)
        worker_count = max(1, min(workers, len(chunks)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(fetch_pubchem_smiles_chunk, chunk, retries, sleep): chunk
                for chunk in chunks
            }
            for future in progress(
                as_completed(futures),
                desc="Resolving PubChem SMILES",
                total=len(futures),
                unit="chunk",
            ):
                chunk = futures[future]
                try:
                    cache.update(future.result())
                except Exception as exc:
                    preview = ", ".join(chunk[:3])
                    print(
                        "Warning: failed to resolve PubChem CID chunk "
                        f"({preview}, ...; n={len(chunk)}): {type(exc).__name__}: {exc}"
                    )
    values = [cache.get(cid, "") if cid and cid.lower() != "nan" else "" for cid in normalized]
    resolved_count = sum(bool(value) for value in values)
    if normalized and not skip_network and resolved_count == 0:
        raise ValueError(
            "PubChem SMILES lookup did not resolve any CIDs. "
            "Check network access, PubChem rate limiting, or pass --smiles-column with a local SMILES column."
        )
    save_json_cache(cache_json, cache)
    return values


def expanded_families(args: argparse.Namespace) -> set[str]:
    families = set(args.families)
    if "all" in families:
        families.update({"structure", "transformers", "random", "minimol", "pretrained_molfeat", "pubchem"})
    if args.include_openai:
        families.update({"openai_paper", "openai_sota"})
    families.discard("all")
    if args.skip_network:
        families.discard("pubchem")
    return families


def append_if_exists(
    commands: list[list[object]],
    command: list[object],
    executable: Path | None,
    skipped: list[dict[str, str]],
    family: str,
) -> bool:
    if executable is not None and not executable.exists():
        skipped.append({"family": family, "reason": f"missing executable: {executable}"})
        return False
    commands.append(command)
    return True


def build_commands(
    compound_h5ad: Path,
    out_dir: Path,
    prefix: str,
    families: set[str],
    args: argparse.Namespace,
    manifest: dict[str, object],
) -> tuple[list[list[object]], list[dict[str, str]], list[Path]]:
    commands: list[list[object]] = []
    skipped: list[dict[str, str]] = []
    generated: list[Path] = []

    if "structure" in families:
        embeddings = list(STRUCTURE_EMBEDDINGS)
        if args.skip_3d and "cats3D" in embeddings:
            embeddings.remove("cats3D")
        if args.legacy_secfp_python is not None and "secfp" in embeddings:
            embeddings.remove("secfp")
        output = out_dir / f"{prefix}_structure_embeddings.h5ad"
        commands.append(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_molfeat_embeddings",
                "--dataset",
                "sciplex",
                "--input-h5ad",
                compound_h5ad,
                "--output-h5ad",
                output,
                "--smiles-column",
                "canonical_smiles",
                "--embeddings",
                *embeddings,
            ]
        )
        generated.append(output)
        if args.legacy_secfp_python is not None:
            output = out_dir / f"{prefix}_legacy_secfp_embeddings.h5ad"
            added = append_if_exists(
                commands,
                [
                    args.legacy_secfp_python,
                    "-m",
                    "chemical_embedding_pipeline.generate_molfeat_embeddings",
                    "--dataset",
                    "sciplex",
                    "--input-h5ad",
                    compound_h5ad,
                    "--output-h5ad",
                    output,
                    "--smiles-column",
                    "canonical_smiles",
                    "--embeddings",
                    "secfp",
                ],
                args.legacy_secfp_python,
                skipped,
                "legacy_secfp",
            )
            if added:
                generated.append(output)

    if "transformers" in families:
        for token, pooling, names in [
            ("chemberta_mlm", "mean_no_special", ["ChemBERTa-77M-MLM"]),
            ("chemberta_mtr", "mean", ["ChemBERTa-77M-MTR"]),
            ("molt5", "cls", ["MolT5"]),
        ]:
            output = out_dir / f"{prefix}_{token}_embeddings.h5ad"
            command: list[object] = [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_transformer_embeddings",
                "--dataset",
                "tahoe",
                "--input-h5ad",
                compound_h5ad,
                "--output-h5ad",
                output,
                "--smiles-column",
                "canonical_smiles",
                "--pooling",
                pooling,
                "--embeddings",
                *names,
            ]
            if args.local_files_only:
                command.append("--local-files-only")
            commands.append(command)
            generated.append(output)

    if "random" in families:
        output = out_dir / f"{prefix}_random_embeddings.h5ad"
        commands.append(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_random_embeddings",
                "--dataset",
                "tahoe",
                "--input-h5ad",
                compound_h5ad,
                "--output-h5ad",
                output,
            ]
        )
        generated.append(output)

    if "minimol" in families:
        output = out_dir / f"{prefix}_minimol_embeddings.h5ad"
        added = append_if_exists(
            commands,
            [
                args.minimol_python,
                "-m",
                "chemical_embedding_pipeline.generate_minimol_embeddings",
                "--dataset",
                "tahoe",
                "--input-h5ad",
                compound_h5ad,
                "--output-h5ad",
                output,
                "--smiles-column",
                "canonical_smiles",
            ],
            args.minimol_python,
            skipped,
            "minimol",
        )
        if added:
            generated.append(output)

    if "pretrained_molfeat" in families:
        output = out_dir / f"{prefix}_pretrained_molfeat_embeddings.h5ad"
        added = append_if_exists(
            commands,
            [
                args.pretrained_molfeat_python,
                "-m",
                "chemical_embedding_pipeline.generate_pretrained_molfeat_embeddings",
                "--dataset",
                "tahoe",
                "--input-h5ad",
                compound_h5ad,
                "--output-h5ad",
                output,
                "--smiles-column",
                "canonical_smiles",
            ],
            args.pretrained_molfeat_python,
            skipped,
            "pretrained_molfeat",
        )
        if added:
            generated.append(output)

    if "pubchem" in families:
        if "pubchem_cid" not in manifest["obs_columns"]:
            skipped.append({"family": "pubchem", "reason": "no PubChem CID column inferred"})
        else:
            output = out_dir / f"{prefix}_pubchem_embeddings.h5ad"
            commands.append(
                [
                    PYTHON,
                    "-m",
                    "chemical_embedding_pipeline.generate_pubchem_embeddings",
                    "--dataset",
                    "tahoe",
                    "--input-h5ad",
                    compound_h5ad,
                    "--output-h5ad",
                    output,
                    "--cache-json",
                    args.pubchem_cache_json,
                ]
            )
            generated.append(output)

    for family, model, embedding_name, suffix in [
        ("openai_paper", "text-embedding-3-small", "chatgpt", "openai_3_small"),
        ("openai_sota", "text-embedding-3-large", "openai_text_embedding_3_large", "openai_3_large"),
    ]:
        if family not in families:
            continue
        if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
            skipped.append({"family": family, "reason": "OPENAI_API_KEY is not set"})
            continue
        output = out_dir / f"{prefix}_{suffix}_embeddings.h5ad"
        commands.append(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_openai_embeddings",
                "--dataset",
                "sciplex",
                "--input-h5ad",
                compound_h5ad,
                "--output-h5ad",
                output,
                "--model",
                model,
                "--embedding-name",
                embedding_name,
                "--template",
                "{canonical_smiles}",
            ]
        )
        generated.append(output)

    if generated:
        commands.append(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.merge_embedding_h5ads",
                "--output-h5ad",
                out_dir / f"{prefix}_compound_embeddings.h5ad",
                "--collision-policy",
                "skip_identical",
                *generated,
            ]
        )
    return commands, skipped, generated


def main() -> None:
    load_dotenv_key("OPENAI_API_KEY", REPO_ROOT / ".env")
    args = parse_args()
    inputs = local_inputs(args)
    if not inputs:
        raise FileNotFoundError("No input H5AD files found.")
    families = expanded_families(args)

    run_manifest = {"inputs": [], "families": sorted(families), "dry_run": args.dry_run}
    for input_h5ad in progress(inputs, desc="Input H5ADs", unit="file"):
        prefix = slugify(input_h5ad.stem)
        out_dir = args.output_dir / prefix
        compound_h5ad = out_dir / f"{prefix}_compounds.h5ad"
        manifest = extract_compounds(
            input_h5ad=input_h5ad,
            output_h5ad=compound_h5ad,
            compound_id_column=args.compound_id_column,
            smiles_column=args.smiles_column,
            cid_column=args.cid_column,
            smiles_cache_json=args.smiles_cache_json,
            pubchem_sleep=args.pubchem_sleep,
            pubchem_workers=args.pubchem_workers,
            pubchem_chunk_size=args.pubchem_chunk_size,
            pubchem_retries=args.pubchem_retries,
            skip_network=args.skip_network,
        )
        commands, skipped, generated = build_commands(compound_h5ad, out_dir, prefix, families, args, manifest)
        for command in progress(commands, desc=f"{prefix} commands", unit="cmd"):
            run(command, args.dry_run)
        run_manifest["inputs"].append(
            {
                "input_h5ad": str(input_h5ad),
                "compound_h5ad": str(compound_h5ad),
                "output_dir": str(out_dir),
                "merged_h5ad": str(out_dir / f"{prefix}_compound_embeddings.h5ad"),
                "generated_h5ads": [str(path) for path in generated],
                "skipped": skipped,
                "compound_manifest": manifest,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "dataset_compound_embedding_run.manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
