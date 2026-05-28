from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from chemical_embedding_pipeline.progress import progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tahoe PubChem Cactvs fingerprints.")
    parser.add_argument("--dataset", choices=["tahoe"], default="tahoe")
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--cid-column", default="pubchem_cid")
    parser.add_argument(
        "--cache-json",
        type=Path,
        default=Path("generated_lfc_embeddings/pubchem_fingerprint2d_cache.json"),
    )
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--zero-cids", nargs="*", default=["71481097", "67462786"])
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def fingerprint2d_for_cid(cid: str, cache: dict[str, str], sleep: float) -> str:
    if cid in cache:
        return cache[cid]
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Fingerprint2D/JSON"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    value = payload["PropertyTable"]["Properties"][0]["Fingerprint2D"]
    cache[cid] = value
    if sleep:
        time.sleep(sleep)
    return value


def chunked(values: list[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [values[start : start + size] for start in range(0, len(values), size)]


def fetch_fingerprint_chunk(cids: list[str], retries: int, sleep: float) -> dict[str, str]:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{','.join(cids)}/property/Fingerprint2D/JSON"
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = payload.get("PropertyTable", {}).get("Properties", [])
            result = {
                str(row.get("CID", "")).strip(): row["Fingerprint2D"]
                for row in rows
                if row.get("CID") is not None and row.get("Fingerprint2D")
            }
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


def decode_cactvs(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    bits = np.unpackbits(np.frombuffer(raw[4:], dtype=np.uint8))
    return bits[:881].astype(np.float64)


def cid_values(adata: ad.AnnData, column: str) -> list[str]:
    if column in adata.obs:
        return pd.Series(adata.obs[column].astype(object)).fillna("").astype(str).tolist()
    if column == "pubchem_cid":
        return [str(value) for value in adata.obs_names]
    raise KeyError(f"{column!r} is not present in input H5AD obs.")


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input_h5ad)
    cids = cid_values(adata, args.cid_column)
    cache = load_cache(args.cache_json)
    zero_cids = {str(cid) for cid in args.zero_cids}
    normalized_cids = [str(cid).strip() for cid in cids]
    missing = sorted(
        {
            cid
            for cid in normalized_cids
            if cid and cid.lower() != "nan" and cid not in zero_cids and cid not in cache
        }
    )
    failed_cids: list[str] = []
    for chunk in progress(chunked(missing, args.chunk_size), desc="PubChem fingerprint chunks", unit="chunk"):
        try:
            cache.update(fetch_fingerprint_chunk(chunk, args.retries, args.sleep))
            save_cache(args.cache_json, cache)
        except Exception as exc:
            failed_cids.extend(chunk)
            preview = ", ".join(chunk[:3])
            print(
                "Warning: failed to resolve PubChem Fingerprint2D chunk "
                f"({preview}, ...; n={len(chunk)}): {type(exc).__name__}: {exc}"
            )

    rows = []
    for cid in progress(normalized_cids, desc="PubChem fingerprints", unit="cid"):
        if not cid or cid.lower() == "nan" or cid in zero_cids:
            rows.append(np.zeros(881, dtype=np.float64))
            continue
        encoded = cache.get(cid)
        if not encoded:
            rows.append(np.zeros(881, dtype=np.float64))
            continue
        rows.append(decode_cactvs(encoded))

    out = ad.AnnData(obs=adata.obs.copy())
    out.obsm["cactvs"] = np.vstack(rows)
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output_h5ad)
    save_cache(args.cache_json, cache)
    manifest = {
        "input_h5ad": str(args.input_h5ad),
        "dataset": args.dataset,
        "cache_json": str(args.cache_json),
        "failed_cid_count": len(failed_cids),
        "failed_cids": failed_cids[:100],
        "embeddings": {
            "cactvs": {
                "shape": list(out.obsm["cactvs"].shape),
                "dtype": str(out.obsm["cactvs"].dtype),
                "source": "PubChem PUG REST Fingerprint2D",
                "zero_cids": sorted(zero_cids),
            }
        },
    }
    manifest_path = args.output_h5ad.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
