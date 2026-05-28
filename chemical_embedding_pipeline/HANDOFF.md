# Chemical Embedding Pipeline Handoff

This directory is a new, non-overwriting pipeline for regenerating practical
chemical embeddings used in the SciPlex/Tahoe LFC benchmark figures.

## Main Commands

Generate the default practical embedding set:

```bash
uv run python -m chemical_embedding_pipeline.generate_lfc_embeddings \
  --dataset both \
  --output-dir generated_lfc_embeddings \
  --local-files-only \
  --verify \
  --merge
```

Run the full implemented non-OpenAI reproducibility suite:

```bash
uv run python -m chemical_embedding_pipeline.run_reproducibility_suite \
  --output-dir generated_lfc_embeddings/reproducibility_suite \
  --all-feasible
```

Audit the exact embeddings that appear in the A5/A8 LFC score files:

```bash
uv run python -m chemical_embedding_pipeline.audit_lfc_score_coverage
```

Generate embeddings for a new cell-level or compound-level dataset H5AD:

```bash
uv run python -m chemical_embedding_pipeline.generate_dataset_compound_embeddings \
  --input-h5ad path/to/dataset.h5ad \
  --output-dir generated_lfc_embeddings/new_dataset \
  --local-files-only
```

The wrapper extracts unique compounds from `.obs`, infers common compound ID,
SMILES, and PubChem CID columns, runs the available generators, and writes one
merged compound embedding H5AD per input dataset. It also supports Hugging Face
dataset repos with `--hf-dataset`, `--hf-file`, and `--hf-pattern`. For
TEMP-style files with `perturbagen` and `pubchem_cid` but no SMILES, it can fill
`canonical_smiles` from PubChem CID and cache the results. Use
`--list-hf-files` before downloading broad Hugging Face patterns; by default the
wrapper refuses to download more than one matched H5AD unless `--max-hf-files`
is raised.

Download selected `theislab/temp` top-level datasets:

```bash
uv run python -m chemical_embedding_pipeline.download_temp_hf_datasets \
  sciplex tahoe \
  --output-dir data/theislab_temp
```

By default this downloads only the canonical processed dataset H5AD for each
selected dataset. No dataset names means no download; pass `--all` for all
known datasets. Add `--include-sep-rep`, `--include-group-rep`, or
`--include-all-filetypes` for the downstream DEG benchmark H5ADs.

## Side Environments

The main `uv run` project environment remains the default environment. Dependency-specific
embeddings use separate side environments:

- `.chemembed-legacy-py311`: legacy RDKit/molfeat SECFP parity path
- `.chemembed-minimol-py311`: MiniMol
- `.chemembed-pretrained-molfeat-py311`: DGL/molfeat pretrained GIN

Create or repair those side environments with:

```bash
uv run python -m chemical_embedding_pipeline.setup_embedding_envs --env all
```

## Exact Or Near-Exact Reproduction

Exact against released H5ADs:

- SciPlex: `cats2D`, `ecfp`, `fcfp`, `maccs`, `topological`
- SciPlex with legacy side env: `secfp`
- Tahoe: `ecfp:2`, `maccs`, `topological`, `moa_onehot`, `target_onehot`,
  `cactvs`, `random`
- Tahoe with legacy side env: `secfp`

Allclose or near-exact:

- SciPlex/Tahoe `erg`
- Tahoe `gin_supervised_contextpred`
- Tahoe `gin_supervised_edgepred`

Implemented candidates that do not reproduce the released matrices exactly:

- SciPlex/Tahoe `avalon`
- SciPlex `cats3D`
- SciPlex/Tahoe `ChemBERTa-77M-MLM`
- SciPlex/Tahoe `ChemBERTa-77M-MTR`
- Tahoe `MolT5`
- Tahoe `MiniMol`

Optional API-backed:

- `chatgpt`: `text-embedding-3-small`
- `openai_text_embedding_3_large`: `text-embedding-3-large`

## Intentionally Excluded Or Out Of Scope

- Boltz affinity embeddings
- UniMol2 heavy model path
- LLM target prediction embeddings
- target-FM aggregated embeddings
- LPM embeddings, per user instruction
- PCA expression baseline, because it is not a reusable compound embedding

See `COVERAGE.md` for detailed verification notes and
`audit_lfc_score_coverage.py` for the machine-readable score-file coverage map.
