# LFC chemical embedding generation

This directory contains a small, non-overwriting set of scripts for generating
the chemical embeddings used in the SciPlex and Tahoe LFC benchmarks.
Long-running commands show `tqdm` progress bars for datasets, files, command
groups, model loops, batches, and network-backed lookups.

The main entry point is:

```bash
uv run python -m chemical_embedding_pipeline.generate_lfc_embeddings \
  --dataset both \
  --output-dir generated_lfc_embeddings \
  --local-files-only \
  --verify \
  --merge
```

That generates the practical LFC embedding families, compares them to the
released H5ADs when possible, and optionally writes one merged H5AD per dataset.
When `--verify` is used, each dataset directory also gets
`<prefix>_verification_summary.csv`, a compact table with `exact`,
`allclose_1e-6`, `allclose_1e-5`, and mean-difference columns.

## What It Generates

SciPlex:

- Structure: `avalon`, `cats2D`, `cats3D`, `ecfp`, `erg`, `fcfp`,
  `maccs`, `secfp`, `topological`
- Transformers: `ChemBERTa-77M-MLM`, `ChemBERTa-77M-MTR`
- Negative control: `random`
- Optional OpenAI: `chatgpt`, `openai_text_embedding_3_large`

Tahoe:

- Structure: `avalon`, `ecfp:2`, `erg`, `maccs`, `secfp`, `topological`
- Transformers: `ChemBERTa-77M-MLM`, `ChemBERTa-77M-MTR`, `MolT5`
- MiniMol: `MiniMol`
- Ready extras: `moa_onehot`, `target_onehot`, `cactvs`,
  `gin_supervised_contextpred`, `gin_supervised_edgepred`, `random`
- Optional OpenAI: `chatgpt`, `openai_text_embedding_3_large`

Boltz, target-FM aggregation, LLM target prediction, broad released-H5AD audit
reports, and other heavier reconstruction experiments are intentionally not part
of the simplified LFC path.

Use `--steps random` for the deterministic random negative control. Use
`--steps extras` for the Tahoe-only ready extras. Add `--skip-network` if
you want to omit PubChem `cactvs` and keep the run local.
The pretrained GIN extras use molfeat's DGL wrapper in their own side
environment and populate a local molfeat model cache from Datamol's public HTTP
mirror.
If you already have a PubChem Fingerprint2D cache, pass it with
`--pubchem-cache-json`.

Use `--steps all_molfeat` when you want the broader feasible molfeat structure
set from the released H5ADs, rather than only the LFC figure subset. This emits
the generator defaults for each dataset.

## New Datasets

For a cell-level or compound-level H5AD with compound metadata in `.obs`, use
the dataset wrapper. It extracts unique compounds, infers common compound ID and
SMILES columns, runs the configured generators, and writes a merged compound
embedding H5AD:

```bash
uv run python -m chemical_embedding_pipeline.generate_dataset_compound_embeddings \
  --input-h5ad path/to/dataset.h5ad \
  --output-dir generated_lfc_embeddings/new_dataset \
  --local-files-only
```

For Hugging Face dataset repos such as `theislab/temp`, either name a specific
H5AD file or let the wrapper match `*.h5ad`:

```bash
uv run python -m chemical_embedding_pipeline.generate_dataset_compound_embeddings \
  --hf-dataset theislab/temp \
  --hf-file path/in/repo/example.h5ad \
  --output-dir generated_lfc_embeddings/theislab_temp \
  --local-files-only
```

Use `--list-hf-files` to inspect matches without downloading. When `--hf-file`
is not supplied, `--max-hf-files` defaults to `1` so a broad pattern does not
accidentally download a full repository.

To download selected top-level datasets from `theislab/temp` first:

```bash
uv run python -m chemical_embedding_pipeline.download_temp_hf_datasets \
  sciplex tahoe \
  --output-dir data/theislab_temp
```

By default this downloads only the canonical processed dataset H5AD for each
selected dataset, which is the right source for enumerating compounds. With no
dataset names, it downloads nothing. Use `--all` to select every known dataset,
and `--list-only` to preview matched files. Add `--include-sep-rep`,
`--include-group-rep`, or `--include-all-filetypes` when you also need the
downstream DEG benchmark H5ADs.

Use `--smiles-column`, `--compound-id-column`, or `--cid-column` when inference
needs help. Use `--families` to limit work, for example `--families structure
transformers random`. Add `--include-openai` to include the OpenAI API-backed
embedding candidates.

TEMP-style H5ADs that have `perturbagen`/`pubchem_cid` but no SMILES are also
supported: the wrapper can fill `canonical_smiles` from PubChem CID and cache
the results in `--smiles-cache-json`. PubChem CID-to-SMILES lookup is fetched
in chunked requests with retry/backoff; tune this with `--pubchem-workers`,
`--pubchem-chunk-size`, `--pubchem-retries`, and `--pubchem-sleep`. Add
`--skip-network` to require that all needed SMILES are already cached.

## L1000 LFC Eval

After generating L1000 phase embeddings, run molecule-heldout KNN/LASSO LFC
evaluation with the 20 generated embeddings plus the LPM export from
`lpm_paper10_ft_morgan_learned_fixmol_best_embeddings/lincs_phase1`:

```bash
uv run python -m chemical_embedding_pipeline.general_lfc_eval \
  --l1000 both \
  --output-csv results/scores/l1000_lfc_embedding_eval.csv \
  --resume
```

By default this evaluates each `cell_type` as a separate context, excludes
controls, uses the predefined `heldout_molecules.tsv` molecular split, trains on
all non-test compounds including `val`, evaluates on `test`, and writes rows
incrementally so the run can be resumed. The expression H5ADs are read from
`data/theislab_temp/...`; the compound embedding H5ADs are read from
`data/generated_lfc_embeddings/l1000/...`, with fallbacks to the original
`generated_lfc_embeddings/l1000/...` layout. For a quick smoke test, narrow the
run:

```bash
uv run python -m chemical_embedding_pipeline.general_lfc_eval \
  --l1000 phase2 \
  --embeddings random \
  --estimators knn \
  --max-contexts 1 \
  --min-compounds 10 \
  --n-splits 2 \
  --inner-splits 2 \
  --output-csv /tmp/l1000_lfc_smoke.csv
```

For a new SciPlex-like or Tahoe-like compound H5AD where you already know the
benchmark profile:

```bash
uv run python -m chemical_embedding_pipeline.generate_lfc_embeddings \
  --dataset tahoe \
  --input-h5ad path/to/new_compounds.h5ad \
  --output-dir generated_lfc_embeddings/new_dataset \
  --prefix new_dataset \
  --smiles-column smiles \
  --local-files-only \
  --merge
```

Use `--dataset sciplex` if the input has SciPlex-style metadata. Defaults are:

- SciPlex SMILES column: `canonical_smiles`
- Tahoe SMILES column: `main_molecule_smiles`

Pass `--smiles-column` when your H5AD uses a different column name.

## Optional OpenAI

The paper-style OpenAI embedding is kept as `chatgpt` with
`text-embedding-3-small`. The additional SOTA embedding is written as
`openai_text_embedding_3_large` with `text-embedding-3-large`.

```bash
export OPENAI_API_KEY=...

uv run python -m chemical_embedding_pipeline.generate_lfc_embeddings \
  --dataset tahoe \
  --input-h5ad path/to/new_compounds.h5ad \
  --steps structure transformers minimol openai_paper openai_sota \
  --output-dir generated_lfc_embeddings/new_dataset \
  --merge
```

## Dependencies

The structure embeddings use current RDKit/molfeat from `uv run` project environment.
Transformer embeddings use Hugging Face checkpoints; pass `--local-files-only`
when checkpoints are already cached. Cache or verify those checkpoints with:

```bash
uv run python -m chemical_embedding_pipeline.download_hf_checkpoints
```

MiniMol is Tahoe-only and uses the optional `.chemembed-minimol-py311/bin/python`
environment.
The pretrained GIN extras can use a separate environment when the main
environment has incompatible DGL/torch binaries. The setup helper pins
`dgl==2.1.0` with `torch==2.2.1` and `torchdata==0.7.1` for this reason:

```bash
uv run python -m chemical_embedding_pipeline.setup_embedding_envs \
  --env pretrained-molfeat

uv run python -m chemical_embedding_pipeline.generate_lfc_embeddings \
  --dataset tahoe \
  --steps pretrained_molfeat \
  --pretrained-molfeat-python .chemembed-pretrained-molfeat-py311/bin/python \
  --verify \
  --merge
```

By default `secfp` also uses the current RDKit/molfeat stack. If you want the
legacy parity path that matched the released `secfp` matrices more closely, add:

```bash
--legacy-secfp-python .chemembed-legacy-py311/bin/python
```

## Lower-Level Scripts

The simple CLI is just a thin wrapper around these focused generators:

- `generate_molfeat_embeddings.py`
- `generate_transformer_embeddings.py`
- `generate_minimol_embeddings.py`
- `generate_metadata_embeddings.py`
- `generate_pubchem_embeddings.py`
- `generate_pretrained_molfeat_embeddings.py`
- `generate_random_embeddings.py`
- `generate_openai_embeddings.py`
- `verify_embeddings.py`
- `merge_embedding_h5ads.py`

The recommended path is the `generate_lfc_embeddings` command above.
See `COVERAGE.md` for the compact list of implemented, optional, and
intentionally excluded embeddings.

To sanity-check that the simplified package still includes the expected feasible
generators:

```bash
uv run python -m chemical_embedding_pipeline.check_coverage
```

To audit the exact embeddings appearing in the SciPlex/Tahoe LFC score files:

```bash
uv run python -m chemical_embedding_pipeline.audit_lfc_score_coverage
```

That writes `generated_lfc_embeddings/lfc_score_coverage.csv` and `.md`, and
fails if a score-file embedding is not classified.

For a slightly stronger local smoke test that generates and verifies the cheap
exact Tahoe extras:

```bash
uv run python -m chemical_embedding_pipeline.smoke_test
```

For a focused reproducibility run that combines the coverage guard, score-file
audit, and cheap exact Tahoe verification:

```bash
uv run python -m chemical_embedding_pipeline.run_reproducibility_suite
```

Add flags to include heavier checks:

- `--include-sciplex-structure` verifies the expanded SciPlex A8 structure set.
- `--include-pubchem` verifies Tahoe `cactvs`.
- `--include-transformers` verifies cached SciPlex/Tahoe ChemBERTa/MolT5.
- `--include-minimol` verifies Tahoe MiniMol in `.chemembed-minimol-py311`.
- `--include-pretrained-molfeat` verifies Tahoe pretrained GIN in
  `.chemembed-pretrained-molfeat-py311`.
- `--include-legacy-secfp` verifies exact SECFP in `.chemembed-legacy-py311`.
- `--all-feasible` runs all of the above non-OpenAI checks.
