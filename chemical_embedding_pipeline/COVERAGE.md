# Implemented Embedding Coverage

This file is intentionally small. It lists the embeddings this simplified
pipeline can generate, without restoring the older broad audit/report machinery.

Run `python -m chemical_embedding_pipeline.audit_lfc_score_coverage` to audit
the exact embeddings present in the SciPlex/Tahoe LFC score files. The current
classification has no unclassified score-file embeddings: implemented,
optional API-backed, intentionally excluded, or out-of-scope.

Run `python -m chemical_embedding_pipeline.run_reproducibility_suite` for the
focused local gates: coverage guard, score-file audit, and exact Tahoe metadata
plus random verification. Add flags for PubChem and SciPlex structure checks.
The suite also has opt-in checks for cached transformers, MiniMol, pretrained
GIN, legacy SECFP, and an `--all-feasible` mode for the full implemented
non-OpenAI path.

Run `python -m chemical_embedding_pipeline.generate_dataset_compound_embeddings`
to apply the configured generators to arbitrary cell-level or compound-level
H5AD datasets. It creates a unique-compound H5AD first, then writes individual
and merged compound embedding H5AD outputs.

Latest full implemented non-OpenAI run:

```bash
python -m chemical_embedding_pipeline.run_reproducibility_suite \
  --output-dir /private/tmp/lfc_repro_suite_all_feasible \
  --all-feasible
```

This completed successfully and wrote verification summaries for exact Tahoe
metadata/random, PubChem, SciPlex structure, cached transformers, MiniMol,
pretrained GIN, and legacy SECFP.

## Core LFC Set

SciPlex:

- `avalon` - current molfeat, candidate; released bits do not exactly match
- `cats2D` - current molfeat, exact against released
- `cats3D` - deterministic ETKDG 3D candidate; released conformers do not match
- `ecfp` - current molfeat, exact against released
- `erg` - current molfeat, allclose against released
- `fcfp` - current molfeat, exact against released
- `maccs` - current molfeat, exact against released
- `secfp` - current molfeat by default; optional legacy path is exact against released
- `topological` - current molfeat, exact against released
- `ChemBERTa-77M-MLM` - Hugging Face public checkpoint candidate
- `ChemBERTa-77M-MTR` - Hugging Face public checkpoint candidate
- `random` - deterministic negative-control generator; not present in released SciPlex H5AD
- `chatgpt` - OpenAI `text-embedding-3-small`, API required
- `openai_text_embedding_3_large` - OpenAI SOTA add-on, API required

Tahoe:

- `avalon` - current molfeat, candidate; released bits do not exactly match
- `ecfp:2` - current molfeat, exact against released
- `erg` - current molfeat, allclose against released
- `maccs` - current molfeat, exact against released
- `secfp` - current molfeat by default; optional legacy path is exact against released
- `topological` - current molfeat, exact against released
- `ChemBERTa-77M-MLM` - Hugging Face public checkpoint candidate
- `ChemBERTa-77M-MTR` - Hugging Face public checkpoint candidate
- `MolT5` - Hugging Face public checkpoint candidate
- `MiniMol` - public MiniMol package candidate; wired and verified, not exact
- `chatgpt` - OpenAI `text-embedding-3-small`, API required
- `openai_text_embedding_3_large` - OpenAI SOTA add-on, API required

## Feasible Extras

Cached transformer checkpoints verified locally:

- `DeepChem/ChemBERTa-77M-MLM`: `ed8a5374f2024ec8da53760af91a33fb8f6a15ff`
- `DeepChem/ChemBERTa-77M-MTR`: `66b895cab8adebea0cb59a8effa66b2020f204ca`
- `laituan245/molt5-large`: `1ad0b044adde4a7d11b9429427c97626945abbe1`

Wrapper-level transformer verification against the released H5ADs completed
with cached checkpoints. These public-checkpoint regenerations are wired in but
do not exactly reproduce the released matrices:

- SciPlex `ChemBERTa-77M-MLM`: mean abs diff `0.1092468743`
- SciPlex `ChemBERTa-77M-MTR`: mean abs diff `0.2687014228`
- Tahoe `ChemBERTa-77M-MLM`: mean abs diff `0.1048186100`
- Tahoe `ChemBERTa-77M-MTR`: mean abs diff `0.2409305175`
- Tahoe `MolT5`: mean abs diff `0.0309655128`
- Tahoe `MiniMol`: mean abs diff `0.0026964227`, binary Jaccard mean `0.9995942295`

Tahoe:

- `moa_onehot` - exact against released
- `target_onehot` - exact against released
- `cactvs` - exact against released with PubChem cache/network and released zero-CID handling
- `gin_supervised_contextpred` - implemented in a side env; allclose at `1e-6`
- `gin_supervised_edgepred` - implemented in a side env; allclose at `1e-5`
- `random` - exact against released

Pretrained GIN verification used `.chemembed-pretrained-molfeat-py311` with
`dgl==2.1.0`, `torch==2.2.1`, and `torchdata==0.7.1`. The generator populates a
local molfeat model store from the public Datamol HTTP mirror because anonymous
GCS bucket listing is not available:

- Tahoe `gin_supervised_contextpred`: mean abs diff `3.5852608e-08`
- Tahoe `gin_supervised_edgepred`: mean abs diff `4.5176180e-08`

SciPlex A8 structure verification with the default current RDKit/molfeat path:

- exact: `cats2D`, `ecfp`, `fcfp`, `maccs`, `topological`
- allclose at `1e-6`: `erg`
- candidate but not exact: `avalon`, `cats3D`, `secfp`
- optional legacy `secfp` path is exact

SciPlex and Tahoe:

- `all_molfeat` can emit the broader feasible molfeat defaults for each dataset.
  Current verification with RDKit/molfeat from the `uv run` project
  environment recovered:
  - SciPlex: 17 exact, 18 allclose at `1e-6` out of 23 generated molfeat keys.
  - Tahoe: 7 exact, 8 allclose at `1e-6` out of 11 generated molfeat keys.

SciPlex `all_molfeat` exact/allclose keys:

- `atompair`
- `atompair-count`
- `cats2D`
- `ecfp`
- `ecfp-count`
- `erg`
- `estate`
- `fcfp`
- `fcfp-count`
- `layered`
- `maccs`
- `pattern`
- `pharm2D`
- `rdkit`
- `scaffoldkeys`
- `skeys`
- `topological`
- `topological-count`

Tahoe `all_molfeat` exact/allclose keys:

- `cats2d`
- `ecfp:2`
- `ecfp:4`
- `erg`
- `estate`
- `fcfp`
- `maccs`
- `topological`

## Intentionally Excluded

- Boltz affinity embeddings
- LLM target prediction embeddings
- target-FM aggregated embeddings
- UniMol2 heavy model path
- Broad reconstruction/audit reports
