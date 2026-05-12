# Tahoe LFC No-Tahoe Embedding Report

Lower `L2` is better. Cell-line/fold tables use every cell-line/fold unit; fold-only tables average cell lines within fold first.

## Coverage

| dataset | rows | rows_with_embedding | unique_cids | unique_cids_with_embedding |
| --- | --- | --- | --- | --- |
| op3 | 139 | 136 | 139 | 136 |
| sciplex3 | 190 | 177 | 188 | 175 |
| tahoe | 380 | 288 | 379 | 287 |

## KNN - cell-line/fold variation

| rank | embedding | model_type | n | mean_l2 | median_l2 | sd | sem | ci95_half_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Idealized Baseline | Positive Control | 225.0 | 4.234 | 4.211 | 0.415 | 0.028 | 0.055 |
| 2 | MiniMol | SMILES Transformer | 225.0 | 5.885 | 5.863 | 0.510 | 0.034 | 0.067 |
| 3 | AIDO.Cell 100M - Norman | Gene Target | 225.0 | 5.887 | 5.887 | 0.526 | 0.035 | 0.069 |
| 4 | Boltz Binding Probability (Protein) | Protein Affinity | 225.0 | 5.901 | 5.900 | 0.521 | 0.035 | 0.068 |
| 5 | MACCS | Molecule Structure | 225.0 | 5.903 | 5.910 | 0.520 | 0.035 | 0.068 |
| 6 | Targets Weighted (Name, PubChem) | Gene Target | 225.0 | 5.904 | 5.916 | 0.529 | 0.035 | 0.070 |
| 7 | scPRINT - Norman | Gene Target | 225.0 | 5.904 | 5.901 | 0.522 | 0.035 | 0.069 |
| 8 | Targets Weighted (Name) | Gene Target | 225.0 | 5.905 | 5.920 | 0.532 | 0.035 | 0.070 |
| 9 | Targets Binary (Name) | Gene Target | 225.0 | 5.909 | 5.923 | 0.532 | 0.035 | 0.070 |
| 10 | ErG | Molecule Structure | 225.0 | 5.912 | 5.934 | 0.520 | 0.035 | 0.068 |
| 11 | ECFP:2 | Molecule Structure | 225.0 | 5.913 | 5.919 | 0.522 | 0.035 | 0.069 |
| 12 | SECFP | Molecule Structure | 225.0 | 5.916 | 5.928 | 0.536 | 0.036 | 0.070 |
| 13 | ECFP:2 (pkl) | Molecule Structure | 225.0 | 5.920 | 5.933 | 0.520 | 0.035 | 0.068 |
| 14 | Boltz (Protein) | Protein Affinity | 225.0 | 5.920 | 5.930 | 0.517 | 0.034 | 0.068 |
| 15 | Avalon | Molecule Structure | 225.0 | 5.921 | 5.930 | 0.528 | 0.035 | 0.069 |
| 16 | Boltz (Fragment) | Protein Affinity | 225.0 | 5.922 | 5.915 | 0.515 | 0.034 | 0.068 |
| 17 | Topological | Molecule Structure | 225.0 | 5.922 | 5.936 | 0.525 | 0.035 | 0.069 |
| 18 | Targets Binary (Name, PubChem) | Gene Target | 225.0 | 5.924 | 5.933 | 0.522 | 0.035 | 0.069 |
| 19 | TranscriptFormer - Norman | Gene Target | 225.0 | 5.925 | 5.915 | 0.519 | 0.035 | 0.068 |
| 20 | Boltz Binding Probability (Fragment) | Protein Affinity | 225.0 | 5.925 | 5.936 | 0.519 | 0.035 | 0.068 |
| 21 | LPM | Response Embedding | 225.0 | 5.930 | 5.930 | 0.504 | 0.034 | 0.066 |
| 22 | ChemBERTa-77M-MTR | SMILES Transformer | 225.0 | 5.933 | 5.936 | 0.520 | 0.035 | 0.068 |
| 23 | ChatGPT | LLM | 225.0 | 5.941 | 5.943 | 0.510 | 0.034 | 0.067 |
| 24 | Random Embeddings | Negative Control | 225.0 | 5.943 | 5.933 | 0.517 | 0.034 | 0.068 |
| 25 | Uni-Mol2 (570M-H) | Molecule Structure | 225.0 | 5.944 | 5.948 | 0.511 | 0.034 | 0.067 |
| 26 | MolT5 | SMILES Transformer | 225.0 | 5.952 | 5.950 | 0.514 | 0.034 | 0.068 |
| 27 | ChemBERTa-77M-MLM | SMILES Transformer | 225.0 | 5.989 | 5.978 | 0.511 | 0.034 | 0.067 |

## KNN - fold-only variation

| rank | embedding | model_type | n_folds | mean_l2 | median_l2 | sd | sem | ci95_half_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Idealized Baseline | Positive Control | 5.0 | 4.234 | 4.293 | 0.267 | 0.119 | 0.331 |
| 2 | MiniMol | SMILES Transformer | 5.0 | 5.885 | 5.988 | 0.303 | 0.135 | 0.376 |
| 3 | AIDO.Cell 100M - Norman | Gene Target | 5.0 | 5.887 | 6.022 | 0.337 | 0.151 | 0.418 |
| 4 | Boltz Binding Probability (Protein) | Protein Affinity | 5.0 | 5.901 | 6.024 | 0.330 | 0.147 | 0.409 |
| 5 | MACCS | Molecule Structure | 5.0 | 5.903 | 6.006 | 0.321 | 0.144 | 0.399 |
| 6 | Targets Weighted (Name, PubChem) | Gene Target | 5.0 | 5.904 | 6.028 | 0.338 | 0.151 | 0.419 |
| 7 | scPRINT - Norman | Gene Target | 5.0 | 5.904 | 6.029 | 0.327 | 0.146 | 0.406 |
| 8 | Targets Weighted (Name) | Gene Target | 5.0 | 5.905 | 6.017 | 0.348 | 0.155 | 0.432 |
| 9 | Targets Binary (Name) | Gene Target | 5.0 | 5.909 | 6.026 | 0.344 | 0.154 | 0.427 |
| 10 | ErG | Molecule Structure | 5.0 | 5.912 | 6.019 | 0.324 | 0.145 | 0.402 |
| 11 | ECFP:2 | Molecule Structure | 5.0 | 5.913 | 6.026 | 0.329 | 0.147 | 0.409 |
| 12 | SECFP | Molecule Structure | 5.0 | 5.916 | 6.027 | 0.354 | 0.158 | 0.439 |
| 13 | ECFP:2 (pkl) | Molecule Structure | 5.0 | 5.920 | 6.024 | 0.325 | 0.145 | 0.403 |
| 14 | Boltz (Protein) | Protein Affinity | 5.0 | 5.920 | 6.022 | 0.318 | 0.142 | 0.395 |
| 15 | Avalon | Molecule Structure | 5.0 | 5.921 | 6.042 | 0.340 | 0.152 | 0.423 |
| 16 | Boltz (Fragment) | Protein Affinity | 5.0 | 5.922 | 6.014 | 0.313 | 0.140 | 0.389 |
| 17 | Topological | Molecule Structure | 5.0 | 5.922 | 6.029 | 0.332 | 0.148 | 0.412 |
| 18 | Targets Binary (Name, PubChem) | Gene Target | 5.0 | 5.924 | 6.041 | 0.327 | 0.146 | 0.406 |
| 19 | TranscriptFormer - Norman | Gene Target | 5.0 | 5.925 | 6.043 | 0.320 | 0.143 | 0.397 |
| 20 | Boltz Binding Probability (Fragment) | Protein Affinity | 5.0 | 5.925 | 6.031 | 0.323 | 0.144 | 0.401 |
| 21 | LPM | Response Embedding | 5.0 | 5.930 | 6.018 | 0.288 | 0.129 | 0.358 |
| 22 | ChemBERTa-77M-MTR | SMILES Transformer | 5.0 | 5.933 | 6.032 | 0.316 | 0.142 | 0.393 |
| 23 | ChatGPT | LLM | 5.0 | 5.941 | 6.037 | 0.303 | 0.136 | 0.376 |
| 24 | Random Embeddings | Negative Control | 5.0 | 5.943 | 6.061 | 0.316 | 0.141 | 0.392 |
| 25 | Uni-Mol2 (570M-H) | Molecule Structure | 5.0 | 5.944 | 6.034 | 0.305 | 0.136 | 0.378 |
| 26 | MolT5 | SMILES Transformer | 5.0 | 5.952 | 6.044 | 0.306 | 0.137 | 0.379 |
| 27 | ChemBERTa-77M-MLM | SMILES Transformer | 5.0 | 5.989 | 6.071 | 0.296 | 0.133 | 0.368 |

## Lasso - cell-line/fold variation

| rank | embedding | model_type | n | mean_l2 | median_l2 | sd | sem | ci95_half_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Idealized Baseline | Positive Control | 225.0 | 2.812 | 2.741 | 0.436 | 0.029 | 0.057 |
| 2 | ChatGPT | LLM | 225.0 | 5.878 | 5.881 | 0.518 | 0.035 | 0.068 |
| 3 | MiniMol | SMILES Transformer | 225.0 | 5.905 | 5.881 | 0.514 | 0.034 | 0.067 |
| 4 | LPM | Response Embedding | 225.0 | 5.909 | 5.896 | 0.516 | 0.034 | 0.068 |
| 5 | Boltz (Protein) | Protein Affinity | 225.0 | 5.911 | 5.894 | 0.518 | 0.035 | 0.068 |
| 6 | Boltz Binding Probability (Protein) | Protein Affinity | 225.0 | 5.912 | 5.903 | 0.519 | 0.035 | 0.068 |
| 7 | Boltz (Fragment) | Protein Affinity | 225.0 | 5.912 | 5.892 | 0.517 | 0.034 | 0.068 |
| 8 | AIDO.Cell 100M - Norman | Gene Target | 225.0 | 5.914 | 5.929 | 0.516 | 0.034 | 0.068 |
| 9 | ECFP:2 | Molecule Structure | 225.0 | 5.914 | 5.930 | 0.514 | 0.034 | 0.068 |
| 10 | Topological | Molecule Structure | 225.0 | 5.915 | 5.917 | 0.513 | 0.034 | 0.067 |
| 11 | ECFP:2 (pkl) | Molecule Structure | 225.0 | 5.915 | 5.930 | 0.515 | 0.034 | 0.068 |
| 12 | SECFP | Molecule Structure | 225.0 | 5.916 | 5.901 | 0.520 | 0.035 | 0.068 |
| 13 | MACCS | Molecule Structure | 225.0 | 5.918 | 5.920 | 0.515 | 0.034 | 0.068 |
| 14 | ChemBERTa-77M-MLM | SMILES Transformer | 225.0 | 5.919 | 5.925 | 0.515 | 0.034 | 0.068 |
| 15 | Boltz Binding Probability (Fragment) | Protein Affinity | 225.0 | 5.919 | 5.918 | 0.518 | 0.035 | 0.068 |
| 16 | TranscriptFormer - Norman | Gene Target | 225.0 | 5.920 | 5.925 | 0.516 | 0.034 | 0.068 |
| 17 | MolT5 | SMILES Transformer | 225.0 | 5.920 | 5.913 | 0.517 | 0.034 | 0.068 |
| 18 | Avalon | Molecule Structure | 225.0 | 5.920 | 5.903 | 0.519 | 0.035 | 0.068 |
| 19 | Targets Binary (Name, PubChem) | Gene Target | 225.0 | 5.921 | 5.930 | 0.516 | 0.034 | 0.068 |
| 20 | Targets Weighted (Name, PubChem) | Gene Target | 225.0 | 5.921 | 5.929 | 0.516 | 0.034 | 0.068 |
| 21 | ChemBERTa-77M-MTR | SMILES Transformer | 225.0 | 5.921 | 5.908 | 0.516 | 0.034 | 0.068 |
| 22 | Random Embeddings | Negative Control | 225.0 | 5.921 | 5.930 | 0.516 | 0.034 | 0.068 |
| 23 | ErG | Molecule Structure | 225.0 | 5.921 | 5.930 | 0.516 | 0.034 | 0.068 |
| 24 | scPRINT - Norman | Gene Target | 225.0 | 5.922 | 5.928 | 0.517 | 0.034 | 0.068 |
| 25 | Uni-Mol2 (570M-H) | Molecule Structure | 225.0 | 5.923 | 5.945 | 0.514 | 0.034 | 0.068 |
| 26 | Targets Binary (Name) | Gene Target | 225.0 | 5.929 | 5.900 | 0.496 | 0.033 | 0.065 |
| 27 | Targets Weighted (Name) | Gene Target | 225.0 | 5.959 | 5.946 | 0.501 | 0.033 | 0.066 |

## Lasso - fold-only variation

| rank | embedding | model_type | n_folds | mean_l2 | median_l2 | sd | sem | ci95_half_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Idealized Baseline | Positive Control | 5.0 | 2.812 | 2.781 | 0.208 | 0.093 | 0.258 |
| 2 | ChatGPT | LLM | 5.0 | 5.878 | 5.992 | 0.325 | 0.145 | 0.404 |
| 3 | MiniMol | SMILES Transformer | 5.0 | 5.905 | 6.014 | 0.311 | 0.139 | 0.386 |
| 4 | LPM | Response Embedding | 5.0 | 5.909 | 6.020 | 0.317 | 0.142 | 0.394 |
| 5 | Boltz (Protein) | Protein Affinity | 5.0 | 5.911 | 6.022 | 0.318 | 0.142 | 0.395 |
| 6 | Boltz Binding Probability (Protein) | Protein Affinity | 5.0 | 5.912 | 6.022 | 0.320 | 0.143 | 0.397 |
| 7 | Boltz (Fragment) | Protein Affinity | 5.0 | 5.912 | 6.022 | 0.316 | 0.141 | 0.392 |
| 8 | AIDO.Cell 100M - Norman | Gene Target | 5.0 | 5.914 | 6.013 | 0.314 | 0.140 | 0.390 |
| 9 | ECFP:2 | Molecule Structure | 5.0 | 5.914 | 6.014 | 0.315 | 0.141 | 0.391 |
| 10 | Topological | Molecule Structure | 5.0 | 5.915 | 6.024 | 0.311 | 0.139 | 0.386 |
| 11 | ECFP:2 (pkl) | Molecule Structure | 5.0 | 5.915 | 6.020 | 0.316 | 0.141 | 0.392 |
| 12 | SECFP | Molecule Structure | 5.0 | 5.916 | 6.006 | 0.325 | 0.145 | 0.403 |
| 13 | MACCS | Molecule Structure | 5.0 | 5.918 | 6.033 | 0.315 | 0.141 | 0.391 |
| 14 | ChemBERTa-77M-MLM | SMILES Transformer | 5.0 | 5.919 | 6.029 | 0.313 | 0.140 | 0.389 |
| 15 | Boltz Binding Probability (Fragment) | Protein Affinity | 5.0 | 5.919 | 6.032 | 0.318 | 0.142 | 0.395 |
| 16 | TranscriptFormer - Norman | Gene Target | 5.0 | 5.920 | 6.032 | 0.317 | 0.142 | 0.393 |
| 17 | MolT5 | SMILES Transformer | 5.0 | 5.920 | 6.038 | 0.315 | 0.141 | 0.392 |
| 18 | Avalon | Molecule Structure | 5.0 | 5.920 | 6.030 | 0.319 | 0.143 | 0.397 |
| 19 | Targets Binary (Name, PubChem) | Gene Target | 5.0 | 5.921 | 6.035 | 0.316 | 0.141 | 0.392 |
| 20 | Targets Weighted (Name, PubChem) | Gene Target | 5.0 | 5.921 | 6.035 | 0.316 | 0.141 | 0.393 |
| 21 | ChemBERTa-77M-MTR | SMILES Transformer | 5.0 | 5.921 | 6.035 | 0.316 | 0.141 | 0.392 |
| 22 | Random Embeddings | Negative Control | 5.0 | 5.921 | 6.036 | 0.316 | 0.141 | 0.393 |
| 23 | ErG | Molecule Structure | 5.0 | 5.921 | 6.035 | 0.315 | 0.141 | 0.391 |
| 24 | scPRINT - Norman | Gene Target | 5.0 | 5.922 | 6.042 | 0.317 | 0.142 | 0.394 |
| 25 | Uni-Mol2 (570M-H) | Molecule Structure | 5.0 | 5.923 | 6.035 | 0.313 | 0.140 | 0.389 |
| 26 | Targets Binary (Name) | Gene Target | 5.0 | 5.929 | 5.997 | 0.263 | 0.118 | 0.327 |
| 27 | Targets Weighted (Name) | Gene Target | 5.0 | 5.959 | 6.023 | 0.280 | 0.125 | 0.348 |
