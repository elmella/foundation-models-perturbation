# LFC Embedding Performance Report

Lower `L2` is better. The main tables compute `SD`, `SEM`, and `95% CI +/-` across matched cell-line/fold units, not seed-level repeats.

The fold-only tables first average cell lines within each fold for each embedding, then compute the same descriptive statistics across folds.

## SciPlex LFC

### SciPlex LFC KNN - cell-line/fold variation

| rank | embedding          | model_type         | n_units | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline | Positive Control   | 15      | 5.805   | 6.052     | 0.910 | 0.235 | 0.504           |
| 2    | LPM                | Response Embedding | 15      | 6.852   | 7.148     | 0.990 | 0.256 | 0.548           |
| 3    | ErG                | Molecule Structure | 15      | 6.956   | 7.308     | 0.976 | 0.252 | 0.541           |
| 4    | CATS2D             | Molecule Structure | 15      | 6.958   | 7.234     | 0.918 | 0.237 | 0.509           |
| 5    | ChemBERTa-77M-MTR  | SMILES Transformer | 15      | 6.980   | 7.203     | 0.902 | 0.233 | 0.500           |
| 6    | MACCS              | Molecule Structure | 15      | 6.998   | 7.268     | 0.817 | 0.211 | 0.453           |
| 7    | ChatGPT            | LLM                | 15      | 6.999   | 6.971     | 0.722 | 0.186 | 0.400           |
| 8    | CATS3D             | Molecule Structure | 15      | 7.022   | 7.289     | 0.893 | 0.230 | 0.494           |
| 9    | Random Embeddings  | Negative Control   | 15      | 7.073   | 7.319     | 0.965 | 0.249 | 0.534           |
| 10   | Avalon             | Molecule Structure | 15      | 7.090   | 7.340     | 0.933 | 0.241 | 0.517           |
| 11   | ECFP:2             | Molecule Structure | 15      | 7.094   | 7.321     | 0.940 | 0.243 | 0.521           |
| 12   | Topological        | Molecule Structure | 15      | 7.101   | 7.354     | 0.931 | 0.241 | 0.516           |
| 13   | FCFP               | Molecule Structure | 15      | 7.101   | 7.342     | 0.927 | 0.239 | 0.513           |
| 14   | SECFP              | Molecule Structure | 15      | 7.108   | 7.347     | 0.910 | 0.235 | 0.504           |
| 15   | ChemBERTa-77M-MLM  | SMILES Transformer | 15      | 7.189   | 7.434     | 0.878 | 0.227 | 0.486           |

### SciPlex LFC KNN - fold-only variation

| rank | embedding          | model_type         | n_folds | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline | Positive Control   | 5       | 5.805   | 6.193     | 0.900 | 0.403 | 1.118           |
| 2    | LPM                | Response Embedding | 5       | 6.852   | 7.323     | 0.972 | 0.435 | 1.207           |
| 3    | ErG                | Molecule Structure | 5       | 6.956   | 7.461     | 0.979 | 0.438 | 1.215           |
| 4    | CATS2D             | Molecule Structure | 5       | 6.958   | 7.375     | 0.921 | 0.412 | 1.144           |
| 5    | ChemBERTa-77M-MTR  | SMILES Transformer | 5       | 6.980   | 7.328     | 0.891 | 0.398 | 1.106           |
| 6    | MACCS              | Molecule Structure | 5       | 6.998   | 7.386     | 0.799 | 0.357 | 0.992           |
| 7    | ChatGPT            | LLM                | 5       | 6.999   | 7.125     | 0.641 | 0.286 | 0.795           |
| 8    | CATS3D             | Molecule Structure | 5       | 7.022   | 7.456     | 0.885 | 0.396 | 1.099           |
| 9    | Random Embeddings  | Negative Control   | 5       | 7.073   | 7.547     | 0.964 | 0.431 | 1.197           |
| 10   | Avalon             | Molecule Structure | 5       | 7.090   | 7.517     | 0.925 | 0.414 | 1.148           |
| 11   | ECFP:2             | Molecule Structure | 5       | 7.094   | 7.512     | 0.936 | 0.419 | 1.162           |
| 12   | Topological        | Molecule Structure | 5       | 7.101   | 7.531     | 0.926 | 0.414 | 1.150           |
| 13   | FCFP               | Molecule Structure | 5       | 7.101   | 7.515     | 0.918 | 0.411 | 1.140           |
| 14   | SECFP              | Molecule Structure | 5       | 7.108   | 7.502     | 0.903 | 0.404 | 1.122           |
| 15   | ChemBERTa-77M-MLM  | SMILES Transformer | 5       | 7.189   | 7.574     | 0.869 | 0.388 | 1.079           |

### SciPlex LFC LASSO - cell-line/fold variation

| rank | embedding          | model_type         | n_units | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline | Positive Control   | 15      | 4.670   | 5.009     | 0.795 | 0.205 | 0.440           |
| 2    | LPM                | Response Embedding | 15      | 6.785   | 7.044     | 0.916 | 0.236 | 0.507           |
| 3    | ChatGPT            | LLM                | 15      | 6.896   | 7.133     | 0.728 | 0.188 | 0.403           |
| 4    | FCFP               | Molecule Structure | 15      | 6.989   | 7.237     | 0.907 | 0.234 | 0.502           |
| 5    | SECFP              | Molecule Structure | 15      | 7.023   | 7.234     | 0.859 | 0.222 | 0.476           |
| 6    | Topological        | Molecule Structure | 15      | 7.029   | 7.291     | 0.948 | 0.245 | 0.525           |
| 7    | ECFP:2             | Molecule Structure | 15      | 7.039   | 7.267     | 0.924 | 0.238 | 0.511           |
| 8    | MACCS              | Molecule Structure | 15      | 7.042   | 7.248     | 0.891 | 0.230 | 0.493           |
| 9    | ErG                | Molecule Structure | 15      | 7.044   | 7.349     | 0.936 | 0.242 | 0.518           |
| 10   | Random Embeddings  | Negative Control   | 15      | 7.059   | 7.308     | 0.963 | 0.249 | 0.533           |
| 11   | CATS3D             | Molecule Structure | 15      | 7.059   | 7.308     | 0.963 | 0.249 | 0.533           |
| 12   | CATS2D             | Molecule Structure | 15      | 7.070   | 7.364     | 0.937 | 0.242 | 0.519           |
| 13   | ChemBERTa-77M-MTR  | SMILES Transformer | 15      | 7.072   | 7.291     | 0.918 | 0.237 | 0.508           |
| 14   | Avalon             | Molecule Structure | 15      | 7.078   | 7.274     | 0.872 | 0.225 | 0.483           |
| 15   | ChemBERTa-77M-MLM  | SMILES Transformer | 15      | 7.078   | 7.283     | 0.911 | 0.235 | 0.504           |

### SciPlex LFC LASSO - fold-only variation

| rank | embedding          | model_type         | n_folds | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline | Positive Control   | 5       | 4.670   | 4.716     | 0.537 | 0.240 | 0.667           |
| 2    | LPM                | Response Embedding | 5       | 6.785   | 7.174     | 0.899 | 0.402 | 1.116           |
| 3    | ChatGPT            | LLM                | 5       | 6.896   | 7.155     | 0.691 | 0.309 | 0.858           |
| 4    | FCFP               | Molecule Structure | 5       | 6.989   | 7.388     | 0.886 | 0.396 | 1.100           |
| 5    | SECFP              | Molecule Structure | 5       | 7.023   | 7.401     | 0.838 | 0.375 | 1.040           |
| 6    | Topological        | Molecule Structure | 5       | 7.029   | 7.510     | 0.949 | 0.424 | 1.178           |
| 7    | ECFP:2             | Molecule Structure | 5       | 7.039   | 7.432     | 0.904 | 0.404 | 1.122           |
| 8    | MACCS              | Molecule Structure | 5       | 7.042   | 7.465     | 0.881 | 0.394 | 1.093           |
| 9    | ErG                | Molecule Structure | 5       | 7.044   | 7.465     | 0.933 | 0.417 | 1.159           |
| 10   | Random Embeddings  | Negative Control   | 5       | 7.059   | 7.531     | 0.963 | 0.431 | 1.196           |
| 11   | CATS3D             | Molecule Structure | 5       | 7.059   | 7.531     | 0.963 | 0.431 | 1.196           |
| 12   | CATS2D             | Molecule Structure | 5       | 7.070   | 7.521     | 0.933 | 0.417 | 1.158           |
| 13   | ChemBERTa-77M-MTR  | SMILES Transformer | 5       | 7.072   | 7.473     | 0.913 | 0.408 | 1.134           |
| 14   | Avalon             | Molecule Structure | 5       | 7.078   | 7.498     | 0.857 | 0.383 | 1.064           |
| 15   | ChemBERTa-77M-MLM  | SMILES Transformer | 5       | 7.078   | 7.521     | 0.908 | 0.406 | 1.127           |

### SciPlex LFC KNN vs Lasso

| embedding_key     | knn                | lasso              | best_head | lasso_minus_knn       |
| ----------------- | ------------------ | ------------------ | --------- | --------------------- |
| pca               | 5.804626906186502  | 4.669796563671011  | lasso     | -1.1348303425154906   |
| fcfp              | 7.10115911284107   | 6.989228007418739  | lasso     | -0.11193110542233065  |
| ChemBERTa-77M-MLM | 7.188732844687761  | 7.078320803721595  | lasso     | -0.11041204096616664  |
| chatgpt           | 6.9994400132574315 | 6.896276783129442  | lasso     | -0.10316323012798989  |
| secfp             | 7.107513204832425  | 7.023112904663964  | lasso     | -0.08440030016846034  |
| topological       | 7.101023658799651  | 7.029225449274233  | lasso     | -0.07179820952541771  |
| LPM_emb           | 6.852237147191578  | 6.7845158099470675 | lasso     | -0.06772133724451024  |
| ECFP:2_pkl        | 7.093759733178476  | 7.039215199481361  | lasso     | -0.05454453369711487  |
| random            | 7.0728531032128314 | 7.058822075211668  | lasso     | -0.014031028001163293 |
| avalon            | 7.089506570320664  | 7.077557978135658  | lasso     | -0.011948592185005324 |
| cats3D            | 7.021723221973446  | 7.058847718260826  | knn       | 0.03712449628738046   |
| maccs             | 6.997721416058571  | 7.041672230537544  | knn       | 0.043950814478972866  |
| erg               | 6.9562623509514205 | 7.043637603627934  | knn       | 0.08737525267651325   |
| ChemBERTa-77M-MTR | 6.979860630552978  | 7.072422838701934  | knn       | 0.09256220814895588   |
| cats2D            | 6.95834524453002   | 7.069913348864577  | knn       | 0.1115681043345571    |

## Tahoe LFC

### Tahoe LFC KNN - cell-line/fold variation

| rank | embedding                            | model_type         | n_units | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline                   | Positive Control   | 225     | 4.063   | 4.059     | 0.338 | 0.023 | 0.044           |
| 2    | LPM                                  | Response Embedding | 225     | 4.770   | 4.770     | 0.370 | 0.025 | 0.049           |
| 3    | AIDO.Cell 100M - Norman              | Gene Target        | 225     | 5.792   | 5.788     | 0.438 | 0.029 | 0.058           |
| 4    | Boltz Binding Probability (Protein)  | Protein Affinity   | 225     | 5.800   | 5.800     | 0.439 | 0.029 | 0.058           |
| 5    | MiniMol                              | SMILES Transformer | 225     | 5.800   | 5.787     | 0.437 | 0.029 | 0.057           |
| 6    | Targets Binary (Name)                | Gene Target        | 225     | 5.810   | 5.809     | 0.439 | 0.029 | 0.058           |
| 7    | Targets Weighted (Name, PubChem)     | Gene Target        | 225     | 5.813   | 5.799     | 0.439 | 0.029 | 0.058           |
| 8    | scPRINT - Norman                     | Gene Target        | 225     | 5.813   | 5.818     | 0.445 | 0.030 | 0.058           |
| 9    | ErG                                  | Molecule Structure | 225     | 5.814   | 5.808     | 0.437 | 0.029 | 0.057           |
| 10   | Avalon                               | Molecule Structure | 225     | 5.817   | 5.815     | 0.440 | 0.029 | 0.058           |
| 11   | Targets Weighted (Name)              | Gene Target        | 225     | 5.817   | 5.811     | 0.440 | 0.029 | 0.058           |
| 12   | ECFP:2                               | Molecule Structure | 225     | 5.820   | 5.826     | 0.441 | 0.029 | 0.058           |
| 13   | MACCS                                | Molecule Structure | 225     | 5.821   | 5.814     | 0.437 | 0.029 | 0.057           |
| 14   | Boltz Binding Probability (Fragment) | Protein Affinity   | 225     | 5.824   | 5.822     | 0.438 | 0.029 | 0.058           |
| 15   | ECFP:2 (pkl)                         | Molecule Structure | 225     | 5.825   | 5.825     | 0.440 | 0.029 | 0.058           |
| 16   | SECFP                                | Molecule Structure | 225     | 5.826   | 5.828     | 0.442 | 0.029 | 0.058           |
| 17   | Targets Binary (Name, PubChem)       | Gene Target        | 225     | 5.828   | 5.812     | 0.439 | 0.029 | 0.058           |
| 18   | Topological                          | Molecule Structure | 225     | 5.830   | 5.823     | 0.440 | 0.029 | 0.058           |
| 19   | TranscriptFormer - Norman            | Gene Target        | 225     | 5.831   | 5.828     | 0.435 | 0.029 | 0.057           |
| 20   | Boltz (Fragment)                     | Protein Affinity   | 225     | 5.836   | 5.819     | 0.434 | 0.029 | 0.057           |
| 21   | ChatGPT                              | LLM                | 225     | 5.839   | 5.828     | 0.436 | 0.029 | 0.057           |
| 22   | Boltz (Protein)                      | Protein Affinity   | 225     | 5.841   | 5.825     | 0.439 | 0.029 | 0.058           |
| 23   | ChemBERTa-77M-MTR                    | SMILES Transformer | 225     | 5.841   | 5.843     | 0.442 | 0.029 | 0.058           |
| 24   | Random Embeddings                    | Negative Control   | 225     | 5.850   | 5.850     | 0.438 | 0.029 | 0.058           |
| 25   | Uni-Mol2 (570M-H)                    | Molecule Structure | 225     | 5.853   | 5.847     | 0.436 | 0.029 | 0.057           |
| 26   | MolT5                                | SMILES Transformer | 225     | 5.853   | 5.848     | 0.441 | 0.029 | 0.058           |
| 27   | ChemBERTa-77M-MLM                    | SMILES Transformer | 225     | 5.898   | 5.894     | 0.442 | 0.029 | 0.058           |

### Tahoe LFC KNN - fold-only variation

| rank | embedding                            | model_type         | n_folds | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline                   | Positive Control   | 5       | 4.063   | 4.102     | 0.108 | 0.048 | 0.134           |
| 2    | LPM                                  | Response Embedding | 5       | 4.770   | 4.798     | 0.140 | 0.062 | 0.173           |
| 3    | AIDO.Cell 100M - Norman              | Gene Target        | 5       | 5.792   | 5.855     | 0.153 | 0.068 | 0.189           |
| 4    | Boltz Binding Probability (Protein)  | Protein Affinity   | 5       | 5.800   | 5.856     | 0.150 | 0.067 | 0.186           |
| 5    | MiniMol                              | SMILES Transformer | 5       | 5.800   | 5.857     | 0.139 | 0.062 | 0.173           |
| 6    | Targets Binary (Name)                | Gene Target        | 5       | 5.810   | 5.858     | 0.146 | 0.065 | 0.181           |
| 7    | Targets Weighted (Name, PubChem)     | Gene Target        | 5       | 5.813   | 5.879     | 0.148 | 0.066 | 0.184           |
| 8    | scPRINT - Norman                     | Gene Target        | 5       | 5.813   | 5.897     | 0.167 | 0.075 | 0.207           |
| 9    | ErG                                  | Molecule Structure | 5       | 5.814   | 5.879     | 0.139 | 0.062 | 0.173           |
| 10   | Avalon                               | Molecule Structure | 5       | 5.817   | 5.857     | 0.155 | 0.069 | 0.192           |
| 11   | Targets Weighted (Name)              | Gene Target        | 5       | 5.817   | 5.863     | 0.151 | 0.068 | 0.188           |
| 12   | ECFP:2                               | Molecule Structure | 5       | 5.820   | 5.864     | 0.154 | 0.069 | 0.191           |
| 13   | MACCS                                | Molecule Structure | 5       | 5.821   | 5.867     | 0.138 | 0.062 | 0.172           |
| 14   | Boltz Binding Probability (Fragment) | Protein Affinity   | 5       | 5.824   | 5.884     | 0.142 | 0.063 | 0.176           |
| 15   | ECFP:2 (pkl)                         | Molecule Structure | 5       | 5.825   | 5.859     | 0.151 | 0.068 | 0.187           |
| 16   | SECFP                                | Molecule Structure | 5       | 5.826   | 5.866     | 0.159 | 0.071 | 0.198           |
| 17   | Targets Binary (Name, PubChem)       | Gene Target        | 5       | 5.828   | 5.878     | 0.144 | 0.064 | 0.178           |
| 18   | Topological                          | Molecule Structure | 5       | 5.830   | 5.858     | 0.152 | 0.068 | 0.188           |
| 19   | TranscriptFormer - Norman            | Gene Target        | 5       | 5.831   | 5.876     | 0.131 | 0.059 | 0.163           |
| 20   | Boltz (Fragment)                     | Protein Affinity   | 5       | 5.836   | 5.897     | 0.124 | 0.055 | 0.154           |
| 21   | ChatGPT                              | LLM                | 5       | 5.839   | 5.886     | 0.143 | 0.064 | 0.177           |
| 22   | Boltz (Protein)                      | Protein Affinity   | 5       | 5.841   | 5.900     | 0.140 | 0.063 | 0.174           |
| 23   | ChemBERTa-77M-MTR                    | SMILES Transformer | 5       | 5.841   | 5.878     | 0.148 | 0.066 | 0.183           |
| 24   | Random Embeddings                    | Negative Control   | 5       | 5.850   | 5.901     | 0.141 | 0.063 | 0.175           |
| 25   | Uni-Mol2 (570M-H)                    | Molecule Structure | 5       | 5.853   | 5.885     | 0.126 | 0.056 | 0.157           |
| 26   | MolT5                                | SMILES Transformer | 5       | 5.853   | 5.912     | 0.141 | 0.063 | 0.176           |
| 27   | ChemBERTa-77M-MLM                    | SMILES Transformer | 5       | 5.898   | 5.969     | 0.136 | 0.061 | 0.169           |

### Tahoe LFC LASSO - cell-line/fold variation

| rank | embedding                            | model_type         | n_units | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline                   | Positive Control   | 225     | 2.725   | 2.670     | 0.383 | 0.026 | 0.050           |
| 2    | LPM                                  | Response Embedding | 225     | 4.565   | 4.550     | 0.344 | 0.023 | 0.045           |
| 3    | ChatGPT                              | LLM                | 225     | 5.784   | 5.781     | 0.436 | 0.029 | 0.057           |
| 4    | AIDO.Cell 100M - Norman              | Gene Target        | 225     | 5.804   | 5.804     | 0.437 | 0.029 | 0.057           |
| 5    | MiniMol                              | SMILES Transformer | 225     | 5.812   | 5.804     | 0.437 | 0.029 | 0.057           |
| 6    | Boltz Binding Probability (Protein)  | Protein Affinity   | 225     | 5.812   | 5.805     | 0.438 | 0.029 | 0.058           |
| 7    | Boltz (Protein)                      | Protein Affinity   | 225     | 5.816   | 5.815     | 0.437 | 0.029 | 0.057           |
| 8    | scPRINT - Norman                     | Gene Target        | 225     | 5.817   | 5.815     | 0.440 | 0.029 | 0.058           |
| 9    | Boltz (Fragment)                     | Protein Affinity   | 225     | 5.817   | 5.812     | 0.437 | 0.029 | 0.057           |
| 10   | MolT5                                | SMILES Transformer | 225     | 5.817   | 5.820     | 0.437 | 0.029 | 0.057           |
| 11   | Topological                          | Molecule Structure | 225     | 5.819   | 5.820     | 0.437 | 0.029 | 0.057           |
| 12   | Avalon                               | Molecule Structure | 225     | 5.820   | 5.820     | 0.438 | 0.029 | 0.058           |
| 13   | Boltz Binding Probability (Fragment) | Protein Affinity   | 225     | 5.820   | 5.816     | 0.439 | 0.029 | 0.058           |
| 14   | ChemBERTa-77M-MLM                    | SMILES Transformer | 225     | 5.821   | 5.820     | 0.437 | 0.029 | 0.057           |
| 15   | MACCS                                | Molecule Structure | 225     | 5.821   | 5.821     | 0.437 | 0.029 | 0.057           |
| 16   | ErG                                  | Molecule Structure | 225     | 5.822   | 5.823     | 0.437 | 0.029 | 0.057           |
| 17   | ChemBERTa-77M-MTR                    | SMILES Transformer | 225     | 5.823   | 5.826     | 0.439 | 0.029 | 0.058           |
| 18   | TranscriptFormer - Norman            | Gene Target        | 225     | 5.824   | 5.824     | 0.437 | 0.029 | 0.057           |
| 19   | Uni-Mol2 (570M-H)                    | Molecule Structure | 225     | 5.824   | 5.826     | 0.437 | 0.029 | 0.057           |
| 20   | Random Embeddings                    | Negative Control   | 225     | 5.824   | 5.826     | 0.437 | 0.029 | 0.057           |
| 21   | Targets Weighted (Name, PubChem)     | Gene Target        | 225     | 5.824   | 5.826     | 0.437 | 0.029 | 0.057           |
| 22   | Targets Binary (Name, PubChem)       | Gene Target        | 225     | 5.824   | 5.819     | 0.437 | 0.029 | 0.057           |
| 23   | ECFP:2                               | Molecule Structure | 225     | 5.826   | 5.826     | 0.445 | 0.030 | 0.058           |
| 24   | SECFP                                | Molecule Structure | 225     | 5.832   | 5.826     | 0.450 | 0.030 | 0.059           |
| 25   | ECFP:2 (pkl)                         | Molecule Structure | 225     | 5.833   | 5.828     | 0.450 | 0.030 | 0.059           |
| 26   | Targets Weighted (Name)              | Gene Target        | 225     | 5.833   | 5.826     | 0.431 | 0.029 | 0.057           |
| 27   | Targets Binary (Name)                | Gene Target        | 225     | 5.836   | 5.828     | 0.432 | 0.029 | 0.057           |

### Tahoe LFC LASSO - fold-only variation

| rank | embedding                            | model_type         | n_folds | mean_l2 | median_l2 | sd    | sem   | ci95_half_width |
| ---- | ------------------------------------ | ------------------ | ------- | ------- | --------- | ----- | ----- | --------------- |
| 1    | Idealized Baseline                   | Positive Control   | 5       | 2.725   | 2.705     | 0.087 | 0.039 | 0.108           |
| 2    | LPM                                  | Response Embedding | 5       | 4.565   | 4.580     | 0.104 | 0.047 | 0.130           |
| 3    | ChatGPT                              | LLM                | 5       | 5.784   | 5.841     | 0.147 | 0.066 | 0.182           |
| 4    | AIDO.Cell 100M - Norman              | Gene Target        | 5       | 5.804   | 5.850     | 0.146 | 0.065 | 0.182           |
| 5    | MiniMol                              | SMILES Transformer | 5       | 5.812   | 5.858     | 0.140 | 0.062 | 0.173           |
| 6    | Boltz Binding Probability (Protein)  | Protein Affinity   | 5       | 5.812   | 5.864     | 0.143 | 0.064 | 0.178           |
| 7    | Boltz (Protein)                      | Protein Affinity   | 5       | 5.816   | 5.861     | 0.140 | 0.063 | 0.174           |
| 8    | scPRINT - Norman                     | Gene Target        | 5       | 5.817   | 5.872     | 0.144 | 0.064 | 0.179           |
| 9    | Boltz (Fragment)                     | Protein Affinity   | 5       | 5.817   | 5.867     | 0.137 | 0.061 | 0.170           |
| 10   | MolT5                                | SMILES Transformer | 5       | 5.817   | 5.873     | 0.141 | 0.063 | 0.176           |
| 11   | Topological                          | Molecule Structure | 5       | 5.819   | 5.865     | 0.141 | 0.063 | 0.175           |
| 12   | Avalon                               | Molecule Structure | 5       | 5.820   | 5.871     | 0.146 | 0.065 | 0.182           |
| 13   | Boltz Binding Probability (Fragment) | Protein Affinity   | 5       | 5.820   | 5.870     | 0.142 | 0.064 | 0.177           |
| 14   | ChemBERTa-77M-MLM                    | SMILES Transformer | 5       | 5.821   | 5.866     | 0.140 | 0.063 | 0.174           |
| 15   | MACCS                                | Molecule Structure | 5       | 5.821   | 5.871     | 0.141 | 0.063 | 0.175           |
| 16   | ErG                                  | Molecule Structure | 5       | 5.822   | 5.872     | 0.140 | 0.063 | 0.174           |
| 17   | ChemBERTa-77M-MTR                    | SMILES Transformer | 5       | 5.823   | 5.870     | 0.143 | 0.064 | 0.178           |
| 18   | TranscriptFormer - Norman            | Gene Target        | 5       | 5.824   | 5.871     | 0.141 | 0.063 | 0.176           |
| 19   | Uni-Mol2 (570M-H)                    | Molecule Structure | 5       | 5.824   | 5.872     | 0.140 | 0.063 | 0.174           |
| 20   | Random Embeddings                    | Negative Control   | 5       | 5.824   | 5.872     | 0.141 | 0.063 | 0.175           |
| 21   | Targets Weighted (Name, PubChem)     | Gene Target        | 5       | 5.824   | 5.872     | 0.140 | 0.063 | 0.174           |
| 22   | Targets Binary (Name, PubChem)       | Gene Target        | 5       | 5.824   | 5.870     | 0.139 | 0.062 | 0.173           |
| 23   | ECFP:2                               | Molecule Structure | 5       | 5.826   | 5.864     | 0.146 | 0.065 | 0.181           |
| 24   | SECFP                                | Molecule Structure | 5       | 5.832   | 5.858     | 0.161 | 0.072 | 0.200           |
| 25   | ECFP:2 (pkl)                         | Molecule Structure | 5       | 5.833   | 5.866     | 0.155 | 0.069 | 0.193           |
| 26   | Targets Weighted (Name)              | Gene Target        | 5       | 5.833   | 5.868     | 0.119 | 0.053 | 0.148           |
| 27   | Targets Binary (Name)                | Gene Target        | 5       | 5.836   | 5.860     | 0.103 | 0.046 | 0.128           |

### Tahoe LFC KNN vs Lasso

| embedding_key                                          | knn                | lasso              | best_head | lasso_minus_knn        |
| ------------------------------------------------------ | ------------------ | ------------------ | --------- | ---------------------- |
| pca                                                    | 4.063280730236886  | 2.7245467238628134 | lasso     | -1.3387340063740725    |
| LPM_emb                                                | 4.770296436501457  | 4.5648112786202    | lasso     | -0.20548515788125687   |
| ChemBERTa-77M-MLM                                      | 5.898490235619512  | 5.820647671924434  | lasso     | -0.0778425636950777    |
| chatgpt                                                | 5.838794436927492  | 5.783711764980096  | lasso     | -0.055082671947395134  |
| MolT5                                                  | 5.852687813714425  | 5.817237231922344  | lasso     | -0.03545058179208116   |
| unimol2-570m-H                                         | 5.852620537527687  | 5.823838338012912  | lasso     | -0.02878219951477501   |
| random                                                 | 5.849788288857459  | 5.8238709010260985 | lasso     | -0.02591738783136055   |
| boltz_affinity_pred_value_protein                      | 5.840819347491071  | 5.816059982723028  | lasso     | -0.024759364768042857  |
| boltz_affinity_pred_value_fragment                     | 5.836141117156348  | 5.817068129409009  | lasso     | -0.019072987747339454  |
| ChemBERTa-77M-MTR                                      | 5.841254811596294  | 5.823301962928956  | lasso     | -0.017952848667338372  |
| topological                                            | 5.830004599320567  | 5.819466428471497  | lasso     | -0.010538170849070383  |
| TranscriptFormer_Norman_K-562_controls_(D=2048)_concat | 5.830608396315827  | 5.823687804238454  | lasso     | -0.006920592077372945  |
| gene_targets_binary_with_pubchem                       | 5.828009492617624  | 5.824140382263072  | lasso     | -0.0038691103545520633 |
| boltz_affinity_probability_binary_fragment             | 5.824014864520722  | 5.820446913610587  | lasso     | -0.003567950910134954  |
| maccs                                                  | 5.820725966370617  | 5.820878853292659  | knn       | 0.00015288692204240562 |
| avalon                                                 | 5.816644185950338  | 5.819806437272431  | knn       | 0.003162251322093468   |
| scPRINT_Norman_K-562_controls_(D=512)_concat           | 5.813289485310783  | 5.81690920278737   | knn       | 0.00361971747658707    |
| secfp                                                  | 5.825893984337032  | 5.831627209825965  | knn       | 0.005733225488932625   |
| ecfp:2                                                 | 5.820399983228939  | 5.826381382793889  | knn       | 0.005981399564949363   |
| ECFP:2_pkl                                             | 5.825333140751283  | 5.83252698298434   | knn       | 0.007193842233056635   |
| erg                                                    | 5.814224439023761  | 5.8223494999537815 | knn       | 0.008125060930020744   |
| gene_targets_confidence_with_pubchem                   | 5.812514721876398  | 5.823914346783609  | knn       | 0.011399624907211248   |
| MiniMol                                                | 5.800284836420617  | 5.811721440987745  | knn       | 0.011436604567128583   |
| AIDOcell_100M_Norman_Aligned_(D=640)_concat            | 5.7920009976208435 | 5.803677926288079  | knn       | 0.011676928667235131   |
| boltz_affinity_probability_binary_protein              | 5.800154057908647  | 5.812454142943618  | knn       | 0.01230008503497082    |
| gene_targets_confidence_name_only                      | 5.816957127353296  | 5.8326306852746965 | knn       | 0.01567355792140024    |
| gene_targets_binary_name_only                          | 5.810382703631672  | 5.836375628595901  | knn       | 0.025992924964229225   |
