#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/matthew-mella/valinor/foundation-models-perturbation}"
SCORE_DIR="${SCORE_DIR:-${REPO_ROOT}/results/scores}"
SCIPLEX_FIG_DIR="${SCIPLEX_FIG_DIR:-${REPO_ROOT}/results/figures/fig_sciplex_lfc_heldout_molecules_test_only}"
TAHOE_FIG_DIR="${TAHOE_FIG_DIR:-${REPO_ROOT}/results/figures/fig_tahoe_lfc_heldout_molecules_test_only}"

SCIPLEX_SUBMISSION_ROOT="${SCIPLEX_SUBMISSION_ROOT:-${REPO_ROOT}/benchmark/benchmark/submissions_sciplex_lfc_heldout_molecules_test_only}"
TAHOE_SUBMISSION_ROOT="${TAHOE_SUBMISSION_ROOT:-${REPO_ROOT}/benchmark/benchmark/submissions_tahoe_lfc_heldout_molecules_test_only}"

SCIPLEX_OUTPUT_CSV="${SCIPLEX_OUTPUT_CSV:-${SCORE_DIR}/sciplex_lfc_heldout_molecules_test_only.csv}"
TAHOE_OUTPUT_CSV="${TAHOE_OUTPUT_CSV:-${SCORE_DIR}/tahoe_lfc_heldout_molecules_test_only.csv}"

UV_PROJECT="${UV_PROJECT:-${REPO_ROOT}/benchmark}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
mkdir -p "${MPLCONFIGDIR}"

uv --project "${UV_PROJECT}" run python "${REPO_ROOT}/scripts/collect_heldout_lfc_results.py" \
  --dataset sciplex \
  --submission-root "${SCIPLEX_SUBMISSION_ROOT}" \
  --output-csv "${SCIPLEX_OUTPUT_CSV}" \
  --test-split-labels test

uv --project "${UV_PROJECT}" run python "${REPO_ROOT}/scripts/collect_heldout_lfc_results.py" \
  --dataset tahoe \
  --submission-root "${TAHOE_SUBMISSION_ROOT}" \
  --output-csv "${TAHOE_OUTPUT_CSV}" \
  --test-split-labels test

uv --project "${UV_PROJECT}" run python "${REPO_ROOT}/scripts/plot_heldout_lfc_results.py" \
  --repo-root "${REPO_ROOT}" \
  --sciplex-csv "${SCIPLEX_OUTPUT_CSV}" \
  --tahoe-csv "${TAHOE_OUTPUT_CSV}" \
  --sciplex-fig-dir "${SCIPLEX_FIG_DIR}" \
  --tahoe-fig-dir "${TAHOE_FIG_DIR}" \
  --include-pca
