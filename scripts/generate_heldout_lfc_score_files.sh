#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/matthew-mella/valinor/foundation-models-perturbation}"
SCORE_DIR="${SCORE_DIR:-${REPO_ROOT}/results/scores}"

SCIPLEX_SUBMISSION_ROOT="${SCIPLEX_SUBMISSION_ROOT:-${REPO_ROOT}/benchmark/benchmark/submissions_sciplex_lfc_heldout_molecules_test_only}"
TAHOE_SUBMISSION_ROOT="${TAHOE_SUBMISSION_ROOT:-${REPO_ROOT}/benchmark/benchmark/submissions_tahoe_lfc_heldout_molecules_test_only}"

SCIPLEX_OUTPUT_CSV="${SCIPLEX_OUTPUT_CSV:-${SCORE_DIR}/sciplex_lfc_heldout_molecules_test_only.csv}"
TAHOE_OUTPUT_CSV="${TAHOE_OUTPUT_CSV:-${SCORE_DIR}/tahoe_lfc_heldout_molecules_test_only.csv}"

cd "${REPO_ROOT}"

uv run python scripts/collect_heldout_lfc_results.py \
  --dataset sciplex \
  --submission-root "${SCIPLEX_SUBMISSION_ROOT}" \
  --output-csv "${SCIPLEX_OUTPUT_CSV}" \
  --test-split-labels test

uv run python scripts/collect_heldout_lfc_results.py \
  --dataset tahoe \
  --submission-root "${TAHOE_SUBMISSION_ROOT}" \
  --output-csv "${TAHOE_OUTPUT_CSV}" \
  --test-split-labels test
