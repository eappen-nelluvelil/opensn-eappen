#!/usr/bin/env bash
# Drive a profiling matrix across the CBCD GPU regression tests, using
# runtime-report (cheap, hierarchical, MPI-aggregated). Each run lands in its
# own output file so tests can be compared.
#
# Usage:
#   run_matrix.sh [num_ranks] [output_dir]
#
# Default num_ranks=4. Default output_dir=<repo>/scripts/profiling/cbcd/results/<timestamp>.
#
# The matrix covers:
#   - baseline non-cyclic CBC GPU cases (no delayed path exercised)
#   - 3d_4_cycles_1 CBC GPU (the validated cyclic CBCD case)
# Edit TESTS below to add/remove cases.
set -euo pipefail

NP="${1:-4}"
OUT_DIR_ARG="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_BASE="${REPO_ROOT}/test/python/modules/linear_boltzmann_solvers/transport_steady"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${OUT_DIR_ARG}" ]]; then
  OUT_DIR="${HERE}/results/${TIMESTAMP}"
else
  if [[ "${OUT_DIR_ARG}" = /* ]]; then OUT_DIR="${OUT_DIR_ARG}"; else OUT_DIR="$(pwd)/${OUT_DIR_ARG}"; fi
fi
mkdir -p "${OUT_DIR}"

# Baseline non-cyclic CBCD GPU test(s) + cyclic CBCD GPU test.
# The 5-cycle cases are expected to fail on the pre-existing GPU max-DOF-per-
# cell limit (not a CBCD defect) — commented out by default.
TESTS=(
  "transport_3d_1b_ortho_cbc_gpu.py"
  "transport_3d_2_unstructured_cbc_gpu.py"
  "transport_3d_4_cycles_1_cbc_gpu.py"
  # "transport_3d_5_cycles_2_cbc_gpu.py"             # expect failure
  # "transport_3d_5_cycles_2_bicgstab_cbc_gpu.py"    # expect failure
  # "transport_3d_5_cycles_2_crichardson_cbc_gpu.py" # expect failure
)

echo "[run_matrix] out_dir = ${OUT_DIR}"
echo "[run_matrix] np      = ${NP}"
for t in "${TESTS[@]}"; do
  echo
  echo "=================================================================="
  echo "[run_matrix] running ${t}"
  echo "=================================================================="
  stem="${t%.py}"
  "${HERE}/run_runtime_report.sh" \
    "${TEST_BASE}/${t}" "${NP}" \
    "${OUT_DIR}/runtime_report_${stem}.txt" || echo "[run_matrix] ${t} FAILED (continuing)"
done

echo
echo "[run_matrix] results in ${OUT_DIR}"
ls -la "${OUT_DIR}"
