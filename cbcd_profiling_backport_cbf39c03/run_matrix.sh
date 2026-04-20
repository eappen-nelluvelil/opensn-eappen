#!/usr/bin/env bash
# Drive a profiling matrix for the CBCD implementation at commit cbf39c03.
#
# Unlike the later branch, this matrix intentionally avoids the cyclic CBCD
# cases because they were added later. The goal here is to profile the
# minimally-sized-FLUDS CBCD implementation on the cases that actually exist on
# that branch.
#
# Usage:
#   run_matrix.sh [num_ranks] [output_dir]
#
# Default num_ranks = 4.
# Default output_dir = <this_dir>/results/<timestamp>.
set -euo pipefail

NP="${1:-4}"
OUT_DIR_ARG="${2:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
TEST_BASE="${REPO_ROOT}/test/python/modules/linear_boltzmann_solvers/transport_steady"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${OUT_DIR_ARG}" ]]; then
  OUT_DIR="${HERE}/results/${TIMESTAMP}"
else
  if [[ "${OUT_DIR_ARG}" = /* ]]; then
    OUT_DIR="${OUT_DIR_ARG}"
  else
    OUT_DIR="$(pwd)/${OUT_DIR_ARG}"
  fi
fi
mkdir -p "${OUT_DIR}"

TESTS=(
  "transport_1d_1_cbc_gpu.py"
  "transport_2d_2_unstructured_cbc_gpu.py"
  "transport_3d_1a_extruder_cbc_gpu.py"
  "transport_3d_1b_ortho_cbc_gpu.py"
  "transport_3d_2_unstructured_cbc_gpu.py"
)

echo "[run_matrix] out_dir = ${OUT_DIR}"
echo "[run_matrix] np      = ${NP}"

for t in "${TESTS[@]}"; do
  stem="${t%.py}"

  echo
  echo "=================================================================="
  echo "[run_matrix] runtime report: ${t}"
  echo "=================================================================="
  "${HERE}/run_runtime_report.sh" \
    "${TEST_BASE}/${t}" "${NP}" \
    "${OUT_DIR}/runtime_report_${stem}.txt" || echo "[run_matrix] runtime ${t} FAILED"

  echo
  echo "=================================================================="
  echo "[run_matrix] launch report: ${t}"
  echo "=================================================================="
  "${HERE}/run_launch_report.sh" \
    "${TEST_BASE}/${t}" "${NP}" \
    "${OUT_DIR}/launch_report_${stem}.txt" || echo "[run_matrix] launch ${t} FAILED"
done

echo
echo "[run_matrix] results in ${OUT_DIR}"
ls -la "${OUT_DIR}"
