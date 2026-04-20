#!/usr/bin/env bash
# Run a CBCD GPU regression test with Caliper's runtime-report service.
#
# Produces a hierarchical per-region timing summary aggregated across MPI ranks
# (Caliper's runtime-report reports min/avg/max across ranks from rank 0).
#
# Usage:
#   run_runtime_report.sh <test_script> [num_ranks] [output_file]
#
# Defaults:
#   num_ranks   = 4
#   output_file = ./runtime_report_<test_stem>.txt  (in the test script's directory)
#
# The test script path may be absolute or relative; mpirun is invoked from the
# directory containing the test script (required by OpenSn input conventions).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_script.py> [num_ranks] [output_file]" >&2
  exit 1
fi

TEST_SCRIPT_ARG="$1"
NP="${2:-4}"
OUT_ARG="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENSN_BIN="${OPENSN_BIN:-${REPO_ROOT}/build/python/opensn}"

if [[ ! -x "${OPENSN_BIN}" ]]; then
  echo "opensn binary not found/executable at ${OPENSN_BIN}" >&2
  echo "Set OPENSN_BIN env var to override." >&2
  exit 1
fi

# Resolve the test script to absolute, then chdir to its directory.
if [[ "${TEST_SCRIPT_ARG}" = /* ]]; then
  TEST_ABS="${TEST_SCRIPT_ARG}"
else
  TEST_ABS="$(cd "$(dirname "${TEST_SCRIPT_ARG}")" && pwd)/$(basename "${TEST_SCRIPT_ARG}")"
fi
TEST_DIR="$(dirname "${TEST_ABS}")"
TEST_NAME="$(basename "${TEST_ABS}")"
TEST_STEM="${TEST_NAME%.py}"

if [[ -z "${OUT_ARG}" ]]; then
  OUT_ABS="${TEST_DIR}/runtime_report_${TEST_STEM}.txt"
else
  if [[ "${OUT_ARG}" = /* ]]; then OUT_ABS="${OUT_ARG}"; else OUT_ABS="$(pwd)/${OUT_ARG}"; fi
fi

# Caliper config: inclusive times, wider columns, order by inclusive time.
CALI_CFG="${CALI_CFG:-runtime-report(calc.inclusive=true,region.count),max_column_width=100}"

echo "[run_runtime_report] repo        = ${REPO_ROOT}"
echo "[run_runtime_report] binary      = ${OPENSN_BIN}"
echo "[run_runtime_report] test_dir    = ${TEST_DIR}"
echo "[run_runtime_report] test_script = ${TEST_NAME}"
echo "[run_runtime_report] num_ranks   = ${NP}"
echo "[run_runtime_report] output      = ${OUT_ABS}"
echo "[run_runtime_report] cali_config = ${CALI_CFG}"
echo

cd "${TEST_DIR}"

# Caliper writes the runtime-report to stderr. Redirect both streams to tee so
# the report and any solver stdout both land in the output file.
mpirun --np "${NP}" "${OPENSN_BIN}" \
  --caliper="${CALI_CFG}" \
  -i "${TEST_NAME}" 2>&1 | tee "${OUT_ABS}"

echo
echo "[run_runtime_report] wrote ${OUT_ABS}"
