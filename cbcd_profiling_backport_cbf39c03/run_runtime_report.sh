#!/usr/bin/env bash
# Run an OpenSn regression test with Caliper's runtime-report service.
#
# This is the cheapest profiling mode in this bundle. It produces a hierarchical
# MPI-aggregated timing report that is useful for identifying where the CBCD
# execution path spends wall time.
#
# Usage:
#   run_runtime_report.sh <test_script.py> [num_ranks] [output_file]
#
# Defaults:
#   num_ranks   = 4
#   output_file = <test_dir>/runtime_report_<test_stem>.txt
#
# Environment overrides:
#   OPENSN_BIN  = absolute path to the OpenSn console binary
#   CALI_CFG    = Caliper configuration string
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_script.py> [num_ranks] [output_file]" >&2
  exit 1
fi

TEST_SCRIPT_ARG="$1"
NP="${2:-4}"
OUT_ARG="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENSN_BIN="${OPENSN_BIN:-${REPO_ROOT}/build/python/opensn}"

if [[ ! -x "${OPENSN_BIN}" ]]; then
  echo "opensn binary not found/executable at ${OPENSN_BIN}" >&2
  echo "Set OPENSN_BIN to override." >&2
  exit 1
fi

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
  if [[ "${OUT_ARG}" = /* ]]; then
    OUT_ABS="${OUT_ARG}"
  else
    OUT_ABS="$(pwd)/${OUT_ARG}"
  fi
fi

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

mpirun --np "${NP}" "${OPENSN_BIN}" \
  --caliper="${CALI_CFG}" \
  -i "${TEST_NAME}" 2>&1 | tee "${OUT_ABS}"

echo
echo "[run_runtime_report] wrote ${OUT_ABS}"
