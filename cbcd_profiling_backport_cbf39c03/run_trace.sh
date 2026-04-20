#!/usr/bin/env bash
# Run an OpenSn regression test with Caliper's event-trace service.
#
# This produces per-rank .cali files for offline analysis. It is heavier than
# runtime-report, but it is the right tool if you need to go beyond the flat
# launch summary and inspect raw event counts or feed the traces to custom
# post-processing.
#
# Usage:
#   run_trace.sh <test_script.py> [num_ranks] [output_dir]
#
# Defaults:
#   num_ranks = 4
#   output_dir = <test_dir>/cali_<test_stem>_<timestamp>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_script.py> [num_ranks] [output_dir]" >&2
  exit 1
fi

TEST_SCRIPT_ARG="$1"
NP="${2:-4}"
OUT_DIR_ARG="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENSN_BIN="${OPENSN_BIN:-${REPO_ROOT}/build/python/opensn}"

if [[ ! -x "${OPENSN_BIN}" ]]; then
  echo "opensn binary not found/executable at ${OPENSN_BIN}" >&2
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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${OUT_DIR_ARG}" ]]; then
  OUT_DIR="${TEST_DIR}/cali_${TEST_STEM}_${TIMESTAMP}"
else
  if [[ "${OUT_DIR_ARG}" = /* ]]; then
    OUT_DIR="${OUT_DIR_ARG}"
  else
    OUT_DIR="$(pwd)/${OUT_DIR_ARG}"
  fi
fi

mkdir -p "${OUT_DIR}"

CALI_CFG="${CALI_CFG:-event-trace(outdir=${OUT_DIR},time.inclusive=true)}"

echo "[run_trace] repo        = ${REPO_ROOT}"
echo "[run_trace] binary      = ${OPENSN_BIN}"
echo "[run_trace] test_dir    = ${TEST_DIR}"
echo "[run_trace] test_script = ${TEST_NAME}"
echo "[run_trace] num_ranks   = ${NP}"
echo "[run_trace] out_dir     = ${OUT_DIR}"
echo "[run_trace] cali_config = ${CALI_CFG}"
echo

cd "${TEST_DIR}"

mpirun --np "${NP}" "${OPENSN_BIN}" \
  --caliper="${CALI_CFG}" \
  -i "${TEST_NAME}" 2>&1 | tee "${OUT_DIR}/stdout.txt"

echo
echo "[run_trace] cali files:"
find "${OUT_DIR}" -maxdepth 1 -name "*.cali" -type f | sort | sed "s/^/  /" || true
if ! find "${OUT_DIR}" -maxdepth 1 -name "*.cali" -type f | read -r _; then
  echo "  (none — check cali_config and whether the source patch was applied)"
fi
echo
echo "[run_trace] summarize with: $(dirname "$0")/summarize_cali.sh ${OUT_DIR}"
