#!/usr/bin/env bash
# Produce a Caliper runtime-report that includes per-region ENTRY COUNTS and
# MIN/MAX/AVG time PER VISIT, then post-process it into a flat CBCD-only table.
# This tells us, e.g., how many KernelLaunch calls there are per solve and the
# cost distribution per call, without the noisy generic tree output.
#
# Usage:
#   run_launch_report.sh <test_script.py> [num_ranks] [output_file]
#
# Defaults:
#   num_ranks  = 4
#   output     = <test_dir>/launch_report_<test_stem>.txt
#   summary    = <test_dir>/launch_report_<test_stem>_cbcd_regions.txt
#
# Env overrides: OPENSN_BIN
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
  OUT_ABS="${TEST_DIR}/launch_report_${TEST_STEM}.txt"
else
  if [[ "${OUT_ARG}" = /* ]]; then OUT_ABS="${OUT_ARG}"; else OUT_ABS="$(pwd)/${OUT_ARG}"; fi
fi
SUMMARY_ABS="${OUT_ABS%.txt}_cbcd_regions.txt"

# runtime-report with region.count + region.stats gives, per region:
#   - entry count
#   - min/max/avg time per visit (exclusive)
#   - inclusive/exclusive time aggregated across ranks
CALI_CFG="${CALI_CFG:-runtime-report(calc.inclusive=true,region.count=true,region.stats=true),max_column_width=100}"

echo "[run_launch_report] binary      = ${OPENSN_BIN}"
echo "[run_launch_report] test        = ${TEST_NAME}"
echo "[run_launch_report] num_ranks   = ${NP}"
echo "[run_launch_report] output      = ${OUT_ABS}"
echo "[run_launch_report] summary     = ${SUMMARY_ABS}"
echo "[run_launch_report] cali_config = ${CALI_CFG}"
echo

cd "${TEST_DIR}"

# Caliper writes runtime-report output to stderr. Redirect both streams so the
# launch/count report and solver stdout land in the same output file.
mpirun --np "${NP}" "${OPENSN_BIN}" \
  --caliper="${CALI_CFG}" \
  -i "${TEST_NAME}" 2>&1 | tee "${OUT_ABS}"

gawk '
function is_target_region(region) {
  return region ~ /^(CBCD_AngleSet::(TryInitialize|TryAdvanceOneStep|RetireBatch|ProcessIncoming|LaunchBatch|FlushBatch|Finalize|DelayedPhaseQueue|ProcessDelayedIncoming|FinalizeCompletion)|CBCD_AsynchronousCommunicator::(CommThreadLoop|SerializeAndSend|ProbeAndReceive|PollInFlightSends)|CBCDSweepChunk::Sweep(::ArgsRefresh|::KernelLaunch)?|CBCD_FLUDS::(CopyIncomingBoundaryPsiToDevice|CopyOutgoingPsiBackToHost|CopyDelayedOutgoingPsiBackToHost|CopySavedPsiFromDevice|CopySavedPsiToDestinationPsi))$/
}
function print_header() {
  printf "%-56s %12s %12s %12s %12s %14s %14s %14s %14s %16s %16s\n", \
         "Region", "Min(s/rk)", "Max(s/rk)", "Avg(s/rk)", "Time(%)", \
         "Calls/rk(avg)", "Calls/rk(max)", "Calls(total)", "Visits", "Min(ns)", "Max(ns)"
}
function print_row(region, line) {
  gsub(/^[[:space:]]+/, "", line)
  n = split(line, f, /[[:space:]]+/)
  if (n < 11) return
  printf "%-56s %12s %12s %12s %12s %14s %14s %14s %14s %16s %16s\n", \
         region, f[1], f[2], f[3], f[4], f[6], f[7], f[8], f[9], f[10], f[11]
}
BEGIN {
  print_header()
}
{
  if (match($0, /^[[:space:]]*([A-Za-z0-9_:]+)[[:space:]]+([0-9.]+)/, m)) {
    region = m[1]
    if (is_target_region(region) && !seen[region]++) print_row(region, substr($0, RLENGTH - length(m[2])))
    pending = ""
    next
  }

  if (match($0, /^[[:space:]]*([A-Za-z0-9_:]+)[[:space:]]*$/, m)) {
    region = m[1]
    pending = is_target_region(region) ? region : ""
    next
  }

  if (pending != "" && $0 ~ /^[[:space:]]*\|-[[:space:]]+[0-9.]/) {
    line = $0
    sub(/^[[:space:]]*\|-[[:space:]]+/, "", line)
    if (!seen[pending]++) print_row(pending, line)
    pending = ""
    next
  }
}
' "${OUT_ABS}" > "${SUMMARY_ABS}"

echo
echo "[run_launch_report] wrote ${OUT_ABS}"
echo "[run_launch_report] wrote ${SUMMARY_ABS}"
