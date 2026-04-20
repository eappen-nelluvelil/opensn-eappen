#!/usr/bin/env bash
# Summarize one or more Caliper .cali files produced by run_trace.sh.
#
# Usage:
#   summarize_cali.sh <dir_or_file> [output_file]
#
# Produces two views from raw event-trace .cali files:
#   1. Hierarchical region tree with entry counts and inclusive time
#   2. Flat table with counts plus min/max/avg/exclusive/inclusive timing
#
# If <dir_or_file> is a directory, all *.cali files inside are aggregated.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dir_or_file> [output_file]" >&2
  exit 1
fi

TARGET="$1"
OUT_ARG="${2:-}"

CALI_QUERY="${CALI_QUERY:-/home/eappen/opensn-deps-latest-3/bin/cali-query}"
if [[ ! -x "${CALI_QUERY}" ]]; then
  CALI_QUERY="$(command -v cali-query || true)"
fi
if [[ -z "${CALI_QUERY}" || ! -x "${CALI_QUERY}" ]]; then
  echo "cali-query not found; set CALI_QUERY env var" >&2
  exit 1
fi

if [[ -d "${TARGET}" ]]; then
  mapfile -t CALI_FILES < <(find "${TARGET}" -maxdepth 2 -name "*.cali" | sort)
else
  CALI_FILES=("${TARGET}")
fi

if [[ ${#CALI_FILES[@]} -eq 0 ]]; then
  echo "No .cali files found in ${TARGET}" >&2
  exit 1
fi

if [[ -z "${OUT_ARG}" ]]; then
  if [[ -d "${TARGET}" ]]; then
    OUT_ABS="${TARGET%/}/summary.txt"
  else
    OUT_ABS="${TARGET%.cali}_summary.txt"
  fi
else
  if [[ "${OUT_ARG}" = /* ]]; then OUT_ABS="${OUT_ARG}"; else OUT_ABS="$(pwd)/${OUT_ARG}"; fi
fi

{
  echo "============================================================"
  echo "Caliper summary"
  echo "Inputs:"
  for f in "${CALI_FILES[@]}"; do echo "  ${f}"; done
  echo "============================================================"
  echo
  echo "--- Hierarchical inclusive time + counts (aggregated across ranks) ---"
  "${CALI_QUERY}" -q "SELECT count(), inclusive_sum(time.duration)
                      GROUP BY path
                      FORMAT tree(path)
                      ORDER BY inclusive#time.duration DESC" \
    "${CALI_FILES[@]}" || true
  echo
  echo "--- Flat region table (aggregated across ranks) ---"
  "${CALI_QUERY}" -q "SELECT path,
                             count(),
                             min(time.duration),
                             avg(time.duration),
                             max(time.duration),
                             sum(time.duration),
                             inclusive_sum(time.duration)
                      GROUP BY path
                      FORMAT table
                      ORDER BY inclusive#time.duration DESC" \
    "${CALI_FILES[@]}" || true
} | tee "${OUT_ABS}"

echo
echo "[summarize_cali] wrote ${OUT_ABS}"
