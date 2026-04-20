#!/usr/bin/env bash
# Summarize raw Caliper event traces into a flat region table.
#
# This script is intentionally conservative: it does not try to reconstruct a
# full tree report, because that proved expensive and brittle on large CBCD
# traces. Instead it emits an aggregated flat table for the regions matching
# REGION_REGEX.
#
# Usage:
#   summarize_cali.sh <dir_or_file> [output_file]
#
# Environment overrides:
#   CALI_QUERY   = path to cali-query
#   REGION_REGEX = awk-style regex for region filtering
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
  echo "cali-query not found; set CALI_QUERY" >&2
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
  if [[ "${OUT_ARG}" = /* ]]; then
    OUT_ABS="${OUT_ARG}"
  else
    OUT_ABS="$(pwd)/${OUT_ARG}"
  fi
fi

REGION_REGEX="${REGION_REGEX:-^(CBCD_AngleSet::|CBCD_AsynchronousCommunicator::|CBCDSweepChunk::Sweep|CBCD_FLUDS::)}"

{
  echo "============================================================"
  echo "Caliper trace summary"
  echo "Inputs:"
  for f in "${CALI_FILES[@]}"; do
    echo "  ${f}"
  done
  echo "Region regex: ${REGION_REGEX}"
  echo "============================================================"
  echo

  printf "%-56s %12s %18s %18s %16s %16s %18s\n" \
    "Region" "Count" "Sum excl(ns)" "Sum incl(ns)" "Min excl(ns)" "Max excl(ns)" "Avg excl(ns)"

  for f in "${CALI_FILES[@]}"; do
    "${CALI_QUERY}" -e \
      --print-attributes=region,time.duration.ns,time.inclusive.duration.ns \
      "${f}"
  done | gawk -F, -v region_regex="${REGION_REGEX}" '
    function getv(key,   i, a, n) {
      for (i = 1; i <= NF; ++i) {
        n = split($i, a, "=")
        if (a[1] == key)
          return substr($i, length(key) + 2)
      }
      return ""
    }
    {
      region = getv("region")
      if (region == "" || region !~ region_regex)
        next

      ex = getv("time.duration.ns")
      inc = getv("time.inclusive.duration.ns")
      if (ex == "")
        next
      if (inc == "")
        inc = ex

      ex += 0
      inc += 0
      count[region]++
      sum_ex[region] += ex
      sum_inc[region] += inc
      if (!(region in min_ex) || ex < min_ex[region])
        min_ex[region] = ex
      if (ex > max_ex[region])
        max_ex[region] = ex
    }
    END {
      for (r in count) {
        avg = sum_ex[r] / count[r]
        printf "%-56s %12d %18.0f %18.0f %16.0f %16.0f %18.2f\n", \
          r, count[r], sum_ex[r], sum_inc[r], min_ex[r], max_ex[r], avg
      }
    }
  ' | sort -k3,3nr
} | tee "${OUT_ABS}"

echo
echo "[summarize_cali] wrote ${OUT_ABS}"
