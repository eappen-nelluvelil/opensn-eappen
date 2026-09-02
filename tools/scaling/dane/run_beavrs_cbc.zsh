#!/usr/bin/env zsh

set -eu
setopt pipe_fail

PROGRAM=$0
SCRIPT_DIR=${0:A:h}
DRIVER=$SCRIPT_DIR/beavrs.py
RESULTS=${OPENSN_DANE_RESULTS:-/p/lustre1/$USER/opensn-results}
BENCHMARK=${OPENSN_DANE_BEAVRS_SOURCE:-}
NODES=${OPENSN_DANE_BEAVRS_NODES:-32}
RANKS_PER_NODE=${OPENSN_DANE_BEAVRS_RANKS_PER_NODE:-64}
TIME_LIMIT=${OPENSN_DANE_BEAVRS_TIME_LIMIT:-24:00:00}

usage()
{
  print -u2 -- "Usage: $PROGRAM {launch|prepare|submit|status|collect|paths} SCALING_LABEL BEAVRS_LABEL"
  print -u2 -- "Set OPENSN_DANE_BEAVRS_SOURCE before launch or prepare."
  exit 2
}

valid_label()
{
  [[ -n $1 && $1 != *[^A-Za-z0-9_.-]* ]]
}

(( $# == 3 )) || usage
COMMAND=$1
SCALING_LABEL=$2
BEAVRS_LABEL=$3
valid_label "$SCALING_LABEL" && valid_label "$BEAVRS_LABEL" || usage

SCALING_ROOT=$RESULTS/$SCALING_LABEL
OUTPUT=$RESULTS/$BEAVRS_LABEL

prepare()
{
  [[ -d $BENCHMARK ]] || {
    print -u2 -- 'Set OPENSN_DANE_BEAVRS_SOURCE to the original BEAVRS directory.'
    exit 2
  }
  python3 "$DRIVER" prepare \
    --scaling-root "$SCALING_ROOT" \
    --output "$OUTPUT" \
    --benchmark-source "$BENCHMARK" \
    --nodes "$NODES" \
    --ranks-per-node "$RANKS_PER_NODE" \
    --time-limit "$TIME_LIMIT"
}

case $COMMAND in
  launch) prepare; python3 "$DRIVER" submit --output "$OUTPUT" ;;
  prepare) prepare ;;
  submit) python3 "$DRIVER" submit --output "$OUTPUT" ;;
  status) python3 "$DRIVER" status --output "$OUTPUT" ;;
  collect) python3 "$DRIVER" collect --output "$OUTPUT" ;;
  paths)
    print -- "scaling=$SCALING_ROOT"
    print -- "benchmark=${BENCHMARK:-unset}"
    print -- "results=$OUTPUT"
    print -- "nodes=$NODES"
    print -- "ranks_per_node=$RANKS_PER_NODE"
    print -- "time_limit=$TIME_LIMIT"
    ;;
  *) usage ;;
esac
