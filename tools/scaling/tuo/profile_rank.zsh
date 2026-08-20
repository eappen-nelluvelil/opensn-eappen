#!/bin/zsh

set -euo pipefail

: ${OPENSN_PROFILE_MODE:?OPENSN_PROFILE_MODE is required}
: ${OPENSN_PROFILE_BINARY:?OPENSN_PROFILE_BINARY is required}
: ${OPENSN_PROFILE_INPUT:?OPENSN_PROFILE_INPUT is required}
: ${OPENSN_PROFILE_OUTPUT:?OPENSN_PROFILE_OUTPUT is required}

rank=${FLUX_TASK_RANK:-${PMI_RANK:-0}}
rank_output=$OPENSN_PROFILE_OUTPUT/rank-$rank
mkdir -p -- "$rank_output"

case $OPENSN_PROFILE_MODE in
  rocprof)
    exec rocprofv3 \
      --hip-runtime-trace \
      --kernel-trace \
      --memory-copy-trace \
      --memory-allocation-trace \
      --stats \
      --output-format csv \
      --output-directory "$rank_output" \
      -- "$OPENSN_PROFILE_BINARY" -i "$OPENSN_PROFILE_INPUT"
    ;;
  *)
    print -u2 "unsupported profile mode: $OPENSN_PROFILE_MODE"
    exit 2
    ;;
esac
