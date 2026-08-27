#!/bin/zsh

set -euo pipefail

: ${OPENSN_PROFILE_MODE:?OPENSN_PROFILE_MODE is required}
: ${OPENSN_PROFILE_BINARY:?OPENSN_PROFILE_BINARY is required}
: ${OPENSN_PROFILE_INPUT:?OPENSN_PROFILE_INPUT is required}
: ${OPENSN_PROFILE_OUTPUT:?OPENSN_PROFILE_OUTPUT is required}
: ${FLUX_TASK_RANK:?FLUX_TASK_RANK is required}
: ${FLUX_TASK_LOCAL_ID:?FLUX_TASK_LOCAL_ID is required}
: ${ROCR_VISIBLE_DEVICES:?ROCR_VISIBLE_DEVICES is required}

[[ $ROCR_VISIBLE_DEVICES != *,* ]] || {
  print -u2 "rank $FLUX_TASK_RANK was assigned multiple GPUs"
  exit 1
}

rank_output=$OPENSN_PROFILE_OUTPUT/rank-$FLUX_TASK_RANK
selected_ranks=${OPENSN_ROCPROF_RANKS:-0}
if [[ $selected_ranks != all && ,$selected_ranks, != *,$FLUX_TASK_RANK,* ]]; then
  exec "$OPENSN_PROFILE_BINARY" --verbose 1 -i "$OPENSN_PROFILE_INPUT"
fi
mkdir -p -- "$rank_output"
{
  print -- "hostname=$(hostname -s)"
  print -- "rank=$FLUX_TASK_RANK"
  print -- "local_rank=$FLUX_TASK_LOCAL_ID"
  print -- "gpu=$ROCR_VISIBLE_DEVICES"
  print -- "omp_num_threads=${OMP_NUM_THREADS:?OMP_NUM_THREADS is required}"
  grep '^Cpus_allowed_list:' /proc/self/status
  rocprofv3 --version 2>&1 || true
} >| "$rank_output/metadata.txt"

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
      -- "$OPENSN_PROFILE_BINARY" --verbose 1 -i "$OPENSN_PROFILE_INPUT"
    ;;
  *)
    print -u2 "unsupported profile mode: $OPENSN_PROFILE_MODE"
    exit 2
    ;;
esac
