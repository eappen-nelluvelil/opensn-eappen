#!/usr/bin/env zsh

set -euo pipefail
trap 'exit 130' INT
trap 'exit 143' TERM

(( $# >= 2 && $# <= 3 )) || {
  print -u2 'usage: run_flux_copy_profile.zsh {run|resume|collect|status|paths} LABEL [PROFILE]'
  exit 2
}
action=$1
label=$2
[[ $action == run || $action == resume || $action == collect || $action == status || $action == paths ]] || exit 2
[[ -n $label && $label != *[^A-Za-z0-9_.-]* ]] || exit 2
source_dir=$(git -C "${0:A:h}" rev-parse --show-toplevel)
revision=$(git -C "$source_dir" rev-parse HEAD)
short=${revision[1,9]}
study_root=${OPENSN_TUO_STUDY_ROOT:-/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies}

export OPENSN_SOURCE=$source_dir
export OPENSN_TUO_REUSE_ROOT=${OPENSN_TUO_REUSE_ROOT:-$study_root/builds/gfx942-minfluds-profiling-6386bec5e}
export OPENSN_TUO_REUSE_VENV=${OPENSN_TUO_REUSE_VENV:-$study_root/builds/flux-copy-f60e0ecbd/venv}
export OPENSN_TUO_ROOT=$study_root/builds/flux-copy-$short
export OPENSN_TUO_BUILD=$OPENSN_TUO_ROOT/build-opensn
export OPENSN_TUO_RESULTS=${OPENSN_TUO_RESULTS:-/p/lustre5/$USER/opensn-results}
export OPENSN_TUO_MESH_DIR=${OPENSN_TUO_MESH_DIR:-$study_root/builds/gfx942/mesh-cache}
export OPENSN_TUO_BANK=${OPENSN_TUO_BANK:-cbronze}
export OPENSN_TUO_LABEL=$label
export OPENSN_TUO_PROFILE_ROOT=$OPENSN_TUO_RESULTS/$label-profile/resource-aware
export OPENSN_TUO_PROFILE_NODES=1,2,4,8
export OPENSN_TUO_PROFILE_KINDS=strong,weak
export OPENSN_TUO_PROFILE_DIVISOR=39
export OPENSN_TUO_PROFILE_ITERATIONS=10
export OPENSN_TUO_PROFILE_REPETITIONS=3
export OPENSN_TUO_NUM_THREADS=21
export OPENSN_TUO_TIME_LIMIT=60m
export OPENSN_TUO_PROFILE_TIME_LIMIT=60m
export OPENSN_TUO_QUEUE=pdebug
export OPENSN_TUO_PROGRESS_INTERVAL=60
export OPENSN_TUO_PROFILES=baseline,cbcd-metrics,caliper,caliper-mpi,pmpi,rocprof
export OPENSN_ROCPROF_RANKS=${OPENSN_ROCPROF_RANKS:-0}
unset OPENSN_CBCD_NUM_WORKERS OPENSN_CBCD_PROFILE_DIR
unset CALI_CONFIG CALI_SERVICES_ENABLE
helper=$source_dir/tools/scaling/tuo/interactive_cbcd.zsh

profiles=("${(@s:,:)OPENSN_TUO_PROFILES}")
if (( $# == 3 )); then
  case $3 in
    baseline|cbcd-metrics|caliper|caliper-mpi|pmpi|rocprof) profiles=("$3") ;;
    *) print -u2 "Unsupported profile: $3"; exit 2 ;;
  esac
fi

case $action in
  paths) exec zsh "$helper" paths ;;
  status)
    print -- "Results: $OPENSN_TUO_PROFILE_ROOT"
    exec flux jobs -u "$USER"
    ;;
  collect) exec zsh "$helper" collect-profile ;;
esac

[[ -z $(git -C "$source_dir" status --porcelain) ]] || {
  print -u2 'Use a clean, revision-specific source worktree.'
  exit 2
}
if [[ $action == run && -e $OPENSN_TUO_PROFILE_ROOT ]]; then
  print -u2 'This result directory already exists. Use resume or a new label.'
  exit 2
fi
if [[ $action == resume && ! -f $OPENSN_TUO_PROFILE_ROOT/manifest.json ]]; then
  print -u2 'No prepared campaign exists for this label. Use run first.'
  exit 2
fi

zsh "$helper" build
[[ -r $OPENSN_TUO_BUILD/source-revision.txt &&
   $(<"$OPENSN_TUO_BUILD/source-revision.txt") == $revision ]] || {
  print -u2 'The build does not match this checkout. Do not run the campaign.'
  exit 2
}

rc=0
for profile in "${profiles[@]}"; do
  if [[ $action == resume ]]; then
    command_name=resume-profile-interactive
  else
    command_name=run-profile-interactive
  fi
  if ! zsh "$helper" "$command_name" "$profile"; then
    print -u2 "Profile $profile failed. Retaining its logs and continuing with other profiles."
    rc=1
  fi
done
zsh "$helper" collect-profile || rc=1
print -- "Results: $OPENSN_TUO_PROFILE_ROOT"
exit $rc
