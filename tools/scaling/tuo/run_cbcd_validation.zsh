#!/bin/zsh

set -euo pipefail

script=${0:A}
source_dir=${OPENSN_SOURCE:-${script:h:h:h:h}}
helper=$source_dir/tools/scaling/tuo/interactive_cbcd.zsh
study_root=${OPENSN_TUO_STUDY_ROOT:-/usr/workspace/${USER}/opensn-gpu/cbcd-v2-studies}
export OPENSN_SOURCE=$source_dir
export OPENSN_TUO_RESULTS=${OPENSN_TUO_RESULTS:-$study_root/results}

usage()
{
  cat >&2 <<EOF
usage: $0 COMMAND LABEL

Commands:
  smoke-profile  rebuild OpenSn, run/collect resource-aware 1/2/4-node strong
                 scaling, then run/collect the selected profiles interactively
  submit-scaling prepare and submit resource-aware strong/weak pbatch studies
  collect        collect the resource-aware results present for LABEL

The helper honors the existing OPENSN_TUO_ROOT, OPENSN_TUO_BUILD,
OPENSN_TUO_MESH_DIR, OPENSN_TUO_RESULTS, and OPENSN_TUO_BANK settings.
LABEL selects separate result directories for this executable/configuration.
EOF
  exit 2
}

stage()
{
  print
  print -- "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ==="
}

show_configuration()
{
  stage 'Selected validation configuration'
  zsh "$helper" paths
  print -- "interactive_iterations=$OPENSN_TUO_INTERACTIVE_ITERATIONS"
  print -- "profile_nodes=$OPENSN_TUO_PROFILE_NODES"
  print -- "profile_iterations=$OPENSN_TUO_PROFILE_ITERATIONS"
  print -- "profiles=$OPENSN_TUO_PROFILES"
}

smoke_profile()
{
  unset OPENSN_CBCD_NUM_WORKERS
  export OPENSN_TUO_INTERACTIVE_ITERATIONS=${OPENSN_TUO_INTERACTIVE_ITERATIONS:-2}
  export OPENSN_TUO_PROFILE_NODES=${OPENSN_TUO_PROFILE_NODES:-1,2,4}
  export OPENSN_TUO_PROFILE_DIVISOR=${OPENSN_TUO_PROFILE_DIVISOR:-39}
  export OPENSN_TUO_PROFILE_ITERATIONS=${OPENSN_TUO_PROFILE_ITERATIONS:-3}
  export OPENSN_TUO_PROFILES=${OPENSN_TUO_PROFILES:-baseline,caliper-mpi,pmpi}
  export OPENSN_TUO_TIME_LIMIT=${OPENSN_TUO_TIME_LIMIT:-60m}

  show_configuration

  stage 'Rebuilding OpenSn from the selected checkout'
  zsh "$helper" rebuild

  stage 'Running resource-aware 1/2/4-node strong-scaling smoke study'
  zsh "$helper" run-interactive resource-aware

  stage 'Collecting strong-scaling smoke results'
  zsh "$helper" collect-interactive resource-aware

  stage 'Running resource-aware profiles in pdebug allocations'
  zsh "$helper" run-profile-interactive

  stage 'Collecting profile inventory'
  zsh "$helper" collect-profile

  stage 'Smoke and profile workflow completed'
  print -- "interactive_results=${OPENSN_TUO_RESULTS}/$OPENSN_TUO_LABEL-interactive/resource-aware"
  print -- "profile_results=${OPENSN_TUO_PROFILE_ROOT:-${OPENSN_TUO_RESULTS}/$OPENSN_TUO_LABEL-profile/resource-aware}"
}

submit_scaling()
{
  unset OPENSN_CBCD_NUM_WORKERS
  stage 'Preparing and submitting resource-aware strong/weak pbatch studies'
  zsh "$helper" submit-batch resource-aware
  stage 'Scaling jobs submitted'
  print -- "batch_results=${OPENSN_TUO_RESULTS}/$OPENSN_TUO_LABEL-batch/resource-aware"
  print -- 'Monitor with: flux jobs -u $USER'
}

collect_results()
{
  local found=0
  local interactive_root=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-interactive/resource-aware
  local batch_root=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-batch/resource-aware

  if [[ -r $interactive_root/manifest.json ]]; then
    stage 'Collecting resource-aware interactive results'
    zsh "$helper" collect-interactive resource-aware
    found=1
  fi
  if [[ -r $OPENSN_TUO_PROFILE_ROOT/manifest.json ]]; then
    stage 'Collecting profile inventory'
    zsh "$helper" collect-profile
    found=1
  fi
  if [[ -r $batch_root/manifest.json ]]; then
    stage 'Collecting resource-aware batch results'
    zsh "$helper" collect-batch resource-aware
    found=1
  fi
  (( found == 1 )) || {
    print -u2 -- "No prepared results were found for label '$OPENSN_TUO_LABEL'."
    exit 1
  }
  stage 'Collection completed'
}

(( $# == 2 )) || usage
command=$1
export OPENSN_TUO_LABEL=$2
export OPENSN_TUO_PROFILE_ROOT=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile/resource-aware

[[ -x $helper ]] || {
  print -u2 "Tuo helper is not executable: $helper"
  exit 1
}

case $command in
  smoke-profile) smoke_profile ;;
  submit-scaling) submit_scaling ;;
  collect) collect_results ;;
  *) usage ;;
esac
