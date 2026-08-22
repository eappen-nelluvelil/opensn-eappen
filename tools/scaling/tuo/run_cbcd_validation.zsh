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
  submit-campaign
                 rebuild OpenSn, then prepare and submit the complete
                 resource-aware strong/weak and full-node profile campaign
  status         write partial scaling/profile summaries and show queued jobs
  collect        collect the resource-aware results present for LABEL

The helper honors the existing OPENSN_TUO_ROOT, OPENSN_TUO_BUILD,
OPENSN_TUO_MESH_DIR, OPENSN_TUO_RESULTS, and OPENSN_TUO_BANK settings.
LABEL selects separate result directories for this executable/configuration.
EOF
  exit 2
}

campaign_paths()
{
  campaign_batch_root=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-batch/resource-aware
  campaign_profile_root=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile/resource-aware
}

require_fresh_campaign_label()
{
  campaign_paths
  if [[ -e $campaign_batch_root || -e $campaign_profile_root ]]; then
    print -u2 -- "Campaign label '$OPENSN_TUO_LABEL' already has a study directory."
    print -u2 -- 'Choose a new label; repeating a full submission would duplicate jobs.'
    print -u2 -- "batch=$campaign_batch_root"
    print -u2 -- "profile=$campaign_profile_root"
    exit 1
  fi
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

submit_campaign()
{
  unset OPENSN_CBCD_NUM_WORKERS
  export OPENSN_TUO_NODES=${OPENSN_TUO_NODES:-1,2,4,8,16,32,64,128,256}
  export OPENSN_TUO_REPETITIONS=${OPENSN_TUO_REPETITIONS:-3}
  export OPENSN_TUO_MAX_ITERATIONS=${OPENSN_TUO_MAX_ITERATIONS:-10}
  export OPENSN_TUO_PROFILE_NODES=$OPENSN_TUO_NODES
  export OPENSN_TUO_PROFILE_DIVISOR=39
  export OPENSN_TUO_PROFILE_ITERATIONS=$OPENSN_TUO_MAX_ITERATIONS
  export OPENSN_TUO_PROFILES=baseline,caliper-mpi,pmpi
  export OPENSN_TUO_BATCH_TIME_LIMIT=1h
  export OPENSN_TUO_PROFILE_TIME_LIMIT=1h
  export OPENSN_TUO_BATCH_ROOT=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-batch
  export OPENSN_TUO_PROFILE_ROOT=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile/resource-aware

  require_fresh_campaign_label

  local -a campaign_nodes campaign_profiles
  campaign_nodes=("${(@s:,:)OPENSN_TUO_NODES}")
  campaign_profiles=("${(@s:,:)OPENSN_TUO_PROFILES}")

  stage 'Selected full-scale campaign configuration'
  zsh "$helper" paths
  print -- "scaling_jobs=$((2 * ${#campaign_nodes}))"
  print -- "profile_jobs=$((${#campaign_profiles} * ${#campaign_nodes}))"
  print -- 'allocation_time_limit=1h'

  stage 'Rebuilding OpenSn from the selected checkout'
  zsh "$helper" rebuild

  stage 'Preparing all studies before submitting any jobs'
  zsh "$helper" prepare-batch resource-aware
  zsh "$helper" prepare-profile

  stage 'Submitting uninstrumented strong/weak scaling jobs'
  zsh "$campaign_batch_root/submit.zsh"

  stage 'Submitting strong-scaling diagnostic profile jobs'
  zsh "$campaign_profile_root/submit.zsh" \
    --nodes "$OPENSN_TUO_PROFILE_NODES" \
    --profiles "$OPENSN_TUO_PROFILES"

  stage 'Full-scale campaign submitted'
  print -- "batch_results=$campaign_batch_root"
  print -- "profile_results=$campaign_profile_root"
  print -- 'Monitor with: flux jobs -u $USER'
  print -- "Collect progress with: $script status $OPENSN_TUO_LABEL"
  print -- "Collect final results with: $script collect $OPENSN_TUO_LABEL"
}

campaign_status()
{
  campaign_paths
  local found=0
  local environment=$OPENSN_TUO_ROOT/env.zsh
  [[ -r $environment ]] || {
    print -u2 -- "Dependency environment is not ready: $environment"
    return 1
  }
  source "$environment"

  if [[ -r $campaign_batch_root/manifest.json ]]; then
    stage 'Collecting currently available scaling measurements'
    python "$source_dir/tools/scaling/tuo/study.py" collect \
      --study "$campaign_batch_root" --allow-incomplete
    sed -n '1,80p' "$campaign_batch_root/summary.md"
    found=1
  fi
  if [[ -r $campaign_profile_root/manifest.json ]]; then
    stage 'Collecting currently available profile inventory'
    python "$source_dir/tools/scaling/tuo/study.py" collect-profile \
      --study "$campaign_profile_root" --allow-incomplete
    sed -n '1,80p' "$campaign_profile_root/profile-summary.md"
    found=1
  fi
  (( found == 1 )) || {
    print -u2 -- "No prepared campaign was found for label '$OPENSN_TUO_LABEL'."
    return 1
  }
  stage 'Current Flux jobs'
  flux jobs -u "$USER" || true
}

collect_results()
{
  local found=0 overall_rc=0
  local interactive_root=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-interactive/resource-aware
  local batch_root=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-batch/resource-aware

  if [[ -r $batch_root/manifest.json ]]; then
    stage 'Collecting resource-aware batch results'
    zsh "$helper" collect-batch resource-aware || overall_rc=1
    found=1
  fi
  if [[ -r $OPENSN_TUO_PROFILE_ROOT/manifest.json ]]; then
    stage 'Collecting profile inventory'
    zsh "$helper" collect-profile || overall_rc=1
    found=1
  fi
  if [[ -r $interactive_root/manifest.json ]]; then
    stage 'Collecting resource-aware interactive results'
    zsh "$helper" collect-interactive resource-aware || overall_rc=1
    found=1
  fi
  (( found == 1 )) || {
    print -u2 -- "No prepared results were found for label '$OPENSN_TUO_LABEL'."
    exit 1
  }
  if (( overall_rc == 0 )); then
    stage 'Collection completed'
  else
    print -u2 -- 'Collection is incomplete; available summaries were still written.'
  fi
  return $overall_rc
}

(( $# == 2 )) || usage
command=$1
export OPENSN_TUO_LABEL=$2
export OPENSN_TUO_BATCH_ROOT=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-batch
export OPENSN_TUO_PROFILE_ROOT=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile/resource-aware

[[ -x $helper ]] || {
  print -u2 "Tuo helper is not executable: $helper"
  exit 1
}

case $command in
  smoke-profile) smoke_profile ;;
  submit-scaling) submit_scaling ;;
  submit-campaign) submit_campaign ;;
  status) campaign_status ;;
  collect) collect_results ;;
  *) usage ;;
esac
