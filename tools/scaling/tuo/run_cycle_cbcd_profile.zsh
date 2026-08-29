#!/bin/zsh

set -euo pipefail

usage()
{
  cat >&2 <<EOF
usage: $0 ACTION [PROFILE]

Actions:
  build      build this exact checkout in the shared Tuo dependency stack
  run        run fresh 1/2/4-node strong profiles
  resume     run only incomplete 1/2/4-node strong profile cases
  collect    collect the profile inventory and CBCD metrics
  status     collect available results and show the user's Flux jobs
  paths      print the resolved checkout, build, and result paths

PROFILE is optional for run and resume. Without it, the runner processes
cbcd-metrics, baseline, pmpi, caliper, and rocprof in that order. Set
OPENSN_TUO_BANK before running. OPENSN_TUO_CYCLE_LABEL may select a new
campaign; its default includes the exact source revision.
EOF
  exit 2
}

(( $# >= 1 && $# <= 2 )) || usage
action=$1
profile=${2:-}

script=${0:A}
source_dir=${script:h:h:h:h}
sha=$(git -C "$source_dir" rev-parse HEAD)
short=${sha[1,9]}
study_root=${OPENSN_TUO_STUDY_ROOT:-/usr/workspace/${USER}/opensn-gpu/cbcd-v2-studies}
work_root=${OPENSN_TUO_ROOT:-$study_root/builds/gfx942-update-3-clean-1}

[[ -n ${OPENSN_TUO_BANK:-} ]] || {
  print -u2 -- 'Set OPENSN_TUO_BANK to the LC bank for this campaign.'
  exit 2
}

export OPENSN_SOURCE=$source_dir
export OPENSN_TUO_ROOT=$work_root
export OPENSN_TUO_BUILD=${OPENSN_TUO_CYCLE_BUILD:-$work_root/build-opensn-cbcd-v2-cycles-$short}
export OPENSN_TUO_MESH_DIR=${OPENSN_TUO_MESH_DIR:-$study_root/builds/gfx942/mesh-cache}
export OPENSN_TUO_RESULTS=${OPENSN_TUO_RESULTS:-/p/lustre5/${USER}/opensn-results}
export OPENSN_TUO_LABEL=${OPENSN_TUO_CYCLE_LABEL:-cbcd-v2-cycles-$short-pdebug-1}
export OPENSN_TUO_PROFILE_ROOT=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile/resource-aware
export OPENSN_TUO_PROFILE_NODES=1,2,4
export OPENSN_TUO_PROFILE_KINDS=strong
export OPENSN_TUO_PROFILE_DIVISOR=39
export OPENSN_TUO_PROFILE_ITERATIONS=${OPENSN_TUO_CYCLE_ITERATIONS:-10}
export OPENSN_TUO_PROFILE_TIME_LIMIT=${OPENSN_TUO_CYCLE_TIME_LIMIT:-60m}
export OPENSN_TUO_TIME_LIMIT=$OPENSN_TUO_PROFILE_TIME_LIMIT
export OPENSN_TUO_PROGRESS_INTERVAL=${OPENSN_TUO_CYCLE_PROGRESS_INTERVAL:-60}
export OPENSN_TUO_PROFILES=${OPENSN_TUO_CYCLE_PROFILES:-cbcd-metrics,baseline,pmpi,caliper,rocprof}
unset OPENSN_CBCD_NUM_WORKERS

helper=$source_dir/tools/scaling/tuo/interactive_cbcd.zsh
study=$OPENSN_TUO_PROFILE_ROOT

ensure_build()
{
  if [[ -x $OPENSN_TUO_BUILD/python/opensn ]]; then
    print -- "Reusing completed build: $OPENSN_TUO_BUILD/python/opensn"
  elif [[ -r $OPENSN_TUO_ROOT/env.zsh ]]; then
    zsh "$helper" rebuild
  else
    zsh "$helper" build
  fi
}

collect_available()
{
  [[ -r $study/manifest.json ]] || {
    print -u2 -- "Profile study is not prepared: $study"
    return 1
  }
  [[ -r $OPENSN_TUO_ROOT/env.zsh ]] || {
    print -u2 -- "Dependency environment is not ready: $OPENSN_TUO_ROOT/env.zsh"
    return 1
  }
  source "$OPENSN_TUO_ROOT/env.zsh"
  python "$source_dir/tools/scaling/tuo/study.py" collect-profile \
    --study "$study" --allow-incomplete
  sed -n '1,100p' "$study/profile-summary.md"
  [[ ! -r $study/cbcd-metrics-summary.md ]] || \
    sed -n '1,100p' "$study/cbcd-metrics-summary.md"
}

run_selected()
{
  local command=$1
  ensure_build
  if [[ -n $profile ]]; then
    zsh "$helper" "$command" "$profile"
  else
    zsh "$helper" "$command"
  fi
}

case $action in
  build)
    (( $# == 1 )) || usage
    ensure_build
    ;;
  run)
    run_selected run-profile-interactive
    ;;
  resume)
    run_selected resume-profile-interactive
    ;;
  collect)
    (( $# == 1 )) || usage
    ensure_build
    zsh "$helper" collect-profile
    ;;
  status)
    (( $# == 1 )) || usage
    collect_available || true
    flux jobs -u "$USER" || true
    ;;
  paths)
    (( $# == 1 )) || usage
    print -- "source_sha=$sha"
    zsh "$helper" paths
    ;;
  *) usage ;;
esac
