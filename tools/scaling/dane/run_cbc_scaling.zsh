#!/usr/bin/env zsh

set -eu
setopt pipe_fail

PROGRAM=$0
SCRIPT_DIR=${0:A:h}
SOURCE_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
STUDY=$SCRIPT_DIR/study.py
BOOTSTRAP=$SCRIPT_DIR/bootstrap_opensn.zsh

usage()
{
  print -u2 -- "Usage:"
  print -u2 -- "  $PROGRAM setup"
  print -u2 -- "  $PROGRAM {setup-launch|launch|prepare|submit|status|collect|paths} LABEL"
  print -u2 -- ""
  print -u2 -- "Required environment for launch/prepare:"
  print -u2 -- "  OPENSN_DANE_BANK       Slurm account/bank"
  print -u2 -- ""
  print -u2 -- "Optional overrides:"
  print -u2 -- "  OPENSN_DANE_RESULTS     Campaign parent (default: /p/lustre1/\$USER/opensn-results)"
  print -u2 -- "  OPENSN_DANE_WORK_ROOT   Sources/builds (default: /usr/workspace/\$USER/opensn-dane-cbc-scaling)"
  print -u2 -- "  OPENSN_DANE_ENVIRONMENT Bash-sourceable environment setup file"
  print -u2 -- "  OPENSN_DANE_REPETITIONS Trials per allocation (default: 3)"
  print -u2 -- "  OPENSN_DANE_TIME_LIMIT  Scaling-job limit (default: 01:00:00)"
  print -u2 -- "  OPENSN_DANE_BUILD_JOBS  Parallel build jobs (default: 16)"
  print -u2 -- "  OPENSN_DANE_TOOLCHAIN    Isolated toolchain tag (default: isolated-1)"
}

require_command()
{
  command -v "$1" >/dev/null 2>&1 || {
    print -u2 -- "ERROR: required command not found: $1"
    return 1
  }
}

require_label()
{
  if [[ $# -ne 1 || -z "$1" || "$1" == *[^A-Za-z0-9_.-]* ]]; then
    usage
    return 1
  fi
}

set_paths()
{
  local label=$1
  export OPENSN_DANE_RESULTS=${OPENSN_DANE_RESULTS:-/p/lustre1/$USER/opensn-results}
  export OPENSN_DANE_WORK_ROOT=${OPENSN_DANE_WORK_ROOT:-/usr/workspace/$USER/opensn-dane-cbc-scaling}
  export OPENSN_DANE_TOOLCHAIN=${OPENSN_DANE_TOOLCHAIN:-isolated-1}
  local default_environment=$OPENSN_DANE_WORK_ROOT/toolchains/$OPENSN_DANE_TOOLCHAIN/opensn-dane-env.sh
  export OPENSN_DANE_ENVIRONMENT=${OPENSN_DANE_ENVIRONMENT:-$default_environment}
  export OPENSN_DANE_ROOT=$OPENSN_DANE_RESULTS/$label
}

show_paths()
{
  local label=$1
  set_paths "$label"
  print -- "source=$SOURCE_ROOT"
  print -- "work=$OPENSN_DANE_WORK_ROOT"
  print -- "results=$OPENSN_DANE_ROOT"
  print -- "environment=$OPENSN_DANE_ENVIRONMENT"
  print -- "revision=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
  print -- "nodes=1,2,4,8,16,32,64,128,256"
  print -- "ranks_per_node=64"
  print -- "build_type=Native"
  print -- "repetitions=${OPENSN_DANE_REPETITIONS:-3}"
}

prepare_campaign()
{
  local label=$1
  set_paths "$label"
  : ${OPENSN_DANE_BANK:?Set OPENSN_DANE_BANK to the Slurm account/bank}

  require_command git
  require_command python3

  if [[ ! -r $OPENSN_DANE_ENVIRONMENT ]]; then
    print -u2 -- "ERROR: isolated Dane environment is not ready: $OPENSN_DANE_ENVIRONMENT"
    print -u2 -- "Run: $PROGRAM setup"
    return 1
  fi
  source "$OPENSN_DANE_ENVIRONMENT"
  require_command "${OPENSN_DANE_GMSH:-gmsh}"

  if [[ -n $(git -C "$SOURCE_ROOT" status --porcelain) ]]; then
    print -u2 -- "ERROR: the CBC-cycle source worktree is not clean."
    return 1
  fi

  local branch_sha=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
  local branch_short=${branch_sha[1,9]}
  local branch_source=$OPENSN_DANE_WORK_ROOT/sources/cbc-minfluds-$branch_short
  local build_root=$OPENSN_DANE_WORK_ROOT/builds/$label
  local branch_build=$build_root/cbc-minfluds-$branch_short-native

  mkdir -p "$OPENSN_DANE_WORK_ROOT/sources" "$OPENSN_DANE_WORK_ROOT/builds"
  if [[ -e $branch_source ]]; then
    local actual=$(git -C "$branch_source" rev-parse HEAD)
    if [[ $actual != $branch_sha ]]; then
      print -u2 -- "ERROR: $branch_source is at $actual, expected $branch_sha"
      return 1
    fi
  else
    git -C "$SOURCE_ROOT" worktree add --detach "$branch_source" "$branch_sha"
  fi

  python3 "$STUDY" prepare \
    --root "$OPENSN_DANE_ROOT" \
    --bank "$OPENSN_DANE_BANK" \
    --source "$branch_source" \
    --sha "$branch_sha" \
    --build "$branch_build" \
    --geometry "$SOURCE_ROOT/tools/scaling/lib/cube.geo" \
    --cross-sections "$SOURCE_ROOT/tools/scaling/lib/xs_168g.xs" \
    --gmsh "${OPENSN_DANE_GMSH:-gmsh}" \
    --nodes 1,2,4,8,16,32,64,128,256 \
    --ranks-per-node 64 \
    --repetitions "${OPENSN_DANE_REPETITIONS:-3}" \
    --time-limit "${OPENSN_DANE_TIME_LIMIT:-01:00:00}" \
    --build-time-limit "${OPENSN_DANE_BUILD_TIME_LIMIT:-01:00:00}" \
    --build-jobs "${OPENSN_DANE_BUILD_JOBS:-16}" \
    --environment "$OPENSN_DANE_ENVIRONMENT"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command_name=$1
shift

if [[ $command_name == setup ]]; then
  if [[ $# -ne 0 ]]; then
    usage
    exit 2
  fi
  zsh "$BOOTSTRAP" setup
  exit
fi

require_label "$@"
label=$1
set_paths "$label"

case $command_name in
  setup-launch)
    require_command sbatch
    zsh "$BOOTSTRAP" setup
    prepare_campaign "$label"
    python3 "$STUDY" submit --root "$OPENSN_DANE_ROOT"
    ;;
  launch)
    require_command sbatch
    prepare_campaign "$label"
    python3 "$STUDY" submit --root "$OPENSN_DANE_ROOT"
    ;;
  prepare)
    prepare_campaign "$label"
    ;;
  submit)
    require_command sbatch
    python3 "$STUDY" submit --root "$OPENSN_DANE_ROOT"
    ;;
  status)
    python3 "$STUDY" status --root "$OPENSN_DANE_ROOT"
    ;;
  collect)
    python3 "$STUDY" collect --root "$OPENSN_DANE_ROOT"
    ;;
  paths)
    show_paths "$label"
    ;;
  *)
    usage
    exit 2
    ;;
esac
