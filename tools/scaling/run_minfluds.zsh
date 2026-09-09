#!/usr/bin/env zsh

set -eu
setopt pipe_fail
(( $# >= 3 && $# <= 4 )) || {
  print -u2 'usage: run_minfluds.zsh {dane|tuo} {scaling|status|collect|beavrs} SCALING_LABEL [BEAVRS_LABEL]'
  exit 2
}
cluster=$1
action=$2
label=$3
[[ $cluster == dane || $cluster == tuo ]] || exit 2
[[ $action == scaling || $action == status || $action == collect || $action == beavrs ]] || exit 2
[[ -n $label && $label != *[^A-Za-z0-9_.-]* ]] || exit 2
tools_dir=${0:A:h}
repo=$(git -C "$tools_dir" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse HEAD)
short=${revision[1,9]}
bank=${OPENSN_STUDY_BANK:-cbronze}
benchmark=${OPENSN_BEAVRS_SOURCE:-/usr/workspace/$USER/opensn-gpu/beavrs-benchmark}

if [[ $cluster == dane ]]; then
  export OPENSN_DANE_SOURCE=$repo OPENSN_DANE_BANK=$bank
  export OPENSN_DANE_WORK_ROOT=${OPENSN_DANE_WORK_ROOT:-/usr/workspace/$USER/opensn-dane-cbc-scaling}
  export OPENSN_DANE_RESULTS=${OPENSN_DANE_RESULTS:-/p/lustre1/$USER/opensn-results}
  export OPENSN_DANE_TOOLCHAIN=${OPENSN_DANE_TOOLCHAIN:-clang19-openmpi412-python314-1}
  export OPENSN_DANE_ENVIRONMENT=${OPENSN_DANE_ENVIRONMENT:-$OPENSN_DANE_WORK_ROOT/toolchains/$OPENSN_DANE_TOOLCHAIN/opensn-dane-env.sh}
  export OPENSN_DANE_TIME_LIMIT=01:00:00
  runner=$tools_dir/dane/run_cbc_scaling.zsh
  if [[ $action != beavrs ]]; then
    [[ $action != scaling ]] || action=launch
    exec zsh "$runner" "$action" "$label"
  fi
  manifest=$OPENSN_DANE_RESULTS/$label/manifest.json
  [[ -r $manifest ]] || { print -u2 "Missing scaling manifest: $manifest"; exit 2; }
  values=("${(@f)$(python3 - "$manifest" "$revision" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
data = manifest['implementations']['branch']
if data['sha'] != sys.argv[2]:
    raise SystemExit('Scaling manifest uses a different revision; use its checkout.')
print(data['source'])
print(data['build'])
print(manifest['environment'])
PY
  )}")
  (( ${#values} == 3 )) || exit 2
  source_dir=$values[1]
  build=$values[2]
  environment=$values[3]
  nodes=32
  time_limit=24:00:00
  results=$OPENSN_DANE_RESULTS
else
  study_root=${OPENSN_TUO_STUDY_ROOT:-/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies}
  export OPENSN_SOURCE=$repo OPENSN_TUO_BANK=$bank
  export OPENSN_TUO_REUSE_ROOT=${OPENSN_TUO_REUSE_ROOT:-$study_root/builds/gfx942-minfluds-profiling-6386bec5e}
  export OPENSN_TUO_ROOT=$study_root/builds/minfluds2-$short
  export OPENSN_TUO_BUILD=$OPENSN_TUO_ROOT/build-opensn
  export OPENSN_TUO_MESH_DIR=${OPENSN_TUO_MESH_DIR:-$study_root/builds/gfx942/mesh-cache}
  export OPENSN_TUO_RESULTS=${OPENSN_TUO_RESULTS:-/p/lustre5/$USER/opensn-results}
  export OPENSN_TUO_LABEL=$label OPENSN_TUO_NUM_THREADS=21
  export OPENSN_TUO_NODES=1,2,4,8,16,32,64,128,256
  export OPENSN_TUO_PROFILES=cbcd-metrics,caliper-mpi
  export OPENSN_TUO_BATCH_TIME_LIMIT=1h OPENSN_TUO_PROFILE_TIME_LIMIT=1h
  export OPENSN_TUO_BATCH_ROOT=$OPENSN_TUO_RESULTS/$label-batch
  export OPENSN_TUO_PROFILE_ROOT=$OPENSN_TUO_RESULTS/$label-profile/resource-aware
  unset OPENSN_CBCD_NUM_WORKERS
  if [[ $action != beavrs ]]; then
    [[ $action != scaling ]] || action=submit-scaling-metrics
    exec zsh "$tools_dir/tuo/run_cbcd_validation.zsh" "$action" "$label"
  fi
  source_dir=$repo
  build=$OPENSN_TUO_BUILD
  environment=$OPENSN_TUO_ROOT/env.zsh
  nodes=16
  time_limit=12h
  results=$OPENSN_TUO_RESULTS
fi

(( $# == 4 )) || { print -u2 'Supply a distinct fourth argument: BEAVRS_LABEL'; exit 2; }
beavrs_label=$4
[[ -n $beavrs_label && $beavrs_label != *[^A-Za-z0-9_.-]* ]] || exit 2
[[ -x $build/python/opensn ]] || {
  print -u2 'The Native scaling build must finish successfully before BEAVRS submission.'
  exit 2
}
python3 "$tools_dir/beavrs_retry.py" --cluster "$cluster" \
  --source "$source_dir" --build "$build" --environment "$environment" \
  --benchmark "$benchmark" --output "$results/$beavrs_label" \
  --nodes "$nodes" --time-limit "$time_limit" --bank "$bank" --submit
