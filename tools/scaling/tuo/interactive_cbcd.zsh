#!/bin/zsh

set -euo pipefail

script=${0:A}
source_dir=${OPENSN_SOURCE:-${script:h:h:h:h}}
revision=$(git -C "$source_dir" rev-parse --verify HEAD^{commit})
tag=${revision[1,12]}
work_root=${OPENSN_TUO_ROOT:-/usr/workspace/${USER}/opensn-gpu/tuo/gfx942}
build_dir=${OPENSN_TUO_BUILD:-$work_root/build-$tag}
results_root=${OPENSN_TUO_RESULTS:-$work_root/results}
mesh_cache=${OPENSN_TUO_MESH_CACHE:-$work_root/mesh-cache}
worker_policy=${OPENSN_CBCD_WORKER_POLICY:-hardware}
worker_count=${OPENSN_CBCD_NUM_WORKERS:-}
worker_tag=$worker_policy
[[ -z $worker_count ]] || worker_tag=$worker_tag-w$worker_count
label=${OPENSN_TUO_LABEL:-cbcd-v2-$tag-$worker_tag}
study_dir=${OPENSN_TUO_INTERACTIVE_STUDY:-$results_root/$label}
baseline_study=${OPENSN_TUO_BASELINE_STUDY:-}
queue=${OPENSN_TUO_QUEUE:-pdebug}
time_limit=${OPENSN_TUO_TIME_LIMIT:-60m}
interactive_iterations=${OPENSN_TUO_INTERACTIVE_ITERATIONS:-2}
bank=${OPENSN_TUO_BANK:-}
scalar_flux_rtol=${OPENSN_TUO_SCALAR_FLUX_RTOL:-1.0e-10}
scalar_flux_atol=${OPENSN_TUO_SCALAR_FLUX_ATOL:-1.0e-12}

usage()
{
  cat >&2 <<EOF
usage: $0 COMMAND [NODES]

Commands:
  build                 build this exact revision in a one-node allocation
  build-here            build in the current allocation
  prepare               prepare immutable 1/2/4-node strong-scaling jobs
  run NODES             run one prepared case in an exact-size allocation
  run-here NODES        run one case in the current allocation
  run-all               run 1/2/4 sequentially in one four-node allocation
  run-all-here          run 1/2/4 in the current four-node allocation
  paired-run-all        alternate baseline/candidate in one allocation
  paired-run-all-here   paired run in the current four-node allocation
  summary               strictly collect, report, and optionally compare
  paths                 print all derived paths and worker settings

Set OPENSN_CBCD_WORKER_POLICY=hardware|resource-aware. An optional positive
OPENSN_CBCD_NUM_WORKERS fixes the worker count. Set OPENSN_TUO_BASELINE_STUDY
for paired runs. Historical baseline studies must use hardware policy unless
OPENSN_TUO_ALLOW_BASELINE_POLICY_OVERRIDE=1 is deliberately set.
EOF
  exit 2
}

require_clean_source()
{
  [[ ${#revision} == 40 && $revision != *[^0-9a-f]* ]] || {
    print -u2 "Unable to resolve a full source SHA: $revision"
    exit 1
  }
  local status=$(git -C "$source_dir" status --porcelain --untracked-files=normal)
  [[ -z $status ]] || {
    print -u2 "Source tree is not clean: $source_dir"
    print -u2 -- "$status"
    exit 1
  }
}

validate_settings()
{
  case $worker_policy in
    hardware|resource-aware) ;;
    *) print -u2 "Invalid OPENSN_CBCD_WORKER_POLICY: $worker_policy"; exit 2 ;;
  esac
  [[ -z $worker_count || $worker_count == <1-> ]] || {
    print -u2 'OPENSN_CBCD_NUM_WORKERS must be a positive integer.'
    exit 2
  }
  [[ $interactive_iterations == <1-> ]] || {
    print -u2 'OPENSN_TUO_INTERACTIVE_ITERATIONS must be a positive integer.'
    exit 2
  }
  [[ $queue == pdebug ]] || {
    print -u2 'The interactive helper is intentionally restricted to pdebug.'
    exit 2
  }
}

environment_file()
{
  local manifest=$build_dir/tuo-build-manifest.json
  [[ -r $manifest ]] || {
    print -u2 "Build manifest does not exist: $manifest"
    exit 1
  }
  python3 - "$manifest" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["environment"])
PY
}

binary_path()
{
  print -- "$build_dir/python/opensn"
}

allocation()
{
  local nodes=$1
  local ranks=$((nodes * 4))
  local -a command=(
    flux alloc -N "$nodes" -n "$ranks" -q "$queue"
    --exclusive --amd-gpumode=SPX -t "$time_limit"
  )
  [[ -z $bank ]] || command+=(-B "$bank")
  print -r -l -- "${command[@]}"
}

check_nodes()
{
  case ${1:-} in
    1|2|4) ;;
    *) print -u2 'NODES must be 1, 2, or 4.'; exit 2 ;;
  esac
}

verify_study()
{
  local target=$1
  [[ -r $target/manifest.json ]] || {
    print -u2 "Study does not exist: $target"
    exit 1
  }
  local environment=$(python3 - "$target/manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["environment"])
PY
)
  (
    source "$environment"
    python "$target/assets/study.py" verify --study "$target"
  )
}

verify_candidate_study()
{
  verify_study "$study_dir"
  python3 - "$study_dir/manifest.json" "$revision" "$worker_policy" \
    "$worker_count" "$interactive_iterations" <<'PY'
import json
import sys

path, revision, worker_policy, worker_count, iterations = sys.argv[1:]
manifest = json.load(open(path))
compatibility = manifest.get("compatibility", {})
expected_workers = int(worker_count) if worker_count else None
if (
    manifest.get("revision") != revision
    or manifest.get("type") != "scaling"
    or manifest.get("repetitions") != 1
    or manifest.get("worker_policy") != worker_policy
    or manifest.get("cbcd_workers") != expected_workers
    or compatibility.get("nodes") != [1, 2, 4]
    or compatibility.get("kinds") != ["strong"]
    or compatibility.get("max_iterations") != int(iterations)
):
    raise SystemExit("prepared candidate does not match the helper settings")
PY
}

build_here()
{
  require_clean_source
  export OPENSN_SOURCE=$source_dir
  export OPENSN_TUO_ROOT=$work_root
  export OPENSN_TUO_BUILD=$build_dir
  export OPENSN_TUO_REVISION=$revision
  zsh "$source_dir/tools/scaling/tuo/bootstrap.zsh" all
}

build()
{
  require_clean_source
  local -a command=("${(@f)$(allocation 1)}")
  print -- 'Requesting one pdebug node for a clean build.'
  "$command[@]" zsh "$script" build-here
}

prepare()
{
  require_clean_source
  if [[ -r $study_dir/manifest.json ]]; then
    verify_candidate_study
    print -- "Reusing verified immutable study: $study_dir"
    return
  fi

  local binary=$(binary_path)
  local environment=$(environment_file)
  source "$environment"
  local gmsh=$(command -v gmsh)
  local -a worker_args=(--worker-policy "$worker_policy")
  [[ -z $worker_count ]] || worker_args+=(--cbcd-workers "$worker_count")
  local -a bank_args=()
  [[ -z $bank ]] || bank_args=(--bank "$bank")
  python "$source_dir/tools/scaling/tuo/study.py" prepare \
    --source "$source_dir" \
    --binary "$binary" \
    --environment "$environment" \
    --output "$study_dir" \
    --mesh-cache "$mesh_cache" \
    --gmsh "$gmsh" \
    --label "$label" \
    --nodes 1,2,4 \
    --kinds strong \
    --repetitions 1 \
    --max-iterations "$interactive_iterations" \
    --queue pdebug \
    --time-limit "$time_limit" \
    --no-save-angular-flux \
    "${worker_args[@]}" "${bank_args[@]}"
}

run_here()
{
  local nodes=$1
  check_nodes "$nodes"
  : ${FLUX_JOB_ID:?run-here requires an active Flux allocation}
  verify_candidate_study
  zsh "$study_dir/jobs/strong-$nodes.zsh"
}

run()
{
  local nodes=$1
  check_nodes "$nodes"
  verify_candidate_study
  local -a command=("${(@f)$(allocation "$nodes")}")
  "$command[@]" zsh "$script" run-here "$nodes"
}

run_all_here()
{
  : ${FLUX_JOB_ID:?run-all-here requires an active Flux allocation}
  local nodes
  for nodes in 1 2 4; do
    run_here "$nodes"
  done
}

run_all()
{
  verify_candidate_study
  local -a command=("${(@f)$(allocation 4)}")
  print -- 'Requesting one four-node pdebug allocation for all three cases.'
  "$command[@]" zsh "$script" run-all-here
}

baseline_policy()
{
  python3 - "$baseline_study/manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1])).get("worker_policy", "unknown"))
PY
}

require_baseline()
{
  [[ -n $baseline_study ]] || {
    print -u2 'Set OPENSN_TUO_BASELINE_STUDY for a paired run.'
    exit 1
  }
  verify_study "$baseline_study"
  verify_candidate_study
  [[ ${baseline_study:A} != ${study_dir:A} ]] || {
    print -u2 'Baseline and candidate study paths must be different.'
    exit 1
  }
  local policy=$(baseline_policy)
  if [[ $policy != hardware &&
        ${OPENSN_TUO_ALLOW_BASELINE_POLICY_OVERRIDE:-0} != 1 ]]; then
    print -u2 "Historical baseline policy must be hardware; found: $policy"
    exit 1
  fi
  python3 - "$baseline_study/manifest.json" "$study_dir/manifest.json" <<'PY'
import json
import sys

baseline, candidate = (json.load(open(path)) for path in sys.argv[1:])
for name, manifest in (("baseline", baseline), ("candidate", candidate)):
    points = {
        (case.get("kind"), case.get("nodes"))
        for case in manifest.get("cases", [])
    }
    if (
        manifest.get("type") != "scaling"
        or manifest.get("repetitions") != 1
        or points != {("strong", 1), ("strong", 2), ("strong", 4)}
    ):
        raise SystemExit(f"{name} is not a one-trial strong 1/2/4 study")
    compatibility = dict(manifest["compatibility"])
    compatibility.pop("worker_policy", None)
    compatibility.pop("cbcd_workers", None)
    manifest["_paired_compatibility"] = compatibility
if baseline["_paired_compatibility"] != candidate["_paired_compatibility"]:
    raise SystemExit("baseline and candidate paired-study configurations differ")
PY
}

run_case_from()
{
  local target=$1
  local nodes=$2
  zsh "$target/jobs/strong-$nodes.zsh"
}

paired_run_all_here()
{
  : ${FLUX_JOB_ID:?paired-run-all-here requires an active Flux allocation}
  require_baseline
  verify_candidate_study
  run_case_from "$baseline_study" 1
  run_case_from "$study_dir" 1
  run_case_from "$study_dir" 2
  run_case_from "$baseline_study" 2
  run_case_from "$baseline_study" 4
  run_case_from "$study_dir" 4
}

paired_run_all()
{
  require_baseline
  verify_candidate_study
  local -a command=("${(@f)$(allocation 4)}")
  print -- 'Requesting one four-node pdebug allocation for the paired sequence.'
  "$command[@]" zsh "$script" paired-run-all-here
}

collect_one()
{
  local target=$1
  local environment=$(python3 - "$target/manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["environment"])
PY
)
  (
    source "$environment"
    python "$target/assets/study.py" collect --study "$target"
  )
  sed -n '1,40p' "$target/summary.md"
}

summary()
{
  verify_candidate_study
  collect_one "$study_dir"
  [[ -n $baseline_study ]] || return
  require_baseline
  collect_one "$baseline_study"
  local comparison=$results_root/compare-${baseline_study:t}-to-${study_dir:t}
  local -a policy_args=()
  [[ ${OPENSN_TUO_ALLOW_POLICY_COMPARISON:-0} != 1 ]] ||
    policy_args=(--allow-worker-policy-difference)
  [[ ${OPENSN_TUO_ALLOW_BASELINE_POLICY_OVERRIDE:-0} != 1 ]] ||
    policy_args+=(--allow-nonhardware-baseline)
  local environment=$(environment_file)
  (
    source "$environment"
    python "$study_dir/assets/study.py" compare \
      --baseline "$baseline_study" \
      --candidate "$study_dir" \
      --output "$comparison" \
      --max-slowdown "${OPENSN_TUO_MAX_SLOWDOWN:-1.03}" \
      --scalar-flux-rtol "$scalar_flux_rtol" \
      --scalar-flux-atol "$scalar_flux_atol" \
      "${policy_args[@]}"
  )
  sed -n '1,80p' "$comparison/comparison.md"
}

paths()
{
  print -- "source=$source_dir"
  print -- "revision=$revision"
  print -- "work_root=$work_root"
  print -- "build=$build_dir"
  print -- "study=$study_dir"
  print -- "baseline_study=${baseline_study:-unset}"
  print -- "mesh_cache=$mesh_cache"
  print -- "worker_policy=$worker_policy"
  print -- "requested_cbcd_workers=${worker_count:-policy-derived}"
  print -- "interactive_iterations=$interactive_iterations"
  print -- "scalar_flux_rtol=$scalar_flux_rtol"
  print -- "scalar_flux_atol=$scalar_flux_atol"
}

validate_settings
(( $# >= 1 )) || usage
command=$1
shift
case $command in
  build) (( $# == 0 )) || usage; build ;;
  build-here) (( $# == 0 )) || usage; build_here ;;
  prepare) (( $# == 0 )) || usage; prepare ;;
  run) (( $# == 1 )) || usage; run "$1" ;;
  run-here) (( $# == 1 )) || usage; run_here "$1" ;;
  run-all) (( $# == 0 )) || usage; run_all ;;
  run-all-here) (( $# == 0 )) || usage; run_all_here ;;
  paired-run-all) (( $# == 0 )) || usage; paired_run_all ;;
  paired-run-all-here) (( $# == 0 )) || usage; paired_run_all_here ;;
  summary) (( $# == 0 )) || usage; summary ;;
  paths) (( $# == 0 )) || usage; paths ;;
  *) usage ;;
esac
