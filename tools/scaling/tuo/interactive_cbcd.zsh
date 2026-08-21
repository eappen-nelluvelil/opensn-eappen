#!/bin/zsh

set -euo pipefail

script=${0:A}
source_dir=${OPENSN_SOURCE:-${script:h:h:h:h}}
revision=$(git -C "$source_dir" rev-parse HEAD)
tag=$(git -C "$source_dir" rev-parse --short=12 "$revision")

study_root=${OPENSN_TUO_STUDY_ROOT:-/usr/workspace/${USER}/opensn-gpu/cbcd-v2-studies}
deps_root=${OPENSN_TUO_ROOT:-$study_root/builds/gfx942}
build_dir=${OPENSN_TUO_BUILD:-$deps_root/build-candidate-$tag}
results_root=${OPENSN_TUO_RESULTS:-$study_root/results}
study_dir=${OPENSN_TUO_INTERACTIVE_STUDY:-$results_root/cbcd-v2-interactive-$tag}
mesh_cache=${OPENSN_TUO_MESH_CACHE:-$deps_root/mesh-cache}
env_file=$deps_root/env.zsh
binary=$build_dir/python/opensn
queue=${OPENSN_TUO_QUEUE:-pdebug}
time_limit=${OPENSN_TUO_TIME_LIMIT:-60m}
bank=${OPENSN_TUO_BANK:-}

usage()
{
  cat >&2 <<EOF
usage: $0 COMMAND [NODES]

Commands:
  build          build this revision in a one-node pdebug allocation
  build-here     build this revision in the current allocation
  prepare        generate cached strong-scaling inputs for 1, 2, and 4 nodes
  run NODES      run CBCD V2 in a pdebug allocation (NODES must be 1, 2, or 4)
  run-all        run the 1-, 2-, and 4-node cases sequentially
  run-here NODES run CBCD V2 in the current allocation
  summary        print metrics from the latest run at each node count
  paths          print the derived source, build, and result paths

Optional environment overrides:
  OPENSN_SOURCE, OPENSN_TUO_STUDY_ROOT, OPENSN_TUO_ROOT, OPENSN_TUO_BUILD,
  OPENSN_TUO_RESULTS, OPENSN_TUO_INTERACTIVE_STUDY, OPENSN_TUO_MESH_CACHE,
  OPENSN_TUO_QUEUE, OPENSN_TUO_TIME_LIMIT, and OPENSN_TUO_BANK
EOF
  exit 2
}

check_nodes()
{
  case ${1:-} in
    1|2|4) ;;
    *)
      print -u2 'NODES must be 1, 2, or 4.'
      exit 2
      ;;
  esac
}

allocation_command()
{
  local nodes=$1
  local -a command=(flux alloc -N "$nodes" -q "$queue" --exclusive -t "$time_limit")
  [[ -n $bank ]] && command+=(-B "$bank")
  print -r -l -- "${command[@]}"
}

require_dependencies()
{
  [[ -r $env_file ]] || {
    print -u2 "Dependency environment does not exist: $env_file"
    print -u2 'Build the OpenSn dependencies before using this helper.'
    exit 1
  }
}

require_build()
{
  require_dependencies
  [[ -x $binary ]] || {
    print -u2 "CBCD V2 binary does not exist: $binary"
    print -u2 "Run '$0 build' first."
    exit 1
  }
}

build_here()
{
  require_dependencies
  mkdir -p -- "$deps_root/logs"
  export OPENSN_SOURCE=$source_dir
  export OPENSN_TUO_ROOT=$deps_root
  export OPENSN_TUO_BUILD=$build_dir

  print -- "Building CBCD V2 revision $revision"
  print -- "Build directory: $build_dir"
  zsh "$source_dir/tools/scaling/tuo/bootstrap.zsh" build-opensn \
    |& tee "$deps_root/logs/build-candidate-$tag.log"
}

build()
{
  require_dependencies
  local -a allocation=("${(@f)$(allocation_command 1)}")
  print -- "Requesting a one-node $queue allocation for the build."
  "$allocation[@]" zsh "$script" build-here
}

prepare()
{
  require_build

  local reuse=true
  if [[ -r $study_dir/manifest.json &&
        -r $study_dir/inputs/strong-1.py &&
        -r $study_dir/inputs/strong-2.py &&
        -r $study_dir/inputs/strong-4.py ]]; then
    local nodes
    for nodes in 1 2 4; do
      grep -q '"save_angular_flux": False' \
        "$study_dir/inputs/strong-$nodes.py" || reuse=false
    done
  else
    reuse=false
  fi
  if $reuse; then
    print -- "Reusing prepared inputs in $study_dir"
    return
  fi
  if [[ -d $study_dir && -n $(ls -A -- "$study_dir") ]]; then
    print -u2 "Incomplete or incompatible study directory: $study_dir"
    print -u2 'Set OPENSN_TUO_INTERACTIVE_STUDY to a new directory.'
    exit 1
  fi

  source "$env_file"
  local gmsh
  gmsh=$(command -v gmsh 2>/dev/null) || {
    print -u2 'Gmsh is not available in the OpenSn Python environment.'
    exit 1
  }
  "$gmsh" --version >/dev/null
  local -a bank_argument=()
  [[ -n $bank ]] && bank_argument=(--bank "$bank")

  python "$source_dir/tools/scaling/tuo/study.py" prepare \
    --binary "$binary" \
    --environment "$env_file" \
    --output "$study_dir" \
    --mesh-cache "$mesh_cache" \
    --gmsh "$gmsh" \
    --label CBCD-V2-interactive \
    --revision "$revision" \
    --nodes 1,2,4 \
    --repetitions 1 \
    --queue "$queue" \
    --time-limit "$time_limit" \
    --no-save-angular-flux \
    "${bank_argument[@]}"

  print -- "Prepared CBCD V2 inputs in $study_dir"
}

run_here()
{
  local nodes=$1
  check_nodes "$nodes"
  require_build
  local input=$study_dir/inputs/strong-$nodes.py
  [[ -r $input ]] || {
    print -u2 "Strong-scaling input does not exist: $input"
    print -u2 "Run '$0 prepare' first."
    exit 1
  }

  source "$env_file"
  export MPICH_GPU_SUPPORT_ENABLED=1
  export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
  unset OPENSN_NUM_THREADS OPENSN_CBCD_NUM_WORKERS OMP_NUM_THREADS

  local ranks=$((nodes * 4))
  local parent=$study_dir/interactive/nodes-$nodes
  local run_dir=$parent/run-$(date -u +%Y%m%dT%H%M%SZ)-$$
  mkdir -p -- "$run_dir"
  ln -sfn "${run_dir:t}" "$parent/latest"

  {
    print -- "revision=$revision"
    print -- "binary=$binary"
    print -- "input=$input"
    print -- "nodes=$nodes"
    print -- "ranks=$ranks"
    print -- 'gpus_per_rank=1'
    print -- 'save_angular_flux=false'
  } >| "$run_dir/metadata.txt"

  print -- "Running CBCD V2 on $nodes node(s), $ranks ranks, and one GPU per rank."
  print -- "Results: $run_dir"
  /usr/bin/time -f 'wall_seconds=%e max_rss_kb=%M' -o "$run_dir/time.txt" \
    flux run -N "$nodes" -n "$ranks" -g1 --exclusive \
      "$binary" -i "$input" |& tee "$run_dir/output.txt"
}

run()
{
  local nodes=$1
  check_nodes "$nodes"
  require_build
  [[ -r $study_dir/inputs/strong-$nodes.py ]] || {
    print -u2 "Run '$0 prepare' before requesting an allocation."
    exit 1
  }

  local -a allocation=("${(@f)$(allocation_command "$nodes")}")
  print -- "Requesting a $nodes-node $queue allocation."
  "$allocation[@]" zsh "$script" run-here "$nodes"
}

run_all()
{
  local nodes
  for nodes in 1 2 4; do
    run "$nodes"
  done
}

summary()
{
  local nodes latest output timing
  for nodes in 1 2 4; do
    latest=$study_dir/interactive/nodes-$nodes/latest
    output=$latest/output.txt
    timing=$latest/time.txt
    print -- "== $nodes node(s) =="
    if [[ ! -r $output ]]; then
      print -- 'no completed or active run'
      continue
    fi
    grep -E 'avg_sweep_time|lagged_unknowns|(^|[[:space:]])unknowns[[:space:]]*=' \
      "$output" || print -- 'OpenSn metrics not found'
    [[ -r $timing ]] && cat -- "$timing"
  done
}

paths()
{
  print -- "source=$source_dir"
  print -- "revision=$revision"
  print -- "dependencies=$deps_root"
  print -- "build=$build_dir"
  print -- "binary=$binary"
  print -- "study=$study_dir"
  print -- "mesh_cache=$mesh_cache"
}

(( $# >= 1 )) || usage
command=$1
shift
case $command in
  build) (( $# == 0 )) || usage; build ;;
  build-here) (( $# == 0 )) || usage; build_here ;;
  prepare) (( $# == 0 )) || usage; prepare ;;
  run) (( $# == 1 )) || usage; run "$1" ;;
  run-all) (( $# == 0 )) || usage; run_all ;;
  run-here) (( $# == 1 )) || usage; run_here "$1" ;;
  summary) (( $# == 0 )) || usage; summary ;;
  paths) (( $# == 0 )) || usage; paths ;;
  *) usage ;;
esac
