#!/bin/zsh

set -euo pipefail

script=${0:A}
source_dir=${OPENSN_SOURCE:-${script:h:h:h:h}}
study_root=${OPENSN_TUO_STUDY_ROOT:-/usr/workspace/${USER}/opensn-gpu/cbcd-v2-studies}
work_root=${OPENSN_TUO_ROOT:-$study_root/builds/gfx942-update-3}
build_dir=${OPENSN_TUO_BUILD:-$work_root/build-opensn}
environment=$work_root/env.zsh
binary=$build_dir/python/opensn
results_root=${OPENSN_TUO_RESULTS:-/p/lustre5/${USER}/opensn-results}
benchmark_source=${OPENSN_TUO_BEAVRS_SOURCE:-}
bank=${OPENSN_TUO_BANK:-}
nodes=${OPENSN_TUO_BEAVRS_NODES:-32}
time_limit=${OPENSN_TUO_BEAVRS_TIME_LIMIT:-24h}
num_threads=${OPENSN_TUO_NUM_THREADS:-21}
preparer=$source_dir/tools/scaling/tuo/prepare_beavrs_cbcd.py

usage()
{
  cat >&2 <<EOF
usage: $0 {launch|prepare|submit|status|collect|paths} LABEL

launch prepares and submits one independent BEAVRS CBCD job. Set
OPENSN_TUO_BEAVRS_SOURCE to the directory containing:
  beavrs_quarter_core_gpu.py
  beavrs_quarter_core_partitioned.obj
  beavrs_CASMO-70.h5

The selected OpenSn executable must already be built. The job uses at least
32 pbatch nodes, four MPI ranks per node, one MI300A per rank, and a Native
build. OPENSN_TUO_BEAVRS_NODES and OPENSN_TUO_BEAVRS_TIME_LIMIT override the
32-node and 24-hour defaults.
EOF
  exit 2
}

check_label()
{
  [[ -n $1 && $1 != *[^A-Za-z0-9_.-]* ]] || {
    print -u2 -- "Invalid label: $1"
    exit 2
  }
}

set_paths()
{
  local label=$1
  export campaign_root=$results_root/$label-beavrs
  export job=$campaign_root/beavrs-cbcd.zsh
}

check_configuration()
{
  [[ $nodes == <32-> ]] || {
    print -u2 'OPENSN_TUO_BEAVRS_NODES must be an integer of at least 32.'
    exit 2
  }
  [[ $num_threads == <2-> ]] || {
    print -u2 'OPENSN_TUO_NUM_THREADS must be at least 2 for CBCD.'
    exit 2
  }
  [[ -n $bank ]] || {
    print -u2 'Set OPENSN_TUO_BANK.'
    exit 2
  }
  [[ -r $environment && -x $binary ]] || {
    print -u2 -- "OpenSn is not ready: $binary"
    exit 1
  }
  grep -qx 'CMAKE_BUILD_TYPE:STRING=Native' "$build_dir/CMakeCache.txt" || {
    print -u2 -- "OpenSn is not a Native build: $build_dir"
    exit 1
  }
  [[ -d $benchmark_source ]] || {
    print -u2 'Set OPENSN_TUO_BEAVRS_SOURCE to the BEAVRS benchmark directory.'
    exit 2
  }
  local file
  for file in beavrs_quarter_core_gpu.py beavrs_quarter_core_partitioned.obj beavrs_CASMO-70.h5; do
    [[ -s $benchmark_source/$file ]] || {
      print -u2 -- "Missing BEAVRS input: $benchmark_source/$file"
      exit 1
    }
  done
}

prepare_campaign()
{
  check_configuration
  [[ ! -e $campaign_root ]] || {
    print -u2 -- "Campaign directory already exists: $campaign_root"
    exit 1
  }
  mkdir -p -- "$campaign_root/scheduler"
  source "$environment"
  python "$preparer" \
    "$benchmark_source/beavrs_quarter_core_gpu.py" \
    "$campaign_root/beavrs_quarter_core_cbcd.py"

  local ranks=$((4 * nodes))
  cat >| "$job" <<EOF
#!/bin/zsh
#flux: --job-name=${campaign_root:t}
#flux: -N $nodes
#flux: -n $ranks
#flux: -q pbatch
#flux: -B $bank
#flux: --exclusive
#flux: -t $time_limit
#flux: --output=$campaign_root/scheduler/{{id}}.out
#flux: --error=$campaign_root/scheduler/{{id}}.err

set -euo pipefail
source ${(q)environment}
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
export OPENSN_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads

grep -qx 'CMAKE_BUILD_TYPE:STRING=Native' ${(q)build_dir}/CMakeCache.txt
run=$campaign_root/results/run-${FLUX_JOB_ID:-allocation}-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p -- "$run"
ln -s ${(q)benchmark_source}/beavrs_quarter_core_partitioned.obj "$run/"
ln -s ${(q)benchmark_source}/beavrs_CASMO-70.h5 "$run/"
cp ${(q)campaign_root}/beavrs_quarter_core_cbcd.py "$run/input.py"
{
  print -- 'revision=$(git -C "$source_dir" rev-parse HEAD)'
  print -- 'build_type=Native'
  print -- 'nodes=$nodes'
  print -- 'ranks=$ranks'
  print -- 'ranks_per_node=4'
  print -- 'gpus_per_rank=1'
  print -- 'opensn_num_threads=$num_threads'
  print -- "BEAVRS_QC_N_POLAR=${BEAVRS_QC_N_POLAR:-4}"
  print -- "BEAVRS_QC_N_AZIMUTHAL=${BEAVRS_QC_N_AZIMUTHAL:-32}"
  print -- "BEAVRS_QC_SCATTERING_ORDER=${BEAVRS_QC_SCATTERING_ORDER:-1}"
  print -- "BEAVRS_QC_USE_CMFD=${BEAVRS_QC_USE_CMFD:-False}"
} >| "$run/metadata.txt"

cd "$run"
set +e
/usr/bin/time -f 'wall_seconds=%e launcher_max_rss_kb=%M' -o time.txt \
  flux run -N $nodes -n $ranks --exclusive -o exit-on-error \
    ${(q)binary} --verbose 1 -i input.py > stdout.txt 2> stderr.txt
exit_code=$?
set -e
print -- "$exit_code" >| exit_code.txt
if (( exit_code != 0 )) || ! grep -q 'OpenSn finished execution\.' stdout.txt; then
  touch FAILED
  exit $((exit_code == 0 ? 1 : exit_code))
fi
touch SUCCESS
EOF
  chmod 700 "$job"
  print -- "Prepared $job"
}

submit_campaign()
{
  source "$environment"
  flux batch "$job"
}

status_campaign()
{
  flux jobs -u "$USER"
  find "$campaign_root/results" -maxdepth 2 -type f \
    \( -name SUCCESS -o -name FAILED \) -print 2>/dev/null || true
}

collect_campaign()
{
  local run=$(find "$campaign_root/results" -mindepth 1 -maxdepth 1 -type d \
    -name 'run-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n $run ]] || {
    print -u2 -- "No BEAVRS run found under $campaign_root"
    exit 1
  }
  local state=INCOMPLETE
  [[ -f $run/SUCCESS ]] && state=SUCCESS
  [[ -f $run/FAILED ]] && state=FAILED
  {
    print -- '# BEAVRS CBCD result'
    print
    print -- "- State: $state"
    print -- "- Result: \`$run\`"
    print
    print -- '```text'
    grep -E 'k_eff|Identified .* pins|Pin power min/mean/max|avg_sweep_time|sweep_time_per_unknown|OpenSn finished' \
      "$run/stdout.txt" 2>/dev/null || true
    print -- '```'
  } >| "$campaign_root/summary.md"
  cat "$campaign_root/summary.md"
}

(( $# == 2 )) || usage
command=$1
label=$2
check_label "$label"
set_paths "$label"

case $command in
  launch) prepare_campaign; submit_campaign ;;
  prepare) prepare_campaign ;;
  submit) submit_campaign ;;
  status) status_campaign ;;
  collect) collect_campaign ;;
  paths)
    print -- "source=$source_dir"
    print -- "build=$build_dir"
    print -- "binary=$binary"
    print -- "benchmark_source=${benchmark_source:-unset}"
    print -- "results=$campaign_root"
    print -- "nodes=$nodes"
    print -- "ranks_per_node=4"
    print -- "opensn_num_threads=$num_threads"
    ;;
  *) usage ;;
esac
