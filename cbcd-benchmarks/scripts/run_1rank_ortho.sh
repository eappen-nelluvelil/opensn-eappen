#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_root=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$bench_root/.." && pwd)
case_dir="$repo_root/test/python/modules/linear_boltzmann_solvers/transport_steady"
case_file="transport_3d_1b_ortho_cbc_gpu.py"

old_bin=${OLD_OPENSN:-"$repo_root/build-cbc-and-cbcd-with-minimally-sized-fluds-cfsp-update/python/opensn"}
new_bin=${NEW_OPENSN:-"$repo_root/build-cbc-and-cbcd-with-minimally-sized-fluds-cfsp-update-2/python/opensn"}
mpi_np=${MPI_NP:-1}
mpirun_cmd=${MPIRUN:-mpirun}
caliper_scope=${CALIPER_SCOPE:-both}
verbose=${VERBOSE:-0}
run_label=${RUN_LABEL:-}
time_cmd=()
if [[ -x /usr/bin/time ]]; then
  time_cmd=(/usr/bin/time -v)
fi

while (($#)); do
  case "$1" in
    --no-caliper)
      caliper_scope=none
      ;;
    --caliper-scope)
      caliper_scope=$2
      shift
      ;;
    --old)
      old_bin=$2
      shift
      ;;
    --new)
      new_bin=$2
      shift
      ;;
    --np)
      mpi_np=$2
      shift
      ;;
    --label)
      run_label=$2
      shift
      ;;
    --verbose)
      verbose=1
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--no-caliper] [--caliper-scope old|new|both|none] [--old PATH] [--new PATH] [--np N] [--label NAME] [--verbose]

Environment overrides:
  OLD_OPENSN   Old OpenSn binary path
  NEW_OPENSN   New OpenSn binary path
  MPIRUN       MPI launcher, default: mpirun
  MPI_NP       MPI ranks, default: 1
  CALIPER_SCOPE Caliper runs to perform, default: both
  RUN_LABEL    Suffix for the timestamped result directory
  VERBOSE      Set to 1 to stream full OpenSn logs to the terminal
EOF
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

case "$caliper_scope" in
  old|new|both|none) ;;
  *)
    echo "error: --caliper-scope must be old, new, both, or none" >&2
    exit 2
    ;;
esac

if [[ ! -x "$old_bin" ]]; then
  echo "error: old binary is not executable: $old_bin" >&2
  exit 1
fi
if [[ ! -x "$new_bin" ]]; then
  echo "error: new binary is not executable: $new_bin" >&2
  exit 1
fi

timestamp=$(date +%Y%m%d-%H%M%S)
if [[ -n "$run_label" ]]; then
  timestamp="$timestamp-$run_label"
fi
out_dir="$bench_root/results/1rank-ortho-$timestamp"
if [[ -e "$out_dir" ]]; then
  out_dir="$out_dir-$$"
fi
mkdir -p "$out_dir"

capture_binary_info() {
  local label=$1
  local binary=$2
  local info="$out_dir/${label}_binary_info.txt"
  local help="$out_dir/${label}_help.txt"

  {
    echo "path: $binary"
    stat "$binary"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$binary"
    fi
    echo
    echo "linked libraries:"
    ldd "$binary" | grep -Ei 'mpi|mpich|openmpi|caliper|cuda|cudart|cupti|nvtx|hip|hsa|fabric|xpmem' || true
  } > "$info" 2>&1

  "$binary" -h > "$help" 2>&1 || true
}

caliper_args_for() {
  local label=$1
  local help="$out_dir/${label}_help.txt"

  if grep -q -- '--caliper-preset' "$help"; then
    printf '%s\n' '--caliper-preset' 'cuda'
  elif grep -q -- '--caliper' "$help"; then
    printf '%s\n' '--caliper=runtime-report(calc.inclusive=true,max_column_width=120)'
  else
    return 1
  fi
}

run_case() {
  local label=$1
  local binary=$2
  local log=$3
  shift 3

  body() {
    echo "=== $label ==="
    echo "timestamp: $(date --iso-8601=seconds)"
    echo "host: $(hostname)"
    echo "binary: $binary"
    echo "command: $mpirun_cmd --np $mpi_np $binary -c $* -i $case_file"
    echo
    cd "$case_dir"
    "${time_cmd[@]}" "$mpirun_cmd" --np "$mpi_np" "$binary" -c "$@" -i "$case_file"
  }

  if [[ "$verbose" == "1" ]]; then
    body "$@" 2>&1 | tee "$log"
  else
    printf 'running: %s\n' "$label"
    body "$@" > "$log" 2>&1
    printf 'finished: %s\n' "$label"
  fi
}

summarize_log() {
  local label=$1
  local log=$2

  awk -v label="$label" '
    BEGIN {
      avg = grind = unknowns = elapsed = slot_wall = "";
      comm = "no";
      stream_query = "";
      try_initialize = "";
      copy_boundary = "";
      try_advance = "";
      launch_batch = "";
      retire_batch = "";
      flush_batch = "";
    }
    /Wall time[[:space:]]+:/ { slot_wall = $(NF - 1) }
    /avg_sweep_time =/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "avg_sweep_time") avg = $(i + 2);
        if ($i == "sweep_time_per_unknown") grind = $(i + 2);
      }
    }
    /unknowns =/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "unknowns") {
          unknowns = $(i + 2);
          gsub(/,/, "", unknowns);
        }
      }
    }
    /Average sweep time/ { avg = $NF }
    /Sweep Time\/Unknown/ { grind = $NF }
    /Number of unknowns per sweep/ { unknowns = $NF }
    /Elapsed execution time:/ { elapsed = $NF }
    /CBCD_AsynchronousCommunicator::CommThreadLoop/ { comm = "yes" }
    /^cudaStreamQuery[[:space:]]/ { stream_query = $2 }
    /^CBCD_AngleSet::TryInitialize[[:space:]]/ { try_initialize = $2 }
    /^  CBCD_FLUDS::CopyIncomingBoundaryPsiToDevice[[:space:]]/ { copy_boundary = $2 }
    /^CBCD_AngleSet::TryAdvanceOneStep/ { getline; if ($2 != "") try_advance = $2 }
    /^  CBCD_AngleSet::LaunchBatch[[:space:]]/ { getline; if ($2 != "") launch_batch = $2 }
    /^  CBCD_AngleSet::RetireBatch[[:space:]]/ { retire_batch = $2 }
    /^  CBCD_AngleSet::FlushBatch[[:space:]]/ { flush_batch = $2 }
    END {
      printf "%-24s avg_sweep_s=%s grind_ns=%s unknowns=%s elapsed=%s slot_plan_s=%s comm_thread=%s cudaStreamQuery_s=%s tryInitialize_s=%s copyBoundary_s=%s tryAdvance_s=%s launchBatch_s=%s retireBatch_s=%s flushBatch_s=%s\n",
             label, avg, grind, unknowns, elapsed, slot_wall, comm, stream_query,
             try_initialize, copy_boundary, try_advance, launch_batch, retire_batch, flush_batch;
    }
  ' "$log"
}

extract_hot_regions() {
  local log=$1
  local out=$2
  grep -E \
    'avg_sweep_time|sweep_time_per_unknown|Elapsed execution time|CBCD_AngleSet|CBCD_AsynchronousCommunicator|CBCD_FLUDS::Copy|CBCDSweepChunk|SweepScheduler::S|cudaStreamQuery|cudaStreamSynchronize|cudaMemcpyAsync|cudaLaunchKernel|Kernel|Average sweep time|Sweep Time/Unknown|Number of unknowns per sweep' \
    "$log" > "$out" || true
}

{
  echo "CBCD 1-rank orthogonal benchmark"
  echo "timestamp: $timestamp"
  echo "repo: $repo_root"
  echo "case: $case_dir/$case_file"
  echo "old_binary: $old_bin"
  echo "new_binary: $new_bin"
  echo "mpi_np: $mpi_np"
  echo "caliper_scope: $caliper_scope"
  echo
  echo "git_branch: $(git -C "$repo_root" rev-parse --abbrev-ref HEAD)"
  echo "git_commit: $(git -C "$repo_root" rev-parse HEAD)"
  echo "git_status:"
  git -C "$repo_root" status --short --untracked-files=no
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    echo
  fi
  echo "system:"
  uname -a
  command -v lscpu >/dev/null 2>&1 && lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\\(s\\)|NUMA' || true
  echo
  echo "mpi:"
  command -v "$mpirun_cmd" >/dev/null 2>&1 && "$mpirun_cmd" --version | head -5 || true
} | tee "$out_dir/summary.txt"

cp "$case_dir/$case_file" "$out_dir/$case_file"
git -C "$repo_root" diff > "$out_dir/git_diff.patch"
git -C "$repo_root" status --short --untracked-files=all > "$out_dir/git_status.txt"
env | sort | grep -E '^(CUDA|NVIDIA|OMP|OPENMP|OMPI|PMI|PMIX|MPI|MPICH|UCX|FI_|LD_LIBRARY_PATH|PATH|CMAKE_PREFIX_PATH)=' \
  > "$out_dir/environment.txt" || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$out_dir/nvidia-smi.txt" 2>&1 || true
fi

capture_binary_info old "$old_bin"
capture_binary_info new "$new_bin"

run_case "old non-profiled" "$old_bin" "$out_dir/old_nonprofile.log"
run_case "new non-profiled" "$new_bin" "$out_dir/new_nonprofile.log"

case "$caliper_scope" in
  old|both)
    if mapfile -t old_caliper_args < <(caliper_args_for old); then
      run_case "old caliper" "$old_bin" "$out_dir/old_caliper.log" "${old_caliper_args[@]}"
    else
      echo "old binary does not expose Caliper; skipping old Caliper run" | tee "$out_dir/old_caliper.log"
    fi
    ;;
esac

case "$caliper_scope" in
  new|both)
    if mapfile -t new_caliper_args < <(caliper_args_for new); then
      run_case "new caliper" "$new_bin" "$out_dir/new_caliper.log" "${new_caliper_args[@]}"
    else
      echo "new binary does not expose Caliper; skipping new Caliper run" | tee "$out_dir/new_caliper.log"
    fi
    ;;
esac

for log in "$out_dir"/*.log; do
  extract_hot_regions "$log" "${log%.log}_hot_regions.txt"
done

{
  echo
  echo "metrics:"
  summarize_log "old_nonprofile" "$out_dir/old_nonprofile.log"
  summarize_log "new_nonprofile" "$out_dir/new_nonprofile.log"
  [[ -f "$out_dir/old_caliper.log" ]] && summarize_log "old_caliper" "$out_dir/old_caliper.log"
  [[ -f "$out_dir/new_caliper.log" ]] && summarize_log "new_caliper" "$out_dir/new_caliper.log"
  echo
  echo "analysis files:"
  printf "  %s\n" "$out_dir"/*_hot_regions.txt
  echo
  echo "logs: $out_dir"
} | tee -a "$out_dir/summary.txt"

echo "$out_dir"
