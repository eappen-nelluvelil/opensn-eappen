#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_root=$(cd -- "$script_dir/.." && pwd)
worker="$script_dir/run_1rank_ortho.sh"

repeats=${REPEATS:-5}
jobs=${JOBS:-1}
mpi_np=${MPI_NP:-1}
mpirun_cmd=${MPIRUN:-mpirun}
old_bin=${OLD_OPENSN:-}
new_bin=${NEW_OPENSN:-}
caliper_args=(--no-caliper)

usage() {
  cat <<EOF
Usage: $0 [--repeats N] [--jobs N] [--old PATH] [--new PATH] [--np N] [--mpirun CMD] [--with-caliper|--caliper-scope old|new|both|none]

Defaults:
  --repeats ${REPEATS:-5}
  --jobs ${JOBS:-1}
  --np ${MPI_NP:-1}
  --no-caliper

Notes:
  Use --jobs 1 for uncontended timing on a single GPU.
  Use --jobs N only when concurrent runs have isolated resources or when throughput matters more than timing fidelity.
EOF
}

while (($#)); do
  case "$1" in
    --repeats)
      repeats=$2
      shift
      ;;
    --jobs)
      jobs=$2
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
    --mpirun)
      mpirun_cmd=$2
      shift
      ;;
    --with-caliper)
      caliper_args=()
      ;;
    --caliper-scope)
      caliper_args=(--caliper-scope "$2")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! [[ "$repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --repeats must be a positive integer" >&2
  exit 2
fi
if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --jobs must be a positive integer" >&2
  exit 2
fi

if ((jobs > repeats)); then
  jobs=$repeats
fi

timestamp=$(date +%Y%m%d-%H%M%S)
batch_dir="$bench_root/results/1rank-ortho-repeats-$timestamp"
if [[ -e "$batch_dir" ]]; then
  batch_dir="$batch_dir-$$"
fi
mkdir -p "$batch_dir"
metrics_tsv="$batch_dir/metrics.tsv"
printf 'repeat\tstatus\tduration_s\told_avg_s\tnew_avg_s\told_grind_ns\tnew_grind_ns\tunknowns\telapsed_old\telapsed_new\tspeedup_pct\tresult_dir\n' \
  > "$metrics_tsv"

worker_args=(--np "$mpi_np" "${caliper_args[@]}")
if [[ -n "$old_bin" ]]; then
  worker_args+=(--old "$old_bin")
fi
if [[ -n "$new_bin" ]]; then
  worker_args+=(--new "$new_bin")
fi

progress_bar() {
  local done=$1
  local total=$2
  local width=28
  local filled=$((done * width / total))
  local empty=$((width - filled))

  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %d/%d' "$done" "$total"
}

format_seconds() {
  local seconds=$1
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))

  if ((hours > 0)); then
    printf '%dh%02dm%02ds' "$hours" "$minutes" "$secs"
  elif ((minutes > 0)); then
    printf '%dm%02ds' "$minutes" "$secs"
  else
    printf '%ds' "$secs"
  fi
}

extract_metrics() {
  local summary=$1
  awk '
    /^old_nonprofile/ {
      for (i = 1; i <= NF; ++i) {
        split($i, a, "=");
        if (a[1] == "avg_sweep_s") old_avg = a[2];
        if (a[1] == "grind_ns") old_grind = a[2];
        if (a[1] == "unknowns") unknowns = a[2];
        if (a[1] == "elapsed") old_elapsed = a[2];
      }
    }
    /^new_nonprofile/ {
      for (i = 1; i <= NF; ++i) {
        split($i, a, "=");
        if (a[1] == "avg_sweep_s") new_avg = a[2];
        if (a[1] == "grind_ns") new_grind = a[2];
        if (a[1] == "elapsed") new_elapsed = a[2];
      }
    }
    END {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s",
             old_avg, new_avg, old_grind, new_grind, unknowns, old_elapsed, new_elapsed;
    }
  ' "$summary"
}

run_one() {
  local repeat=$1
  local label
  local stdout_log
  local status_file
  label=$(printf 'repeat-%03d' "$repeat")
  stdout_log="$batch_dir/$label.out"
  status_file="$batch_dir/$label.status"

  set +e
  local start_epoch
  start_epoch=$(date +%s)
  RUN_LABEL="$label" VERBOSE=0 MPIRUN="$mpirun_cmd" "$worker" "${worker_args[@]}" > "$stdout_log" 2>&1
  local status=$?
  local duration_s
  duration_s=$(($(date +%s) - start_epoch))

  local result_dir
  result_dir=$(tail -n 1 "$stdout_log")

  if ((status == 0)) && [[ -d "$result_dir" && -f "$result_dir/summary.txt" ]]; then
    local parsed
    parsed=$(extract_metrics "$result_dir/summary.txt")
    local old_avg new_avg old_grind new_grind unknowns old_elapsed new_elapsed speedup
    IFS=$'\t' read -r old_avg new_avg old_grind new_grind unknowns old_elapsed new_elapsed <<< "$parsed"
    speedup=$(awk -v old="$old_avg" -v new="$new_avg" 'BEGIN { printf "%.4f", 100.0 * (old - new) / old }')
    printf '%d\tok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$repeat" "$duration_s" "$old_avg" "$new_avg" "$old_grind" "$new_grind" "$unknowns" \
      "$old_elapsed" "$new_elapsed" "$speedup" "$result_dir" > "$status_file"
  else
    printf '%d\tfail\t%s\t\t\t\t\t\t\t\t\t%s\n' "$repeat" "$duration_s" "$stdout_log" > "$status_file"
  fi

  exit 0
}

print_completion() {
  local completed=$1
  local status_file=$2
  local repeat status duration_s old_avg new_avg old_grind new_grind unknowns old_elapsed new_elapsed speedup result_dir

  IFS=$'\t' read -r repeat status duration_s old_avg new_avg old_grind new_grind unknowns old_elapsed new_elapsed speedup result_dir < "$status_file"
  cat "$status_file" >> "$metrics_tsv"

  local elapsed_s remaining_s mean_speedup
  elapsed_s=$(($(date +%s) - batch_start_epoch))
  remaining_s=$((elapsed_s * (repeats - completed) / completed))
  mean_speedup=$(awk -F '\t' '
    NR > 1 && $2 == "ok" { ++n; sum += $11 }
    END { if (n > 0) printf "%.4f", sum / n; else printf "0.0000" }
  ' "$metrics_tsv")

  if [[ "$status" == "ok" ]]; then
    printf '%s elapsed=%s eta=%s repeat=%03d dt=%s old=%ss/%sns new=%ss/%sns speedup=%+0.4f%% mean=%+0.4f%%\n' \
      "$(progress_bar "$completed" "$repeats")" \
      "$(format_seconds "$elapsed_s")" "$(format_seconds "$remaining_s")" \
      "$repeat" "$(format_seconds "$duration_s")" \
      "$old_avg" "$old_grind" "$new_avg" "$new_grind" "$speedup" "$mean_speedup"
  else
    printf '%s elapsed=%s eta=%s repeat=%03d failed dt=%s log=%s\n' \
      "$(progress_bar "$completed" "$repeats")" \
      "$(format_seconds "$elapsed_s")" "$(format_seconds "$remaining_s")" \
      "$repeat" "$(format_seconds "$duration_s")" "$result_dir"
  fi
}

summarize_batch() {
  awk -F '\t' '
    NR == 1 { next }
    $2 == "ok" {
      ++n;
      duration += $3;
      old_avg += $4;
      new_avg += $5;
      old_grind += $6;
      new_grind += $7;
      speedup += $11;
    }
    END {
      if (n == 0) {
        print "No successful repeats.";
        exit;
      }
      printf "successful_repeats=%d\n", n;
      printf "duration_s_mean=%.9g\n", duration / n;
      printf "old_avg_s_mean=%.9g\n", old_avg / n;
      printf "new_avg_s_mean=%.9g\n", new_avg / n;
      printf "old_grind_ns_mean=%.9g\n", old_grind / n;
      printf "new_grind_ns_mean=%.9g\n", new_grind / n;
      printf "speedup_pct_mean=%.6f\n", speedup / n;
    }
  ' "$metrics_tsv"
}

printf 'CBCD 1-rank orthogonal repeats: repeats=%d jobs=%d np=%s caliper=%s\n' \
  "$repeats" "$jobs" "$mpi_np" "${caliper_args[*]:-both}"
printf 'batch_dir=%s\n' "$batch_dir"
if ((jobs > 1)); then
  printf 'warning: concurrent GPU benchmark runs can perturb timings; use --jobs 1 for uncontended measurements.\n'
fi

declare -A repeat_by_pid=()
next=1
running=0
completed=0
batch_start_epoch=$(date +%s)

while ((completed < repeats)); do
  while ((running < jobs && next <= repeats)); do
    run_one "$next" &
    pid=$!
    repeat_by_pid[$pid]=$next
    ((++next))
    ((++running))
  done

  done_pid=
  wait -n -p done_pid
  repeat=${repeat_by_pid[$done_pid]}
  unset 'repeat_by_pid[$done_pid]'
  running=$((running - 1))
  completed=$((completed + 1))

  status_file=$(printf '%s/repeat-%03d.status' "$batch_dir" "$repeat")
  print_completion "$completed" "$status_file"
done

summary_file="$batch_dir/aggregate-summary.txt"
summarize_batch | tee "$summary_file"
printf 'metrics=%s\n' "$metrics_tsv"
printf 'summary=%s\n' "$summary_file"
