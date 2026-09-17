# Repeated-problem memory diagnostics

This diagnostic branch leaves the numerical solver unchanged. Tracing is off
unless `OPENSN_MEMORY_TRACE_DIR` names an existing writable directory.
Each rank writes a separate log with timestamps, host, PID, object address,
current RSS, peak RSS, anonymous RSS, and virtual, locked, and pinned memory
reported by Linux. Host values retain the `/proc` units, kB (KiB).

The trace covers problem construction, base runtime, sweep data structures,
FLUDS, solve entry/exit, solver cleanup, angleset cleanup, and completed problem
deletion. Destructor-body markers precede implicit member destruction.
`problem.delete.complete` is emitted only after `delete` returns, including
member and base-class destruction. Object addresses can be reused across trials.

CUDA/HIP runs also record the current device, device-query status codes, and
free/total device memory in bytes. These are device-wide measurements, not
per-process allocation totals. Failed queries must not be interpreted as zero
available memory. There are no additional MPI collectives or explicit device
synchronizations. CPU runs do not query the GPU even in a GPU-enabled build.

Set `OPENSN_MEMORY_ALLOCATOR=1` to write glibc `malloc_info` XML at each native
sample. These snapshots include all arenas and direct mmap allocations, but do
not identify allocation call sites, all thread caches, or GPU/runtime allocation
pools. Arena storage minus reported free storage is not an exact live-object
count. They can help distinguish retained free arena space from other growth.

`memory_study.py prepare` generates the standard 64-group, 448-direction CBC
input with function-scoped trials, 17 repetitions and 16 WGS iterations by
default. CPU inputs default to `allow_cycles=True`. GPU inputs default to
`allow_cycles=False`. Use `--no-allow-cycles` for comparison with acyclic scaling
runs. Angular flux saving is disabled. The mesh and cross
sections remain shared between trials. Python markers record trial numbers,
process and node memory, and cgroup membership before and after each trial.
No forced garbage collection is performed. Memory records include NUMA mappings
and scheduler CPU/wait times to help distinguish placement and contention from
retained memory.

`memory_study.py run --case CASE --binary EXE -- LAUNCHER ...` records the
launch arguments, relevant environment, and SHA-256 hashes, then launches OpenSn
from the input directory. It enables native and allocator tracing and refuses to
reuse a case with an existing launch record. Standard output, standard error, and
the launcher exit code are kept in the case directory. `SUCCESS` requires every
trial to finish and report a sweep timing. This checks completion, not convergence
or numerical accuracy. It does not infer success from a timeout or suppress
nonzero launcher exit codes. Use `run --trim` only for
the separately labeled trimming control.

Use separate cases for each `run --mode`:

- `baseline` disables memory tracing and Caliper reports.
- `memory` enables lifetime and allocator tracing. This is the default.
- `caliper` collects region and MPI timings in `caliper.txt` without memory tracing.
- `rocprof` uses `rocprofv3` to collect rank 0 HIP runtime, kernel, memory-copy,
  and memory-allocation traces. It requires a HIP build and `rocprofv3` on PATH.
  Other ranks execute without the profiler. Version, affinity and visible-device
  settings are recorded with the traces. Missing traces cause the run to fail.

Use three trials for short Caliper and rocprof captures. Their instrumented
timings are not baseline measurements. With Flux, pass `-o exit-timeout=none`
to `flux run` for rocprof so that untraced ranks exiting first do not interrupt
trace-file writing. Keep the allocation walltime and `exit-on-error` enabled.
Memory-specific options require `--mode memory`. All modes clear inherited
Caliper and memory-tracing settings before enabling their own configuration.

Start with `OPENSN_MEMORY_TRIM` unset. An optional, separate control with
`OPENSN_MEMORY_TRIM=1` calls glibc `malloc_trim(0)` after the post-trial marker
and records another marker. A reduction in RSS then demonstrates reclaimable
allocator storage, not the absence of leaks. Never combine control and default
measurements as performance results.

`memory_study.py build` creates a separate build using compiler, MPI, Python,
backend, architecture, and dependency hints from an existing CMake cache, with
`CMAKE_BUILD_TYPE=Native`. Source
the matching environment first. It does not install Python packages or rebuild
dependencies, and records its configure command and the reused cache. It refuses
to overwrite an existing build. Preserve both caches, module lists, binary/input
checksums, launch commands, allocation records, stdout, stderr, and all traces.

A timeout or OOM can leave the final sample incomplete. Flushes reduce lost log
output but do not guarantee a final sample after SIGKILL. Missing destructor
markers after an externally killed process are not proof of an ownership leak.
Instrumented runs are diagnostics and must not be used as scaling-performance samples.
