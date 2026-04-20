# CBCD Profiling Backport for `cbf39c03`

This directory is a self-contained profiling bundle for the CBCD implementation
at commit:

- `cbf39c03eaf09b0c819099eff729a4f0eeb7efcf`

That commit is the head you identified for the
`cbc-and-cbcd-with-minimally-sized-fluds` branch state you want to profile.
It predates the CBC/CBCD cycle work added from `597ed6e1` onward, so it is the
right baseline if the goal is:

1. profile the original minimally-sized-FLUDS CBCD implementation,
2. compare it against the later branch state,
3. understand which later changes improved or hurt scheduler/communicator
   behavior.

This bundle gives you:

- profiling scripts,
- a minimal source patch to add the missing Caliper hooks and shutdown
  behavior,
- a repeatable workflow for generating runtime and launch-count reports,
- a path to trace-based analysis if the runtime reports are not sufficient.

## What problem this backport solves

At `cbf39c03`, OpenSn already had a fair amount of Caliper instrumentation, but
it was missing several details that turned out to be necessary for useful CBCD
profiling:

1. `framework/runtime.cc` did not flush/stop the `cali::ConfigManager` at
   shutdown.
   - That makes trace-producing services unreliable.
   - It also makes profiling behavior harder to reason about.

2. The CBCD scheduler path did not expose enough internal structure.
   - `CBCD_AngleSet::TryAdvanceOneStep` existed as one large region.
   - That is not enough to distinguish:
     - retiring completed work,
     - processing incoming face data,
     - launch decisions,
     - flush/copy stages,
     - final completion.

3. The CBCD sweep chunk did not split:
   - argument refresh
   - kernel launch
   into separate profiling regions.

4. The CBCD communicator did not mark `PollInFlightSends()`.
   - Without that, you cannot cleanly separate:
     - actual message serialization/receive work
     - in-flight send bookkeeping/poll churn

5. CBCD FLUDS copy points were not individually visible.
   - That makes it much harder to answer whether a run is
     kernel-bound, scheduler-bound, or copy-bound.

The patch in this directory adds only those profiling-specific pieces. It is
not intended to pull in any algorithmic changes from later commits.

## Files in this directory

- [cbcd_caliper_backport_cbf39c03.patch](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/cbcd_caliper_backport_cbf39c03.patch)
  - minimal source patch for `cbf39c03`
- [run_runtime_report.sh](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/run_runtime_report.sh)
  - hierarchical runtime report
- [run_launch_report.sh](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/run_launch_report.sh)
  - runtime report plus flat CBCD launch/count summary
- [run_trace.sh](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/run_trace.sh)
  - raw `.cali` event traces
- [summarize_cali.sh](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/summarize_cali.sh)
  - flat trace post-processing
- [run_matrix.sh](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/run_matrix.sh)
  - convenience driver for the tests that actually exist on `cbf39c03`

## What the patch changes

The backport patch modifies only five source files.

### 1. `framework/runtime.cc`

Adds:

- Caliper config error checking:
  - `if (cali_mgr.error()) throw ...`
- explicit shutdown handling:
  - `cali_mgr.flush();`
  - `cali_mgr.stop();`

This is required for reliable trace generation and makes profiling failures
fail fast instead of silently falling back into confusing behavior.

### 2. `cbcd_angle_set.cu`

Adds scopes for:

- `CBCD_AngleSet::TryInitialize`
- `CBCD_AngleSet::RetireBatch`
- `CBCD_AngleSet::ProcessIncoming`
- `CBCD_AngleSet::LaunchBatch`
- `CBCD_AngleSet::FlushBatch`
- `CBCD_AngleSet::FinalizeCompletion`

These are the core regions needed to answer the central CBCD performance
question:

- is time being spent in useful GPU work,
- or is CBCD burning time in host-side progress-engine churn?

### 3. `cbcd_async_comm.cu`

Adds:

- `CBCD_AsynchronousCommunicator::PollInFlightSends`

This separates:

- useful communicator work:
  - `SerializeAndSend`
  - `ProbeAndReceive`
- from send-completion polling overhead.

### 4. `cbcd_fluds.cu`

Adds scopes for:

- `CBCD_FLUDS::CopyIncomingBoundaryPsiToDevice`
- `CBCD_FLUDS::CopyOutgoingPsiBackToHost`
- `CBCD_FLUDS::CopySavedPsiFromDevice`
- `CBCD_FLUDS::CopySavedPsiToDestinationPsi`

These are the main host/device transfer points worth exposing on the original
branch.

### 5. `cbcd_sweep_chunk.cu`

Splits `CBCDSweepChunk::Sweep` into:

- `CBCDSweepChunk::Sweep::ArgsRefresh`
- `CBCDSweepChunk::Sweep::KernelLaunch`

This is important because later profiling on the newer branch showed that
`ArgsRefresh` was not the dominant issue. You need the same split on the old
branch if you want that comparison to be meaningful.

## How to use this on `cbc-and-cbcd-with-minimally-sized-fluds`

The intended workflow is:

1. check out the old branch,
2. reset to the exact commit,
3. apply the patch,
4. rebuild OpenSn,
5. run the scripts from this bundle.

### 1. Check out the target branch/commit

From the repo root:

```bash
git checkout cbc-and-cbcd-with-minimally-sized-fluds
git checkout cbf39c03eaf09b0c819099eff729a4f0eeb7efcf
```

If you want a working branch for profiling:

```bash
git checkout -b cbcd-profiling-cbf39c03
```

### 2. Apply the patch

From the repo root:

```bash
git apply cbcd_profiling_backport_cbf39c03/cbcd_caliper_backport_cbf39c03.patch
```

If you want Git to stage the result immediately:

```bash
git apply --index cbcd_profiling_backport_cbf39c03/cbcd_caliper_backport_cbf39c03.patch
```

### 3. Rebuild OpenSn

Make sure your dependency environment is loaded:

```bash
source /home/eappen/opensn-deps-latest-3/bin/set_opensn_env.sh
```

Then rebuild:

```bash
cd /home/eappen/opensn-eappen-prs/opensn-eappen/build
make -j6
```

If you keep separate build directories for branch comparisons, that is even
better. The scripts allow you to override the binary via `OPENSN_BIN`.

## Which tests to profile on `cbf39c03`

The later cyclic CBCD tests do not exist on that branch state. The tests that
do exist there are:

- `transport_1d_1_cbc_gpu.py`
- `transport_2d_2_unstructured_cbc_gpu.py`
- `transport_3d_1a_extruder_cbc_gpu.py`
- `transport_3d_1b_ortho_cbc_gpu.py`
- `transport_3d_2_unstructured_cbc_gpu.py`

That is why [run_matrix.sh](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/run_matrix.sh)
uses those tests instead of the later cyclic case mix.

If you later add or cherry-pick additional CBCD tests into that profiling
branch, extend `TESTS=(...)` in `run_matrix.sh`.

## Recommended profiling workflow

### A. Cheap baseline: runtime reports

Run from the repo root:

```bash
cbcd_profiling_backport_cbf39c03/run_runtime_report.sh \
  test/python/modules/linear_boltzmann_solvers/transport_steady/transport_3d_1b_ortho_cbc_gpu.py
```

This produces:

- `runtime_report_transport_3d_1b_ortho_cbc_gpu.txt`

Use this first when you want a broad timing decomposition.

### B. CBCD-specific count/launch analysis

Run from the repo root:

```bash
cbcd_profiling_backport_cbf39c03/run_launch_report.sh \
  test/python/modules/linear_boltzmann_solvers/transport_steady/transport_3d_1b_ortho_cbc_gpu.py
```

This produces two files:

- the full raw runtime report:
  - `launch_report_<test>.txt`
- a flat CBCD-only summary:
  - `launch_report_<test>_cbcd_regions.txt`

The flat summary is the main artifact for answering:

- how many batch-progress steps occur,
- how many actual kernel launches occur,
- whether host-side orchestration dominates useful GPU work.

### C. Full old-branch matrix

Run from the repo root:

```bash
cbcd_profiling_backport_cbf39c03/run_matrix.sh
```

This will generate both runtime and launch reports for all CBCD GPU tests that
exist on `cbf39c03`.

## How to analyze the generated results

### 1. Start with the flat launch summary

For each test, inspect:

- `CBCD_AngleSet::RetireBatch`
- `CBCD_AngleSet::ProcessIncoming`
- `CBCD_AngleSet::LaunchBatch`
- `CBCD_AngleSet::FlushBatch`
- `CBCDSweepChunk::Sweep::ArgsRefresh`
- `CBCDSweepChunk::Sweep::KernelLaunch`
- `CBCD_AsynchronousCommunicator::SerializeAndSend`
- `CBCD_AsynchronousCommunicator::ProbeAndReceive`
- `CBCD_AsynchronousCommunicator::PollInFlightSends`

The first ratio to compute is:

- host progress calls / kernel launches

For example:

- `Calls(total)` for `RetireBatch`
- divided by `Calls(total)` for `KernelLaunch`

If that ratio is large, CBCD is spending many host-side progress cycles per
actual launch.

That was exactly the pattern on the later branch: CBCD was host-overhead bound,
not kernel-bound.

### 2. Use runtime reports for macro structure

The runtime report is useful for seeing:

- how much total time sits in `TryAdvanceOneStep`
- how much total time sits in `CommThreadLoop`
- how big `BuildRuntime`, `InitializeSweepDataStructures`, and solver setup are

That helps separate:

- startup cost
- steady sweep cost
- progress-engine cost

### 3. Compare old vs new branch at the same test scale

The comparison that matters is not absolute wall time across arbitrarily
different test inputs. It is:

- same test
- same number of MPI ranks
- same binary mode
- same `save_angular_flux` setting
- same quadrature / groups

Then compare:

- total `KernelLaunch` time
- total `ArgsRefresh` time
- total host lifecycle time:
  - `RetireBatch`
  - `ProcessIncoming`
  - `LaunchBatch`
  - `FlushBatch`
- communicator time:
  - `SerializeAndSend`
  - `ProbeAndReceive`
  - `PollInFlightSends`

That is the clean way to determine whether later branch changes improved:

- launch granularity,
- batching efficiency,
- communicator churn,
- host/device copy behavior.

### 4. Treat `CopySavedPsiToDestinationPsi` carefully

This matters only if the profiled test actually enables `save_angular_flux`.

If your test is intended to represent steady-state grind time and does not need
saved angular fluxes, that region should not drive optimization decisions.

This is important because save-psi overhead can dominate the report and obscure
the real steady-state bottlenecks.

## Trace workflow

If runtime reports and flat launch summaries are not enough, use raw traces.

### Generate traces

```bash
cbcd_profiling_backport_cbf39c03/run_trace.sh \
  test/python/modules/linear_boltzmann_solvers/transport_steady/transport_3d_1b_ortho_cbc_gpu.py
```

This produces a directory containing:

- `stdout.txt`
- one `.cali` file per MPI rank

### Summarize traces

```bash
cbcd_profiling_backport_cbf39c03/summarize_cali.sh <trace_dir>
```

This summary is intentionally flat, not hierarchical. That was a deliberate
choice:

- tree reconstruction from raw event traces was too brittle and too expensive
  on large CBCD traces
- a flat table is enough for region-count and per-region aggregate timing

### When traces are worth using

Use traces when you need something that runtime-report cannot provide cleanly:

- custom region filtering
- event-count sanity checks
- alternative aggregation logic
- comparison against future, more detailed Caliper configurations

## Extending this framework

There are several legitimate ways to extend the analysis.

### 1. Change the Caliper config

All scripts accept `CALI_CFG`.

Examples:

```bash
CALI_CFG='runtime-report(calc.inclusive=true,region.count=true,region.stats=true,profile.mpi)' \
  cbcd_profiling_backport_cbf39c03/run_launch_report.sh <test>
```

```bash
CALI_CFG='event-trace(outdir=/tmp/mytrace,time.inclusive=true)' \
  cbcd_profiling_backport_cbf39c03/run_trace.sh <test>
```

If you want MPI profiling details, Caliper’s runtime-report recipe supports:

- `profile.mpi`
- `mpi.message.count`
- `mpi.message.size`

Those are useful if the communicator becomes the dominant suspect.

### 2. Extend the flat launch summary

If you add more CBCD scopes later, update the regex in:

- [run_launch_report.sh](/home/eappen/opensn-eappen-prs/opensn-eappen/cbcd_profiling_backport_cbf39c03/run_launch_report.sh)

Specifically, update `is_target_region(region)`.

### 3. Add finer-grain instrumentation

If later analysis shows that one top-level region is still too broad, the next
reasonable instrumentation boundaries are:

- inside `CBCD_AngleSet::TryAdvanceOneStep()`
- inside communicator queue processing
- around specific FLUDS scatter or pack operations

But be disciplined. Do not add per-cell or per-face scopes casually. That
would perturb the runtime enough to damage the usefulness of the measurements.

### 4. Rebuild dependencies only if there is a concrete reason

OpenSn’s dependency build recipe already pulls Caliper `v2.13.0` with MPI
enabled:

- [tools/dependencies/CMakeLists.txt](/home/eappen/opensn-eappen-prs/opensn-eappen/tools/dependencies/CMakeLists.txt)

On this machine, the profiling issues were not caused by a weak Caliper build.
They were caused by:

- missing runtime flush/stop,
- insufficient CBCD scope granularity,
- script/query issues.

So rebuilding Caliper is not the first move. Do it only if you have a specific
feature gap:

- missing service,
- broken formatter,
- or a known tool bug you want to fix.

## Practical advice

Use this sequence:

1. apply the patch
2. rebuild OpenSn
3. run `run_matrix.sh`
4. inspect the `*_cbcd_regions.txt` files first
5. use runtime reports for context
6. use traces only if the flat launch summaries still leave ambiguity

That gives the highest signal for the least time spent.
