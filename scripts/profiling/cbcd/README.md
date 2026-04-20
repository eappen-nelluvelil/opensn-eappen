# CBCD Profiling Harness

Scripts for profiling the CBCD (device CBC) sweep path using the Caliper
instrumentation already compiled into OpenSn. No code is recompiled by these
scripts — you must have a working `build/python/opensn` first.

## Binary & deps

- `opensn` binary: defaults to `<repo>/build/python/opensn`; override with
  `OPENSN_BIN`.
- `cali-query`: defaults to `/home/eappen/opensn-deps-latest-3/bin/cali-query`;
  override with `CALI_QUERY`.

All CBCD GPU regression tests are MPI with `--np 4`. mpirun must be invoked
from the directory containing the test script; the scripts handle that.

## Caliper regions instrumented for CBCD

Added in this branch on top of the pre-existing scopes:

Angle-set state machine (`cbcd_angle_set.cu`)
- `CBCD_AngleSet::AngleSetAdvance` *(existing)*
- `CBCD_AngleSet::TryAdvanceOneStep` *(existing)*
- `CBCD_AngleSet::TryInitialize` *(new)*
- `CBCD_AngleSet::RetireBatch` *(new)*
- `CBCD_AngleSet::ProcessIncoming` *(new)*
- `CBCD_AngleSet::LaunchBatch` *(new)*
- `CBCD_AngleSet::FlushBatch` *(new)*
- `CBCD_AngleSet::Finalize`, `DelayedPhaseQueue`, `ProcessDelayedIncoming`,
  `FinalizeCompletion` *(new)*

Sweep chunk (`cbcd_sweep_chunk.cu`)
- `CBCDSweepChunk::Sweep` *(existing)*
- `CBCDSweepChunk::Sweep::ArgsRefresh` *(new — per-launch kernel-arg rebuild)*
- `CBCDSweepChunk::Sweep::KernelLaunch` *(new)*

Communicator (`cbcd_async_comm.cu`)
- `CommThreadLoop`, `SerializeAndSend`, `ProbeAndReceive` *(existing)*
- `PollInFlightSends` *(new)*

FLUDS (`cbcd_fluds.cu`)
- `CopyIncomingBoundaryPsiToDevice` *(new)*
- `CopyOutgoingPsiBackToHost` *(new)*
- `CopyDelayedOutgoingPsiBackToHost` *(new)*
- `CopySavedPsiFromDevice` *(new)*
- `CopySavedPsiToDestinationPsi` *(new)*

The per-entry `ScatterReceivedFaceData` / `ScatterDelayedReceivedFaceData`
functions are intentionally *not* individually instrumented — they run inside
tight per-entry loops and Caliper scope overhead would distort their cost.
Their aggregate time shows up inside `ProcessIncoming` /
`ProcessDelayedIncoming`.

## Scripts

### `run_runtime_report.sh` — cheap, MPI-aggregated hierarchical report

```
scripts/profiling/cbcd/run_runtime_report.sh \
  test/python/modules/linear_boltzmann_solvers/transport_steady/transport_3d_4_cycles_1_cbc_gpu.py
```

Runs with Caliper `runtime-report(calc.inclusive=true,region.count)` and prints
a tree of regions with aggregated min/avg/max inclusive time across the four
ranks. Output file defaults to
`runtime_report_<test_stem>.txt` next to the test.

Override the Caliper config at the env level:

```
CALI_CFG='runtime-report(calc.inclusive=true,profile.mpi)' \
  scripts/profiling/cbcd/run_runtime_report.sh <test>
```

### `run_trace.sh` — per-rank `.cali` binary trace

```
scripts/profiling/cbcd/run_trace.sh \
  test/python/modules/linear_boltzmann_solvers/transport_steady/transport_3d_4_cycles_1_cbc_gpu.py
```

Writes `<out_dir>/trace.cali` (Caliper `spot` config). Use
`summarize_cali.sh <out_dir>` to render.

### `summarize_cali.sh` — post-hoc aggregation

```
scripts/profiling/cbcd/summarize_cali.sh <out_dir_or_cali_file>
```

Produces:
1. A region tree ordered by aggregated time.
2. A flat top-40 table by exclusive time.

### `run_matrix.sh` — full CBCD profiling sweep

Drives `run_runtime_report.sh` across:
- `transport_3d_1b_ortho_cbc_gpu.py` (non-cyclic baseline, structured)
- `transport_3d_2_unstructured_cbc_gpu.py` (non-cyclic baseline, unstructured)
- `transport_3d_4_cycles_1_cbc_gpu.py` (cyclic, delayed-flux path exercised)

The 5-cycle GPU tests are omitted by default (pre-existing max-DOF-per-cell
limitation, per the branch summary). Uncomment them in `run_matrix.sh` if you
want to confirm the failure mode.

## Recommended profiling procedure

For an honest first-pass comparison of where CBCD time goes:

1. **Baseline vs. cyclic**: run `run_matrix.sh`. Compare the `CBCD_*` region
   tree between a non-cyclic test and the 4-cycle test. Anything that grows
   disproportionately with the delayed path is a candidate for optimization.

2. **Sweep-chunk attribution**: check the ratio of
   `CBCDSweepChunk::Sweep::ArgsRefresh` to `::KernelLaunch`. ArgsRefresh is the
   host-side `gpu_kernel::Arguments<>` rebuild per launch; if it's an
   appreciable fraction of Sweep, the refresh may be worth caching more
   aggressively.

3. **Communicator cost**: compare `CommThreadLoop` time against
   `SerializeAndSend` + `ProbeAndReceive` + `PollInFlightSends`. The
   difference is pure spinning/yielding (cost of the lock-free poll loop).

4. **Host<->device copy cost**: `CBCD_FLUDS::CopySavedPsiToDestinationPsi`
   and `CopyOutgoingPsiBackToHost` are the main host-side memcpy regions.
   Large values there point at zero-copy / layout opportunities.

5. **Scheduler wait**: `ScheduleAlgoAsyncFIFO` minus the sum of the above is
   time spent in `yield()` loops — high values suggest under-supply of ready
   work (dependency bubble) rather than inefficient per-step code.

Caliper overhead per scope is on the order of tens of nanoseconds, so scopes
that enter millions of times per sweep *will* perturb timings. The scopes
added here are all at angle-set-step or sweep-chunk granularity, not per-cell
or per-face, so the perturbation is expected to be small — but always
cross-check by running the same case without Caliper (`mpirun --np 4 opensn -i
<test>`) and comparing wall time.
