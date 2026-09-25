# Host CBC study harness

`host_cbc_study.py` compares pinned CPU implementations without modifying their
baseline sources. A configuration supplies the source worktree, an existing
CPU dependency build, mesh paths, MPI launcher argument list, node counts,
ranks per node, threads, iteration limit, modes, and number of trials.
Machine paths and scheduler/account choices belong in an external site launcher,
not in this repository.

## Execution and builds

Each trial is a **new MPI application** with one problem and one steady-state
solver. Trials execute sequentially, ordered by trial number then mode:
baseline, Caliper/MPI, PMPI. Thus early allocation expiry can still leave
examples of each mode. The allocation never runs implementations concurrently.
The Python loop in the driver launches processes; it does not retain solver
objects between trials.

Two isolated Native builds reuse dependency/compiler/Python settings through
`study_build.py`; dependencies are not rebuilt or replaced:

- `native`: the exact supplied source revision, for baseline and PMPI runs.
- `profile`: a detached worktree of that same revision with five Caliper scope
  annotations, for Caliper/MPI runs. `profile-overlay.patch` records the entire
  source difference. It separates runtime construction, SPDS, FLUDS local alpha
  planning, and beta finalization. No MPI operations, dependencies, kernels,
  or numerical algorithms are changed.

Caliper/MPI collects an aggregated region/MPI report with inclusive timing,
counts and rank statistics. PMPI here means Caliper's **MPI report** on the
unmodified Native binary, not a separate custom interposition library. It gives
per-MPI-function counts and times but does not identify every message size or
network contention source. Profiling overhead must never be mixed into baseline
performance statistics. A process-wide MPI report includes the harness's two
metadata gathers as well as solver communication.

Baseline disables Caliper channels, allocator tuning and common injected
profilers. Both library and executable SHA-256 fingerprints are recorded.
Native host compilation must take place on the target compute architecture.
Caliper is already a required OpenSn dependency; no GPU profiler is needed.

## Problem and interpretation

The input uses 64 groups, 448 directions, single-angle aggregation, CBC on the
host, `allow_cycles=False`, and `save_angular_flux=False`. It applies unit incident
flux in group zero on xmin, with the other default vacuum boundaries. The
configured Richardson iteration limit is fixed, with a very small tolerance.
These are **fixed-work timing experiments**, not comparisons of converged
time-to-solution. Do not infer accuracy from the timing inputs or from agreement
between implementations.

The harness rejects early convergence, missing output, nonfinite metrics,
nonzero lagged unknowns, incomplete iteration histories and nonzero process exit.
Flux maxima in groups 0 and 63 and the residual history are compared against
baseline trial one of the same implementation/decomposition. The tolerances
check reproducibility of the same algebraic problem, not discretization error.
Cross-implementation and cross-decomposition comparisons require separate
inspection of those recorded signatures. Truly cyclic cell dependencies require
a separate cycle-enabled campaign; they are not silently enabled on failure.

Mesh files are hashed and reused, not regenerated. In weak-scaling analysis,
use the recorded actual global unknown/cell counts: integer mesh divisors do
not give exactly constant local work. Keep the raw weak time ratio and any
work-normalized efficiency distinct. Keep all baseline samples; report their
spread, not merely the fastest run. Separate setup, solve and sweep costs.

## Provenance, memory and placement

Each attempt retains input, launch command, source/build fingerprints, stdout,
stderr, exit status, timing/flux/residual results, rank placement and memory
summaries. Two gathers in the input collect placement before mesh construction
and memory after the solve, writing one JSON file each rather than one file per
rank. They are outside solver/sweep timing but included in application elapsed
time and whole-process MPI counts. The environment requires ABI-compatible
`mpi4py`; the site build preflight must check it alongside MPI and Python.

`memory.json` records Linux `ru_maxrss` in KiB for every rank. These are process
high-water marks, **not simultaneous node peaks**. Summing them is not a valid
measurement of peak node usage. The site launcher should also retain Slurm job
and step accounting, node hostnames, CPU/NUMA topology, module/compiler/MPI
versions and allocation stderr for OOM/timeout diagnosis. A failed process may
not reach the final memory gather; lack of that file is not proof of an OOM.
Fresh processes prevent cross-trial retention, not single-trial memory exhaustion.

## Layout and resumption

```
manifest.json
inputs/{strong,weak}-N.py
inputs/xs_168g.xs
profile-overlay.patch
builds/{native,profile}/{configure.json,CMakeCache.txt,build.json}
results/KIND/nodes-N/MODE/trial-I/attempt-TIMESTAMP-UUID/
    input.py, launch.json, stdout.txt, stderr.txt, exit_code.txt
    placement.json, memory.json, result.json, SUCCESS or FAILED
    mpi-regions.txt or mpi.txt (profiling modes)
```

Sources and actual build products live outside the result root. Download the
result root, not the source/build workspace. Metadata files under `builds/`
are small and should be retained.

`prepare` refuses existing roots. `build` resumes only recorded build plans.
`run` locks one scaling case, checks source/input/helper/binary fingerprints,
and skips only attempts whose completion evidence still validates. It retains
every failed/partial attempt and creates a new timestamp/UUID on retry. A hard
scheduler kill can leave an attempt without a terminal marker; it is incomplete.
Budget exhaustion returns 3; application/profile-validation failure returns
nonzero and stops the case. The site launcher must check active submissions
before resubmitting. There is no automatic requeue or resubmission.

## Validation

Run `python tools/developer/test_host_cbc_study.py` for harness checks. Run a small
two-rank case in all modes before submitting large jobs; this verifies the input
API, MPI/Python compatibility, generated annotations and Caliper report options
in the actual target environment. Inspect the scope-only overlay and verify
baseline/profile scalar-flux and residual signatures. This smoke test is not a
replacement for independent transport correctness regressions.

### Comparing host communication polling changes

Use a new campaign when changing source revisions; do not edit a completed
campaign's manifests or reuse its completion markers with a different binary.
Keep the original Native baseline samples, and compare like-for-like meshes,
rank counts, thread counts, iteration limits and build settings. A rebase onto
a new upstream revision is another changed variable; measure against the same
upstream base when isolating a communication optimization.

Host CBC can avoid normal receive probes once every incoming face is complete,
and can reserve per-cell send progress for cells that queued new messages.
Its angle-set entry/exit completion checks and delayed-data drain remain active.
Compare `MPI_Iprobe` and `MPI_Testsome` counts as well as Native sweep timings;
unchanged message contents do not imply identical probe counts or identical
communication/computation overlap on every MPI implementation. Caliper scope
and MPI-call overhead can amplify the apparent benefit of fewer polls, so
profile timings are not evidence of the baseline speedup.

The registered `communication_equilibrium.py` regressions independently check
a uniform reflecting absorber, fragmented/full faces, multiple groupsets,
cyclic partitions, repeated execution and balance. Run these and the existing
nonuniform cyclic/restart/balance regressions before performance comparisons.
Local shared-memory results do not establish multi-node fabric performance;
retain the target-machine baseline and profiling modes for that validation.
