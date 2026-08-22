# Tuolumne CBCD V2 studies

This directory provides a short path from a clean Tuo checkout to interactive
and batch CBCD V2 studies. It uses four MPI ranks per node, one MI300A per rank,
the default native SPX mode, and Flux/mpibind placement. The two policy studies differ only
in CBCD thread allocation:

- `hardware` uses the direct `std::thread::hardware_concurrency()` behavior.
- `resource-aware` bounds workers using the CPU resources assigned to each rank
  and reserves a core for its communicator.

The helper prepares and runs both policies so the binary, mesh, solver settings,
rank count, and allocation are otherwise the same.

## 1. Select the checkout and paths

On a Tuo login node, update the checkout to
`cbc-cbcd-minimally-sized-fluds-update-3`, then set the paths once in the shell
used for the study:

```zsh
STUDY_ROOT=/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies

export OPENSN_SOURCE=$STUDY_ROOT/source-update-3
export OPENSN_TUO_ROOT=$STUDY_ROOT/builds/gfx942-update-3
export OPENSN_TUO_BUILD=$OPENSN_TUO_ROOT/build-opensn
export OPENSN_TUO_MESH_DIR=$STUDY_ROOT/builds/gfx942/mesh-cache
export OPENSN_TUO_RESULTS=$STUDY_ROOT/results
export OPENSN_TUO_LABEL=update-3
export OPENSN_TUO_BANK=YOUR_LC_BANK

HELPER=$OPENSN_SOURCE/tools/scaling/tuo/interactive_cbcd.zsh
zsh "$HELPER" paths
```

The default mesh path is the cache created by the earlier Tuo framework. Study
preparation never regenerates meshes. Interactive strong scaling needs only
`cube-d39.msh`. A full 1--256-node strong/weak campaign uses these files:

```text
cube-d6.msh   cube-d8.msh   cube-d10.msh  cube-d12.msh
cube-d15.msh  cube-d19.msh  cube-d25.msh  cube-d31.msh
cube-d39.msh
```

Check the existing files before starting:

```zsh
for divisor in 6 8 10 12 15 19 25 31 39; do
  [[ -s $OPENSN_TUO_MESH_DIR/cube-d$divisor.msh ]] || \
    print -u2 "missing cube-d$divisor.msh"
done
```

## 2. Build a fresh Tuo stack

Choose a new `OPENSN_TUO_ROOT`; do not point it at an older dependency prefix or
virtual environment. Then run:

```zsh
zsh "$HELPER" build
```

The helper obtains one exclusive `pdebug` node and builds a new Python virtual
environment, OpenSn dependencies, and the HIP-enabled OpenSn executable. The
ambient Python and dependency paths are cleared before configuration. Caliper
is built with MPI support and its ROCm activity service by default.

If the one-hour interactive allocation ends during the dependency build, run
the same command again. It resumes the selected build directories. The
lower-level resumable commands are:

```zsh
zsh "$OPENSN_SOURCE/tools/scaling/tuo/bootstrap.zsh" configure-deps
zsh "$OPENSN_SOURCE/tools/scaling/tuo/bootstrap.zsh" build-deps
zsh "$OPENSN_SOURCE/tools/scaling/tuo/bootstrap.zsh" build-opensn
```

Run those lower-level commands on a compute node with `OPENSN_SOURCE`,
`OPENSN_TUO_ROOT`, and `OPENSN_TUO_BUILD` exported as above.

After pulling source changes, retain the fresh dependency stack and rebuild
only OpenSn in a new one-node allocation:

```zsh
zsh "$HELPER" rebuild
```

Unlike `build`, `rebuild` never treats an existing executable as proof that the
new source has already been compiled.

## One-command validation workflow

After exporting the checkout, build, mesh, results, and bank paths from section 1,
the executable wrapper can rebuild the selected checkout and perform the complete
resource-aware smoke/profile pass:

```zsh
RUNNER=$OPENSN_SOURCE/tools/scaling/tuo/run_cbcd_validation.zsh

export OPENSN_TUO_INTERACTIVE_ITERATIONS=2
export OPENSN_TUO_PROFILE_ITERATIONS=3
export OPENSN_TUO_PROFILE_NODES=1,2,4
export OPENSN_TUO_PROFILES=baseline,caliper-mpi,pmpi

"$RUNNER" smoke-profile update-3-topology-flow
```

The wrapper runs only the resource-aware policy. It rebuilds OpenSn, runs and
collects the 1/2/4-node strong-scaling smoke study, then obtains a separate
`pdebug` allocation for each selected profile and collects the profile inventory.
It prints timestamps, the generated job and result paths, each completed case's
WGS/sweep metrics, and the final study locations. While a case is running it
prints a heartbeat or the newest WGS/communication progress line once per minute.
Set `OPENSN_TUO_PROGRESS_INTERVAL` to another number of seconds, or to zero to
disable the heartbeat. Existing meshes are reused.

Once that validation is satisfactory, submit the larger resource-aware strong/
weak campaign with the same executable and a distinct label:

```zsh
export OPENSN_TUO_NODES=1,2,4,8,16,32,64,128,256
export OPENSN_TUO_REPETITIONS=3
export OPENSN_TUO_MAX_ITERATIONS=10
export OPENSN_TUO_BATCH_TIME_LIMIT=4h

"$RUNNER" submit-scaling update-3-topology-flow-scaling
```

After the selected jobs finish, the same wrapper collects whichever result
types exist for that label:

```zsh
"$RUNNER" collect update-3-topology-flow-scaling
```

The lower-level helper commands remain available for targeted reruns and custom
study configurations.

## 3. Interactive 1/2/4-node policy comparison

Do not set a fixed worker count for this comparison:

```zsh
unset OPENSN_CBCD_NUM_WORKERS
export OPENSN_TUO_INTERACTIVE_ITERATIONS=10

zsh "$HELPER" run-interactive
zsh "$HELPER" collect-interactive
```

To study only the resource-aware implementation, use the optional policy
argument. This runs 1, 2, and 4 nodes inside one four-node allocation and avoids
creating or running hardware-policy jobs:

```zsh
unset OPENSN_CBCD_NUM_WORKERS
export OPENSN_TUO_INTERACTIVE_ITERATIONS=10

zsh "$HELPER" run-interactive resource-aware
zsh "$HELPER" collect-interactive resource-aware
```

Repeat `run-interactive resource-aware` at least three times before evaluating
a small scaling reversal. Every invocation appends a new independent run and
the collector reports the median, MAD, and IQR across all successful runs.

`prepare-interactive` creates separate hardware and resource-aware studies that
use the same strong-scaling mesh. Rerunning it refreshes generated inputs and
job scripts while preserving existing result directories. `run-interactive`
obtains one exclusive four-node allocation and runs:

```text
hardware 1 node
resource-aware 1 node
resource-aware 2 nodes
hardware 2 nodes
hardware 4 nodes
resource-aware 4 nodes
```

Alternating the first policy reduces a consistent warm/cold-order bias. Ten WGS
iterations match the earlier scaling protocol; use a smaller value only for a
smoke test, and select a new `OPENSN_TUO_LABEL` before preparing that different
configuration. Repeating `run-interactive` adds another set of measurements
rather than replacing completed output.

Run one selected case when diagnosing or retrying a point:

```zsh
zsh "$HELPER" run hardware 2
zsh "$HELPER" run resource-aware 2
```

Each run prints the scheduler's selected worker count. On the established Tuo
placement, the expected comparison is the hardware counter's full value versus
the affinity-bounded resource-aware value. Treat the printed value as the
authoritative result for a particular allocation.

The collector writes `results.csv`, `summary.csv`, `summary.md`, scaling plots,
and a hardware-to-resource-aware comparison under the interactive results
directory. It checks clean completion, the final WGS state, scalar-flux
observables, and the selected worker count. Iteration counts may differ between
node counts because CBCD cycles and lagged flux change with decomposition; the
two policies must agree at the same node count.

Within repeated measurements of one point, discrete properties (unknown and
lagged-unknown counts, WGS status and iterations, and worker count) must match
exactly. Scalar-flux maxima are summarized by their median, minimum, maximum,
and exact binary64 ULP span. This exposes scheduling-scale floating variation
without embedding an empirical acceptance tolerance in the collector.

## 4. Larger pbatch strong/weak studies

The defaults prepare strong and weak cases at 1, 2, 4, 8, 16, 32, 64, 128, and
256 nodes, with three repetitions per allocation and ten WGS iterations:

```zsh
export OPENSN_TUO_NODES=1,2,4,8,16,32,64,128,256
export OPENSN_TUO_REPETITIONS=3
export OPENSN_TUO_MAX_ITERATIONS=10
export OPENSN_TUO_BATCH_TIME_LIMIT=4h

zsh "$HELPER" prepare-batch
zsh "$HELPER" submit-batch
```

Without a policy argument the helper prepares and submits both policies. To
run only the resource-aware campaign requested for CBCD V2, use:

```zsh
zsh "$HELPER" submit-batch resource-aware
```

All strong cases use
`cube-d39.msh`; the weak cases use divisors 6, 8, 10, 12, 15, 19, 25, 31, and
39 in increasing node order. No mesh generation is performed. Each invocation
of `submit-batch` submits the complete selected campaign, so do not repeat it
unless another full set of measurements is intentional; submit an individual
generated job with `flux batch` when retrying only that point.

Monitor the jobs with:

```zsh
flux jobs -A
```

After both studies finish:

```zsh
zsh "$HELPER" collect-batch
```

For the resource-aware-only campaign, collect with:

```zsh
zsh "$HELPER" collect-batch resource-aware
```

To use a smaller first campaign, set `OPENSN_TUO_NODES` before
`prepare-batch`, for example `1,2,4,8,16`. Use a new `OPENSN_TUO_LABEL` or
`OPENSN_TUO_BATCH_ROOT` when changing nodes, iteration count, repetitions, or
the executable; prepared study directories are intentionally not rewritten.

## 5. Profiling jobs

The helper prepares a separate resource-aware profiling directory, keeping
profiler overhead out of the scaling measurements. By default it runs the d39
strong-scaling problem for ten iterations at 1, 2, and 4 nodes with:

- an uninstrumented baseline,
- a Caliper runtime region report,
- a Caliper region/MPI-function timing report, and
- a Caliper PMPI call report.

Prepare and submit all four profiles with:

```zsh
unset OPENSN_CBCD_NUM_WORKERS
export OPENSN_TUO_PROFILE_NODES=1,2,4
export OPENSN_TUO_PROFILE_DIVISOR=39
export OPENSN_TUO_PROFILE_ITERATIONS=10
export OPENSN_TUO_PROFILES=baseline,caliper,caliper-mpi,pmpi

zsh "$HELPER" submit-profile
```

Monitor with `flux jobs -A`. Once all jobs finish, collect their validated
inventory with:

```zsh
zsh "$HELPER" collect-profile
```

The combined report is named `mpi-regions.txt`; use it to compare MPI function
and OpenSn region timings at 2 versus 4 nodes. The profile deliberately omits
Caliper's message-request and communication-pattern tracking. Those services
add request bookkeeping to the communicator's progress loop, while OpenSn does
not define the Caliper communication regions needed by `comm.stats`. Exact CBCD
message, record, section, and byte counts are reported by the communicator
itself. The `pmpi` report is named `mpi.txt`. Use the uninstrumented baseline
and scaling studies for performance conclusions; profiler timings are
diagnostic only.

To run the same selected profiles through `pdebug` instead of submitting
`pbatch` jobs, use:

```zsh
zsh "$HELPER" run-profile-interactive
zsh "$HELPER" collect-profile
```

One allocation is requested per profile, sized to the largest node count selected
for that profile. Inside it, the generated 1-, 2-, and 4-node jobs run in order,
with progress and output paths printed after every case. A single profile can be
selected, for example `zsh "$HELPER" run-profile-interactive pmpi`.

If a node or allocation fails partway through the sequence, resume it with:

```zsh
zsh "$HELPER" resume-profile-interactive
zsh "$HELPER" collect-profile
```

The resume command keeps the complete manifest, skips only cases with a
validated successful run and zero exit code, and creates a new result directory
for every retry. It requests a fresh allocation per incomplete profile and
continues to later profiles if one allocation fails. Select one profile with,
for example, `zsh "$HELPER" resume-profile-interactive baseline`.

`caliper-rocm`, `rocprof`, and `hpctoolkit` also preserve the requested
four-ranks-per-node layout when explicitly selected. For example, a focused
2/4-node ROCm trace can be prepared in a separate result directory with:

```zsh
export OPENSN_TUO_PROFILE_ROOT=$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile-rocm
export OPENSN_TUO_PROFILE_NODES=2,4
export OPENSN_TUO_PROFILE_ITERATIONS=2
export OPENSN_TUO_PROFILES=rocprof

zsh "$HELPER" submit-profile
```

`omniperf` remains intentionally restricted to one node and one rank because
kernel replay is a microanalysis and does not preserve the production MPI
decomposition.

Current Tuo queue limits and launch recommendations should be checked before a
large campaign:

- [Tuolumne platform](https://hpc.llnl.gov/hardware/compute-platforms/tuolumne)
- [Flux and MPI](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/running-jobs-flux-and-mpi)
- [El Capitan systems GPU programming](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/using-el-capitan-systems-gpu-programming)
