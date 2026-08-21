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

## 3. Interactive 1/2/4-node policy comparison

Do not set a fixed worker count for this comparison:

```zsh
unset OPENSN_CBCD_NUM_WORKERS
export OPENSN_TUO_INTERACTIVE_ITERATIONS=10

zsh "$HELPER" prepare-interactive
zsh "$HELPER" run-interactive
zsh "$HELPER" collect-interactive
```

`prepare-interactive` creates separate hardware and resource-aware studies that
use the same strong-scaling mesh. `run-interactive` obtains one exclusive
four-node allocation and runs:

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

The helper prepares and submits both policies. All strong cases use
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

To use a smaller first campaign, set `OPENSN_TUO_NODES` before
`prepare-batch`, for example `1,2,4,8,16`. Use a new `OPENSN_TUO_LABEL` or
`OPENSN_TUO_BATCH_ROOT` when changing nodes, iteration count, repetitions, or
the executable; prepared study directories are intentionally not rewritten.

## 5. Profiling jobs

Source the new environment and prepare a separate profiling directory. This
keeps profiler overhead out of the scaling measurements:

```zsh
source "$OPENSN_TUO_ROOT/env.zsh"

python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" prepare-profile \
  --binary "$OPENSN_TUO_BUILD/python/opensn" \
  --environment "$OPENSN_TUO_ROOT/env.zsh" \
  --output "$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile-resource-aware" \
  --mesh-dir "$OPENSN_TUO_MESH_DIR" \
  --label "$OPENSN_TUO_LABEL-profile-resource-aware" \
  --profile-nodes 1,2,4 \
  --worker-policy resource-aware \
  --queue pbatch \
  --bank "$OPENSN_TUO_BANK" \
  --no-save-angular-flux

zsh "$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile-resource-aware/submit.zsh"

python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" collect-profile \
  --study "$OPENSN_TUO_RESULTS/$OPENSN_TUO_LABEL-profile-resource-aware"
```

Use the uninstrumented scaling studies for performance conclusions. Caliper,
PMPI, rocprofv3, HPCToolkit, and OmniPerf jobs answer narrower attribution
questions and should be submitted only as needed.

Current Tuo queue limits and launch recommendations should be checked before a
large campaign:

- [Tuolumne platform](https://hpc.llnl.gov/hardware/compute-platforms/tuolumne)
- [Flux and MPI](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/running-jobs-flux-and-mpi)
- [El Capitan systems GPU programming](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/using-el-capitan-systems-gpu-programming)
