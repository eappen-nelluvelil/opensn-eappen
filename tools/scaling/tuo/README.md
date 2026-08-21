# Tuolumne CBCD studies

This directory provides a revision-pinned workflow for CBCD V2 correctness,
strong/weak scaling, and profiling on Tuolumne. The supported production
geometry is four MPI ranks per node, one MI300A and 21 user CPU cores per rank,
native SPX mode, and Flux/mpibind placement over Tuo's 84 user cores. A node
has 96 physical CPU cores; three cores per socket are reserved for the OS,
leaving 84 to jobs, with two hardware threads per core. The binding preflight
requires all 21 user cores (rather than counting hardware threads) and a distinct
local GPU for every rank.

The framework never checks out a branch, sources personal state files, or
installs shell dotfiles. Give it an already checked-out, clean source tree. It
derives and records the exact 40-character Git SHA, hashes the executable and
all study assets, and refuses to run a binary whose build manifest does not
match.

## Build one exact revision

Run from a clean detached checkout or clean branch worktree. Dependency setup
needs network access to the LC package mirrors. Builds belong on a compute
node. `all` refuses an existing stack or build directory: it creates a new,
content-fingerprinted dependency prefix and a new Python venv.

```zsh
export OPENSN_SOURCE=/usr/workspace/$USER/opensn-gpu/source-update-2
export OPENSN_TUO_ROOT=/usr/workspace/$USER/opensn-gpu/tuo/gfx942
export OPENSN_TUO_BUILD=$OPENSN_TUO_ROOT/build-update-2
export OPENSN_TUO_REVISION=$(git -C "$OPENSN_SOURCE" rev-parse HEAD)
export OPENSN_TUO_STACK_ID=update-2-clean-1

flux alloc -N1 -q pdebug --exclusive --amd-gpumode=SPX -t 60m
zsh "$OPENSN_SOURCE/tools/scaling/tuo/bootstrap.zsh" all
```

`all` creates the Python environment, dependencies, OpenSn binary, and
`$OPENSN_TUO_BUILD/tuo-build-manifest.json`. To retry an interrupted build,
use `build-deps`, then `build-opensn`; to rebuild from scratch, select a new
`OPENSN_TUO_STACK_ID` and build path. The fingerprint covers the dependency
inputs, exact module set, exact Python package pins, stack ID, and Caliper
features. Gmsh 4.15.2 is pinned by default.

The dependency build forces Caliper 2.13's MPI support on (needed by PMPI and
cross-rank aggregation), forces NVTX/CUPTI off because Tuo is AMD, and requests
Caliper's rocProfiler service. The actual Caliper `WITH_*` cache entries and
installed feature macros are validated and stored in
`caliper-features.json`; the installed recipe inventory must contain
`runtime-report` and `mpi-report`, plus the exact `rocm-activity-report` recipe
when the ROCm backend is requested. The fresh
Boost 1.86 package configuration is selected explicitly. The launcher and its
complete resolved DSO closure—including `libopensn.so`—are hashed and checked
again during preparation and on every compute allocation. If the installed
ROCm cannot support that service, explicitly select
`OPENSN_TUO_CALIPER_GPU_BACKEND=NONE`; the MPI/PMPI services remain mandatory.

Use identical pinned inputs but fresh stack IDs and distinct build directories
for clean rebuilds. Build the proven
`cbc-and-cbcd-with-minimally-sized-fluds` revision as the primary CBCD V2
performance baseline. Trunk device CBC is a useful separate comparison, not a
substitute for that baseline.

To build an older clean source tree that predates these tools, invoke the
bootstrap from the update-2 tools checkout while setting `OPENSN_SOURCE` to
the old tree. The bootstrap records and requires both exact clean revisions,
and supports the old and current dependency-driver interfaces.

## Prepare an immutable production study

```zsh
BUILD_MANIFEST=$OPENSN_TUO_BUILD/tuo-build-manifest.json
ENVIRONMENT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment"])' \
  "$BUILD_MANIFEST")
source "$ENVIRONMENT"
python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" prepare \
  --source "$OPENSN_SOURCE" \
  --binary "$OPENSN_TUO_BUILD/python/opensn" \
  --environment "$ENVIRONMENT" \
  --output /path/to/results/update-2 \
  --mesh-cache /path/to/mesh-cache \
  --gmsh "$(command -v gmsh)" \
  --label update-2 \
  --nodes 1,2,4,8,16,32,64,128,256 \
  --kinds strong,weak \
  --repetitions 3 \
  --queue pbatch \
  --bank YOUR_LC_BANK \
  --worker-policy hardware \
  --no-save-angular-flux
```

Preparation is atomic. It stages the geometry, cross sections, generator,
runtime wrapper, build manifest, and generated inputs; records hashes and tool
versions; and uses a content-addressed mesh cache. The executable, environment,
dependency manifest, Caliper feature manifest, and OpenSn CMake cache are
revalidated before preparation, submission, collection, and execution.
Full-node allocations and launches use `--exclusive` without explicit `-c` or
`-g`, following current Tuo guidance. The preflight then proves each rank has
one distinct physical GPU and a disjoint 21-user-core CPU affinity.

`--worker-policy hardware` is the default and is required for the historical
CBCD V2 baseline unless deliberately overridden. Use
`--worker-policy resource-aware` for the affinity-aware candidate policy, and
optionally add `--cbcd-workers N` for a fixed positive override. Both values
are recorded in the study manifest and every attempt's metadata. Collection
also requires and records the scheduler's actual `workers=N` diagnostic.

Submit selected cases with the recorded, idempotent submitter:

```zsh
/path/to/results/update-2/submit.zsh --kinds strong --nodes 1,2,4
/path/to/results/update-2/submit.zsh --kinds strong,weak
```

Repeating the command skips cases already recorded in `submissions.jsonl`.
Use `--resubmit` explicitly for failed, incomplete, or invalid `SUCCESS`
attempts. Valid successful cases are never overwritten; every attempt is
stored under its Flux job ID. Collection validates every success-marked
attempt instead of silently choosing the newest one, and authenticates the
collected result files plus every contributing attempt artifact.

After all cases finish:

```zsh
python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" collect \
  --study /path/to/results/update-2 --require-monotonic

python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" compare \
  --baseline /path/to/results/proven-old-cbcd \
  --candidate /path/to/results/update-2 \
  --output /path/to/results/comparison \
  --max-slowdown 1.03 \
  --monotonic-tolerance 0.0 \
  --scalar-flux-rtol 1.0e-10 \
  --scalar-flux-atol 1.0e-12
```

Collection requires an exit-zero marker, clean `OpenSn finished execution`, a
valid final WGS record, and a valid rank/GPU/CPU binding map. It reports median,
MAD, and IQR. Each run must also report finite scalar-flux maxima for groups 0
and 63. Repetitions at the same `(kind, nodes)` point must reproduce those
maxima exactly; baseline/candidate maxima at that same point are checked with
the explicit `--scalar-flux-rtol` and `--scalar-flux-atol` thresholds.
Comparison rejects incompatible studies, numerical/iteration
mismatches, non-monotonic candidate strong scaling, and slowdowns beyond the
requested threshold. `--allow-incomplete` is diagnostic only; incomplete
collections cannot be used by `compare`.

WGS iteration counts are compared only across repetitions at an identical
decomposition and between baseline/candidate at the same `(kind, nodes)`.
They are intentionally not required to match across node counts: additional
CBCD cycles and lagged flux can legitimately increase iteration counts as the
decomposition changes.

`compare` refuses a non-hardware baseline by default. Policy experiments must
opt into `--allow-worker-policy-difference`; a deliberately non-hardware
baseline additionally requires `--allow-nonhardware-baseline`.

## Short 1/2/4-node check

The helper uses one four-node `pdebug` allocation for `run-all`, rather than
chaining allocations. Tuo's documented pdebug limit is 16 nodes per user
and one hour, interactive only; the helper deliberately stays at four nodes
and defaults to 60 minutes. Production pbatch jobs can request up to 256
nodes. The currently documented allocation flag is
`--amd-gpumode=SPX|TPX|CPX`; these studies explicitly select the native/default
SPX mode.

Interactive studies default to two WGS iterations so a paired 1/2/4 sequence
has margin under pdebug's one-hour limit. Override
`OPENSN_TUO_INTERACTIVE_ITERATIONS` only if both paired studies use the same
value. Production studies default to ten iterations.

```zsh
export OPENSN_SOURCE=/path/to/clean/update-2
export OPENSN_TUO_ROOT=/path/to/tuo/gfx942
export OPENSN_TUO_BUILD=$OPENSN_TUO_ROOT/build-update-2
export OPENSN_TUO_LABEL=update-2
export OPENSN_CBCD_WORKER_POLICY=hardware

HELPER=$OPENSN_SOURCE/tools/scaling/tuo/interactive_cbcd.zsh
zsh "$HELPER" build
zsh "$HELPER" prepare
zsh "$HELPER" run-all
zsh "$HELPER" summary
```

For a paired run, prepare a one-repetition strong study for the proven old CBCD
binary too, then run both studies in the same four-node allocation:

```zsh
export OPENSN_TUO_BASELINE_STUDY=/path/to/proven-old-interactive-study
zsh "$HELPER" paired-run-all
```

The order alternates by node count to reduce systematic first-run bias. For a
strong acceptance decision, repeat paired allocations and use the production
collector/comparator.

Policy A/B studies are the same workflow with two distinct study directories:

```zsh
export OPENSN_TUO_BASELINE_STUDY=/path/to/hardware-policy-study
export OPENSN_TUO_INTERACTIVE_STUDY=/path/to/resource-aware-policy-study
export OPENSN_TUO_ALLOW_POLICY_COMPARISON=1
zsh "$HELPER" paired-run-all
zsh "$HELPER" summary
```

The comparison still requires identical geometry, launch dimensions, solver
settings, requested repetition policy, per-case meshes, dependency recipe,
unknown counts, WGS status, same-point iteration count, residual, and
scalar-flux maxima. The
opt-in only removes worker policy/count from the compatibility fingerprint.

## Profiling

The default profile set is the uninstrumented baseline, Caliper runtime report,
and Caliper PMPI report at 1, 2, and 4 nodes. Heavy one-node profilers are opt-in.

```zsh
python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" prepare-profile \
  --source "$OPENSN_SOURCE" \
  --binary "$OPENSN_TUO_BUILD/python/opensn" \
  --environment "$ENVIRONMENT" \
  --output /path/to/results/update-2-profile \
  --mesh-cache /path/to/mesh-cache \
  --gmsh "$(command -v gmsh)" \
  --label update-2-profile \
  --profile-nodes 1,2,4 \
  --profiles baseline,caliper,pmpi,caliper-rocm,rocprof,hpctoolkit,omniperf \
  --bank YOUR_LC_BANK

/path/to/results/update-2-profile/submit.zsh --profiles baseline,caliper,pmpi
python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" collect-profile \
  --study /path/to/results/update-2-profile
```

`submit.zsh` is deliberately disabled for `pdebug` studies because that queue
is interactive-only. Run the generated job inside one `flux alloc` instead.
Submitting a scaling study with `--profiles`, or a profiling study with
`--kinds`, is rejected rather than silently broadening the request.

Profiler timings are never used for scaling conclusions.

The baseline, Caliper runtime-report, and Caliper PMPI profiles work at
multiple node counts. `caliper-rocm` is an opt-in one-node rocProfiler activity
report and requires a Caliper `WITH_ROCPROFILER=ON` build. rocprofv3 and
HPCToolkit launch rank-resolved profiling; Omniperf is intentionally
restricted to one rank/one GPU. Select only the profiles needed for a question.

Current Tuo limits and topology should be rechecked before a campaign:
[Tuolumne platform](https://hpc.llnl.gov/hardware/compute-platforms/tuolumne)
and [El Capitan systems GPU programming](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/using-el-capitan-systems-gpu-programming),
[Flux and mpibind](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/running-jobs-flux-and-mpi),
and [El Capitan systems pro tips](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/pro-tips).
