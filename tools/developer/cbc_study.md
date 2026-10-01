# Paired CBC scaling studies

`cbc_cluster_study.py` prepares isolated source/build/result trees, builds on a
one-node debug allocation, and launches host CBC on Dane and Tuolumne and device
CBC on Tuolumne. Python 3.11 or newer, an existing compatible dependency build,
and an existing mesh manifest are required. No dependencies or meshes are
regenerated. Site paths belong in an untracked configuration, not this repository.

## Configuration and commands

Provide these JSON fields:

| Field | Meaning |
| --- | --- |
| `site` | `dane` or `tuolumne` |
| `repo` | Existing Git repository containing both study revisions |
| `trunk_revision`, `cycles_revision`, `compact_revision` | Commits or refs to pin at preparation |
| `environment` | Existing ABI-compatible `.sh` or `.zsh` environment |
| `work_parent`, `results_parent` | Shared cluster scratch locations |
| `host_reuse_build` | Dependency configuration in an existing build directory |
| `device_reuse_build` | Tuolumne HIP dependency build directory |
| `host_mesh_manifest`, `device_mesh_manifest` | Existing manifests with `meshes.strong` and `meshes.weak` maps keyed by node count |
| `smoke_mesh` | Existing small mesh used before permitting study submissions |
| `account` | Allocation account |
| `pdebug_nodes`, `pbatch_nodes` | Node-count lists, no more than 256 |
| `pdebug_nodes_by_label`, `pbatch_nodes_by_label` | Optional node-list overrides keyed by label |
| `trials` | Trials per profiling mode and selection; default three |
| `walltime_seconds` | Allocation duration; default 3600 |
| `host_ranks_per_node` | Default 64 on Dane, 84 on Tuolumne |
| `build_jobs` | Build concurrency; default 16 |
| `labels` | Optional subset of `trunk-host`, `cycles-host`, `compact-host`, and Tuolumne-only `compact-device` |
| `modes` | Optional modes for every label; use `["baseline"]` for timing-only studies |
| `device_selections` | Optional device cases: `reference`, `fusion-off`, or `fusion-on`; default fusion off/on |

Source the selected cluster environment before invoking the driver. It records
the Python executable and uses that environment again for every allocation.
Run `prepare --config CONFIG.json`; retain the printed settings filename.

```sh
python tools/developer/cbc_cluster_study.py build trunk-host --settings SETTINGS.json
python tools/developer/cbc_cluster_study.py build cycles-host --settings SETTINGS.json
python tools/developer/cbc_cluster_study.py build compact-host --settings SETTINGS.json
python tools/developer/cbc_cluster_study.py build compact-device --settings SETTINGS.json
python tools/developer/cbc_cluster_study.py run compact-host weak 4 --settings SETTINGS.json
python tools/developer/cbc_cluster_study.py submit --settings SETTINGS.json
python tools/developer/cbc_cluster_study.py status --settings SETTINGS.json
```

Build only labels prepared for that cluster. `run` requests an interactive
pdebug allocation; `submit` submits the prepared pbatch node counts for each
label and scaling kind. Dane's trunk/cycles-host strong-one-node cases are skipped
because of the known memory-capacity limitation; its weak-one-node case remains.
Re-running an interactive case skips validated completed trials and preserves
failed attempts. Batch submission records prevent accidental duplicate submits;
a failed submitted allocation requires explicit operator review before resubmission.
Use `submit --kind weak` to submit only weak-scaling jobs, for example for a
separate larger-device-mesh campaign.

## Work and comparisons

Every trial starts a fresh MPI/OpenSn process lifetime: there is no Python loop
constructing multiple transport problems. Each input has 64 groups, 448 angles,
single-angle aggregation, 16 WGS iterations, and a 256 KiB message-size limit.
The Native baseline binary is unmodified. Only requested build variants are
created: a baseline-only campaign builds just Native, without an annotated
profiling worktree. When Caliper/MPI is requested, a separate Native profiling
worktree adds setup regions; its patch and build options are retained.
PMPI measurements use Caliper's `mpi-report` on the Native baseline binary,
not a distinct third-party tracer. Device cases additionally use rank-zero
`rocprofv3` HIP/kernel/copy/allocation tracing; they are not all-rank GPU profiles.

Trunk is a host CBC reference with `allow_cycles=False`; it has only one selection.
Within each label/kind/node allocation, run the selected modes for each trial.
For each mode/trial, run the selections consecutively; reverse their order on
alternate trials. With 17 trials and baseline only, host cases therefore make
34 fresh launches, alternating which lagging selection runs first. Compare
performance using baseline timings, not profiler timing.

Host selections are `allow_cycles=False` and `True`. The measured lagged-unknown
count establishes whether the flag changes the actual dependency graph. Zero
lagged unknowns means that run does not measure a lagging benefit. Nonzero lagging
changes the fixed-iteration numerical result; compare each selection with its
own numerical signature, and compare converged time-to-solution separately.

Device CBC does not implement lagged cycle fluxes. Set
`device_selections=["reference"]` for a single unlagged production device case.
This makes 17 launches in a 17-trial baseline-only allocation. For a source that
implements the experimental worker-fusion switch, paired selections instead
set `OPENSN_CBCD_FUSE_WORKER_LAUNCHES=0` and `1`. Device lagging requests are rejected.
Fusion uses the same cell sweep arithmetic and dependency graph, grouping ready
angle-set batches into a worker launch. Saved angular flux retains the original
launch path. The environment switch is experimental and defaults to off.

Each successful attempt records source/binary/input hashes, average sweep and
grind times, unknown and lagged-unknown counts, residual history, selected scalar
flux maxima, rank CPU/GPU placement, and peak rank RSS. Existing analytically
anchored and conservation regressions establish correctness; the study's flux
checks establish repeatability, not an independent proof of the discretization.
An attempt is complete only with a validated `SUCCESS` marker. Partial live copies
are not evidence that every trial finished.

Report medians and ranges across complete baseline trials. For weak scaling,
report both `T(1)/T(N)` and the recorded unknowns per rank; different mesh families,
rank counts, lagging policies, or GPU architectures must not be silently mixed.
For a fixed global strong problem, report `T(P0)/T(P)` and
`(P0/P) T(P0)/T(P)`. Missing reference points do not justify inventing a one-node
baseline. Separate sweep cost, setup, solver reductions, and end-to-end time.

## Placement

Dane uses 64 MPI ranks per node with one thread per rank, Slurm PMIx, mpibind,
exclusive nodes, and all node memory. Nested serial drivers do not inherit CPU
binding masks into MPI application steps.

Tuolumne uses 84 host ranks per node with one thread, or four device ranks with
21 worker threads each, SPX mode, exclusive nodes, and
`mpibind=greedy:0,smt:1` with CPU-optimized host mappings and GPU-optimized device
mappings. Retain the site's reserved-core restriction. Rank records
must show disjoint physical cores and distinct single GPUs for device cases.
These settings aim for local NUMA/device affinity; they do not guarantee optimal
inter-node topology or eliminate system noise. See the
[LLNL launch guide](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/running-jobs-flux-and-mpi)
and [mpibind options](https://github.com/LLNL/mpibind/blob/master/flux/options.md).
The Tuolumne build controller runs directly as the allocation's initial program,
not as a scheduled `flux run --exclusive` job. Otherwise it would reserve the
node while waiting for its own exclusive MPI smoke jobs to acquire that node.
Application steps retain the site's Spindle configuration. No extra serial
Flux job is submitted around the controller.
