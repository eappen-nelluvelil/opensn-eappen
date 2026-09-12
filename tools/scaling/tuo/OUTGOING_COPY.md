# Outgoing-copy comparison on Tuo

Use `run_outgoing_copy_profile.zsh` from a clean, revision-specific worktree of
`cbc-and-cbcd-with-minimally-sized-fluds-profiling-2`. The comparison baseline is
`6bffcd8b43421fb8813a8748f18afa69e8e48f61`. The driver does not fetch or push Git
branches and does not modify existing source worktrees or builds.

```zsh
export OPENSN_TUO_BANK=cbronze
OUTGOING_RUNNER=$SOURCE/tools/scaling/tuo/run_outgoing_copy_profile.zsh
OUTGOING_LABEL=cbcd-outgoing-$(git -C "$SOURCE" rev-parse --short=9 HEAD)-pdebug-$(date -u +%Y%m%dT%H%M%SZ)
zsh "$OUTGOING_RUNNER" run "$OUTGOING_LABEL"
```

The driver first requests one pdebug node for a Native HIP build. It reuses the
compiled dependencies under
`/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies/builds/gfx942-minfluds-profiling-6386bec5e`.
Set `OPENSN_TUO_REUSE_ROOT` before running if these dependencies are elsewhere.
That directory must contain the dependency `env.zsh` and `deps` directory, not
only an OpenSn build that itself reused another dependency installation.

The new executable, Python environment, and MPI header overlay are separate,
under `builds/outgoing-copy-<revision>`. Existing dependency libraries are not
rebuilt or overwritten. A source-revision marker is checked before launching.

Each profile requests an eight-node pdebug allocation for 60 minutes. Inside
that allocation, the strong and weak problems run at 1, 2, 4, and 8 nodes, with
four ranks per node and one GPU per rank. The thread budget is 21 per rank.
Inputs retain single-angle aggregation, 64 groups, ten solver iterations,
`save_angular_flux=False`, and the existing divisor-39 strong-scaling problem.

Profiles run in this order:

1. `baseline`, without the requested profiling instrumentation
2. `cbcd-metrics`, with per-rank sweep, angleset, and histogram records
3. `caliper`, with runtime-region measurements
4. `caliper-mpi`, with runtime regions and MPI measurements
5. `pmpi`, using Caliper's MPI report
6. `rocprof`, using rocprofv3 HIP runtime, kernel, and memory traces

ROCm tracing selects rank zero by default. Set `OPENSN_ROCPROF_RANKS=all` before
starting a new campaign to trace every rank, at substantially greater cost.
This does not require the optional Caliper ROCm service. rocprofv3 must be
available in the compute-node environment. A missing profiler is reported as
a failure, not replaced by an uninstrumented run.

Run failures are retained, the other profiles are attempted, and collection
runs afterward. The driver exits nonzero if any profile or collection failed.
An interrupt terminates the driver instead of advancing to the next profile.
The allocation time limit covers the entire sequence for one profile, not each
individual problem.

```zsh
zsh "$OUTGOING_RUNNER" status "$OUTGOING_LABEL"
zsh "$OUTGOING_RUNNER" resume "$OUTGOING_LABEL"
zsh "$OUTGOING_RUNNER" collect "$OUTGOING_LABEL"
```

`status` lists the user's Flux jobs and the campaign result path. `resume`
retains the existing prepared jobs. The underlying helper resumes by node:
if either strong or weak is incomplete at a node count, it reruns that pair
into new run directories. To resume only one profile:

```zsh
zsh "$OUTGOING_RUNNER" resume "$OUTGOING_LABEL" rocprof
```

Results are under
`/p/lustre5/$USER/opensn-results/<label>-profile/resource-aware`.
An existing result root is rejected by `run`. Use `resume` or a new label.
Compare uninstrumented baseline sweep times separately from instrumented runs.
This campaign measures the optimization on Tuo. Local CUDA tests cannot
establish a speedup on Tuo or guarantee monotonic scaling.

The driver can be checked locally without a scheduler:

```sh
python3 tools/scaling/tuo/test_outgoing_copy_profile.py
```
