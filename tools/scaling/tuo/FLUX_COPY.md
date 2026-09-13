# Flux-copy comparisons on Tuo

Use `run_flux_copy_profile.zsh` from a clean, revision-specific worktree of
`cbc-and-cbcd-with-minimally-sized-fluds-profiling-2`. The comparison baseline is
`6bffcd8b43421fb8813a8748f18afa69e8e48f61`. The outgoing-copy-only comparison is
`1e69af813dd979dca68fd2ad5450fa11362d5303`. The driver does not fetch or push Git
branches and does not modify existing source worktrees or builds.

```zsh
export OPENSN_TUO_BANK=cbronze
COPY_RUNNER=$SOURCE/tools/scaling/tuo/run_flux_copy_profile.zsh
COPY_LABEL=cbcd-query-$(git -C "$SOURCE" rev-parse --short=9 HEAD)-pdebug-$(date -u +%Y%m%dT%H%M%SZ)
zsh "$COPY_RUNNER" run "$COPY_LABEL"
```

The driver first requests one pdebug node for a Native HIP build. It reuses the
compiled dependencies under
`/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies/builds/gfx942-minfluds-profiling-6386bec5e`.
Set `OPENSN_TUO_REUSE_ROOT` before running if these dependencies are elsewhere.
That directory must contain the dependency `env.zsh` and `deps` directory, not
only an OpenSn build that itself reused another dependency installation.

The new executable and MPI header overlay are separate, under
`builds/flux-copy-<revision>`. Python packages are reused from the latest campaign's
`builds/flux-copy-f60e0ecbd/venv`, without installation or upgrades. Override
`OPENSN_TUO_REUSE_VENV` if that environment was moved. A missing environment,
incompatible Python ABI, or failed package check stops the build rather than
silently installing packages. Keep the shared venv unchanged while jobs use it.
The peer-reuse campaign at `6fa8bee92` and the batched-publication campaign at
`b235155a9` used this same venv. The complete baseline triplets from `b235155a9`
are the immediate comparison for the completion-query experiment.
Existing dependency libraries are not rebuilt or overwritten. A source-revision
marker is checked before launching.

Each profile requests an eight-node pdebug allocation for 60 minutes. Inside
that allocation, the strong and weak problems run at 1, 2, 4, and 8 nodes, with
four ranks per node and one GPU per rank. The thread budget is 21 per rank.
Inputs retain single-angle aggregation, 64 groups, ten solver iterations,
`save_angular_flux=False`, and the existing divisor-39 strong-scaling problem.

Every case runs three consecutive times within that allocation. Each N-node
case is constrained to broker ranks 0 through N-1, so all three trials use the
same physical nodes. Each run records `nodes.txt`, a trial number, and an
allocation trial-group identifier. There are 144 measured runs across the six
profiles. The first trial is retained, not discarded as warm-up. Later trials
can benefit from warmed system caches and devices, but each starts a new OpenSn
process. Compare trial distributions as well as medians.

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

Only rocprof jobs disable Flux's first-task-exit timeout. This allows the traced
rank to finish writing its files after untraced ranks exit. The allocation
walltime and `exit-on-error` remain enabled. Earlier outgoing-copy results
remain unchanged and do not need to be rerun.

Run failures are retained, the other profiles are attempted, and collection
runs afterward. The driver exits nonzero if any profile or collection failed.
An interrupt terminates the driver instead of advancing to the next profile.
The allocation time limit covers the entire sequence for one profile, not each
individual problem.

```zsh
zsh "$COPY_RUNNER" status "$COPY_LABEL"
zsh "$COPY_RUNNER" resume "$COPY_LABEL"
zsh "$COPY_RUNNER" collect "$COPY_LABEL"
```

`status` lists the user's Flux jobs and the campaign result path. `resume`
retains the existing prepared jobs. A case is complete only after three
successes in one trial group. If either strong or weak is incomplete at a node
count, the helper repeats both three-trial sets in new directories. It does
not combine partial groups from different allocations. To resume one profile:

```zsh
zsh "$COPY_RUNNER" resume "$COPY_LABEL" rocprof
```

Results are under
`/p/lustre5/$USER/opensn-results/<label>-profile/resource-aware`.
An existing result root is rejected by `run`. Use `resume` or a new label.
Compare uninstrumented baseline sweep times separately from instrumented runs.
This campaign measures the optimization on Tuo. Local CUDA tests cannot
establish a speedup on Tuo or guarantee monotonic scaling.

The completion-query experiment retains the stream completion check in
`TryAdvanceOneStep` and removes the second query from batch retirement. The
same worker owns the angle set and submits no stream work between those points.
No event, polling delay, worker-count change, or receive-placement change is
introduced. Inspect ROCm query counts alongside kernel counts, not just summed
API durations from concurrent workers. Keep the uninstrumented timings as the
performance comparison.

To keep the campaign driver running through an ordinary SSH disconnect:

```zsh
COPY_LOG=$OPENSN_TUO_RESULTS/$COPY_LABEL-driver.log
nohup zsh "$COPY_RUNNER" run "$COPY_LABEL" > "$COPY_LOG" 2>&1 < /dev/null &!
tail -f "$COPY_LOG"
```

Define `OPENSN_TUO_RESULTS` first, or use the default result parent shown above.
The redirection parent must already exist. Run the background command only
once. Ctrl-C stops `tail`, not the background driver. A login-node reboot or
site process-cleanup policy can still terminate it. Before using `resume`,
check that no previous driver or allocation is still running this campaign.

The driver can be checked locally without a scheduler:

```sh
python3 tools/scaling/tuo/test_flux_copy_profile.py
```
