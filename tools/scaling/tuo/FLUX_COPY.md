# Flux-copy comparisons on Tuo

Use `run_flux_copy_profile.zsh` from a clean, revision-specific worktree of
`cbc-and-cbcd-with-minimally-sized-fluds-profiling-2`. The comparison baseline is
`6bffcd8b43421fb8813a8748f18afa69e8e48f61`. The outgoing-copy-only comparison is
`1e69af813dd979dca68fd2ad5450fa11362d5303`. The driver does not fetch or push Git
branches and does not modify existing source worktrees or builds.

```zsh
export OPENSN_TUO_BANK=cbronze
COPY_RUNNER=$SOURCE/tools/scaling/tuo/run_flux_copy_profile.zsh
COPY_LABEL=cbcd-worker-fusion-$(git -C "$SOURCE" rev-parse --short=9 HEAD)-pdebug-$(date -u +%Y%m%dT%H%M%SZ)
export OPENSN_CBCD_FUSE_WORKER_LAUNCHES=1
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
The peer-reuse, batched-publication, and completion-query campaigns used this
same venv. The complete baseline triplets from `d805742b4` are the immediate
comparison for completed-source retirement. Retain `1d928df28` as a second
reference for section indexing and contiguous queue consumption. The shared
receive-packet implementation is retained.
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

The shared-receive experiment keeps MPI packet bytes alive until every
referenced angle-set section has been placed in FLUDS by its worker. Mailboxes
contain section descriptors instead of copied flux vectors. The communicator
recycles fully consumed packets and retains at most one idle buffer per source
and power-of-two size class.
It does not perform the final FLUDS writes or wait for a worker to free a packet
before receiving another. Compare reported peak host memory, worker incoming
processing, communicator receive processing, and baseline sweep times. Reduced
allocation and copying do not by themselves guarantee lower sweep times.

The section-indexing experiment adds the serialized payload-byte count to
each angle-set section. The communicator skips directly to the next section,
and workers still parse and place the face records. All ranks must use the
same executable because the packet header changed. Packet-size accounting
includes the additional header field. The queue experiment replaces temporary
pointer-vector snapshots with two read-only spans over the ring. Publication
and release ordering are unchanged. Compare receive and flush phase costs,
packet counts and sizes, memory, and the uninstrumented sweep times separately.
The source changes are `01d934418` (section indexing) and `ca4c2ae1a`
(queue spans). The single MPI progress thread per rank, source-specific
receive probing, worker assignments, and device kernels are unchanged.

The shared-receive baseline still has a substantial strong-scaling limit:
from one to eight nodes, ready-cell batches become smaller and launch counts
per rank do not fall in proportion to volume work. These two changes remove
repeated host processing but do not remove the sweep graph's critical path.
Do not infer ideal scaling or a Tuo speedup from isolated local timings.

The completed-source experiment counts incoming face records using the
existing FLUDS source-to-face CSR offsets. Once all records from a source
have arrived for the current sweep, that source is no longer probed. Counts
reset on every sweep. Packet ownership, worker-side placement, source order,
and the end-of-sweep barrier remain unchanged. The new
`skipped_receive_probes` field in `cbcd-metrics/rank-*/sweeps.csv` counts
omitted visits to completed sources. Compare this with MPI probe counts and
receive-phase time, while judging performance from uninstrumented trials.
It does not count all unsuccessful probes or predict a proportional speedup.

## Combined worker launches

`OPENSN_CBCD_FUSE_WORKER_LAUNCHES=1` selects an experimental kernel that combines
the ready angle-set batches owned by one worker into a single launch. The
original launch path remains the default (`0`). No MPI policy, thread budget,
cell dependencies, or slot plans are changed. This is not a persistent kernel.
The same CBC cell solve is used, and only already-ready cells are included.
Problems with `save_angular_flux=True` retain the original launch path even
when this setting is `1`. Local measurements found a saved-flux regression
with combined launches. The scaling inputs above use `False` and exercise
the experiment. A nonzero `fused_kernel_launches` count confirms its use.

Each worker owns one launch descriptor buffer and one stream. Descriptors and
launched cell lists are not changed until that stream completes. Kernel
arguments are refreshed on the device once per sweep. Workers still place
incoming fluxes, update dependencies, and publish outgoing/reflecting fluxes.
Completion queries are shared across all batches in a combined launch. A small
batch can therefore wait for a larger batch in that launch, which is the main
performance tradeoff to measure on Tuo.

The new `fused_kernel_launches` sweep metric counts actual combined GPU launches.
Existing `kernel_launches` and angle-set histograms still count logical
angle-set batches. Use the new count to calculate cells per combined launch.
Compare it with ROCm launch counts, as well as baseline sweep times and fluxes.

The driver saves the mode in `<label>-launch-mode.txt` next to the campaign.
Generated jobs freeze the selected value and record it in `metadata.txt`.
`resume` restores the saved mode if it is unset in the login shell and rejects
a conflicting value. Keep that file with the campaign when copying or moving
results. Use a new label for a different mode.

After the experimental campaign finishes, run a baseline-only control with the
same new executable under a different label. Do not start two drivers that
could build in the same revision-specific build directory concurrently:

```zsh
OPENSN_CBCD_FUSE_WORKER_LAUNCHES=0 zsh "$COPY_RUNNER" run "$COPY_LABEL-classic" baseline
```

This requests a separate allocation. The three trials at each node count use
the same nodes within that allocation, but the control and experimental
allocations need not use the same physical nodes. Treat small differences
accordingly. The existing d805742b4 campaign remains the pre-change comparison.
Do not infer ideal scaling from a reduction in GPU launch counts alone.

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
