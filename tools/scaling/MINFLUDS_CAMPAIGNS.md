# Minimum-FLUDS scaling and BEAVRS campaigns

Use `cbc-and-cbcd-with-minimally-sized-fluds-profiling-2`. Its solver base is the
updated production branch with round-robin CBCD worker ownership. It adds the
`faster-vp` volume-postprocessor change and opt-in profiling, not the experimental
device-closure, fused-dispatch, GPU-aware-MPI, or CBCD cycle implementations.

## 1. Publish from the local machine

After reviewing the changes:

```zsh
git push --force-with-lease origin cbc-and-cbcd-with-minimally-sized-fluds
git push -u origin cbc-and-cbcd-with-minimally-sized-fluds-profiling-2
```

The production branch retains two CBC-cycle commits followed by its three
minimum-FLUDS/CBCD commits. Its pre-update backup is
`cbc-and-cbcd-with-minimally-sized-fluds-backup-20260908`.
Nothing in this workflow pushes automatically or changes existing campaigns.

## 2. Select the published revision on either cluster

Paste this block in the login shell. Dane and Tuo can use the same source
checkout because their workspace filesystem is shared. Their builds and
dependency environments must remain separate.

```zsh
prepare_minfluds_checkout() {
  local base=/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies/source-update-3
  local branch=cbc-and-cbcd-with-minimally-sized-fluds-profiling-2
  git -C "$base" fetch origin "$branch" || return
  export SHA=$(git -C "$base" rev-parse FETCH_HEAD) || return
  export SOURCE=/usr/workspace/$USER/opensn-minfluds2/source-${SHA[1,9]}
  mkdir -p "${SOURCE:h}" || return
  if [[ ! -e "$SOURCE" ]]; then
    git -C "$base" worktree add --detach "$SOURCE" "$SHA" || return
  fi
  [[ $(git -C "$SOURCE" rev-parse HEAD) == "$SHA" ]] || return 1
  [[ -z $(git -C "$SOURCE" status --porcelain) ]] || return 1
  export CAMPAIGN_RUNNER=$SOURCE/tools/scaling/run_minfluds.zsh
  print -- "revision=$SHA"
  print -- "runner=$CAMPAIGN_RUNNER"
}
prepare_minfluds_checkout
```

Stop if this fails. No build is launched by this block. Keep this checkout
unchanged until its jobs finish. The script uses `FETCH_HEAD` deliberately;
it does not depend on a clone's remote-tracking refspec configuration.

## 3. Dane scaling and profiling

The default existing environment is:

```text
/usr/workspace/$USER/opensn-dane-cbc-scaling/toolchains/clang19-openmpi412-python314-1/opensn-dane-env.sh
```

This is the private toolchain used for the successfully submitted CBC-cycle
campaign: Clang 19.1.3, OpenMPI 4.1.2, Python 3.14.6, CMake 3.30.5, private
PETSc/Boost/HDF5/VTK/Caliper/mpicpp-lite, and a matching Python venv. The helper
reuses it without reinstalling packages. It includes the earlier MPI version,
mpi4py linker-order, and PETSc/CMake fixes from the Dane setup work.

```zsh
DANE_LABEL=minfluds2-${SHA[1,9]}-dane-$(date -u +%Y%m%dT%H%M%SZ)
zsh "$CAMPAIGN_RUNNER" dane scaling "$DANE_LABEL"
```

This requests a one-node **pdebug** allocation for mesh preparation, then submits
one Native build job to **pdebug**. It submits 18 baseline and 18 separate
Caliper/MPI profiling jobs to **pbatch**, all dependent on successful completion
of that build and its MPI/Caliper preflight. Every study requests 64 ranks/node,
one thread/rank, exclusive nodes, all allocatable host memory, and one hour.
Node counts are 1, 2, 4, 8, 16, 32, 64, 128, 256. Baselines have three trials;
Caliper has one. Memory sufficiency at the smallest node counts is not guaranteed.

Do not cancel `dane-cbc-build` while waiting for its dependents.
No dependencies are rebuilt on pbatch. If the existing environment is absent,
stop and locate the completed toolchain; only run `bootstrap_opensn.zsh setup`
if a fresh toolchain is actually needed.

```zsh
zsh "$CAMPAIGN_RUNNER" dane status "$DANE_LABEL"
zsh "$CAMPAIGN_RUNNER" dane collect "$DANE_LABEL"
```

Results: `/p/lustre1/$USER/opensn-results/$DANE_LABEL`.
Collection writes `summary.csv`, `summary.md`, `raw-results.csv`, and
`profile-index.md`; it does not turn incomplete runs into successful measurements.

## 4. Tuo scaling and profiling

On the Tuo login shell, run step 2 if these variables are not set, then:

```zsh
TUO_LABEL=minfluds2-${SHA[1,9]}-tuo-$(date -u +%Y%m%dT%H%M%SZ)
zsh "$CAMPAIGN_RUNNER" tuo scaling "$TUO_LABEL"
```

The build runs in a one-node **pdebug** allocation. The default reused root is:

```text
/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies/builds/gfx942-minfluds-profiling-6386bec5e
```

Its `env.zsh` selects the existing ROCm/Cray-MPICH toolchain. The compiled
PETSc/Boost/HDF5/VTK/Caliper libraries are reused read-only. A separate venv
includes SciPy and matching, source-built mpi4py; mpicpp-lite 2.7.1 headers are
installed in a private overlay. The overlay has explicit include precedence
so older MPI wrapper headers under the reused library prefix cannot shadow it.
The Native HIP executable targets gfx942 and is built under:

```text
/usr/workspace/$USER/opensn-gpu/cbcd-v2-studies/builds/minfluds2-SHORT/build-opensn
```

`SHORT` is the first nine characters of the selected SHA. The existing
`builds/gfx942/mesh-cache` is reused. Set `OPENSN_TUO_REUSE_ROOT` before launching
only if a different, compatible completed Tuo dependency root is required.
Never use the Dane dependency tree here. The script does not reconfigure the old
dependency build directory against a new source checkout.

After the build succeeds, the helper submits 18 uninstrumented baseline jobs,
18 CBCD-metrics jobs, and 18 Caliper/MPI jobs to **pbatch**. All request one hour
and the same nine node counts, four ranks/node and one GPU/rank. The launch
requests 21 cores/rank and sets `OPENSN_NUM_THREADS=21`: at most 20 sweep workers
plus one communicator. No fixed worker override is inherited. Baselines run
three trials; each profile runs once. These are separate jobs so instrumented
timings are not mistaken for uninstrumented scaling results.

```zsh
zsh "$CAMPAIGN_RUNNER" tuo status "$TUO_LABEL"
zsh "$CAMPAIGN_RUNNER" tuo collect "$TUO_LABEL"
```

Results:

- `/p/lustre5/$USER/opensn-results/$TUO_LABEL-batch/resource-aware`
- `/p/lustre5/$USER/opensn-results/$TUO_LABEL-profile/resource-aware`

The profile collector writes `cbcd-metrics-histograms.csv` and strong/weak
`cbcd-mpi-message-size-histogram-*` plots. These count serialized outgoing MPI
messages in power-of-two size bins; they are not individual face-flux sizes.
The default campaign does not launch rocprof/omniperf replay studies.

## 5. BEAVRS on Dane

Wait until the **pdebug build** in the Dane scaling campaign has completed
successfully; the scaling jobs themselves need not have finished. In the same
Dane shell:

```zsh
DANE_BEAVRS=beavrs-minfluds2-${SHA[1,9]}-dane-$(date -u +%Y%m%dT%H%M%SZ)
zsh "$CAMPAIGN_RUNNER" dane beavrs "$DANE_LABEL" "$DANE_BEAVRS"
```

This reuses that campaign's Native executable and environment and requests
**32 nodes, 64 ranks/node (2048 ranks), 24 hours, pbatch**, with exclusive nodes
and all allocatable memory. No compilation occurs in this job.

```zsh
DANE_BEAVRS_ROOT=/p/lustre1/$USER/opensn-results/$DANE_BEAVRS
DANE_BEAVRS_ID=$(< "$DANE_BEAVRS_ROOT/job-id.txt")
squeue -j "${DANE_BEAVRS_ID%%;*}"
sacct -j "${DANE_BEAVRS_ID%%;*}" --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
```

## 6. BEAVRS on Tuo

After the Tuo scaling helper has completed its **pdebug build**, in that Tuo shell:

```zsh
TUO_BEAVRS=beavrs-minfluds2-${SHA[1,9]}-tuo-$(date -u +%Y%m%dT%H%M%SZ)
zsh "$CAMPAIGN_RUNNER" tuo beavrs "$TUO_LABEL" "$TUO_BEAVRS"
```

This reuses the same Native HIP binary and requests **16 nodes, four ranks/node
(64 ranks), one device/rank, 21 cores/rank, 12 hours, pbatch**. It does not build
dependencies or OpenSn in the benchmark allocation.

```zsh
TUO_BEAVRS_ROOT=/p/lustre5/$USER/opensn-results/$TUO_BEAVRS
flux jobs -a "$(< "$TUO_BEAVRS_ROOT/job-id.txt")"
```

Both BEAVRS workflows use the **original** input, mesh and cross-section files
in `/usr/workspace/$USER/opensn-gpu/beavrs-benchmark`. If necessary, override
`OPENSN_BEAVRS_SOURCE` with the directory containing those original files.
Original files are never edited. Existing output directories are rejected.
To retry, select a new BEAVRS label; retain the scaling label to reuse its build.

## What the BEAVRS retry changes—and what it does not establish

The old Dane job 7437622 was terminated at the 24-hour limit. The archived
outputs do not locate all of that wall time reliably. The old Tuo jobs exited
with rank failures near startup; those logs do **not** establish a CBCD sweep
bug or a hardware-node failure. Missing imports were a possible explanation,
not a demonstrated cause.

The retry checks the Native build's source-revision stamp, imports NumPy/SciPy,
and runs an all-rank mpi4py collective inside OpenSn before loading the mesh.
It uses OpenSn's already initialized MPI runtime, not a second MPI initialization.
Fuel-centroid records are gathered using mpi4py instead of the console's padded
floating-point reduction. The globally deduplicated record values and ordering
are preserved. Stage timings, early eigenvalue output, rank tracebacks, and
success/failure markers distinguish startup, solve, and postprocessing failures.
Inspect `results/run-*/preflight*`, `stdout.txt`, `stderr.txt`, and, on Dane,
`rank-*.stderr`, as well as scheduler output and accounting.

The physical data, geometry, quadrature defaults, convergence tolerances and pin
clustering rule remain unchanged. Host CBC allows cycles; CBCD uses the
production acyclic implementation. Both disable saved angular flux.
The original centroid-clustering rule needs independent validation before its
CSV is treated as a validated physical pin-power reference. This workflow is
an execution/performance benchmark, not a BEAVRS validation claim.

`faster-vp` can reduce volume postprocessing cost, especially for device sweeps;
it does not guarantee eigenvalue convergence or completion within either time
limit. A `SUCCESS` marker plus normal OpenSn termination and successful scheduler
accounting establishes completion, not physical validation. Preserve failure
artifacts rather than repeatedly resubmitting unexplained failures.
