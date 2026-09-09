# Dane scaling and profiling

Run from a clean checkout of `cbc-and-cbcd-with-minimally-sized-fluds-profiling-2`.
The scripts default to Clang **19.1.3** and OpenMPI **4.1.2**, with the
`python/3.14.6` and `cmake/3.30.5` modules. They explicitly reload these modules
in jobs; an interactive `.zshrc` is not sufficient for batch environments.
MPI version checks use `mpicxx --showme:version`; the scripts launch with
`srun` and do not require `mpirun` to be on `PATH`.
CMake 3.30.5 satisfies OpenSn's minimum version while supporting the older
CMake policies used by PETSc's METIS dependency. PETSc receives the selected
executable through `--with-cmake-exec`, not its Boolean `--with-cmake` option.
The private mpi4py is rebuilt from source with exhaustive MPI feature checks,
using the selected MPI wrapper for compilation and linking. Setup and the saved
environment prioritize the wrapper's library directories. This accommodates
MPI installations that omit optional datatype symbols and avoids reusing a
cached mpi4py wheel built against a different MPI installation.
Setup retains `mpi4py-build.log` and `mpi4py-linkage.txt` in the toolchain root;
it still stops if importing MPI fails. No missing-symbol failure is ignored.
The mpi4py linker command places the selected MPI library directories before
Python's extension-link flags. This prevents an Anaconda `-L` directory from
selecting its bundled MPI with the vendor MPI headers. Setup verifies the
resolved MPI library against the compiler wrapper's directories before importing
the extension; runtime search-path changes alone cannot fix the wrong MPI SONAME.

## Launch

For the existing private toolchain, use the wrapper described in
[`MINFLUDS_CAMPAIGNS.md`](../MINFLUDS_CAMPAIGNS.md):

```zsh
LABEL=minfluds2-$(git rev-parse --short=9 HEAD)-dane-$(date -u +%Y%m%dT%H%M%SZ)
zsh tools/scaling/run_minfluds.zsh dane scaling "$LABEL"
```

Keep the printed label. Results default to
`/p/lustre1/$USER/opensn-results/$LABEL`. Existing campaigns are never overwritten.

If a fresh toolchain is needed, `bootstrap_opensn.zsh setup` requests one pdebug
allocation, builds private dependencies and a venv,
and compiles a Native preflight executable on a compute node. PETSc, Boost,
HDF5, VTK, Caliper, and mpicpp-lite are not taken from system package prefixes.
The outer dependency build is serial, with bounded package parallelism (16).
If setup expires, rerun the same command: completed dependency stages remain.
Use `OPENSN_DANE_SETUP_TIME` to change the allocation limit when queue policy permits.

Mesh preparation requests a second one-node pdebug allocation; it does not run
Gmsh on the login node. The campaign generates the existing tetrahedral mesh
family and submits:

- one Native build job on pdebug;
- 18 baseline jobs: strong and weak scaling at 1, 2, 4, 8, 16, 32, 64, 128, 256 nodes;
- 18 separate Caliper jobs at the same sizes, with annotated-region and MPI time.

All study jobs use 64 ranks per node, one OpenSn thread per rank, exclusive
nodes, all allocatable host memory, and one hour. Baselines run three trials;
Caliper runs one. Study jobs run on pbatch and depend on successful completion of the pdebug build job.
Do not cancel that build job while its dependent jobs are pending.
The build job also verifies a one-rank MPI/Caliper report before releasing the
study jobs. Each profiling job must produce a nonempty report containing MPI
entries before it is marked complete.

Strong problems use divisor 39, 64 groups, 448 directions, single-angle
aggregation, ten maximum WGS iterations, and the existing source/tolerance.
Weak problems retain the prior divisor table. Mesh cell counts are only
approximately proportional to nodes. Saved angular flux is disabled.

MPI is launched with `srun --mpi=pmix --mpibind=on --kill-on-bad-exit=1`.
Check `srun --mpi=list` on Dane if its PMIx plugin configuration changes.

## Monitor and collect

```zsh
zsh tools/scaling/dane/run_cbc_scaling.zsh status "$LABEL"
zsh tools/scaling/dane/run_cbc_scaling.zsh collect "$LABEL"
```

Collection writes baseline `raw-results.csv`, `summary.csv`, and `summary.md`.
It also writes `profile-index.md`, linking the per-case `caliper.txt` reports
under `profiles/caliper`. Instrumented timing is never pooled with baseline timing.
Only trials with the normal OpenSn completion marker are accepted.

The reference is **4 nodes**, not an inferred 1-node measurement:
`Estrong(N) = 4*T(4)/(N*T(N))`; `Eweak(N) = T(4)/T(N)`.
If the reference case has not completed, efficiency remains unavailable.
Native flags, compiler/MPI versions, source SHA, and build cache are retained
in the build logs/build directory and campaign manifest.

To cancel only this campaign, first inspect its exact IDs:

```zsh
column -t "/p/lustre1/$USER/opensn-results/$LABEL/job-ids.tsv"
awk '{print $2}' "/p/lustre1/$USER/opensn-results/$LABEL/job-ids.tsv" | xargs -r scancel
```

## Separate source build (BEAVRS)

`OPENSN_DANE_SOURCE=/absolute/detached/source` makes `bootstrap_opensn.zsh setup`
compile that source using the same private toolchain. Dependency recipes still
come from this tools checkout. The resulting executable is printed and resides
at `toolchains/clang19-openmpi412-python314-1/build-opensn-SHA9-native/python/opensn`
under `/usr/workspace/$USER/opensn-dane-cbc-scaling`.
Unset `OPENSN_DANE_SOURCE` before preparing a cycles scaling campaign.
Never point a build directory at a different source worktree or reuse a toolchain
tag with a different compiler/MPI installation.
