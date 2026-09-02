# Dane host-CBC scaling studies

This workflow measures host CBC from the current, clean source revision. The
campaign covers strong and weak scaling on 1, 2, 4, 8, 16, 32, 64, 128, and
256 Dane nodes, with 64 MPI ranks per node and three trials per allocation.

Both OpenSn binaries are configured from fresh, campaign-specific build
directories with `CMAKE_BUILD_TYPE=Native` in a build job on a Dane compute
node. Scaling jobs depend on that build job and verify the build type before
launching. GPU backends and angular-flux storage are disabled.

## One-command setup and launch

Use a clean checkout on a Dane login node. The setup command requests one
exclusive `pdebug` node, builds an isolated OpenSn dependency prefix and Python
environment, performs a native preflight build, and then submits the campaign.
Only Dane's compiler, MPI, CMake, Python, Git, Flex, and build tools are loaded
from modules. OpenSn's package dependencies are built under
`/usr/workspace/$USER/opensn-dane-cbc-scaling/toolchains/isolated-1`.

```bash
cd /path/to/opensn
git switch cbc-and-cbcd-with-minimally-sized-fluds-profiling
git pull --ff-only origin cbc-and-cbcd-with-minimally-sized-fluds-profiling

export OPENSN_DANE_BANK=YOUR_BANK
export OPENSN_DANE_RESULTS=/p/lustre1/$USER/opensn-results          # or lustre2

zsh tools/scaling/dane/run_cbc_scaling.zsh setup-launch cbc-minfluds-native-1
```

The bootstrap is resumable. If its one-hour `pdebug` allocation expires, run
the same `setup-launch` command again. Completed dependency stages are retained,
and no scaling jobs are submitted until setup succeeds. Once setup is complete,
`setup-launch` creates a detached worktree at the exact current revision,
generates the mesh suite, prepares 18 scaling jobs, submits the native build
job, and submits every scaling job with an `afterok` dependency.

The isolated build includes mpicpp-lite, Boost, PETSc and its configured solver
dependencies, HDF5, VTK, Caliper, GoogleTest, and an MPI-compatible Python venv.
The outer dependency superbuild is kept serial while each package uses bounded
parallelism, avoiding nested unbounded `make -j` invocations.
All compiled dependencies use the selected MPI compiler wrappers. PETSc uses
the loaded CMake rather than downloading and building a second CMake release.

Setup and launch may also be performed separately:

```bash
zsh tools/scaling/dane/run_cbc_scaling.zsh setup
zsh tools/scaling/dane/run_cbc_scaling.zsh launch cbc-minfluds-native-1
```

The defaults are one hour per build/scaling job and three trials per allocation.
Override them before `launch` if necessary:

```bash
export OPENSN_DANE_TIME_LIMIT=01:00:00
export OPENSN_DANE_BUILD_TIME_LIMIT=01:00:00
export OPENSN_DANE_REPETITIONS=3
export OPENSN_DANE_BUILD_JOBS=16
export OPENSN_DANE_TOOLCHAIN=isolated-1
export OPENSN_DANE_MODULES='gcc/10.3.1-magic mvapich2/2.3.7 cmake/3.30.5 python/3.13.2 git/2.46.2'
```

These are also the defaults. In particular, the CMake version is explicit
because Dane's unversioned `cmake` module currently resolves to 3.23.1, which
is older than OpenSn's required CMake 3.29.

Do not reuse a campaign label. The workflow refuses to overwrite an existing
result directory.

## Monitor and collect

```bash
zsh tools/scaling/dane/run_cbc_scaling.zsh status cbc-minfluds-native-1
zsh tools/scaling/dane/run_cbc_scaling.zsh collect cbc-minfluds-native-1
```

Collection is incremental and may be run while jobs are active. It writes:

- `raw-results.csv`: every completed trial;
- `summary.csv`: medians, median absolute deviations, interquartile ranges, and
  scaling efficiencies;
- `summary.md`: readable host-CBC strong- and weak-scaling tables.

Strong-scaling efficiency is computed from the sweep time per unknown as
`g(1)/(N*g(N))`; this is equivalent to `T(1)/(N*T(N))` when the global strong
problem size is fixed. Weak-scaling efficiency is `T(1)/T(N)`. The collector
also reports average sweep time, the global unknown count, and lagged unknowns
when OpenSn prints that field.

## Dane-specific choices

Dane has 112 physical CPU cores per node, but this study deliberately launches
the requested 64 ranks per node. Slurm allocates whole `pbatch` nodes, and the
launch uses LLNL's `mpibind` plugin to distribute those ranks across the node's
NUMA topology. Each rank is single-threaded (`OPENSN_NUM_THREADS=1`).

Check current queue and bank limits with `joblimits` before launch. The default
work area is `/usr/workspace/$USER/opensn-dane-cbc-scaling`; change it with
`OPENSN_DANE_WORK_ROOT` if needed.

## BEAVRS host-CBC run

The companion runner converts the untouched BEAVRS CPU input to cycle-capable
host CBC and reuses the exact Native build submitted by a scaling campaign. Its
job depends on that campaign's build job, so it cannot race the executable.
The default is 32 exclusive nodes, 64 ranks per node, one thread per rank, and
24 hours:

```bash
export OPENSN_DANE_BEAVRS_SOURCE=/usr/workspace/$USER/opensn-gpu/beavrs-benchmark
zsh tools/scaling/dane/run_beavrs_cbc.zsh \
  launch cbc-minfluds-native-1 beavrs-cbc-minfluds-native-32n-1
```

The original input is preserved. The derived input uses single-angle
aggregation, `allow_cycles=True`, `save_angular_flux=False`, and a 256 KiB MPI
message cap. Monitor and collect with:

```bash
zsh tools/scaling/dane/run_beavrs_cbc.zsh \
  status cbc-minfluds-native-1 beavrs-cbc-minfluds-native-32n-1
zsh tools/scaling/dane/run_beavrs_cbc.zsh \
  collect cbc-minfluds-native-1 beavrs-cbc-minfluds-native-32n-1
```
