# Tuolumne device-CBC studies

This temporary framework builds OpenSn for AMD MI300A and prepares independent
Flux jobs for device-CBC scaling and profiling. It uses 4 MPI ranks per node,
one MI300A per rank, native SPX mode, and Flux/mpibind's automatic NUMA-local
CPU/GPU binding. Production jobs leave OpenSn thread-count overrides unset so
CBCD derives its workers from each rank's actual affinity.

The scaling workflow is intentionally branch-independent. Prepare one study
for the trunk binary and one for the candidate binary using the same mesh
cache, then collect and compare them. The profile workflow is intended for the
candidate CBCD V2 binary and creates separate batch jobs so profiler overhead
does not affect production results.

Trunk device CBC requires retention of the full angular-flux vector, whereas
CBCD V2 uses its compact FLUDS and does not. Pass `--save-angular-flux` when
preparing the trunk study. Omit it for CBCD V2 scaling and profiling; generated
CBCD V2 inputs then explicitly set `save_angular_flux=False`.

## Environment and build

Install the files in `dotfiles/` only after backing up the current files. Do
not replace LLNL's `.profile*` or `.login*` files. The Zsh configuration loads
only platform modules; it never sources an old virtual environment or OpenSn
dependency prefix. Load a completed revision-specific environment explicitly
with `opensn_use /path/to/env.zsh`.

Build on a compute node, not a login node. Set `OPENSN_SOURCE` to the selected
source worktree, `OPENSN_TUO_ROOT` to the clean dependency root, and optionally
`OPENSN_TUO_BUILD` to a branch-specific build directory:

```zsh
flux alloc -N 1 -q pdebug -t 60m

export OPENSN_SOURCE=/usr/workspace/$USER/opensn-gpu/opensn-studies/source
export OPENSN_TUO_ROOT=/usr/workspace/$USER/opensn-gpu/builds/opensn-tuolumne
mkdir -p "$OPENSN_TUO_ROOT/logs"

zsh "$OPENSN_SOURCE/tools/scaling/tuo/bootstrap.zsh" configure-deps
zsh "$OPENSN_SOURCE/tools/scaling/tuo/bootstrap.zsh" build-deps
zsh "$OPENSN_SOURCE/tools/scaling/tuo/bootstrap.zsh" build-opensn
```

The dependency build is resumable. It uses a private copy of the OpenSn
dependency driver with PETSc's unnecessary CMake download disabled; repository
CMake files are not changed. The resulting environment is
`$OPENSN_TUO_ROOT/env.zsh`. Reuse that dependency root for trunk and candidate
builds, but use a distinct `OPENSN_TUO_BUILD` for each source revision.

## Scaling

Load Gmsh, source the clean environment, and prepare one study per binary:

```zsh
source "$OPENSN_TUO_ROOT/env.zsh"
module load gmsh

python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" prepare \
  --binary "$OPENSN_TUO_BUILD/python/opensn" \
  --environment "$OPENSN_TUO_ROOT/env.zsh" \
  --output /path/to/results/candidate \
  --mesh-cache /path/to/mesh-cache \
  --gmsh "$(command -v gmsh)" \
  --label CBCD-V2 \
  --revision "$(git -C "$OPENSN_SOURCE" rev-parse HEAD)" \
  --bank YOUR_LC_BANK

/path/to/results/candidate/submit.zsh
```

Use the same command with the trunk binary and add `--save-angular-flux` when
preparing the trunk baseline.

Meshes are generated once from `tools/scaling/lib/cube.geo` and reused. Each
study contains 18 jobs for strong and weak scaling over 1--256 nodes. Failed or
OOM points can be excluded with `collect --allow-incomplete`.

```zsh
python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" collect \
  --study /path/to/results/candidate --allow-incomplete

python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" compare \
  --baseline /path/to/results/trunk \
  --candidate /path/to/results/candidate \
  --output /path/to/results/comparison
```

## Profiling

Use a separate directory and prepare the candidate profile jobs:

```zsh
python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" prepare-profile \
  --binary "$OPENSN_TUO_BUILD/python/opensn" \
  --environment "$OPENSN_TUO_ROOT/env.zsh" \
  --output /path/to/results/candidate-profile \
  --mesh-cache /path/to/mesh-cache \
  --gmsh "$(command -v gmsh)" \
  --label CBCD-V2-profile \
  --revision "$(git -C "$OPENSN_SOURCE" rev-parse HEAD)" \
  --bank YOUR_LC_BANK

/path/to/results/candidate-profile/submit.zsh
```

The generated jobs are independent:

- `baseline` measures the unprofiled case at 1, 2, and 4 nodes.
- `caliper` attributes existing OpenSn CPU regions at 1, 2, and 4 nodes.
- `pmpi` reports intra- and inter-node MPI activity at 1, 2, and 4 nodes.
- `rocprof` records HIP API, kernel, allocation, and copy traces per rank.
- `hpctoolkit` samples CPU call paths and ROCm GPU activity at low overhead.
- `omniperf` profiles `SweepKernel` occupancy and memory behavior on one MI300A.

The default profile mesh uses divisor 15 and two linear iterations. This keeps
setup and profiler replay costs bounded while preserving the CBCD scheduler,
communication thread, device transfers, and sweep kernel. Use production
scaling jobs, never profiler timings, for performance conclusions.

After the jobs finish:

```zsh
python "$OPENSN_SOURCE/tools/scaling/tuo/study.py" collect-profile \
  --study /path/to/results/candidate-profile
```

Use `flux jobs -A` to monitor queued and completed jobs. Analyze OmniPerf data
with `omniperf analyze -p PATH`, and process HPCToolkit measurements with
`hpcstruct` followed by `hpcprof` or `hpcprof-mpi` on Tuo.
