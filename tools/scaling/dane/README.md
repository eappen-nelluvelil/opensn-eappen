# Dane AAH/CBC studies

This temporary framework runs paired host AAH and CBC strong- and weak-scaling
studies on LLNL Dane. It is intentionally separate from the general framework
in `tools/scaling`: one command prepares all jobs, one command collects them,
and one command profiles CBC with OpenSn's existing Caliper regions.

The fixed production configuration is 64 MPI ranks per node, one physical core
per rank, and the node sequence `1,2,4,...,256`. AAH and CBC run sequentially
inside the same exclusive allocation. Their order alternates between trials to
limit systematic first-run bias. Strong and weak cases use the existing
`tools/scaling/lib/cube.geo` and `xs_168g.xs` inputs. Meshes are generated once
in a shared cache and verified by SHA-256 before every run.

## 1. Transfer the source tree to Dane

Use `rsync` over SSH for incremental source uploads and result downloads. Keep
the remote source and results in separate directories so that source cleanup
cannot remove completed studies. Run the following on the Mac:

```zsh
LOCAL=/Users/eappen/opensn-eappen-prs/opensn-eappen
REMOTE_HOST=nelluvelil1@dane.llnl.gov
REMOTE_ROOT=/usr/workspace/nelluvelil1/opensn-dane/opensn-profiling
REMOTE_SOURCE=$REMOTE_ROOT/source
REMOTE_RESULTS=$REMOTE_ROOT/results

ssh "$REMOTE_HOST" \
  "mkdir -p '$REMOTE_SOURCE' '$REMOTE_RESULTS'"
```

Review the local working tree before every upload. The transfer intentionally
copies the working tree rather than `.git`, so intended uncommitted profiling
edits can be tested without publishing the branch.

```zsh
git -C "$LOCAL" status --short
git -C "$LOCAL" rev-parse HEAD
```

Define the exclusions once, inspect a dry run, and then perform the upload:

```zsh
RSYNC_FILTERS=(
  --exclude '/.git/'
  --exclude '/build/'
  --exclude '/build-*/'
  --exclude '/cmake-build-*/'
  --exclude '/tools/scaling/dane/runs/'
  --exclude '**/__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
)

rsync -az --partial --delete-delay --itemize-changes \
  "${RSYNC_FILTERS[@]}" --dry-run \
  "$LOCAL/" "${REMOTE_HOST}:${REMOTE_SOURCE}/"

rsync -az --partial --delete-delay --itemize-changes \
  "${RSYNC_FILTERS[@]}" \
  "$LOCAL/" "${REMOTE_HOST}:${REMOTE_SOURCE}/"
```

`--delete-delay` applies only to the dedicated remote source directory and
removes files deleted locally after the upload succeeds. Never place build or
result data in that directory. Do not add a deletion option when downloading
results.

## 2. Install the Zsh configuration

With Zsh configured as the account shell, `.zprofile` and `.zshrc` replace the
login and interactive roles previously handled through Bash. Leave LLNL's
`.profile`, `.profile.linux`, `.login`, and `.login.linux` files unchanged.
`.bashrc` remains only as a quiet fallback.

On Dane or Tuo:

```zsh
SOURCE=/usr/workspace/nelluvelil1/opensn-dane/opensn-profiling/source
DOTFILES=$SOURCE/tools/scaling/dane/dotfiles
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

for name in .bashrc .zprofile .zshrc; do
  [[ ! -e $HOME/$name ]] || cp -p "$HOME/$name" "$HOME/$name.before-opensn-$STAMP"
  install -m 600 "$DOTFILES/$name" "$HOME/$name"
done

zsh -n "$HOME/.zprofile" "$HOME/.zshrc"
bash -n "$HOME/.bashrc"
exec zsh -l
```

The shared `.zshrc` selects modules by hostname. It does not source any old
Python or dependency installation. Set `OPENSN_DANE_ENV` to a completed
revision-specific `env.zsh` when an interactive shell should load that build
automatically. The Tuo path is reserved for a later clean Tuo environment.

## 3. Build a clean Dane environment

OpenSn's managed dependency superbuild installs mpicpp-lite, Boost, PETSc,
HDF5, VTK, and Caliper. MPI, Python, CMake, Clang, Git, and Gmsh are platform
prerequisites. The bootstrap disables discovery of all six managed packages,
so compatible system installations cannot satisfy them. It uses Dane's CMake
3.30.5 rather than asking PETSc to download and build another CMake.

Request an exclusive compute node; do not compile on a login node:

```zsh
salloc -N 1 -p pdebug --exclusive -t 01:00:00

SOURCE=/usr/workspace/nelluvelil1/opensn-dane/opensn-profiling/source
TAG=$(git -C "$SOURCE" rev-parse --short=12 HEAD)
WORK=/usr/workspace/$USER/opensn-dane/builds/cbc-cycles-update-$TAG
export OPENSN_SOURCE=$SOURCE
export OPENSN_DANE_ROOT=$WORK
mkdir -p "$WORK/logs"

zsh "$SOURCE/tools/scaling/dane/bootstrap.zsh" configure-deps \
  |& tee "$WORK/logs/configure-deps.log"
zsh "$SOURCE/tools/scaling/dane/bootstrap.zsh" build-deps \
  |& tee "$WORK/logs/build-deps.log"
zsh "$SOURCE/tools/scaling/dane/bootstrap.zsh" build-opensn \
  |& tee "$WORK/logs/build-opensn.log"
```

The dependency build is resumable. If the one-hour allocation expires, obtain
another allocation and continue with:

```zsh
SOURCE=/usr/workspace/nelluvelil1/opensn-dane/opensn-profiling/source
export OPENSN_SOURCE=$SOURCE
export OPENSN_DANE_ROOT=$WORK
zsh "$SOURCE/tools/scaling/dane/bootstrap.zsh" build-deps
zsh "$SOURCE/tools/scaling/dane/bootstrap.zsh" build-opensn
```

The resulting paths are below `$WORK`:

```text
$WORK/venv
$WORK/deps
$WORK/build-deps-ninja
$WORK/build
$WORK/env.zsh
```

The dependency superbuild uses Ninja and runs one external project at a time,
while each project uses the allocation's available cores internally. Ninja is
installed in the revision-specific virtual environment. Using it for the outer
`ExternalProject` graph prevents GNU Make jobserver state from reaching nested
HDF5, PETSc, VTK, and Caliper configure or install steps. Such inherited state
can produce `read jobs pipe: Bad file descriptor`. Set
`OPENSN_DANE_BUILD_JOBS` before invoking the bootstrap only if per-package
parallelism must be limited. The OpenSn build uses the same detected worker
count.

The upstream dependency driver requests a private CMake build from PETSc to
support hosts whose installed CMake is version 4. The Dane bootstrap instead
creates a private driver copy under `$WORK/dependency-driver` with only that
request removed. Repository CMake files are not modified. PETSc consequently
uses the loaded CMake 3.30.5 while still building its numerical dependencies
from source.

If an earlier Makefile superbuild failed during HDF5 installation or PETSc
configuration, do not delete it. Update the source framework and retain the
same `$WORK`. Running `configure-deps` creates the independent
`$WORK/build-deps-ninja` tree, so CMake does not reuse the failed PETSc package
state or attempt to change the generator of the old build tree. The common
install prefix remains `$WORK/deps`.

OpenSn is configured with `CMAKE_BUILD_TYPE=Native` and the Python module
enabled. System GoogleTest discovery is disabled because Dane's Python module
exposes a GoogleTest shared library built against an incompatible GNU C++ ABI;
OpenSn instead builds its pinned GoogleTest with the active Clang toolchain.
The virtual environment contains pybind11, NumPy, a source-built mpi4py tied to
the loaded OpenMPI, Matplotlib, and Hatchet.

After completion:

```zsh
source "$WORK/env.zsh"
test -x "$WORK/build/python/opensn"
"$WORK/build/python/opensn" --version
command -v cali-query
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'
```

Gmsh is needed only to prepare meshes. Locate the Dane module with
`module spider gmsh`, load the available version, and verify `command -v gmsh`.
The `prepare` command generates the strong- and weak-scaling meshes in
`$MESH_CACHE`. Later studies reuse existing mesh files without invoking Gmsh
again.

## 4. Prepare and submit scaling studies

Record the revision reported before the source upload. On Dane, first generate
a two-node smoke study:

```zsh
REMOTE_ROOT=/usr/workspace/nelluvelil1/opensn-dane/opensn-profiling
SOURCE=$REMOTE_ROOT/source
RESULTS=$REMOTE_ROOT/results
TAG=$(git -C "$SOURCE" rev-parse --short=12 HEAD)
WORK=/usr/workspace/$USER/opensn-dane/builds/cbc-cycles-update-$TAG
DRIVER=$SOURCE/tools/scaling/dane/study.py
BINARY=$WORK/build/python/opensn
ENVIRONMENT=$WORK/env.zsh
MESH_CACHE=$WORK/mesh-cache
REVISION=$(git -C "$SOURCE" rev-parse HEAD)
source "$ENVIRONMENT"

python "$DRIVER" prepare \
  --binary "$BINARY" \
  --environment "$ENVIRONMENT" \
  --output "$RESULTS/smoke-$REVISION" \
  --mesh-cache "$MESH_CACHE" \
  --gmsh "$(command -v gmsh)" \
  --label cbc-cycles-update-smoke \
  --revision "$REVISION" \
  --nodes 1,2 \
  --repetitions 1 \
  --account YOUR_LC_BANK

"$RESULTS/smoke-$REVISION/submit.zsh"
```

After the smoke jobs pass, prepare the production matrix in a new directory:

```zsh
python "$DRIVER" prepare \
  --binary "$BINARY" \
  --environment "$ENVIRONMENT" \
  --output "$RESULTS/production-$REVISION" \
  --mesh-cache "$MESH_CACHE" \
  --gmsh "$(command -v gmsh)" \
  --label cbc-cycles-update \
  --revision "$REVISION" \
  --nodes 1,2,4,8,16,32,64,128,256 \
  --repetitions 3 \
  --account YOUR_LC_BANK

"$RESULTS/production-$REVISION/submit.zsh"
```

Production jobs use `pbatch`, exclusive nodes, physical-core binding, and a
four-hour limit. Use `--partition`, `--time-limit`, or omit `--account` only
when Dane policy requires it.

The branch changes no AAH implementation files, so AAH from this binary is the
trunk AAH implementation built with exactly the same compiler, dependencies,
and flags as CBC. This removes build-to-build noise from the algorithm
comparison.

Monitor jobs with `squeue -u $USER` and `sacct`. After completion, collect on
Dane:

```zsh
python "$DRIVER" collect \
  --study "$RESULTS/production-$REVISION"
```

The collector writes `results.csv`, `summary.csv`, `summary.md`, `strong.pdf`,
and `weak.pdf`. Strong scaling reports average sweep time per unknown; weak
scaling reports average sweep time and efficiency relative to one node.

Download the collected results from the Mac. This command is incremental,
resumes partial files, and never deletes local data:

```zsh
LOCAL_RESULTS=/Users/eappen/opensn-eappen-prs/dane-results
REMOTE_HOST=nelluvelil1@dane.llnl.gov
REMOTE_RESULTS=/usr/workspace/nelluvelil1/opensn-dane/opensn-profiling/results
mkdir -p "$LOCAL_RESULTS"
rsync -az --partial --itemize-changes \
  "${REMOTE_HOST}:${REMOTE_RESULTS}/" "$LOCAL_RESULTS/"
```

For a small number of individual files, `scp` is a simpler equivalent:

```zsh
scp "${REMOTE_HOST}:${REMOTE_RESULTS}/production-$REVISION/summary.md" \
  "$LOCAL_RESULTS/"
```

## 5. Profile CBC interactively

No C++ source modification is required for the first profiling pass. OpenSn
already contains Caliper regions for sweep construction, CBC SPDS processing,
angle-set advancement, asynchronous sends/receives, and sweep kernels. Caliper
is enabled only by the profile command, so the scaling jobs remain
uninstrumented.

Use the prepared production study and one allocation at a time:

```zsh
salloc -N 1 -p pdebug --exclusive -t 01:00:00
REMOTE_ROOT=/usr/workspace/nelluvelil1/opensn-dane/opensn-profiling
SOURCE=$REMOTE_ROOT/source
RESULTS=$REMOTE_ROOT/results
TAG=$(git -C "$SOURCE" rev-parse --short=12 HEAD)
WORK=/usr/workspace/$USER/opensn-dane/builds/cbc-cycles-update-$TAG
source "$WORK/env.zsh"
DRIVER=$SOURCE/tools/scaling/dane/study.py
REVISION=$(git -C "$SOURCE" rev-parse HEAD)
python "$DRIVER" profile \
  --study "$RESULTS/production-$REVISION" \
  --algorithm CBC --mode summary
python "$DRIVER" profile \
  --study "$RESULTS/production-$REVISION" \
  --algorithm CBC --mode hatchet
exit
```

Repeat with `salloc -N 2` and `salloc -N 4`. Dane permits at most eight pdebug
nodes per user and pdebug allocations last one hour. The profile command infers
the allocation size, uses the matching strong-scaling input, and launches 64
ranks per node. Text summaries and `.cali` files are written under
`profiles/nodes-N/` in the study directory and can be downloaded with the same
result-transfer command above.

Use the production scaling outputs for timing. Caliper profiles attribute time
and count MPI/region events but include profiling overhead. If additional C++
regions are later justified, build them in a separate tree and pass that binary
only to `study.py profile --binary`; do not reuse it for production scaling.
