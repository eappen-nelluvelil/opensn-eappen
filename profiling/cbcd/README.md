# Local CBCD profiling

`run.py` provides bounded, revision-pinned CBCD V2 benchmarks and profiler
launches for one NVIDIA GPU. Every MPI rank uses the same device. Consequently,
the 1/2/4-rank results are **same-GPU contention diagnostics**, not accelerator
strong scaling and not a substitute for Tuo's one-GPU-per-rank studies.

The runner uses argument-vector subprocesses only (`shell=False`), executes
cases sequentially, terminates process groups on timeout, validates every OpenSn
result, and publishes each study directory atomically. A failed or interrupted
study is preserved with a terminal `state.json` and `manifest.json`.

## Fresh CUDA/Caliper environment

Use a new virtual environment, dependency prefix, dependency build directory,
and OpenSn build directory. Do not rely on a Caliper executable or library found
incidentally on `PATH`.

```bash
REPO=/home/eappen/opensn-eappen-prs-3/opensn-eappen
ROOT=/home/eappen/opensn-local-cuda-profile
VENV=$ROOT/venv
DEPS_BUILD=$ROOT/build-deps
DEPS=$ROOT/deps
BUILD=$ROOT/build-opensn
CUDA_ROOT=/usr/local/cuda-13.0
CC=/usr/bin/clang-19
CXX=/usr/bin/clang++-19

test ! -e "$ROOT" || { echo "Refusing to reuse $ROOT" >&2; exit 1; }
unset CMAKE_PREFIX_PATH LD_LIBRARY_PATH PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="$CUDA_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install pybind11 numpy jinja2 matplotlib ninja
MPICC="$(command -v mpicc)" python -m pip install --no-binary=mpi4py mpi4py
python -m pip freeze --all > "$ROOT/python-packages.txt"

cmake -G Ninja -S "$REPO/tools/dependencies" -B "$DEPS_BUILD" \
  -DCMAKE_INSTALL_PREFIX="$DEPS" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DMPI_C_COMPILER="$(command -v mpicc)" \
  -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DPython3_EXECUTABLE="$VENV/bin/python" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DJOBS=1 \
  -DOPENSN_FORCE_DEPENDENCY_REBUILD=ON \
  -DOPENSN_CALIPER_GPU_BACKEND=CUDA
cmake --build "$DEPS_BUILD" --parallel 1

source "$DEPS/bin/set_opensn_env.sh"
cmake -G Ninja -S "$REPO" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Native \
  -DCMAKE_PREFIX_PATH="$DEPS" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DMPI_C_COMPILER="$(command -v mpicc)" \
  -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -Dcaliper_DIR="$DEPS/share/cmake/caliper" \
  -DOPENSN_WITH_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPython3_EXECUTABLE="$VENV/bin/python"
cmake --build "$BUILD" --parallel "$(nproc)"

ldd "$BUILD/python/opensn" | grep -E 'libopensn|libcaliper|libpetsc|libhdf5|libvtk|not found'
```

`-DJOBS=1` is required for a genuinely serial dependency build: the dependency
driver's `ExternalProject` recipes use their own `JOBS` value, so Ninja's outer
`--parallel 1` alone is not sufficient. Keep `python-packages.txt` with the
study; use a hash-pinned requirements file when reproducing it later.

The dependency build validates MPI plus the requested NVTX and CUPTI features.
At study creation, the runner resolves `libcaliper` with `ldd` on the selected
OpenSn binary, finds that library's own prefix and `caliper-config.h`, and probes
that prefix's `cali-query` when present. It never infers capabilities from a
different `cali-query` on `PATH`.

Nsight Systems automatically adds `nvtx` and sets
`CALI_SERVICES_ENABLE=nvtx` only inside ranks selected for profiling when the
linked Caliper has its NVTX service. With an older CPU-only Caliper it records
the degradation and safely traces CUDA+MPI.
Use `--nvtx` to require the feature or `--no-nvtx` to disable it. Caliper CUDA
activity reports similarly require detected CUPTI support.

## Dry run

A dry run collects filesystem, Git, build, tool, and CPU provenance but does not
query or launch a GPU and does not launch MPI:

```bash
python profiling/cbcd/run.py benchmark \
  --dry-run \
  --binary "$BUILD/python/opensn" \
  --ranks 1,2,4 \
  --policies hardware,resource-aware \
  --warmups 1 --trials 5 \
  --schedule-seed 20260821 \
  --pe-map 1=8,2=4,4=2
```

The printed directory contains all planned commands. This is the recommended
first check after changing launch options. `--binary` is always required; the
runner never silently selects an older build tree.

## Uninstrumented benchmark

Run only when the GPU is otherwise idle. Commands are strictly sequential.
Trial blocks randomize rank order with the recorded seed. Within each rank, the
two policies run as an adjacent pair and alternate AB/BA order across blocks,
preventing policy order from being confounded with thermal or clock drift. Omit
`--schedule-seed` to generate and record one automatically.

```bash
python profiling/cbcd/run.py benchmark \
  --binary "$BUILD/python/opensn" \
  --label update-2 \
  --ranks 1,2,4 \
  --policies hardware,resource-aware \
  --warmups 1 --trials 5 \
  --schedule-seed 20260821 \
  --pe-map 1=8,2=4,4=2 \
  --timeout 120
```

`resource-aware` is the fair CPU-allocation diagnostic. `hardware` reproduces
the historical worker policy but can heavily oversubscribe the eight local
physical cores at multiple ranks. Never merge the two policies' samples. A
positive fixed worker experiment can be requested with `--workers N`, but then
exactly one policy must be selected. Results are labeled `fixed-workers-N`, not
as a policy A/B comparison, because the override supersedes both policies'
worker limits. The verbose scheduler record supplies the actual worker count.

The first valid run at each MPI rank count establishes that rank's entry in a
schema-v2 `signature.json`. This is deliberately rank-indexed: cycle/lagged-flux
structure can change the exact WGS iteration count between 1, 2, and 4 ranks.
All subsequent runs at the same rank count must match exact total and lagged
unknown counts, contiguous WGS sequence and convergence status, final WGS
iteration/count, maximum-value signature, configuration, scheduler policy, one
timing record, and one completion marker. Default maximum tolerances are
`atol=1e-10`, `rtol=1e-6` and are configurable. A supplied benchmark reference
must contain every requested rank entry and match the input plus data hashes.

`summary.json` and `summary.csv` report median, MAD, Q1, Q3, IQR, minimum, and
maximum for average sweep time, grind time, and wall time. Speedup and rank
efficiency fields are explicitly named `same_gpu_*`.

## Profile modes

Standalone profiles require the schema-v2 `signature.json` from a successful
uninstrumented benchmark, including every rank being profiled. Profiler timings
are never used as performance acceptance measurements.

Caliper runtime, PMPI, and—when CUPTI is detected—CUDA activity reports. The
default `--mode auto` selects this capability-aware set; `--mode all` requires
CUPTI and fails rather than silently omitting a requested report:

```bash
SIG=/path/to/benchmark-study/signature.json

python profiling/cbcd/run.py caliper \
  --binary "$BUILD/python/opensn" --reference "$SIG" \
  --ranks 1,2,4 --mode auto --pe-map 1=8,2=4,4=2
```

Low-overhead Nsight Systems trace of rank 0 only. CPU sampling and context-switch
collection are unconditionally disabled because this machine has
`perf_event_paranoid=4`:

```bash
python profiling/cbcd/run.py nsys \
  --binary "$BUILD/python/opensn" --reference "$SIG" \
  --ranks 1,2,4 --rank-mode rank0 --pe-map 1=8,2=4,4=2
```

`rank0` leaves all other ranks uninstrumented and sets
`NSYS_MPI_STORE_TEAMS_PER_RANK=1`. Use `--rank-mode all` for one focused
cross-rank imbalance trace. Add `--gpu-metrics` only for an otherwise idle GPU;
the wrapper enables device metrics in exactly one node-local profiler instance,
even in all-rank mode. Each `.nsys-rep` is summarized into CSV for CUDA kernels,
CUDA APIs, memory operations, kernel execution latency, MPI events, and MPI
message sizes for multi-rank runs. Use `--no-stats` only when postprocessing
must be deferred.
When NVTX forwarding is active, region and GPU-projected NVTX summaries are
generated as well.

One-rank Nsight Compute microprofile after skipping an approximately one-WGS
warmup of 32 matching launches:

```bash
python profiling/cbcd/run.py ncu \
  --binary "$BUILD/python/opensn" --reference "$SIG" \
  --launch-skip 32 --launch-count 1 --timeout 180
```

NCU replay is intentionally unavailable at 2/4 ranks. Replaying one rank's
kernel while peers share the GPU can manufacture waits and invalid measurements.

A bounded one-rank Compute Sanitizer diagnostic is separate from all timings:

```bash
python profiling/cbcd/run.py sanitizer \
  --binary "$BUILD/python/opensn" --reference "$SIG" \
  --tool memcheck --launch-skip 32 --launch-count 4 --timeout 180
```

`racecheck`, `synccheck`, and `initcheck` are also available. Sanitizer failures
use a nonzero exit code and preserve all logs.

## Provenance and artifacts

Every study records:

- exact 40-character Git revision, porcelain status, tracked binary-diff hash,
  and a dirty-state hash that includes untracked file contents;
- binary and input SHA-256, sizes and mtimes;
- every resolved DSO in the executable's `ldd` closure, including its real path,
  SHA-256, ELF build ID and RPATH/RUNPATH; the run fails on unresolved libraries
  or when `libopensn`/fresh-prefix dependencies escape their expected roots;
- hashes for the graphite cross-section file (and every repeated
  `--workload-asset` supplied for a custom input);
- CMake cache SHA-256 plus relevant compiler/build/CUDA/Caliper entries;
- linked Caliper library, prefix, version, MPI/NVTX/CUPTI feature evidence;
- MPI, CUDA, Nsight, Compute Sanitizer, perf, and Python tool probes;
- GPU identity/driver/memory/compute capability plus initial utilization,
  memory use, power, SM clock, and temperature for real runs;
- hostname, platform, logical CPUs, physical cores, process affinity, relevant
  environment, MPI binding/mapping, policy, PE, and worker override;
- exact command argument vectors, scoped environment overrides, explicitly
  scrubbed environment names, exit status, timeout status, wall time, logs,
  validation data, profiler artifacts, and selected-GPU state/active contexts
  immediately before and after each OpenSn command.

The runner removes inherited `OPENSN_CBCD_NUM_WORKERS` and every `CALI_*`
variable before each command. It then injects only requested settings (for
example, the NVTX service during an Nsight Systems run). This prevents ambient
Caliper CUPTI collection from double tracing Nsight or contaminating benchmark
timings.

`manifest.json`, `runs.csv`, and `state.json` are rewritten atomically after
each transition. The whole hidden working directory is atomically renamed to its
final timestamped name only when the study reaches a terminal state.

Before interpreting results, verify sustained idle GPU utilization, stable
thermal/clocks, no other CUDA contexts, correct rank binding, matching numerical
signatures, and the same worker policy. MPS is deliberately not managed by this
tool; any MPS experiment must be launched and labeled separately.

## Tests

```bash
python -m py_compile profiling/cbcd/run.py
python -m unittest discover -s profiling/cbcd/tests -v
```
