# Repeated CBC baseline runs

`baseline_study.py` uses the standard unstructured scaling input with CBC,
single-angle aggregation, 448 directions, and 64 groups. The defaults are 17
separate problem solves and 16 Richardson iterations per solve. The iteration
limit is intentional for this timing workload and does not establish convergence.
The mesh and cross sections are reused across trials. Each solver and problem
is function-scoped so the previous trial releases its ownership before the next
problem is constructed. Cycles and angular-flux saving are disabled.

On this branch, add `--fuse-worker-launches 1` to `prepare --gpu` to select
combined worker launches. The default is `0`, the original launch path. The
selection is saved in `input.json` and restored explicitly by `run` after it
clears inherited profiling settings. Use a separate case with `0` for a control
using the same binary. All ranks must use the same selection. Neither mode
enables profiling. Saved angular flux would select the original launch path,
so it remains disabled in these inputs.

Run `build` inside a compute-node allocation. It creates a separate Native build
using the compiler, accelerator architecture, dependencies, Python executable,
and native flags from an existing CMake cache. It does not install Python packages
or rebuild dependencies. Use a production source revision without memory-tracing
hooks, and an existing non-instrumented dependency stack.

```sh
python tools/developer/baseline_study.py build \
  --reuse-build /path/to/existing-build --build /path/to/new-build
python tools/developer/baseline_study.py prepare \
  --mesh /path/to/cube.msh --output /path/to/new-case --gpu
python tools/developer/baseline_study.py run \
  --case /path/to/new-case --binary /path/to/new-build/python/opensn -- \
  mpirun -np 4
```

Omit `--gpu` for host CBC. Choose the scheduler launcher, rank and thread counts,
and binding for the target system. Supply the same mesh at every node count for
strong scaling and the appropriate node-dependent meshes for weak scaling.
The utility does not request allocations or chain jobs. Follow the site's queue
limits. A time-limited run remains partial and is not automatically retried.

The run command disables Caliper measurement channels and removes inherited
profiling injection variables. Caliper remains a required linked dependency,
and existing annotation calls remain in the executable. No profiler, memory
sampler, allocator tuning, or extra OpenSn timing instrumentation is enabled.
Standard OpenSn sweep-time output and rank-zero trial markers are retained.

Each case records its source revision, input and mesh checksums, binary checksum,
launch command, relevant environment, standard output, standard error, and exit
status. `SUCCESS` requires a zero process exit and one timing report per completed
trial. If an allocation kills the driver, output logs remain but exit-status and
summary files may be absent. Check scheduler accounting as well.

`trials.json` includes every completed trial. For the standard repeated-study
summary, discard trial 1 as warm-up and calculate statistics from trials 2--17.
Do not treat a partial trial or missing timing report as a sample. Keep partial
campaigns identified when comparing results.

Validation of the helper:

```sh
python tools/developer/test_baseline_study.py
```
