# Independent-process CBCD scaling studies

`cbcd_study.py` runs fixed-work device CBC strong- and weak-scaling trials.
It is shared by the baseline and memory-profiling branches. It does not change
the transport implementation. Site-specific toolchain paths, scheduler settings,
allocation requests, and transfer commands belong in an external launcher.

## Measurements

The default order is baseline, Caliper+MPI, PMPI, and rocprof, with three trials
per mode. Every trial launches a fresh MPI process set, builds one problem, and
executes one solver. There is no loop that reconstructs problems in one process.
This removes cross-trial allocator retention from the measurements, but it does
not measure long-lived application memory behavior or establish that no leak exists.

One Native build is sufficient for all four modes. The existing Caliper
integration is enabled at runtime only for its two modes. rocprof wraps rank zero
externally. There are no added compile-time counters or separate instrumented
transport kernels. Baseline disables Caliper services, custom CBCD diagnostics,
memory diagnostics, injected profiling libraries, and allocator overrides.

| Mode | Purpose | Required output |
| --- | --- | --- |
| `baseline` | Uninstrumented sweep and grind times | Application output and completion markers |
| `caliper-mpi` | Inclusive application regions and MPI timing | `mpi-regions.txt` |
| `pmpi` | MPI operation summary through Caliper's PMPI service | `mpi.txt` |
| `rocprof` | Rank-zero HIP calls, kernels, copies, and allocations | Four nonempty recursive trace types |

PMPI is a Caliper `mpi-report`, not a second MPI implementation or an additional
preloaded library. rocprof is rank-zero tracing, not a full-rank critical-path
measurement. Tracing perturbs execution. Compare scaling with baseline results,
and use profiles to investigate causes rather than mixing their times with baseline.
Dedicated `cbcd-metrics`, a separate plain Caliper pass, host memory sampling,
destructor logging, and hardware-counter passes are omitted. Use the existing
memory-study tools separately when investigating retained memory. ROCm allocation
traces do not replace host RSS or allocator measurements.

## Problem and interpretation

The input adapts `tools/scaling/lib/unstructured.py`. It uses 64 groups, 448
directions, single-angle aggregation, P0 scattering, the existing cross sections,
and the same isotropic incident source on `xmin`. CBC, GPU execution,
`allow_cycles=False`, and `save_angular_flux=False` are explicit. The default
Richardson limit is 16 iterations with absolute tolerance `1.0e-18`.
These are fixed-work measurements, not converged transport solutions.

Strong scaling reuses one mesh at every node count. Weak scaling selects a mesh
for each count from the configuration. For approximately constant local work,
report actual cell and unknown counts, not just mesh divisors. Rounded mesh sizes
can make weak-scaling time ratios differ from work-normalized efficiency.
Sweep times are in seconds and grind times are in ns per reported global unknown.
The runner checks their normalization and requires zero lagged unknowns.

Within a case, later trials must match the first baseline's group-0 and group-63
scalar-flux maxima to `1.0e-10` relative or `1.0e-12` absolute tolerance. Printed
residual histories must agree to `2.0e-5` relative or `1.0e-12` absolute tolerance.
The residual tolerance accommodates the log's seven significant digits and
floating-point reduction variation. These checks detect changed work or gross
numerical changes. They are not an independent accuracy or conservation test.
The runner also requires the full fixed-iteration history, exactly one sweep
summary, successful process exit, and application completion. Early convergence
is rejected because it would change the measured work.

The three trials are retained individually. There is no unreported warm-up trial
and no automatic removal of the first sample. All modes for one case run on the
same allocation if they fit. A resumed allocation may use different nodes.
Allocation IDs, hostnames, CPU affinity, and device masks remain in the records
so that these samples can be distinguished during analysis.

## Configuration and build

`prepare --config CONFIG --root RESULTS` creates a new results directory and an
immutable manifest. The configuration contains these required fields:

- `nodes`, `ranks_per_node`, and `threads`
- `meshes`, with `strong` and `weak` dictionaries mapping node-count strings to paths
- `launcher`, an argument list with `{nodes}`, `{ranks}`, `{threads}`, and `{seconds}` substitutions
- `reuse_build`, the existing dependency-compatible CMake build
- `build_root`, a new directory outside the results tree

Optional fields are `modes`, `trials`, `iterations`, `require_one_gpu`,
`rocprof_launcher_options`, and `cancel`. Defaults are the four modes above,
three trials, and 16 iterations. `cancel` is a scheduler argument list with a
`{jobid}` placeholder. The launcher must publish MPI rank identities through
Flux, Open MPI, or PMI environment variables. The site adapter must preserve
one-device-per-rank binding and provide its existing runtime environment.

`build --root RESULTS` reuses compiler, MPI, Python, dependency, GPU architecture,
and Native flags from the recorded CMake cache. It does not install dependencies
or Python packages. It creates a separate Native build and records the configure
command, resulting cache, executable hash, and shared-library hashes. A failed
build can be retried with the same command. A completed build is verified rather
than overwritten. Run this command on an allocated compute node, not a login node.

Source revision, helpers, meshes, input files, and cross sections are hashed.
Changed source, inputs, or binaries require a new campaign. A site launcher should
also retain its environment setup, loaded modules, linked libraries, and profiler
version. Local untracked files are not a substitute for committing the study tools
before preparing a remote worktree.

## Run, resume, and inspect

Inside an existing allocation, the site adapter invokes:

```sh
python tools/developer/cbcd_study.py run --root RESULTS \
  --kind strong --nodes N --seconds REMAINING_SECONDS --allocation ALLOCATION_ID
```

Use `--kind weak` for weak scaling. A per-kind, per-node-count lock prevents
duplicate runs of the same case while allowing distinct batch cases to run
concurrently. The driver reserves 60 seconds for cleanup and avoids
starting another trial if the remaining time is insufficient based on completed
trials of that mode. A hard timeout cancels the recorded scheduler step and reaps
the local launcher. The enclosing allocation remains the final resource boundary.

Insufficient time returns status 3. A failed trial stops the case. Reissue the
same site command manually to request another allocation and resume. There is no
automatic allocation retry, background submission, or node-count chain.

```sh
python tools/developer/cbcd_study.py status --root RESULTS
```

Status revalidates each success against its manifest, input, exit code, numerical
output, placement records, and profile artifacts. It can also inspect a downloaded
results directory without access to the original binary or cluster environment.

```text
RESULTS/
  manifest.json
  inputs/
  builds/native/{configure.json,CMakeCache.txt,build.json}
  allocations/                         # written by the site adapter
  results/{strong,weak}/nodes-N/MODE/trial-T/attempt-UTC-UUID/
    input.py
    launch.json
    placement/rank-R.json
    stdout.txt
    stderr.txt
    exit_code.txt
    result.json
    SUCCESS                            # only after all checks pass
    FAILED                             # on a diagnosed failed attempt
    mpi-regions.txt, mpi.txt, or rocprof/ # mode-dependent
```

A killed driver can leave an attempt without either marker. It is incomplete
and will not be skipped on resume. All attempts remain intact. No resume command
deletes or rewrites old results. Transfer the entire results directory, including
failed attempts and allocation logs. Build trees and mesh caches remain outside
that directory. Keep the destination on the external results volume and use a
resumable copy without `--delete`.

## Baseline batch campaigns

Prepare a separate campaign with `modes: ["baseline"]` and `trials: 17`.
The site adapter builds on an allocated compute node and writes `batch.json`:

```json
{
  "max_nodes": 256,
  "assets": {"/path/to/allocated-launcher": "SHA256"},
  "submit": ["flux", "batch", "-q", "pbatch", "-N", "{nodes}",
             "-n", "{ranks}", "--exclusive", "-t", "1h",
             "--output={allocation}/stdout.txt", "--error={allocation}/stderr.txt",
             "--wrap", "/path/to/allocated-launcher", "{root}",
             "{kind}", "{nodes}", "{allocation}"]
}
```

Paths, bank, GPU mode and node limit are site configuration, not portable
defaults. Include environment files in `assets`. Set `max_nodes` to the approved
site limit, not the machine size. The allocated launcher restores the recorded
environment and invokes `run` with the scheduler's actual time remaining.

```sh
python tools/developer/cbcd_study_batch.py --root RESULTS
python tools/developer/cbcd_study_batch.py --root RESULTS --kind strong --nodes 4 --retry
```

The first command submits one job per kind and permitted node count. Each job
runs 17 trials sequentially, each with fresh MPI processes. Different cases
can run concurrently. Repeating submission does not duplicate recorded jobs.
An explicit retry requires all previous submissions of that case to be inactive.
If submission was interrupted before its job ID was recorded, the tool stops
instead of risking duplicate submission. Inspect `submit.stdout`, `submit.stderr`
and the scheduler to resolve it. Never remove an unresolved record to retry blindly.

Seventeen trials share one allocation when they fit within the requested hour.
Setup, mesh loading and process startup count toward this limit. A partial job
retains its trials and can be resumed, but the combined samples then span
allocations. Use a new campaign if all 17 samples must share one allocation.
No automatic resubmission or extension of walltime is performed.

## Validation

```sh
python tools/developer/test_cbcd_study.py
python tools/developer/test_cbcd_study_batch.py
flake8 --jobs 1 tools/developer/cbcd_study.py tools/developer/study_build.py \
  tools/developer/cbcd_study_batch.py tools/developer/test_cbcd_study.py \
  tools/developer/test_cbcd_study_batch.py
```

The tests cover independent process lifetimes, incomplete outputs, failed-attempt
preservation, resume, numerical signatures, placement, timeout cleanup, locking,
recursive ROCm traces, environment isolation, and build provenance. Test the site
adapter with a small allocation before treating a full campaign as complete.
