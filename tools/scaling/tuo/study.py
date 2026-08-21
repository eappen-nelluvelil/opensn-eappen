#!/usr/bin/env python3

"""Prepare, submit, collect, and compare CBCD studies on Tuolumne."""

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCALING_DIR = HERE.parent
XS = SCALING_DIR / "lib/xs_168g.xs"
TEMPLATE = HERE / "transport.py.in"
PROFILE_WRAPPER = HERE / "profile_rank.zsh"

DEFAULT_NODES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WEAK_DIVISORS = dict(zip(DEFAULT_NODES, (6, 8, 10, 12, 15, 19, 25, 31, 39)))
PROFILE_NAMES = (
    "baseline",
    "caliper",
    "pmpi",
    "caliper-rocm",
    "rocprof",
    "hpctoolkit",
    "omniperf",
)
DEFAULT_PROFILES = ("baseline", "caliper", "pmpi")
SCALAR_FLUX_GROUPS = (0, 63)

FLOAT = r"[0-9.eE+-]+"
SWEEP_TIME_RE = re.compile(rf"avg_sweep_time\s*=\s*({FLOAT})\s*s")
UNKNOWNS_RE = re.compile(rf"(?<!lagged_)\bunknowns\s*=\s*({FLOAT})")
LAGGED_RE = re.compile(rf"\blagged_unknowns\s*=\s*({FLOAT})")
WGS_FINAL_RE = re.compile(
    r"WGS groups .* final, status\s*=\s*([^,]+), iterations\s*=\s*(\d+)"
)
WGS_RESIDUAL_RE = re.compile(
    rf"WGS groups .* iteration\s*=\s*\d+, residual\s*=\s*({FLOAT})"
)
CBCD_WORKERS_RE = re.compile(r"CBCD scheduler:.*\bworkers=(\d+)\b")
WALL_RE = re.compile(rf"wall_seconds=({FLOAT})")
RSS_RE = re.compile(r"launcher_max_rss_kb=(\d+)")
FINISHED_RE = re.compile(r"OpenSn finished execution\.")
SCALAR_FLUX_MAX_RE = re.compile(
    rf"^OPENSN_TUO_SCALAR_FLUX_MAX group=(\d+) value=({FLOAT})$", re.MULTILINE
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def quote(value):
    return shlex.quote(str(value))


def executable(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"not an executable file: {path}")
    return path


def positive_integer(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def nonnegative_float(value):
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a nonnegative number") from error
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("expected a nonnegative finite number")
    return result


def positive_float(value):
    result = nonnegative_float(value)
    if result == 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return result


def parse_nodes(value):
    try:
        nodes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("nodes must be comma-separated integers") from error
    if not nodes or nodes[0] <= 0:
        raise argparse.ArgumentTypeError("nodes must be positive")
    return nodes


def parse_choices(value, choices, name):
    selected = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    invalid = set(selected) - set(choices)
    if not selected or invalid:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated values from {','.join(choices)}"
        )
    return selected


def parse_kinds(value):
    return parse_choices(value, ("strong", "weak"), "kinds")


def parse_profiles(value):
    return parse_choices(value, PROFILE_NAMES, "profiles")


def write_executable(path, content):
    path.write_text(content)
    path.chmod(0o700)


def write_rows(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def required_meshes(mesh_dir, divisors):
    mesh_dir = mesh_dir.expanduser().resolve()
    meshes = {
        divisor: mesh_dir / f"cube-d{divisor}.msh" for divisor in sorted(divisors)
    }
    missing = [path for path in meshes.values() if not path.is_file()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise RuntimeError(
            "The study uses existing meshes and never generates them. "
            f"Missing mesh file(s):\n  {formatted}"
        )
    return meshes


def write_input(path, mesh, max_iterations, save_angular_flux):
    content = TEMPLATE.read_text()
    replacements = {
        "@MESH@": repr(str(mesh)),
        "@XS@": repr(str(XS.resolve())),
        "@MAX_ITERATIONS@": str(max_iterations),
        "@SAVE_ANGULAR_FLUX@": repr(save_angular_flux),
    }
    for token, value in replacements.items():
        if token not in content:
            raise RuntimeError(f"input template is missing token {token}")
        content = content.replace(token, value)
    if re.search(r"@[A-Z_]+@", content):
        raise RuntimeError("unresolved token remains in generated input")
    path.write_text(content)


def flux_header(label, nodes, tasks, queue, bank, time_limit, stdout, stderr):
    values = {
        "label": label[:48],
        "queue": queue,
        "bank": bank or "",
        "time limit": time_limit,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    token = re.compile(r"[A-Za-z0-9_./:{}+=,@%-]+")
    invalid = [
        name for name, value in values.items() if value and not token.fullmatch(value)
    ]
    if invalid:
        raise RuntimeError("unsafe Flux directive value(s): " + ", ".join(invalid))
    bank_line = f"#flux: -B {bank}\n" if bank else ""
    return f"""#!/bin/zsh
#flux: --job-name={label[:48]}
#flux: -N {nodes}
#flux: -n {tasks}
#flux: -q {queue}
{bank_line}#flux: --exclusive
#flux: -t {time_limit}
#flux: --output={stdout}
#flux: --error={stderr}

set -euo pipefail
"""


def runtime_environment(args):
    workers = "unset OPENSN_CBCD_NUM_WORKERS\n"
    if args.cbcd_workers is not None:
        workers = f"export OPENSN_CBCD_NUM_WORKERS={args.cbcd_workers}\n"
    return f"""source {quote(args.environment)}
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
export OPENSN_CBCD_WORKER_POLICY={quote(args.worker_policy)}
{workers}unset OPENSN_NUM_THREADS OMP_NUM_THREADS
"""


def run_directory_setup(result_root, metadata):
    metadata_lines = "\n".join(f"  print -- {quote(line)}" for line in metadata)
    return f"""result_root={quote(result_root)}
job_tag=${{FLUX_JOB_ID:-allocation}}
job_tag=${{job_tag//\\//_}}
started=$(date -u +%Y%m%dT%H%M%SZ)
result="$result_root/run-$job_tag-$started-$$"
mkdir -p -- "$result"
completed=0

finish_run()
{{
  local exit_code=$?
  trap - EXIT INT TERM
  if (( ! completed )); then
    print -- "$exit_code" >| "$result/job_exit_code.txt"
    touch "$result/FAILED"
  fi
  exit "$exit_code"
}}
trap finish_run EXIT INT TERM

{{
  print -- "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{metadata_lines}
  print -- "flux_job_id=${{FLUX_JOB_ID:-unset}}"
}} >| "$result/metadata.txt"
"""


def scaling_job(args, study, kind, nodes, input_path):
    ranks = nodes * 4
    result_root = study / "results" / kind / f"nodes-{nodes}"
    scheduler = study / "scheduler"
    header = flux_header(
        f"{args.label}-{kind[0]}-{nodes}-{args.worker_policy}",
        nodes,
        ranks,
        args.queue,
        args.bank,
        args.time_limit,
        scheduler / f"{kind}-{nodes}-{{{{id}}}}.out",
        scheduler / f"{kind}-{nodes}-{{{{id}}}}.err",
    )
    setup = run_directory_setup(
        result_root,
        (
            f"binary={args.binary}",
            f"input={input_path}",
            f"nodes={nodes}",
            f"ranks={ranks}",
            "gpus_per_rank=1",
            "gpu_mode=SPX",
            f"worker_policy={args.worker_policy}",
            f"requested_cbcd_workers={args.cbcd_workers or 'policy-derived'}",
        ),
    )
    return header + runtime_environment(args) + f"""
binary={quote(args.binary)}
input={quote(input_path)}
{setup}
for trial_number in {{1..{args.repetitions}}}; do
  trial="$result/trial-$trial_number"
  mkdir -p -- "$trial"
  set +e
  /usr/bin/time \\
    -f 'wall_seconds=%e launcher_max_rss_kb=%M' \\
    -o "$trial/time.txt" \\
    flux run -N {nodes} -n {ranks} --exclusive -o exit-on-error \\
      "$binary" --verbose 1 -i "$input" \\
      > "$trial/stdout.txt" 2> "$trial/stderr.txt"
  exit_code=$?
  set -e
  print -- "$exit_code" >| "$trial/exit_code.txt"
  if (( exit_code != 0 )) ||
     ! grep -q 'OpenSn finished execution\\.' "$trial/stdout.txt" ||
     ! grep -q 'WGS groups .* final, status' "$trial/stdout.txt" ||
     ! grep -q 'CBCD scheduler:.*workers=' "$trial/stdout.txt" ||
     ! grep -q 'avg_sweep_time' "$trial/stdout.txt" ||
     [[ $(grep -Ec '^OPENSN_TUO_SCALAR_FLUX_MAX group=' "$trial/stdout.txt") -ne 2 ]]; then
    touch "$trial/FAILED"
    exit $(( exit_code == 0 ? 1 : exit_code ))
  fi
  touch "$trial/SUCCESS"
done
print -- 0 >| "$result/job_exit_code.txt"
touch "$result/SUCCESS"
completed=1
trap - EXIT INT TERM
"""


def profile_command(profile, nodes, ranks):
    launch = f"flux run -N {nodes} -n {ranks} --exclusive -o exit-on-error"
    if profile == "baseline":
        return "", f'{launch} "$binary" --verbose 1 -i "$input"'
    if profile == "caliper":
        command = (
            f'{launch} "$binary" --verbose 1 --caliper="runtime-report('
            'output=\\"$result/profile.txt\\",aggregate_across_ranks,'
            'calc.inclusive,print.metadata,order_by_time,max_column_width=180,region.count)" '
            '-i "$input"'
        )
        return "", command
    if profile == "pmpi":
        command = (
            f'{launch} "$binary" --verbose 1 '
            '--caliper="mpi-report(output=\\"$result/mpi.txt\\")" -i "$input"'
        )
        return "", command
    if profile == "caliper-rocm":
        command = (
            f'{launch} "$binary" --verbose 1 --caliper="rocm-activity-report('
            'output=\\"$result/rocm.txt\\",aggregate_across_ranks,show_kernels)" '
            '-i "$input"'
        )
        return "", command
    if profile == "rocprof":
        setup = (
            "export OPENSN_PROFILE_MODE=rocprof\n"
            'export OPENSN_PROFILE_BINARY="$binary"\n'
            'export OPENSN_PROFILE_INPUT="$input"\n'
            'export OPENSN_PROFILE_OUTPUT="$result"'
        )
        return setup, f"{launch} {quote(PROFILE_WRAPPER)}"
    if profile == "hpctoolkit":
        setup = "module load hpctoolkit"
        command = (
            f'{launch} hpcrun -o "$result/measurements" -e CPUTIME@5000 '
            '-e gpu=rocm "$binary" --verbose 1 -i "$input"'
        )
        return setup, command
    if profile == "omniperf":
        setup = 'module load omniperf\ncd "$result"'
        command = (
            "flux run -N 1 -n 1 --exclusive -o exit-on-error "
            "omniperf profile --name cbcd --no-roof -b SQ TCC TCP "
            '-k SweepKernel -- "$binary" --verbose 1 -i "$input"'
        )
        return setup, command
    raise ValueError(profile)


def profile_job(args, study, profile, nodes, input_path):
    ranks = 1 if profile == "omniperf" else nodes * 4
    result_root = study / "results" / profile / f"nodes-{nodes}"
    scheduler = study / "scheduler"
    header = flux_header(
        f"{args.label}-{profile}-{nodes}",
        nodes,
        ranks,
        args.queue,
        args.bank,
        args.time_limit,
        scheduler / f"{profile}-{nodes}-{{{{id}}}}.out",
        scheduler / f"{profile}-{nodes}-{{{{id}}}}.err",
    )
    run_setup = run_directory_setup(
        result_root,
        (
            f"binary={args.binary}",
            f"input={input_path}",
            f"profile={profile}",
            f"nodes={nodes}",
            f"ranks={ranks}",
            f"worker_policy={args.worker_policy}",
        ),
    )
    profiler_setup, command = profile_command(profile, nodes, ranks)
    artifact = ":"
    if profile == "caliper":
        artifact = '[[ -s "$result/profile.txt" ]]'
    elif profile == "pmpi":
        artifact = '[[ -s "$result/mpi.txt" ]]'
    elif profile == "caliper-rocm":
        artifact = '[[ -s "$result/rocm.txt" ]]'
    elif profile == "rocprof":
        artifact = 'find "$result" -path "*/rank-*/*" -type f -print -quit | grep -q .'
    elif profile == "hpctoolkit":
        artifact = 'find "$result/measurements" -type f -print -quit | grep -q .'
    elif profile == "omniperf":
        artifact = 'find "$result/workloads/cbcd" -type f -print -quit | grep -q .'
    return header + runtime_environment(args) + f"""
binary={quote(args.binary)}
input={quote(input_path)}
{run_setup}
{profiler_setup}
set +e
/usr/bin/time \\
  -f 'wall_seconds=%e launcher_max_rss_kb=%M' \\
  -o "$result/time.txt" \\
  {command} > "$result/stdout.txt" 2> "$result/stderr.txt"
exit_code=$?
set -e
print -- "$exit_code" >| "$result/exit_code.txt"
if (( exit_code != 0 )) ||
   ! grep -q 'OpenSn finished execution\\.' "$result/stdout.txt" ||
   ! grep -q 'WGS groups .* final, status' "$result/stdout.txt" ||
   ! grep -q 'WGS groups .* iteration.*residual' "$result/stdout.txt" ||
   ! grep -q 'CBCD scheduler:.*workers=' "$result/stdout.txt" ||
   ! grep -q 'avg_sweep_time' "$result/stdout.txt" ||
   [[ $(grep -Ec '^OPENSN_TUO_SCALAR_FLUX_MAX group=' "$result/stdout.txt") -ne 2 ]] ||
   ! {artifact}; then
  exit $(( exit_code == 0 ? 1 : exit_code ))
fi
print -- 0 >| "$result/job_exit_code.txt"
touch "$result/SUCCESS"
completed=1
trap - EXIT INT TERM
"""


def prepare_directory(output, refresh=False):
    study = output.expanduser().resolve()
    if study.exists() and any(study.iterdir()):
        if not refresh:
            raise RuntimeError(f"study directory is not empty: {study}")
        if not (study / "manifest.json").is_file():
            raise RuntimeError(
                f"cannot refresh a directory without a study manifest: {study}"
            )
    for name in ("inputs", "jobs", "results", "scheduler"):
        (study / name).mkdir(parents=True, exist_ok=True)
    return study


def write_submit_wrapper(study, environment, queue):
    if queue == "pdebug":
        content = """#!/bin/zsh
set -euo pipefail
print -u2 'pdebug is interactive-only; run a generated job inside flux alloc.'
exit 2
"""
    else:
        content = f"""#!/bin/zsh
set -euo pipefail
source {quote(environment)}
exec python {quote(Path(__file__).resolve())} submit --study {quote(study)} "$@"
"""
    write_executable(study / "submit.zsh", content)


def validate_prepare(args, nodes):
    args.environment = args.environment.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    if not args.environment.is_file():
        raise RuntimeError(f"environment script does not exist: {args.environment}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise RuntimeError(f"OpenSn executable does not exist: {args.binary}")
    if args.queue == "pdebug" and max(nodes) > 8:
        raise RuntimeError("Tuolumne pdebug studies cannot exceed 8 nodes per user")


def prepare(args):
    validate_prepare(args, args.nodes)
    unsupported = set(args.nodes) - set(WEAK_DIVISORS)
    if "weak" in args.kinds and unsupported:
        raise RuntimeError(f"no weak-scaling mesh is defined for nodes {sorted(unsupported)}")
    divisors = set()
    if "strong" in args.kinds:
        divisors.add(args.strong_divisor)
    if "weak" in args.kinds:
        divisors.update(WEAK_DIVISORS[node] for node in args.nodes)
    meshes = required_meshes(args.mesh_dir, divisors)
    study = prepare_directory(args.output, args.refresh)

    cases = []
    for kind in args.kinds:
        for nodes in args.nodes:
            divisor = args.strong_divisor if kind == "strong" else WEAK_DIVISORS[nodes]
            input_path = study / "inputs" / f"{kind}-{nodes}.py"
            write_input(input_path, meshes[divisor], args.max_iterations, args.save_angular_flux)
            job_path = study / "jobs" / f"{kind}-{nodes}.zsh"
            write_executable(job_path, scaling_job(args, study, kind, nodes, input_path))
            cases.append(
                {
                    "id": f"{kind}-{nodes}",
                    "kind": kind,
                    "nodes": nodes,
                    "ranks": nodes * 4,
                    "divisor": divisor,
                    "mesh": str(meshes[divisor]),
                    "input": str(input_path),
                    "job": str(job_path),
                }
            )

    write_submit_wrapper(study, args.environment, args.queue)
    record = {
        "format": 1,
        "generated_at_utc": utc_now(),
        "machine": "tuolumne",
        "type": "scaling",
        "label": args.label,
        "binary": str(args.binary),
        "environment": str(args.environment),
        "mesh_dir": str(args.mesh_dir.expanduser().resolve()),
        "nodes": args.nodes,
        "kinds": args.kinds,
        "ranks_per_node": 4,
        "gpus_per_rank": 1,
        "gpu_mode": "SPX",
        "queue": args.queue,
        "bank": args.bank,
        "time_limit": args.time_limit,
        "repetitions": args.repetitions,
        "worker_policy": args.worker_policy,
        "cbcd_workers": args.cbcd_workers,
        "strong_divisor": args.strong_divisor,
        "weak_divisors": {
            str(node): WEAK_DIVISORS[node]
            for node in args.nodes
            if "weak" in args.kinds
        },
        "max_iterations": args.max_iterations,
        "save_angular_flux": args.save_angular_flux,
        "cases": cases,
    }
    (study / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prepared {len(cases)} Flux jobs in {study}")


def prepare_profile(args):
    validate_prepare(args, args.profile_nodes)
    meshes = required_meshes(args.mesh_dir, {args.profile_divisor})
    study = prepare_directory(args.output, args.refresh)
    input_path = study / "inputs" / "profile.py"
    write_input(
        input_path,
        meshes[args.profile_divisor],
        args.max_iterations,
        args.save_angular_flux,
    )

    cases = []
    for profile in args.profiles:
        nodes_values = args.profile_nodes if profile in DEFAULT_PROFILES else (1,)
        for nodes in nodes_values:
            job_path = study / "jobs" / f"{profile}-{nodes}.zsh"
            write_executable(job_path, profile_job(args, study, profile, nodes, input_path))
            cases.append(
                {
                    "id": f"{profile}-{nodes}",
                    "profile": profile,
                    "nodes": nodes,
                    "ranks": 1 if profile == "omniperf" else nodes * 4,
                    "job": str(job_path),
                }
            )

    write_submit_wrapper(study, args.environment, args.queue)
    record = {
        "format": 1,
        "generated_at_utc": utc_now(),
        "machine": "tuolumne",
        "type": "profile",
        "label": args.label,
        "binary": str(args.binary),
        "environment": str(args.environment),
        "mesh_dir": str(args.mesh_dir.expanduser().resolve()),
        "mesh": str(meshes[args.profile_divisor]),
        "queue": args.queue,
        "bank": args.bank,
        "time_limit": args.time_limit,
        "gpu_mode": "SPX",
        "worker_policy": args.worker_policy,
        "cbcd_workers": args.cbcd_workers,
        "profile_divisor": args.profile_divisor,
        "profile_nodes": args.profile_nodes,
        "profiles": args.profiles,
        "max_iterations": args.max_iterations,
        "save_angular_flux": args.save_angular_flux,
        "cases": cases,
    }
    (study / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prepared {len(cases)} profile jobs in {study}")


def load_study(path):
    study = path.expanduser().resolve()
    record_path = study / "manifest.json"
    if not record_path.is_file():
        raise RuntimeError(f"study does not exist: {study}")
    return study, json.loads(record_path.read_text())


def submit(args):
    _, record = load_study(args.study)
    if record["queue"] == "pdebug":
        raise RuntimeError("pdebug is interactive-only; run generated jobs inside flux alloc")
    if record["type"] == "scaling" and args.profiles:
        raise RuntimeError("--profiles cannot select scaling jobs")
    if record["type"] == "profile" and (args.kinds or args.nodes):
        raise RuntimeError("--kinds and --nodes cannot select profile jobs")

    selected = []
    for case in record["cases"]:
        if args.nodes and case.get("nodes") not in args.nodes:
            continue
        if args.kinds and case.get("kind") not in args.kinds:
            continue
        if args.profiles and case.get("profile") not in args.profiles:
            continue
        selected.append(case)
    if not selected:
        raise RuntimeError("the requested filters selected no jobs")
    for case in selected:
        result = subprocess.run(
            ["flux", "batch", case["job"]],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"submitted {case['id']}: {result.stdout.strip()}")
    print(f"Submitted {len(selected)} job(s).")


def read_result(stdout, timing, exit_code, success):
    if not success.is_file():
        raise RuntimeError(f"success marker does not exist: {success}")
    if not exit_code.is_file() or exit_code.read_text().strip() != "0":
        raise RuntimeError(f"nonzero or missing exit code: {exit_code}")
    text = stdout.read_text(errors="replace")
    if not FINISHED_RE.search(text):
        raise RuntimeError(f"OpenSn did not finish cleanly: {stdout}")
    sweep_times = SWEEP_TIME_RE.findall(text)
    unknowns = UNKNOWNS_RE.findall(text)
    lagged = LAGGED_RE.findall(text)
    finals = WGS_FINAL_RE.findall(text)
    residuals = WGS_RESIDUAL_RE.findall(text)
    workers = {int(value) for value in CBCD_WORKERS_RE.findall(text)}
    maxima = {}
    for group_text, value_text in SCALAR_FLUX_MAX_RE.findall(text):
        group = int(group_text)
        if group in maxima:
            raise RuntimeError(f"duplicate scalar-flux maximum for group {group}: {stdout}")
        maxima[group] = float(value_text)
    if (
        not sweep_times
        or not unknowns
        or not finals
        or not residuals
        or len(workers) != 1
        or set(maxima) != set(SCALAR_FLUX_GROUPS)
    ):
        raise RuntimeError(f"required CBCD metrics are missing or inconsistent: {stdout}")
    time_text = timing.read_text(errors="replace")
    wall = WALL_RE.search(time_text)
    rss = RSS_RE.search(time_text)
    if wall is None:
        raise RuntimeError(f"wall time is missing: {timing}")

    status, iterations = finals[-1]
    status = status.strip()
    if any(word in status.lower() for word in ("fail", "diverge", "error")):
        raise RuntimeError(f"unsuccessful WGS status {status}: {stdout}")
    sweep = float(sweep_times[-1])
    unknown_count = float(unknowns[-1])
    lagged_count = float(lagged[-1]) if lagged else 0.0
    residual = float(residuals[-1])
    wall_time = float(wall.group(1))
    values = (sweep, unknown_count, lagged_count, residual, wall_time, *maxima.values())
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"non-finite CBCD metric: {stdout}")
    if (
        sweep <= 0.0
        or unknown_count <= 0.0
        or not unknown_count.is_integer()
        or lagged_count < 0.0
        or not lagged_count.is_integer()
        or wall_time <= 0.0
    ):
        raise RuntimeError(f"invalid CBCD metric: {stdout}")
    result = {
        "avg_sweep_time_s": sweep,
        "unknowns": int(unknown_count),
        "lagged_unknowns": int(lagged_count),
        "wgs_status": status,
        "wgs_iterations": int(iterations),
        "scheduler_workers": next(iter(workers)),
        "final_residual": residual,
        "wall_time_s": wall_time,
        "launcher_max_rss_kb": int(rss.group(1)) if rss else None,
    }
    result.update(
        {f"scalar_flux_max_g{group}": maxima[group] for group in SCALAR_FLUX_GROUPS}
    )
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def spread(values):
    median = statistics.median(values)
    return (
        median,
        statistics.median(abs(value - median) for value in values),
        percentile(values, 0.75) - percentile(values, 0.25),
    )


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["kind"], row["nodes"]), []).append(row)
    metrics = {}
    for key, values in grouped.items():
        samples = (
            [value["avg_sweep_time_s"] / value["unknowns"] * 1.0e9 for value in values]
            if key[0] == "strong"
            else [value["avg_sweep_time_s"] for value in values]
        )
        metrics[key] = statistics.median(samples)

    summary = []
    for kind, nodes in sorted(grouped):
        values = grouped[(kind, nodes)]
        signatures = {
            (
                value["unknowns"],
                value["lagged_unknowns"],
                value["wgs_status"],
                value["wgs_iterations"],
                value["scheduler_workers"],
                *(value[f"scalar_flux_max_g{group}"] for group in SCALAR_FLUX_GROUPS),
            )
            for value in values
        }
        if len(signatures) != 1:
            raise RuntimeError(f"inconsistent numerical signature for {kind}-{nodes}")
        signature = next(iter(signatures))
        samples = (
            [value["avg_sweep_time_s"] / value["unknowns"] * 1.0e9 for value in values]
            if kind == "strong"
            else [value["avg_sweep_time_s"] for value in values]
        )
        metric, metric_mad, metric_iqr = spread(samples)
        sweep, sweep_mad, sweep_iqr = spread(
            [value["avg_sweep_time_s"] for value in values]
        )
        base_nodes = min(
            point_nodes for point_kind, point_nodes in grouped if point_kind == kind
        )
        efficiency = metrics[(kind, base_nodes)] / metric
        if kind == "strong":
            efficiency *= base_nodes / nodes
        row = {
            "kind": kind,
            "nodes": nodes,
            "ranks": nodes * 4,
            "trials": len(values),
            "metric": metric,
            "metric_mad": metric_mad,
            "metric_iqr": metric_iqr,
            "metric_unit": "ns/unknown" if kind == "strong" else "s",
            "efficiency_percent": efficiency * 100.0,
            "median_avg_sweep_time_s": sweep,
            "avg_sweep_time_mad_s": sweep_mad,
            "avg_sweep_time_iqr_s": sweep_iqr,
            "median_unknowns": signature[0],
            "median_lagged_unknowns": signature[1],
            "wgs_status": signature[2],
            "wgs_iterations": signature[3],
            "scheduler_workers": signature[4],
            "median_final_residual": statistics.median(
                value["final_residual"] for value in values
            ),
            "median_wall_time_s": statistics.median(
                value["wall_time_s"] for value in values
            ),
        }
        for index, group in enumerate(SCALAR_FLUX_GROUPS, start=5):
            row[f"scalar_flux_max_g{group}"] = signature[index]
        summary.append(row)
    return summary


def write_summary(path, record, rows):
    lines = [
        f"# {record['label']} Tuolumne scaling results",
        "",
        f"Worker policy: `{record['worker_policy']}`",
        "",
        (
            "| Kind | Nodes | Trials | Metric | MAD | IQR | Unit | Efficiency | "
            "Sweep (s) | Iterations | Workers | Residual | Flux max g0 | Flux max g63 |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['kind']} | {row['nodes']} | {row['trials']} "
            f"| {row['metric']:.8g} | {row['metric_mad']:.3g} | {row['metric_iqr']:.3g} "
            f"| {row['metric_unit']} | {row['efficiency_percent']:.2f}% "
            f"| {row['median_avg_sweep_time_s']:.8g} | {row['wgs_iterations']} "
            f"| {row['scheduler_workers']} | {row['median_final_residual']:.8g} "
            f"| {row['scalar_flux_max_g0']:.17e} | {row['scalar_flux_max_g63']:.17e} |"
        )
    path.write_text("\n".join(lines) + "\n")


def monotonic_failures(rows, tolerance):
    strong = sorted(
        (row for row in rows if row["kind"] == "strong"),
        key=lambda row: row["nodes"],
    )
    failures = []
    for previous, current in zip(strong, strong[1:]):
        limit = previous["median_avg_sweep_time_s"] * (1.0 + tolerance)
        if current["median_avg_sweep_time_s"] > limit:
            failures.append(
                f"strong sweep time increased from {previous['nodes']} to "
                f"{current['nodes']} nodes"
            )
    return failures


def collect(args):
    study, record = load_study(args.study)
    if record["type"] != "scaling":
        raise RuntimeError("collect requires a scaling study")
    rows = []
    missing = []
    failed = []
    invalid = []
    for case in record["cases"]:
        case_rows = []
        result_root = study / "results" / case["kind"] / f"nodes-{case['nodes']}"
        for run in sorted(result_root.glob("run-*")):
            if not run.is_dir():
                continue
            if not (run / "SUCCESS").is_file():
                if (run / "FAILED").is_file():
                    failed.append(str(run))
                continue
            for trial in sorted(run.glob("trial-*")):
                try:
                    values = read_result(
                        trial / "stdout.txt",
                        trial / "time.txt",
                        trial / "exit_code.txt",
                        trial / "SUCCESS",
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    invalid.append(f"{trial}: {error}")
                    continue
                case_rows.append(
                    {
                        "kind": case["kind"],
                        "nodes": case["nodes"],
                        "ranks": case["ranks"],
                        "run": run.name,
                        "trial": trial.name,
                        **values,
                    }
                )
        if len(case_rows) < record["repetitions"]:
            missing.append(
                f"{case['id']}: found {len(case_rows)} of "
                f"{record['repetitions']} required trial(s)"
            )
        rows.extend(case_rows)

    complete = not missing and not invalid
    if not complete and not args.allow_incomplete:
        raise RuntimeError(
            f"collection is incomplete ({len(missing)} missing case(s), "
            f"{len(invalid)} invalid trial(s)); use --allow-incomplete for diagnosis"
        )
    write_rows(study / "results.csv", rows)
    summary = summarize(rows)
    write_rows(study / "summary.csv", summary)
    write_summary(study / "summary.md", record, summary)
    monotonic = monotonic_failures(summary, args.monotonic_tolerance)
    (study / "collection.json").write_text(
        json.dumps(
            {
                "collected_at_utc": utc_now(),
                "complete": complete,
                "missing": missing,
                "failed_runs": failed,
                "invalid_trials": invalid,
                "monotonic_failures": monotonic,
            },
            indent=2,
        )
        + "\n"
    )
    if args.require_monotonic and monotonic:
        raise RuntimeError("; ".join(monotonic))
    print(f"Collected {len(rows)} successful trial(s) in {study}")


def read_summary(study):
    study, record = load_study(study)
    path = study / "summary.csv"
    if not path.is_file():
        raise RuntimeError(f"study has not been collected: {study}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return record, rows


def plot_series(output, title, series):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for kind in ("strong", "weak"):
        selected_series = []
        for label, rows in series:
            selected = sorted(
                (row for row in rows if row["kind"] == kind),
                key=lambda row: int(row["nodes"]),
            )
            if selected:
                selected_series.append((label, selected))
        if not selected_series:
            continue
        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        for label, rows in selected_series:
            axis.plot(
                [int(row["nodes"]) for row in rows],
                [float(row["metric"]) for row in rows],
                marker="o",
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Nodes (4 MPI ranks/node, 1 MI300A/rank)")
        axis.set_ylabel(
            "Average sweep time per unknown (ns)"
            if kind == "strong"
            else "Average sweep time (s)"
        )
        axis.set_title(f"{title} {kind} scaling")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"{kind}.pdf")
        plt.close(figure)


def compare(args):
    baseline_record, baseline = read_summary(args.baseline)
    candidate_record, candidate = read_summary(args.candidate)
    compatible_fields = (
        "nodes",
        "kinds",
        "ranks_per_node",
        "gpus_per_rank",
        "gpu_mode",
        "repetitions",
        "strong_divisor",
        "weak_divisors",
        "max_iterations",
        "save_angular_flux",
    )
    different = [
        field
        for field in compatible_fields
        if baseline_record.get(field) != candidate_record.get(field)
    ]
    if different:
        raise RuntimeError("studies use different settings: " + ", ".join(different))

    baseline_by_point = {
        (row["kind"], int(row["nodes"])): row for row in baseline
    }
    candidate_by_point = {
        (row["kind"], int(row["nodes"])): row for row in candidate
    }
    if set(baseline_by_point) != set(candidate_by_point):
        raise RuntimeError("baseline and candidate contain different scaling points")

    comparisons = []
    failures = []
    for key in sorted(baseline_by_point):
        base = baseline_by_point[key]
        cand = candidate_by_point[key]
        for field in (
            "median_unknowns",
            "median_lagged_unknowns",
            "wgs_status",
            "wgs_iterations",
        ):
            if base[field] != cand[field]:
                failures.append(f"{key}: {field} differs")
        if not math.isclose(
            float(base["median_final_residual"]),
            float(cand["median_final_residual"]),
            rel_tol=args.residual_rtol,
            abs_tol=args.residual_atol,
        ):
            failures.append(f"{key}: final residual differs")
        for group in SCALAR_FLUX_GROUPS:
            field = f"scalar_flux_max_g{group}"
            if not math.isclose(
                float(base[field]),
                float(cand[field]),
                rel_tol=args.scalar_flux_rtol,
                abs_tol=args.scalar_flux_atol,
            ):
                failures.append(f"{key}: scalar-flux maximum for group {group} differs")
        base_sweep = float(base["median_avg_sweep_time_s"])
        candidate_sweep = float(cand["median_avg_sweep_time_s"])
        ratio = candidate_sweep / base_sweep
        if ratio > args.max_slowdown:
            failures.append(
                f"{key}: slowdown {ratio:.4f} exceeds {args.max_slowdown:.4f}"
            )
        comparisons.append(
            {
                "kind": key[0],
                "nodes": key[1],
                "baseline_sweep_s": base_sweep,
                "candidate_sweep_s": candidate_sweep,
                "candidate_over_baseline": ratio,
                "baseline_workers": base["scheduler_workers"],
                "candidate_workers": cand["scheduler_workers"],
            }
        )

    candidate_numeric = [
        {
            "kind": row["kind"],
            "nodes": int(row["nodes"]),
            "median_avg_sweep_time_s": float(row["median_avg_sweep_time_s"]),
        }
        for row in candidate
    ]
    failures.extend(monotonic_failures(candidate_numeric, args.monotonic_tolerance))

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "comparison.csv", comparisons)
    lines = [
        "# Tuolumne CBCD scaling comparison",
        "",
        f"Baseline: `{baseline_record['label']}` ({baseline_record['worker_policy']})",
        "",
        f"Candidate: `{candidate_record['label']}` ({candidate_record['worker_policy']})",
        "",
        "| Study | Nodes | Baseline sweep (s) | Candidate sweep (s) | Candidate / baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['kind']} | {row['nodes']} | {row['baseline_sweep_s']:.8g} "
            f"| {row['candidate_sweep_s']:.8g} | {row['candidate_over_baseline']:.4f} |"
        )
    if failures:
        lines.extend(("", "## Failures", "", *(f"- {failure}" for failure in failures)))
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    plot_series(
        output,
        "Tuolumne CBCD",
        (
            (baseline_record["label"], baseline),
            (candidate_record["label"], candidate),
        ),
    )
    if failures:
        raise RuntimeError("comparison failed: " + "; ".join(failures))
    print(f"Comparison written to {output}")


def collect_profile(args):
    study, record = load_study(args.study)
    if record["type"] != "profile":
        raise RuntimeError("collect-profile requires a profile study")
    rows = []
    for case in record["cases"]:
        root = study / "results" / case["profile"] / f"nodes-{case['nodes']}"
        for run in sorted(root.glob("run-*")):
            if not run.is_dir():
                continue
            try:
                values = read_result(
                    run / "stdout.txt",
                    run / "time.txt",
                    run / "exit_code.txt",
                    run / "SUCCESS",
                )
            except (OSError, RuntimeError, ValueError):
                values = None
            rows.append(
                {
                    "profile": case["profile"],
                    "nodes": case["nodes"],
                    "ranks": case["ranks"],
                    "run": run.name,
                    "completed": values is not None,
                    "avg_sweep_time_s": values["avg_sweep_time_s"] if values else None,
                    "wgs_iterations": values["wgs_iterations"] if values else None,
                    "scheduler_workers": values["scheduler_workers"] if values else None,
                    "scalar_flux_max_g0": values["scalar_flux_max_g0"] if values else None,
                    "scalar_flux_max_g63": values["scalar_flux_max_g63"] if values else None,
                    "result_directory": str(run),
                }
            )
    write_rows(study / "profile-summary.csv", rows)
    lines = [
        f"# {record['label']} Tuolumne profile inventory",
        "",
        (
            "| Profile | Nodes | Ranks | Run | Completed | Sweep (s) | "
            "Iterations | Workers | Flux max g0 | Flux max g63 | Result directory |"
        ),
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        sweep = (
            "n/a"
            if row["avg_sweep_time_s"] is None
            else f"{row['avg_sweep_time_s']:.8g}"
        )
        lines.append(
            f"| {row['profile']} | {row['nodes']} | {row['ranks']} | {row['run']} "
            f"| {row['completed']} | {sweep} | {row['wgs_iterations'] or 'n/a'} "
            f"| {row['scheduler_workers'] or 'n/a'} "
            f"| {row['scalar_flux_max_g0'] if row['completed'] else 'n/a'} "
            f"| {row['scalar_flux_max_g63'] if row['completed'] else 'n/a'} "
            f"| `{row['result_directory']}` |"
        )
    (study / "profile-summary.md").write_text("\n".join(lines) + "\n")
    print(f"Profile inventory written to {study}")


def add_common_prepare_arguments(command, profile=False):
    command.add_argument("--binary", type=executable, required=True)
    command.add_argument("--environment", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--mesh-dir", type=Path, required=True)
    command.add_argument("--label", required=True)
    command.add_argument("--queue", default="pbatch")
    command.add_argument("--bank")
    command.add_argument("--time-limit", default="6h" if profile else "4h")
    command.add_argument(
        "--refresh",
        action="store_true",
        help="replace generated inputs and jobs while preserving result directories",
    )
    command.add_argument(
        "--worker-policy",
        choices=("hardware", "resource-aware"),
        default="hardware",
    )
    command.add_argument("--cbcd-workers", type=positive_integer)
    command.add_argument(
        "--save-angular-flux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain the full angular-flux vector",
    )


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    prepare_command = commands.add_parser("prepare", help="prepare scaling jobs")
    add_common_prepare_arguments(prepare_command)
    prepare_command.add_argument("--nodes", type=parse_nodes, default=DEFAULT_NODES)
    prepare_command.add_argument(
        "--kinds", type=parse_kinds, default=("strong", "weak")
    )
    prepare_command.add_argument("--repetitions", type=positive_integer, default=3)
    prepare_command.add_argument("--strong-divisor", type=positive_integer, default=39)
    prepare_command.add_argument("--max-iterations", type=positive_integer, default=10)
    prepare_command.set_defaults(function=prepare)

    profile_prepare = commands.add_parser(
        "prepare-profile", help="prepare independent profile jobs"
    )
    add_common_prepare_arguments(profile_prepare, profile=True)
    profile_prepare.add_argument("--profile-divisor", type=positive_integer, default=15)
    profile_prepare.add_argument("--profile-nodes", type=parse_nodes, default=(1, 2, 4))
    profile_prepare.add_argument(
        "--profiles", type=parse_profiles, default=DEFAULT_PROFILES
    )
    profile_prepare.add_argument("--max-iterations", type=positive_integer, default=2)
    profile_prepare.set_defaults(function=prepare_profile)

    submit_command = commands.add_parser("submit", help="submit selected jobs")
    submit_command.add_argument("--study", type=Path, required=True)
    submit_command.add_argument("--nodes", type=parse_nodes)
    submit_command.add_argument("--kinds", type=parse_kinds)
    submit_command.add_argument("--profiles", type=parse_profiles)
    submit_command.set_defaults(function=submit)

    collect_command = commands.add_parser("collect", help="collect scaling results")
    collect_command.add_argument("--study", type=Path, required=True)
    collect_command.add_argument("--allow-incomplete", action="store_true")
    collect_command.add_argument("--require-monotonic", action="store_true")
    collect_command.add_argument(
        "--monotonic-tolerance", type=nonnegative_float, default=0.0
    )
    collect_command.set_defaults(function=collect)

    compare_command = commands.add_parser("compare", help="compare collected studies")
    compare_command.add_argument("--baseline", type=Path, required=True)
    compare_command.add_argument("--candidate", type=Path, required=True)
    compare_command.add_argument("--output", type=Path, required=True)
    compare_command.add_argument("--max-slowdown", type=positive_float, default=1.03)
    compare_command.add_argument(
        "--monotonic-tolerance", type=nonnegative_float, default=0.0
    )
    compare_command.add_argument(
        "--residual-rtol", type=nonnegative_float, default=1.0e-6
    )
    compare_command.add_argument(
        "--residual-atol", type=nonnegative_float, default=1.0e-12
    )
    compare_command.add_argument(
        "--scalar-flux-rtol", type=nonnegative_float, default=1.0e-10
    )
    compare_command.add_argument(
        "--scalar-flux-atol", type=nonnegative_float, default=1.0e-12
    )
    compare_command.set_defaults(function=compare)

    profile_collect = commands.add_parser(
        "collect-profile", help="collect profile results"
    )
    profile_collect.add_argument("--study", type=Path, required=True)
    profile_collect.set_defaults(function=collect_profile)
    return top


def main():
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
