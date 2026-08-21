#!/usr/bin/env python3

"""Prepare, submit, collect, and compare revision-pinned Tuolumne studies."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCALING_DIR = HERE.parent
GEOMETRY = SCALING_DIR / "lib/cube.geo"
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
WGS_RESIDUAL_RE = re.compile(rf"WGS groups .* iteration\s*=\s*\d+, residual\s*=\s*({FLOAT})")
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dynamic_library_closure(binary):
    result = subprocess.run(
        ["ldd", str(binary)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ldd failed for {binary}: {result.stdout.strip()}")
    libraries = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "=> not found" in stripped:
            raise RuntimeError(f"unresolved dynamic library for {binary}: {stripped}")
        candidate = stripped.split("=>", 1)[-1].split(" (", 1)[0].strip()
        if not candidate.startswith("/"):
            continue
        path = Path(candidate).resolve()
        if not path.is_file():
            raise RuntimeError(f"dynamic library does not exist: {path}")
        libraries[str(path)] = sha256(path)
    if not any(
        re.match(r"libopensn\.so(?:\.|$)", Path(path).name) for path in libraries
    ):
        raise RuntimeError("OpenSn dynamic closure does not contain libopensn.so")
    return [
        {"path": path, "sha256": digest}
        for path, digest in sorted(libraries.items())
    ]


def json_fingerprint(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_executable(path, content):
    path.write_text(content)
    path.chmod(0o700)


def executable(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"not an executable file: {path}")
    return path


def positive_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def nonnegative_float(value):
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a nonnegative finite number") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a nonnegative finite number")
    return parsed


def positive_float(value):
    parsed = nonnegative_float(value)
    if parsed == 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return parsed


def parse_nodes(value):
    try:
        nodes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("nodes must be comma-separated integers") from error
    if not nodes or nodes[0] <= 0:
        raise argparse.ArgumentTypeError("nodes must be positive")
    return nodes


def parse_choices(value, choices, name):
    selected = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
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


def command_output(command):
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def git_revision(source):
    revision = command_output(["git", "-C", str(source), "rev-parse", "--verify", "HEAD^{commit}"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"source does not resolve to a full Git SHA: {source}")
    status = command_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"]
    )
    if status:
        raise RuntimeError(f"source tree is not clean: {source}\n{status}")
    return revision


def build_manifest_path(binary):
    try:
        return binary.parents[1] / "tuo-build-manifest.json"
    except IndexError as error:
        raise RuntimeError(f"cannot infer build directory from {binary}") from error


def validate_build(source, binary, environment):
    source = source.expanduser().resolve()
    binary = binary.expanduser().resolve()
    environment = environment.expanduser().resolve()
    revision = git_revision(source)
    path = build_manifest_path(binary)
    if not path.is_file():
        raise RuntimeError(f"build manifest does not exist: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise RuntimeError(f"unsupported Tuo build manifest schema in {path}")
    expected = {
        "revision": revision,
        "source": str(source),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "environment": str(environment),
        "environment_sha256": sha256(environment),
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if not manifest.get("source_clean"):
        mismatches.append("source_clean")
    caliper_cache = manifest.get("caliper_features", {}).get("cache_features", {})
    mpi_feature = caliper_cache.get("WITH_MPI", {})
    if mpi_feature.get("type") != "BOOL" or mpi_feature.get("value") != "ON":
        mismatches.append("Caliper WITH_MPI")
    recipes = set(
        manifest.get("caliper_features", {}).get("available_config_recipes", [])
    )
    if not {"runtime-report", "mpi-report"}.issubset(recipes):
        mismatches.append("Caliper runtime-report/mpi-report recipes")
    for path_key, hash_key in (
        ("dependencies_manifest", "dependencies_manifest_sha256"),
        ("caliper_features_manifest", "caliper_features_sha256"),
        ("cmake_cache", "cmake_cache_sha256"),
    ):
        external = Path(manifest.get(path_key, ""))
        expected_hash = manifest.get(hash_key)
        if not external.is_file() or not expected_hash or sha256(external) != expected_hash:
            mismatches.append(path_key)
    dependencies_path = Path(manifest.get("dependencies_manifest", ""))
    if dependencies_path.is_file():
        dependencies = json.loads(dependencies_path.read_text())
        for path_key, hash_key in (
            ("bootstrap", "bootstrap_sha256"),
            ("dependency_driver", "dependency_driver_sha256"),
            ("cmake_cache", "cmake_cache_sha256"),
        ):
            external = Path(dependencies.get(path_key, ""))
            expected_hash = dependencies.get(hash_key)
            if (
                not external.is_file()
                or not expected_hash
                or sha256(external) != expected_hash
            ):
                mismatches.append(f"dependencies.{path_key}")
    closure = manifest.get("linked_library_closure", [])
    if not closure or dynamic_library_closure(binary) != closure:
        mismatches.append("linked_library_closure")
    for entry in closure:
        path = Path(entry.get("path", ""))
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            mismatches.append(f"linked_library:{path}")
    boost = manifest.get("boost", {})
    if boost.get("version") != "1.86.0":
        mismatches.append("boost.version")
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("version_config", "version_config_sha256"),
    ):
        external = Path(boost.get(path_key, ""))
        if not external.is_file() or sha256(external) != boost.get(hash_key):
            mismatches.append(f"boost.{path_key}")
    if mismatches:
        raise RuntimeError(
            f"build manifest {path} does not match: {', '.join(sorted(set(mismatches)))}"
        )
    return revision, manifest, path


def gmsh_identity(gmsh):
    version = command_output([str(gmsh), "--version"]).splitlines()[-1].strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError(f"cannot determine a pinned Gmsh version from: {version}")
    return version


def mesh_for(cache, gmsh, gmsh_version, geometry, divisor):
    cache.mkdir(parents=True, exist_ok=True)
    identity = {
        "geometry_sha256": sha256(geometry),
        "gmsh_version": gmsh_version,
        "divisor": divisor,
    }
    key = json_fingerprint(identity)[:16]
    mesh = cache / f"cube-d{divisor}-{key}.msh"
    record = mesh.with_suffix(".msh.json")
    if mesh.is_file() and record.is_file():
        metadata = json.loads(record.read_text())
        if metadata.get("identity") == identity and metadata.get("mesh_sha256") == sha256(mesh):
            return mesh
        raise RuntimeError(f"mesh cache validation failed: {mesh}")
    if mesh.exists():
        raise RuntimeError(f"mesh cache entry is incomplete or not a file: {mesh}")
    if record.exists():
        raise RuntimeError(f"mesh cache record exists without its mesh: {record}")
    temporary = cache / f".{mesh.name}.tmp-{os.getpid()}"
    subprocess.run(
        [
            str(gmsh),
            "-3",
            "-v",
            "0",
            "-setnumber",
            "divisor",
            str(divisor),
            "-o",
            str(temporary),
            str(geometry),
        ],
        check=True,
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Gmsh did not create a valid mesh: {temporary}")
    temporary.replace(mesh)
    atomic_json(
        record,
        {
            "identity": identity,
            "mesh": str(mesh),
            "mesh_sha256": sha256(mesh),
        },
    )
    return mesh


def write_input(path, template, mesh, xs_path, max_iterations, save_angular_flux):
    content = template.read_text()
    replacements = {
        "@MESH@": repr(str(mesh)),
        "@XS@": repr(str(xs_path)),
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


def flux_header(label, nodes, tasks, queue, bank, time_limit, stdout, stderr, gpu_mode):
    if not label or not queue or not time_limit or nodes <= 0 or tasks <= 0:
        raise RuntimeError("Flux directives require nonempty tokens and positive sizes")
    values = {
        "label": label[:48],
        "queue": queue,
        "bank": bank or "",
        "time limit": time_limit,
        "GPU mode": gpu_mode,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    token_pattern = re.compile(r"[A-Za-z0-9_./:{}+=,@%-]+")
    invalid = [
        name
        for name, value in values.items()
        if value and token_pattern.fullmatch(value) is None
    ]
    if invalid:
        raise RuntimeError(
            "unsafe Flux directive token(s): " + ", ".join(sorted(invalid))
        )
    bank_line = f"#flux: -B {bank}\n" if bank else ""
    return f"""#!/bin/zsh
#flux: --job-name={label[:48]}
#flux: -N {nodes}
#flux: -n {tasks}
#flux: -q {queue}
{bank_line}#flux: --exclusive
#flux: --amd-gpumode={gpu_mode}
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


def checksum_guard(path, expected, label):
    return f"""[[ $(sha256sum {quote(path)} | awk '{{print $1}}') == {quote(expected)} ]] || {{
  print -u2 {quote(label + ' hash mismatch')}
  exit 1
}}
"""


def dynamic_closure_guard(build_manifest):
    return f"""python - {quote(build_manifest)} "$binary" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

manifest_path, binary = sys.argv[1:]
expected = json.loads(Path(manifest_path).read_text())["linked_library_closure"]
result = subprocess.run(
    ["ldd", binary],
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if result.returncode != 0 or "=> not found" in result.stdout:
    raise SystemExit(f"runtime dynamic-library resolution failed: {{result.stdout}}")
actual = {{}}
for line in result.stdout.splitlines():
    candidate = line.strip().split("=>", 1)[-1].split(" (", 1)[0].strip()
    if not candidate.startswith("/"):
        continue
    path = Path(candidate).resolve()
    if not path.is_file():
        raise SystemExit(f"runtime dynamic library is missing: {{path}}")
    actual[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
observed = [
    {{"path": path, "sha256": digest}}
    for path, digest in sorted(actual.items())
]
if observed != expected:
    raise SystemExit("runtime dynamic-library closure differs from the build manifest")
PY
"""


def binding_preflight(nodes, ranks):
    if ranks % nodes:
        raise ValueError("rank count must be divisible by node count")
    ranks_per_node = ranks // nodes
    command = (
        "'printf \"%s %s %s %s %s %s\\n\" \"$(hostname -s)\" "
        "\"${FLUX_TASK_RANK:?}\" \"${FLUX_TASK_LOCAL_ID:?}\" "
        "\"${ROCR_VISIBLE_DEVICES:-unset}\" "
        "\"$(sed -n \"s/^Cpus_allowed_list:[[:space:]]*//p\" /proc/self/status)\" "
        "\"${OMP_NUM_THREADS:-unset}\"'"
    )
    return f"""flux run -N {nodes} -n {ranks} --exclusive -o exit-on-error \\
  zsh -c {command} > "$result/binding.txt"
awk -v expected={ranks} -v expected_hosts={nodes} -v per_node={ranks_per_node} '
  function claim_cpus(host, list, ranges, count, index, bounds, cpu, key) {{
    count=0
    split(list, ranges, ",")
    for (index in ranges) {{
      split(ranges[index], bounds, "-")
      if (bounds[1] !~ /^[0-9]+$/ ||
          (bounds[2] != "" && bounds[2] !~ /^[0-9]+$/)) {{ bad=1; continue }}
      if (bounds[2] == "") bounds[2]=bounds[1]
      if (bounds[2] < bounds[1]) {{ bad=1; continue }}
      for (cpu=bounds[1]; cpu<=bounds[2]; cpu++) {{
        key=host SUBSEP cpu
        if (claimed_cpu[key]++) bad=1
        count++
      }}
    }}
    return count
  }}
  NF != 6 || $2 !~ /^[0-9]+$/ || $2 >= expected ||
    $3 !~ /^[0-9]+$/ || $3 >= per_node ||
    $4 !~ /^[0-9]+$/ || $4 >= 4 || $6 != 21 {{ bad=1 }}
  {{
    total++
    per_host[$1]++
    if (seen_rank[$2]++) bad=1
    local_key=$1 SUBSEP $3
    gpu_key=$1 SUBSEP $4
    if (seen_local[local_key]++ || seen_gpu[gpu_key]++) bad=1
    if (claim_cpus($1, $5) != 21) bad=1
  }}
  END {{
    if (total != expected) bad=1
    for (host in per_host) {{ hosts++; if (per_host[host] != per_node) bad=1 }}
    if (hosts != expected_hosts) bad=1
    exit bad
  }}' "$result/binding.txt"
"""


def job_prologue(args, study, case_name, nodes, ranks, input_path, hashes):
    result_root = study / "results" / case_name
    checks = "".join(
        checksum_guard(path, expected, label)
        for label, path, expected in (
            ("binary", args.binary, hashes["binary"]),
            ("environment", args.environment, hashes["environment"]),
            ("input", input_path, hashes["input"]),
            ("mesh", hashes["mesh_path"], hashes["mesh"]),
            ("cross section", hashes["xs_path"], hashes["xs"]),
            (
                "staged build manifest",
                hashes["build_manifest_path"],
                hashes["build_manifest"],
            ),
        )
    )
    checks += "".join(
        checksum_guard(path, expected, label)
        for label, path, expected in hashes["build_provenance"]
    )
    return runtime_environment(args) + f"""
binary={quote(args.binary)}
input={quote(input_path)}
result_root={quote(result_root)}
: ${{FLUX_JOB_ID:?FLUX_JOB_ID is required}}
job_tag=${{FLUX_JOB_ID//\\//_}}
result="$result_root/job-$job_tag"
[[ ! -e $result ]] || {{
  print -u2 "Result attempt already exists: $result"
  exit 1
}}
mkdir -p -- "$result"
touch "$result/RUNNING"
active_trial=

finish_failed()
{{
  local status=$?
  (( status != 0 )) || status=1
  trap - EXIT INT TERM
  if [[ -n $active_trial && -e $active_trial/RUNNING ]]; then
    mv -- "$active_trial/RUNNING" "$active_trial/FAILED"
  fi
  if [[ -e $result/RUNNING ]]; then
    print -- "$status" >| "$result/job_exit_code.txt"
    mv -- "$result/RUNNING" "$result/FAILED"
  fi
  exit "$status"
}}
trap finish_failed EXIT INT TERM

{checks}
{dynamic_closure_guard(hashes['build_manifest_path'])}
{{
  print -- "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  print -- {quote('revision=' + args.revision)}
  print -- {quote('binary=' + str(args.binary))}
  print -- {quote('binary_sha256=' + hashes['binary'])}
  print -- {quote('input=' + str(input_path))}
  print -- {quote('nodes=' + str(nodes))}
  print -- {quote('ranks=' + str(ranks))}
  print -- 'gpus_per_rank=1'
  print -- 'user_cpu_cores_per_rank=21'
  print -- {quote('gpu_mode=' + args.gpu_mode)}
  print -- {quote('worker_policy=' + args.worker_policy)}
  print -- {quote('requested_cbcd_workers=' + str(args.cbcd_workers or 'policy-derived'))}
  print -- "flux_job_id=$FLUX_JOB_ID"
  print -- 'modules_begin'
  module -t list 2>&1 || true
  print -- 'modules_end'
  flux version || true
}} >| "$result/metadata.txt"

{binding_preflight(nodes, ranks)}
"""


def scaling_job(args, study, kind, nodes, input_path, hashes):
    ranks = nodes * 4
    scheduler = study / "scheduler"
    case_name = f"{kind}/nodes-{nodes}"
    header = flux_header(
        f"{args.label}-{kind[0]}-{nodes}",
        nodes,
        ranks,
        args.queue,
        args.bank,
        args.time_limit,
        scheduler / f"{kind}-{nodes}-{{{{id}}}}.out",
        scheduler / f"{kind}-{nodes}-{{{{id}}}}.err",
        args.gpu_mode,
    )
    prologue = job_prologue(args, study, case_name, nodes, ranks, input_path, hashes)
    return header + prologue + f"""
for trial_number in {{1..{args.repetitions}}}; do
  trial="$result/trial-$trial_number"
  active_trial=$trial
  mkdir -p -- "$trial"
  touch "$trial/RUNNING"
  set +e
  /usr/bin/time \
    -f 'wall_seconds=%e launcher_max_rss_kb=%M' \
    -o "$trial/time.txt" \
    flux run -N {nodes} -n {ranks} --exclusive -o exit-on-error \
      "$binary" --verbose 1 -i "$input" > "$trial/stdout.txt" 2> "$trial/stderr.txt"
  status=$?
  set -e
  print -- "$status" >| "$trial/exit_code.txt"
  if (( status != 0 )) ||
     ! grep -q 'OpenSn finished execution\\.' "$trial/stdout.txt" ||
     ! grep -q 'WGS groups .* final, status' "$trial/stdout.txt" ||
     ! grep -q 'WGS groups .* iteration.*residual' "$trial/stdout.txt" ||
     ! grep -q 'CBCD scheduler:.*workers=' "$trial/stdout.txt" ||
     [[ $(grep -Ec '^OPENSN_TUO_SCALAR_FLUX_MAX group=' \
       "$trial/stdout.txt") -ne 2 ]] ||
     ! grep -q '^OPENSN_TUO_SCALAR_FLUX_MAX group=0 value=' \
       "$trial/stdout.txt" ||
     ! grep -q '^OPENSN_TUO_SCALAR_FLUX_MAX group=63 value=' \
       "$trial/stdout.txt" ||
     grep 'WGS groups .* final, status' "$trial/stdout.txt" |
       tail -n 1 | grep -Eqi 'status[[:space:]]*=[[:space:]]*(fail|diverg|error)' ||
     ! grep -q 'avg_sweep_time' "$trial/stdout.txt"; then
    mv -- "$trial/RUNNING" "$trial/FAILED"
    exit $(( status == 0 ? 1 : status ))
  fi
  mv -- "$trial/RUNNING" "$trial/SUCCESS"
  active_trial=
done
print -- 0 >| "$result/job_exit_code.txt"
mv -- "$result/RUNNING" "$result/SUCCESS"
trap - EXIT INT TERM
"""


def profile_command(profile, nodes, ranks, wrapper):
    launch = f"flux run -N {nodes} -n {ranks} --exclusive -o exit-on-error"
    if profile == "baseline":
        return "", f'{launch} "$binary" --verbose 1 -i "$input"'
    if profile == "caliper":
        command = (
            f'{launch} "$binary" --verbose 1 --caliper="runtime-report('
            'output=\\\"$result/profile.txt\\\",aggregate_across_ranks,'
            'calc.inclusive,print.metadata,order_by_time,max_column_width=180,'
            'region.count)" -i "$input"'
        )
        return "", command
    if profile == "pmpi":
        command = (
            f'{launch} "$binary" --verbose 1 '
            '--caliper="mpi-report(output=\\\"$result/mpi.txt\\\")" -i "$input"'
        )
        return "", command
    if profile == "caliper-rocm":
        command = (
            f'{launch} "$binary" --verbose 1 --caliper="rocm-activity-report('
            'output=\\\"$result/rocm.txt\\\",aggregate_across_ranks,'
            'show_kernels)" -i "$input"'
        )
        return "", command
    if profile == "rocprof":
        setup = (
            'export OPENSN_PROFILE_MODE=rocprof\n'
            'export OPENSN_PROFILE_BINARY="$binary"\n'
            'export OPENSN_PROFILE_INPUT="$input"\n'
            'export OPENSN_PROFILE_OUTPUT="$result"'
        )
        return setup, f"{launch} {quote(wrapper)}"
    if profile == "hpctoolkit":
        command = (
            f'{launch} hpcrun -o "$result/measurements" '
            '-e CPUTIME@5000 -e gpu=rocm "$binary" --verbose 1 -i "$input"'
        )
        return "module load hpctoolkit", command
    if profile == "omniperf":
        setup = 'module load omniperf\ncd "$result"'
        command = (
            "flux run -N 1 -n 1 --exclusive -o exit-on-error "
            "omniperf profile --name cbcd --no-roof -b SQ TCC TCP "
            '-k SweepKernel -- "$binary" --verbose 1 -i "$input"'
        )
        return setup, command
    raise ValueError(profile)


def profile_job(args, profile, nodes, study, input_path, hashes):
    ranks = 1 if profile == "omniperf" else nodes * 4
    scheduler = study / "scheduler"
    case_name = f"{profile}/nodes-{nodes}"
    header = flux_header(
        f"{args.label}-{profile}-{nodes}",
        nodes,
        ranks,
        args.queue,
        args.bank,
        args.time_limit,
        scheduler / f"{profile}-{nodes}-{{{{id}}}}.out",
        scheduler / f"{profile}-{nodes}-{{{{id}}}}.err",
        args.gpu_mode,
    )
    prologue = job_prologue(args, study, case_name, nodes, ranks, input_path, hashes)
    wrapper = study / "assets/profile_rank.zsh"
    setup, command = profile_command(profile, nodes, ranks, wrapper)
    artifact_check = ":"
    if profile == "caliper":
        artifact_check = '[[ -s "$result/profile.txt" ]]'
    elif profile == "pmpi":
        artifact_check = '[[ -s "$result/mpi.txt" ]]'
    elif profile == "caliper-rocm":
        artifact_check = '[[ -s "$result/rocm.txt" ]]'
    elif profile == "rocprof":
        artifact_check = (
            'find "$result" -path "*/rank-*/*" -type f '
            '! -name metadata.txt -print -quit | grep -q .'
        )
    elif profile == "hpctoolkit":
        artifact_check = 'find "$result/measurements" -type f -print -quit | grep -q .'
    elif profile == "omniperf":
        artifact_check = (
            'find "$result/workloads/cbcd" -type f -print -quit | grep -q .'
        )
    return header + prologue + f"""
{setup}
set +e
/usr/bin/time \
  -f 'wall_seconds=%e launcher_max_rss_kb=%M' \
  -o "$result/time.txt" \
  {command} > "$result/stdout.txt" 2> "$result/stderr.txt"
status=$?
set -e
print -- "$status" >| "$result/exit_code.txt"
if (( status != 0 )) ||
   ! grep -q 'OpenSn finished execution\\.' "$result/stdout.txt" ||
   ! grep -q 'WGS groups .* final, status' "$result/stdout.txt" ||
   ! grep -q 'WGS groups .* iteration.*residual' "$result/stdout.txt" ||
   ! grep -q 'CBCD scheduler:.*workers=' "$result/stdout.txt" ||
   [[ $(grep -Ec '^OPENSN_TUO_SCALAR_FLUX_MAX group=' \
     "$result/stdout.txt") -ne 2 ]] ||
   ! grep -q '^OPENSN_TUO_SCALAR_FLUX_MAX group=0 value=' \
     "$result/stdout.txt" ||
   ! grep -q '^OPENSN_TUO_SCALAR_FLUX_MAX group=63 value=' \
     "$result/stdout.txt" ||
   grep 'WGS groups .* final, status' "$result/stdout.txt" |
     tail -n 1 | grep -Eqi 'status[[:space:]]*=[[:space:]]*(fail|diverg|error)' ||
   ! grep -q 'avg_sweep_time' "$result/stdout.txt" ||
   ! {artifact_check}; then
  exit $(( status == 0 ? 1 : status ))
fi
print -- 0 >| "$result/job_exit_code.txt"
mv -- "$result/RUNNING" "$result/SUCCESS"
trap - EXIT INT TERM
"""


def stage_common(args, stage, final_study):
    source = args.source.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    args.environment = args.environment.expanduser().resolve()
    if not args.environment.is_file():
        raise RuntimeError(f"environment does not exist: {args.environment}")
    args.revision, build_manifest, build_path = validate_build(
        source, args.binary, args.environment
    )
    gmsh_version = gmsh_identity(args.gmsh)
    recorded_gmsh = build_manifest.get("gmsh", "").splitlines()[0].strip()
    if recorded_gmsh and recorded_gmsh != gmsh_version:
        raise RuntimeError(
            f"Gmsh version {gmsh_version} differs from build environment {recorded_gmsh}"
        )
    assets = stage / "assets"
    assets.mkdir(parents=True)
    staged = {
        "geometry": assets / "cube.geo",
        "xs": assets / "xs_168g.xs",
        "template": assets / "transport.py.in",
        "study": assets / "study.py",
        "profile_wrapper": assets / "profile_rank.zsh",
        "build_manifest": assets / "tuo-build-manifest.json",
    }
    sources = {
        "geometry": GEOMETRY,
        "xs": XS,
        "template": TEMPLATE,
        "study": Path(__file__).resolve(),
        "profile_wrapper": PROFILE_WRAPPER,
        "build_manifest": build_path,
    }
    for name, destination in staged.items():
        shutil.copy2(sources[name], destination)
    staged["study"].chmod(0o700)
    staged["profile_wrapper"].chmod(0o700)
    final_assets = final_study / "assets"
    hashes = {
        "binary": sha256(args.binary),
        "environment": sha256(args.environment),
        "geometry": sha256(staged["geometry"]),
        "xs": sha256(staged["xs"]),
        "template": sha256(staged["template"]),
        "study": sha256(staged["study"]),
        "profile_wrapper": sha256(staged["profile_wrapper"]),
        "build_manifest": sha256(staged["build_manifest"]),
    }
    build_provenance = [
        (
            label,
            Path(build_manifest[path_key]),
            build_manifest[hash_key],
        )
        for label, path_key, hash_key in (
            (
                "dependencies manifest",
                "dependencies_manifest",
                "dependencies_manifest_sha256",
            ),
            (
                "Caliper feature manifest",
                "caliper_features_manifest",
                "caliper_features_sha256",
            ),
            ("OpenSn CMake cache", "cmake_cache", "cmake_cache_sha256"),
        )
    ]
    dependencies = json.loads(
        Path(build_manifest["dependencies_manifest"]).read_text()
    )
    dependency_recipe = {
        "python_packages": sorted(dependencies["python_packages"]),
        "downloaded_archive_sha256": sorted(
            set(dependencies["downloaded_archives"].values())
        ),
        "caliper_requested": build_manifest["caliper_features"]["requested"],
        "caliper_recipes": build_manifest["caliper_features"][
            "available_config_recipes"
        ],
        "boost_version": build_manifest["boost"]["version"],
        "boost_config_sha256": build_manifest["boost"]["config_sha256"],
        "bootstrap_sha256": dependencies["bootstrap_sha256"],
        "caliper_gpu_backend": dependencies["caliper_gpu_backend"],
        "hip_architecture": build_manifest["hip_architecture"],
        "build_type": build_manifest["build_type"],
        "opensn_cmake_options": {
            "OPENSN_WITH_HIP": "ON",
            "OPENSN_WITH_PYTHON": "ON",
            "OPENSN_WITH_PYTHON_MODULE": "ON",
            "CMAKE_HIP_ARCHITECTURES": "gfx942",
            "CMAKE_BUILD_TYPE": "Native",
        },
        "dependency_dso_sha256": sorted(
            entry["sha256"]
            for entry in build_manifest["linked_library_closure"]
            if not Path(entry["path"]).name.startswith("libopensn.so")
        ),
    }
    performance_environment = {
        "modules": build_manifest["modules"],
        "compiler": build_manifest["compiler"],
        "mpi": build_manifest["mpi"],
        "cmake": build_manifest["cmake"],
        "gmsh": build_manifest["gmsh"],
        "hip_architecture": build_manifest["hip_architecture"],
        "build_type": build_manifest["build_type"],
        "dependency_recipe": dependency_recipe,
    }
    build_provenance.extend(
        (
            label,
            Path(dependencies[path_key]),
            dependencies[hash_key],
        )
        for label, path_key, hash_key in (
            ("Tuo bootstrap", "bootstrap", "bootstrap_sha256"),
            (
                "dependency driver",
                "dependency_driver",
                "dependency_driver_sha256",
            ),
            (
                "dependency CMake cache",
                "cmake_cache",
                "cmake_cache_sha256",
            ),
        )
    )
    build_provenance.extend(
        (
            f"linked library {Path(entry['path']).name}",
            Path(entry["path"]),
            entry["sha256"],
        )
        for entry in build_manifest["linked_library_closure"]
    )
    build_provenance.extend(
        (
            label,
            Path(build_manifest["boost"][path_key]),
            build_manifest["boost"][hash_key],
        )
        for label, path_key, hash_key in (
            ("Boost package config", "config", "config_sha256"),
            (
                "Boost package version config",
                "version_config",
                "version_config_sha256",
            ),
        )
    )
    return {
        "source": source,
        "build_manifest": build_manifest,
        "gmsh_version": gmsh_version,
        "staged": staged,
        "final_assets": final_assets,
        "hashes": hashes,
        "build_provenance": build_provenance,
        "performance_environment": performance_environment,
    }


def copy_mesh(stage, final_study, cached_mesh):
    destination = stage / "meshes" / cached_mesh.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_mesh, destination)
    return destination, final_study / "meshes" / cached_mesh.name


def prepare_directory(output):
    study = output.expanduser().resolve()
    if study.exists():
        raise RuntimeError(f"output path already exists: {study}")
    study.parent.mkdir(parents=True, exist_ok=True)
    stage = study.with_name(f".{study.name}.tmp-{os.getpid()}")
    if stage.exists():
        raise RuntimeError(f"temporary output path already exists: {stage}")
    stage.mkdir()
    for name in ("inputs", "jobs", "results", "scheduler"):
        (stage / name).mkdir()
    return study, stage


def finish_study(stage, study, manifest):
    files = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(stage))] = sha256(path)
    manifest["files"] = files
    atomic_json(stage / "manifest.json", manifest)
    stage.replace(study)
    print(f"Prepared {len(manifest['cases'])} immutable Flux jobs in {study}")


def write_submit_wrapper(stage, study, queue, environment=None):
    if queue == "pdebug":
        script = """#!/bin/zsh
set -euo pipefail
print -u2 'pdebug is interactive-only on Tuolumne; run a generated job inside flux alloc.'
exit 2
"""
    else:
        script = f"""#!/bin/zsh
set -euo pipefail
source {quote(environment)}
exec python {quote(study / 'assets/study.py')} submit --study {quote(study)} "$@"
"""
    write_executable(stage / "submit.zsh", script)


def prepare(args):
    if args.queue == "pdebug" and max(args.nodes) > 16:
        raise RuntimeError("Tuolumne pdebug studies cannot exceed 16 nodes")
    unsupported = set(args.nodes) - set(WEAK_DIVISORS)
    if "weak" in args.kinds and unsupported:
        raise RuntimeError(f"no weak-scaling divisor for nodes {sorted(unsupported)}")
    study, stage = prepare_directory(args.output)
    try:
        context = stage_common(args, stage, study)
        final_assets = context["final_assets"]
        cache = args.mesh_cache.expanduser().resolve()
        divisors = {args.strong_divisor}
        if "weak" in args.kinds:
            divisors.update(WEAK_DIVISORS[node] for node in args.nodes)
        meshes = {}
        for divisor in sorted(divisors):
            cached = mesh_for(
                cache,
                args.gmsh,
                context["gmsh_version"],
                context["staged"]["geometry"],
                divisor,
            )
            staged_mesh, final_mesh = copy_mesh(stage, study, cached)
            meshes[divisor] = (staged_mesh, final_mesh)

        cases = []
        for kind in args.kinds:
            for nodes in args.nodes:
                divisor = args.strong_divisor if kind == "strong" else WEAK_DIVISORS[nodes]
                staged_mesh, final_mesh = meshes[divisor]
                staged_input = stage / "inputs" / f"{kind}-{nodes}.py"
                final_input = study / "inputs" / staged_input.name
                write_input(
                    staged_input,
                    context["staged"]["template"],
                    final_mesh,
                    final_assets / "xs_168g.xs",
                    args.max_iterations,
                    args.save_angular_flux,
                )
                hashes = {
                    "binary": context["hashes"]["binary"],
                    "environment": context["hashes"]["environment"],
                    "input": sha256(staged_input),
                    "mesh": sha256(staged_mesh),
                    "mesh_path": final_mesh,
                    "xs": context["hashes"]["xs"],
                    "xs_path": final_assets / "xs_168g.xs",
                    "build_manifest": context["hashes"]["build_manifest"],
                    "build_manifest_path": final_assets / "tuo-build-manifest.json",
                    "build_provenance": context["build_provenance"],
                }
                staged_job = stage / "jobs" / f"{kind}-{nodes}.zsh"
                final_job = study / "jobs" / staged_job.name
                write_executable(
                    staged_job,
                    scaling_job(args, study, kind, nodes, final_input, hashes),
                )
                cases.append(
                    {
                        "id": f"{kind}-{nodes}",
                        "category": "scaling",
                        "kind": kind,
                        "nodes": nodes,
                        "ranks": nodes * 4,
                        "divisor": divisor,
                        "mesh": str(final_mesh),
                        "mesh_sha256": hashes["mesh"],
                        "input": str(final_input),
                        "input_sha256": hashes["input"],
                        "job": str(final_job),
                    }
                )

        write_submit_wrapper(stage, study, args.queue, args.environment)
        compatibility = {
            "study_type": "scaling",
            "nodes": args.nodes,
            "kinds": args.kinds,
            "ranks_per_node": 4,
            "gpus_per_rank": 1,
            "user_cpu_cores_per_rank": 21,
            "gpu_mode": args.gpu_mode,
            "queue": args.queue,
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
            "repetitions": args.repetitions,
            "trial_policy": "sequential-repetitions-in-one-exclusive-allocation-v1",
            "scalar_flux_groups": SCALAR_FLUX_GROUPS,
            "case_mesh_sha256": {
                case["id"]: case["mesh_sha256"] for case in cases
            },
            "geometry_sha256": context["hashes"]["geometry"],
            "xs_sha256": context["hashes"]["xs"],
            "template_sha256": context["hashes"]["template"],
            "performance_environment": context["performance_environment"],
        }
        manifest = {
            "schema_version": 2,
            "generated_at_utc": utc_now(),
            "machine": "tuolumne",
            "type": "scaling",
            "label": args.label,
            "source": str(context["source"]),
            "revision": args.revision,
            "binary": str(args.binary),
            "binary_sha256": context["hashes"]["binary"],
            "environment": str(args.environment),
            "environment_sha256": context["hashes"]["environment"],
            "build_manifest": context["build_manifest"],
            "gmsh": str(args.gmsh),
            "gmsh_version": context["gmsh_version"],
            "mesh_cache": str(cache),
            "repetitions": args.repetitions,
            "queue": args.queue,
            "bank": args.bank,
            "time_limit": args.time_limit,
            "worker_policy": args.worker_policy,
            "cbcd_workers": args.cbcd_workers,
            "compatibility": compatibility,
            "compatibility_fingerprint": json_fingerprint(compatibility),
            "cases": cases,
        }
        finish_study(stage, study, manifest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare_profile(args):
    if args.queue == "pdebug" and max(args.profile_nodes) > 16:
        raise RuntimeError("Tuolumne pdebug studies cannot exceed 16 nodes")
    study, stage = prepare_directory(args.output)
    try:
        context = stage_common(args, stage, study)
        if "caliper-rocm" in args.profiles:
            cache_features = context["build_manifest"]["caliper_features"][
                "cache_features"
            ]
            if cache_features.get("WITH_ROCPROFILER", {}).get("value") != "ON":
                raise RuntimeError(
                    "caliper-rocm requires a build with Caliper WITH_ROCPROFILER=ON"
                )
            recipes = set(
                context["build_manifest"]["caliper_features"].get(
                    "available_config_recipes", []
                )
            )
            if "rocm-activity-report" not in recipes:
                raise RuntimeError(
                    "caliper-rocm requires the installed rocm-activity-report recipe"
                )
        cache = args.mesh_cache.expanduser().resolve()
        cached = mesh_for(
            cache,
            args.gmsh,
            context["gmsh_version"],
            context["staged"]["geometry"],
            args.profile_divisor,
        )
        staged_mesh, final_mesh = copy_mesh(stage, study, cached)
        staged_input = stage / "inputs/profile.py"
        final_input = study / "inputs/profile.py"
        write_input(
            staged_input,
            context["staged"]["template"],
            final_mesh,
            context["final_assets"] / "xs_168g.xs",
            args.max_iterations,
            args.save_angular_flux,
        )
        hashes = {
            "binary": context["hashes"]["binary"],
            "environment": context["hashes"]["environment"],
            "input": sha256(staged_input),
            "mesh": sha256(staged_mesh),
            "mesh_path": final_mesh,
            "xs": context["hashes"]["xs"],
            "xs_path": context["final_assets"] / "xs_168g.xs",
            "build_manifest": context["hashes"]["build_manifest"],
            "build_manifest_path": context["final_assets"] / "tuo-build-manifest.json",
            "build_provenance": context["build_provenance"],
        }
        cases = []
        for profile in args.profiles:
            profile_nodes = (
                args.profile_nodes if profile in ("baseline", "caliper", "pmpi") else (1,)
            )
            for nodes in profile_nodes:
                staged_job = stage / "jobs" / f"{profile}-{nodes}.zsh"
                final_job = study / "jobs" / staged_job.name
                write_executable(
                    staged_job,
                    profile_job(args, profile, nodes, study, final_input, hashes),
                )
                cases.append(
                    {
                        "id": f"{profile}-{nodes}",
                        "category": "profile",
                        "profile": profile,
                        "nodes": nodes,
                        "ranks": 1 if profile == "omniperf" else nodes * 4,
                        "job": str(final_job),
                    }
                )
        write_submit_wrapper(stage, study, args.queue, args.environment)
        compatibility = {
            "study_type": "profile",
            "profile_nodes": args.profile_nodes,
            "profiles": args.profiles,
            "ranks_per_node": 4,
            "gpus_per_rank": 1,
            "user_cpu_cores_per_rank": 21,
            "gpu_mode": args.gpu_mode,
            "queue": args.queue,
            "worker_policy": args.worker_policy,
            "cbcd_workers": args.cbcd_workers,
            "profile_divisor": args.profile_divisor,
            "max_iterations": args.max_iterations,
            "save_angular_flux": args.save_angular_flux,
            "trial_policy": "single-profile-run-per-exclusive-allocation-v1",
            "scalar_flux_groups": SCALAR_FLUX_GROUPS,
            "case_mesh_sha256": {case["id"]: hashes["mesh"] for case in cases},
            "geometry_sha256": context["hashes"]["geometry"],
            "xs_sha256": context["hashes"]["xs"],
            "template_sha256": context["hashes"]["template"],
            "performance_environment": context["performance_environment"],
        }
        manifest = {
            "schema_version": 2,
            "generated_at_utc": utc_now(),
            "machine": "tuolumne",
            "type": "profile",
            "label": args.label,
            "source": str(context["source"]),
            "revision": args.revision,
            "binary": str(args.binary),
            "binary_sha256": context["hashes"]["binary"],
            "environment": str(args.environment),
            "environment_sha256": context["hashes"]["environment"],
            "build_manifest": context["build_manifest"],
            "gmsh": str(args.gmsh),
            "gmsh_version": context["gmsh_version"],
            "mesh": str(final_mesh),
            "mesh_sha256": hashes["mesh"],
            "input": str(final_input),
            "input_sha256": hashes["input"],
            "queue": args.queue,
            "bank": args.bank,
            "time_limit": args.time_limit,
            "worker_policy": args.worker_policy,
            "cbcd_workers": args.cbcd_workers,
            "compatibility": compatibility,
            "compatibility_fingerprint": json_fingerprint(compatibility),
            "cases": cases,
        }
        finish_study(stage, study, manifest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_manifest(study):
    study = study.expanduser().resolve()
    path = study / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"study manifest does not exist: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise RuntimeError(f"unsupported study manifest schema in {path}")
    return study, manifest


def verify_study_files(study, manifest):
    mismatches = []
    for relative, expected in manifest["files"].items():
        path = study / relative
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(relative)
    binary = Path(manifest["binary"])
    environment = Path(manifest["environment"])
    if not binary.is_file() or sha256(binary) != manifest["binary_sha256"]:
        mismatches.append("external binary")
    if not environment.is_file() or sha256(environment) != manifest["environment_sha256"]:
        mismatches.append("external environment")
    build = manifest["build_manifest"]
    staged_build_path = study / "assets/tuo-build-manifest.json"
    try:
        staged_build = json.loads(staged_build_path.read_text())
    except (OSError, json.JSONDecodeError):
        staged_build = None
    if staged_build != build:
        mismatches.append("staged build manifest contents")
    closure = build.get("linked_library_closure", [])
    if not closure:
        mismatches.append("external linked_library_closure")
    else:
        try:
            if dynamic_library_closure(binary) != closure:
                mismatches.append("external linked_library_closure")
        except RuntimeError:
            mismatches.append("external linked_library_closure")
    for entry in closure:
        path = Path(entry.get("path", ""))
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            mismatches.append(f"external linked library {path}")
    boost = build.get("boost", {})
    if boost.get("version") != "1.86.0":
        mismatches.append("external boost.version")
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("version_config", "version_config_sha256"),
    ):
        path = Path(boost.get(path_key, ""))
        if not path.is_file() or sha256(path) != boost.get(hash_key):
            mismatches.append(f"external boost.{path_key}")
    for path_key, hash_key in (
        ("dependencies_manifest", "dependencies_manifest_sha256"),
        ("caliper_features_manifest", "caliper_features_sha256"),
        ("cmake_cache", "cmake_cache_sha256"),
    ):
        path = Path(build.get(path_key, ""))
        if not path.is_file() or sha256(path) != build.get(hash_key):
            mismatches.append(f"external {path_key}")
    dependencies_path = Path(build.get("dependencies_manifest", ""))
    if dependencies_path.is_file():
        dependencies = json.loads(dependencies_path.read_text())
        for path_key, hash_key in (
            ("bootstrap", "bootstrap_sha256"),
            ("dependency_driver", "dependency_driver_sha256"),
            ("cmake_cache", "cmake_cache_sha256"),
        ):
            path = Path(dependencies.get(path_key, ""))
            if not path.is_file() or sha256(path) != dependencies.get(hash_key):
                mismatches.append(f"external dependencies.{path_key}")
    if mismatches:
        raise RuntimeError("study hash verification failed: " + ", ".join(mismatches))


def verify(args):
    study, manifest = load_manifest(args.study)
    verify_study_files(study, manifest)
    print(
        f"Verified {manifest['type']} study {manifest['label']} "
        f"at revision {manifest['revision']}"
    )


def read_submissions(study):
    path = study / "submissions.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def result_root(study, case):
    if case["category"] == "scaling":
        return study / "results" / case["kind"] / f"nodes-{case['nodes']}"
    return study / "results" / case["profile"] / f"nodes-{case['nodes']}"


def successful_attempts(study, case):
    root = result_root(study, case)
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob("job-*") if (path / "SUCCESS").is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def read_attempt(study, manifest, case, attempt):
    del study
    if parse_exit_code(attempt / "job_exit_code.txt") != 0:
        raise RuntimeError(f"invalid job exit marker in {attempt}")
    validate_binding(attempt / "binding.txt", case["nodes"], case["ranks"])
    if case["category"] == "profile":
        profile = case["profile"]
        required_file = {
            "caliper": attempt / "profile.txt",
            "pmpi": attempt / "mpi.txt",
            "caliper-rocm": attempt / "rocm.txt",
        }.get(profile)
        if required_file is not None and (
            not required_file.is_file() or required_file.stat().st_size == 0
        ):
            raise RuntimeError(f"missing {profile} artifact in {attempt}")
        artifact_root = {
            "hpctoolkit": attempt / "measurements",
            "omniperf": attempt / "workloads/cbcd",
        }.get(profile)
        artifact_exists = artifact_root is None or any(
            path.is_file() and path.name != "metadata.txt"
            for path in artifact_root.rglob("*")
        )
        if profile == "rocprof":
            artifact_exists = any(
                path.is_file()
                and path.name != "metadata.txt"
                and any(part.startswith("rank-") for part in path.relative_to(attempt).parts)
                for path in attempt.rglob("*")
            )
        if not artifact_exists:
            raise RuntimeError(f"missing {profile} artifact in {attempt}")
        return [
            read_result(
                attempt / "stdout.txt",
                attempt / "time.txt",
                attempt / "exit_code.txt",
                attempt / "SUCCESS",
            )
        ]
    return [
        read_result(
            attempt / f"trial-{trial}/stdout.txt",
            attempt / f"trial-{trial}/time.txt",
            attempt / f"trial-{trial}/exit_code.txt",
            attempt / f"trial-{trial}/SUCCESS",
        )
        for trial in range(1, manifest["repetitions"] + 1)
    ]


def artifact_hashes(study, attempts):
    files = {}
    for attempt in attempts:
        for path in sorted(attempt.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(study))] = sha256(path)
    return files


def case_selected(case, args):
    if args.nodes and case["nodes"] not in args.nodes:
        return False
    if case["category"] == "scaling" and args.kinds:
        return case["kind"] in args.kinds
    if case["category"] == "profile" and args.profiles:
        return case["profile"] in args.profiles
    return True


def submit(args):
    study, manifest = load_manifest(args.study)
    verify_study_files(study, manifest)
    if manifest["queue"] == "pdebug":
        raise RuntimeError(
            "pdebug is interactive-only on Tuolumne; use flux alloc and run the job"
        )
    if manifest["type"] == "scaling" and args.profiles is not None:
        raise RuntimeError("--profiles cannot select cases from a scaling study")
    if manifest["type"] == "profile" and args.kinds is not None:
        raise RuntimeError("--kinds cannot select cases from a profile study")
    records = read_submissions(study)
    submitted = {record["case_id"] for record in records}
    log = study / "submissions.jsonl"
    selected = [case for case in manifest["cases"] if case_selected(case, args)]
    if not selected:
        raise RuntimeError("no cases match the submission selection")
    submitted_now = 0
    for case in selected:
        success_marked = successful_attempts(study, case)
        valid = []
        invalid = []
        for attempt in success_marked:
            try:
                read_attempt(study, manifest, case, attempt)
                valid.append(attempt)
            except (OSError, RuntimeError, ValueError) as error:
                invalid.append(f"{attempt.name}: {error}")
        if valid:
            print(f"skip {case['id']}: successful result already exists")
            continue
        if invalid and not args.resubmit:
            raise RuntimeError(
                f"{case['id']} has invalid SUCCESS attempt(s); use --resubmit: "
                + "; ".join(invalid)
            )
        if case["id"] in submitted and not args.resubmit:
            print(f"skip {case['id']}: submission already recorded")
            continue
        output = command_output(["flux", "batch", case["job"]])
        job_id = output.splitlines()[-1].split()[-1]
        if not job_id:
            raise RuntimeError(f"Flux did not return a job ID for {case['id']}")
        record = {
            "submitted_at_utc": utc_now(),
            "case_id": case["id"],
            "job_id": job_id,
            "job": case["job"],
            "resubmission": case["id"] in submitted,
        }
        with log.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        submitted.add(case["id"])
        submitted_now += 1
        print(f"submitted {case['id']}: {job_id}")
    print(f"Submitted {submitted_now} new job(s).")


def parse_exit_code(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid exit-code marker: {path}") from error


def read_result(output, timing, exit_code, success_marker):
    if not success_marker.is_file() or parse_exit_code(exit_code) != 0:
        raise RuntimeError(f"run did not complete successfully: {output.parent}")
    text = output.read_text(errors="replace")
    if not FINISHED_RE.search(text):
        raise RuntimeError(f"missing clean OpenSn termination in {output}")
    sweep_times = SWEEP_TIME_RE.findall(text)
    unknowns = UNKNOWNS_RE.findall(text)
    lagged = LAGGED_RE.findall(text)
    finals = WGS_FINAL_RE.findall(text)
    residuals = WGS_RESIDUAL_RE.findall(text)
    scheduler_worker_matches = CBCD_WORKERS_RE.findall(text)
    scalar_flux_matches = SCALAR_FLUX_MAX_RE.findall(text)
    time_text = timing.read_text(errors="replace")
    wall = WALL_RE.search(time_text)
    rss = RSS_RE.search(time_text)
    if not sweep_times or not unknowns or not finals or not residuals or wall is None:
        raise RuntimeError(f"missing required result metrics in {output}")
    scheduler_workers = {int(value) for value in scheduler_worker_matches}
    if len(scheduler_workers) != 1 or next(iter(scheduler_workers), 0) <= 0:
        raise RuntimeError(f"missing or inconsistent CBCD scheduler worker count in {output}")
    scalar_flux_maxima = {}
    for group_text, value_text in scalar_flux_matches:
        group = int(group_text)
        if group in scalar_flux_maxima:
            raise RuntimeError(f"duplicate scalar-flux maximum for group {group} in {output}")
        scalar_flux_maxima[group] = float(value_text)
    if set(scalar_flux_maxima) != set(SCALAR_FLUX_GROUPS):
        raise RuntimeError(
            f"scalar-flux maxima in {output} must contain exactly groups "
            f"{SCALAR_FLUX_GROUPS}"
        )
    if not all(math.isfinite(value) for value in scalar_flux_maxima.values()):
        raise RuntimeError(f"non-finite scalar-flux maximum in {output}")
    status, iterations = finals[-1]
    status = status.strip()
    if any(word in status.lower() for word in ("fail", "diverge", "error")):
        raise RuntimeError(f"unsuccessful WGS status {status} in {output}")
    sweep_time = float(sweep_times[-1])
    unknown_count = float(unknowns[-1])
    lagged_count = float(lagged[-1]) if lagged else 0.0
    residual = float(residuals[-1])
    wall_time = float(wall.group(1))
    if (
        not math.isfinite(sweep_time)
        or not math.isfinite(unknown_count)
        or not math.isfinite(lagged_count)
        or not math.isfinite(residual)
        or not math.isfinite(wall_time)
        or sweep_time <= 0.0
        or unknown_count <= 0.0
        or lagged_count < 0.0
        or wall_time <= 0.0
        or not unknown_count.is_integer()
        or not lagged_count.is_integer()
    ):
        raise RuntimeError(f"invalid numerical result metrics in {output}")
    result = {
        "avg_sweep_time_s": sweep_time,
        "unknowns": int(unknown_count),
        "lagged_unknowns": int(lagged_count),
        "wgs_status": status,
        "wgs_iterations": int(iterations),
        "scheduler_workers": next(iter(scheduler_workers)),
        "final_residual": residual,
        "wall_time_s": wall_time,
        "launcher_max_rss_kb": int(rss.group(1)) if rss else None,
    }
    result.update(
        {
            f"scalar_flux_max_g{group}": scalar_flux_maxima[group]
            for group in SCALAR_FLUX_GROUPS
        }
    )
    return result


def parse_cpu_affinity(value):
    cpus = set()
    for segment in value.split(","):
        bounds = segment.split("-")
        if len(bounds) > 2 or not all(bound.isdigit() for bound in bounds):
            raise ValueError("invalid CPU range")
        first = int(bounds[0])
        last = int(bounds[-1])
        if last < first:
            raise ValueError("reversed CPU range")
        segment_cpus = set(range(first, last + 1))
        if cpus.intersection(segment_cpus):
            raise ValueError("overlapping CPU ranges")
        cpus.update(segment_cpus)
    return cpus


def validate_binding(path, nodes, ranks):
    if not path.is_file():
        raise RuntimeError(f"binding map does not exist: {path}")
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != ranks or any(len(row) != 6 for row in rows):
        raise RuntimeError(f"binding map has the wrong rank count or format: {path}")
    hosts = {}
    rank_ids = set()
    host_local_ranks = set()
    gpu_assignments = set()
    host_cpus = {}
    for host, rank, local_rank, gpu, cpus, omp_threads in rows:
        try:
            rank_id = int(rank)
            local_rank_id = int(local_rank)
            gpu_index = int(gpu)
            omp_count = int(omp_threads)
            cpu_set = parse_cpu_affinity(cpus)
        except (ValueError, IndexError) as error:
            raise RuntimeError(f"invalid GPU/CPU binding in {path}") from error
        if (
            rank_id < 0
            or rank_id >= ranks
            or local_rank_id < 0
            or gpu_index < 0
            or gpu_index >= 4
            or omp_count != 21
            or len(cpu_set) != 21
        ):
            raise RuntimeError(f"invalid GPU/CPU binding in {path}")
        if rank_id in rank_ids or (host, local_rank_id) in host_local_ranks:
            raise RuntimeError(f"duplicate rank assignment in {path}")
        if (host, gpu_index) in gpu_assignments:
            raise RuntimeError(f"duplicate per-node GPU assignment in {path}")
        used_cpus = host_cpus.setdefault(host, set())
        if used_cpus.intersection(cpu_set):
            raise RuntimeError(f"overlapping per-node CPU assignment in {path}")
        rank_ids.add(rank_id)
        host_local_ranks.add((host, local_rank_id))
        gpu_assignments.add((host, gpu_index))
        used_cpus.update(cpu_set)
        hosts[host] = hosts.get(host, 0) + 1
    if ranks % nodes:
        raise RuntimeError(f"binding map has a nonintegral ranks-per-node count: {path}")
    ranks_per_node = ranks // nodes
    if rank_ids != set(range(ranks)):
        raise RuntimeError(f"binding map does not contain the exact global rank set: {path}")
    if any(local_rank >= ranks_per_node for _, local_rank in host_local_ranks):
        raise RuntimeError(f"binding map has an out-of-range local rank: {path}")
    if len(hosts) != nodes or any(count != ranks_per_node for count in hosts.values()):
        raise RuntimeError(f"binding map has the wrong ranks-per-node count: {path}")


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
    mad = statistics.median(abs(value - median) for value in values)
    iqr = percentile(values, 0.75) - percentile(values, 0.25)
    return median, mad, iqr


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["kind"], row["nodes"]), []).append(row)
    medians = {}
    for key, values in groups.items():
        if key[0] == "strong":
            samples = [value["avg_sweep_time_s"] / value["unknowns"] * 1.0e9 for value in values]
        else:
            samples = [value["avg_sweep_time_s"] for value in values]
        medians[key] = statistics.median(samples)
    summary = []
    for kind, nodes in sorted(groups):
        values = groups[(kind, nodes)]
        unknown_counts = {value["unknowns"] for value in values}
        lagged_counts = {value["lagged_unknowns"] for value in values}
        iterations = {value["wgs_iterations"] for value in values}
        statuses = {value["wgs_status"] for value in values}
        scalar_flux_values = {
            group: {value[f"scalar_flux_max_g{group}"] for value in values}
            for group in SCALAR_FLUX_GROUPS
        }
        if (
            len(unknown_counts) != 1
            or len(lagged_counts) != 1
            or len(iterations) != 1
            or len(statuses) != 1
            or any(
                len(group_values) != 1
                for group_values in scalar_flux_values.values()
            )
        ):
            raise RuntimeError(f"inconsistent numerical signature for {kind}-{nodes}")
        base_nodes = min(node for study_kind, node in groups if study_kind == kind)
        metric = medians[(kind, nodes)]
        base = medians[(kind, base_nodes)]
        efficiency = base / metric * (base_nodes / nodes if kind == "strong" else 1.0)
        metric_samples = [
            value["avg_sweep_time_s"] / value["unknowns"] * 1.0e9
            if kind == "strong"
            else value["avg_sweep_time_s"]
            for value in values
        ]
        metric_median, metric_mad, metric_iqr = spread(metric_samples)
        sweep_median, sweep_mad, sweep_iqr = spread(
            [value["avg_sweep_time_s"] for value in values]
        )
        scheduler_workers = {value["scheduler_workers"] for value in values}
        if len(scheduler_workers) != 1:
            raise RuntimeError(f"inconsistent scheduler worker count for {kind}-{nodes}")
        summary_row = {
            "kind": kind,
            "nodes": nodes,
            "ranks": nodes * 4,
            "trials": len(values),
            "metric": metric_median,
            "metric_mad": metric_mad,
            "metric_iqr": metric_iqr,
            "metric_unit": "ns/unknown" if kind == "strong" else "s",
            "efficiency_percent": efficiency * 100.0,
            "median_avg_sweep_time_s": sweep_median,
            "avg_sweep_time_mad_s": sweep_mad,
            "avg_sweep_time_iqr_s": sweep_iqr,
            "median_unknowns": next(iter(unknown_counts)),
            "wgs_status": next(iter(statuses)),
            "wgs_iterations": next(iter(iterations)),
            "median_final_residual": statistics.median(
                value["final_residual"] for value in values
            ),
            "median_wall_time_s": statistics.median(
                value["wall_time_s"] for value in values
            ),
            "median_lagged_unknowns": next(iter(lagged_counts)),
            "scheduler_workers": next(iter(scheduler_workers)),
        }
        summary_row.update(
            {
                f"scalar_flux_max_g{group}": next(iter(scalar_flux_values[group]))
                for group in SCALAR_FLUX_GROUPS
            }
        )
        summary.append(summary_row)
    return summary


def write_rows(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, manifest, rows):
    lines = [
        f"# {manifest['label']} Tuolumne scaling results",
        "",
        f"Revision: `{manifest['revision']}`",
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
            f"| {row['metric']:.8g} | {row['metric_mad']:.3g} "
            f"| {row['metric_iqr']:.3g} | {row['metric_unit']} "
            f"| {row['efficiency_percent']:.2f}% "
            f"| {row['median_avg_sweep_time_s']:.8g} "
            f"| {row['wgs_iterations']} | {row['scheduler_workers']} "
            f"| {row['median_final_residual']:.8g} "
            f"| {row['scalar_flux_max_g0']:.17e} "
            f"| {row['scalar_flux_max_g63']:.17e} |"
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
                f"strong sweep time increased from {previous['nodes']} to {current['nodes']} nodes"
            )
    return failures


def collect(args):
    study, manifest = load_manifest(args.study)
    if manifest["type"] != "scaling":
        raise RuntimeError("collect requires a scaling study")
    verify_study_files(study, manifest)
    rows = []
    missing = []
    failed = []
    invalid_attempts = []
    used_attempts = []
    for case in manifest["cases"]:
        attempts = successful_attempts(study, case)
        if not attempts:
            missing.append(case["id"])
            continue
        valid_case_attempts = 0
        for attempt in attempts:
            try:
                attempt_values = read_attempt(study, manifest, case, attempt)
            except (OSError, RuntimeError, ValueError) as error:
                invalid_attempts.append(f"{case['id']}/{attempt.name}: {error}")
                continue
            valid_case_attempts += 1
            used_attempts.append(attempt)
            for trial, values in enumerate(attempt_values, start=1):
                rows.append(
                    {
                        "kind": case["kind"],
                        "nodes": case["nodes"],
                        "ranks": case["ranks"],
                        "trial": trial,
                        "attempt": attempt.name,
                        **values,
                    }
                )
        if valid_case_attempts == 0:
            failed.append(f"{case['id']}: no valid successful attempt")
    complete = not missing and not failed
    if not complete and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(missing)} cases are missing and {len(failed)} result sets failed; "
            "use --allow-incomplete only for diagnosis"
        )
    write_rows(study / "results.csv", rows)
    summary = summarize(rows)
    write_rows(study / "summary.csv", summary)
    write_summary(study / "summary.md", manifest, summary)
    monotonic = monotonic_failures(summary, args.monotonic_tolerance)
    collection = {
        "collected_at_utc": utc_now(),
        "complete": complete,
        "missing": missing,
        "failed": failed,
        "invalid_attempts": invalid_attempts,
        "monotonic_failures": monotonic,
        "artifacts": {
            name: sha256(study / name)
            for name in ("results.csv", "summary.csv", "summary.md")
        },
        "attempt_artifacts": artifact_hashes(study, used_attempts),
    }
    atomic_json(study / "collection.json", collection)
    if args.require_monotonic and monotonic:
        raise RuntimeError("; ".join(monotonic))
    print(f"Collected {len(rows)} successful runs in {study}")


def read_collected(study):
    study, manifest = load_manifest(study)
    verify_study_files(study, manifest)
    collection_path = study / "collection.json"
    if not collection_path.is_file():
        raise RuntimeError(f"study has not been collected: {study}")
    collection = json.loads(collection_path.read_text())
    if not collection.get("complete"):
        raise RuntimeError(f"incomplete study cannot be compared: {study}")
    required_artifacts = {"results.csv", "summary.csv", "summary.md"}
    if set(collection.get("artifacts", {})) != required_artifacts:
        raise RuntimeError(f"collection artifact inventory is incomplete: {study}")
    if not collection.get("attempt_artifacts"):
        raise RuntimeError(f"collection attempt inventory is empty: {study}")
    for relative, expected in {
        **collection.get("artifacts", {}),
        **collection.get("attempt_artifacts", {}),
    }.items():
        path = study / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"collected artifact hash mismatch: {path}")
    with (study / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return manifest, rows


def comparable_configuration(manifest, allow_policy_difference):
    configuration = dict(manifest["compatibility"])
    if allow_policy_difference:
        configuration.pop("worker_policy", None)
        configuration.pop("cbcd_workers", None)
    return configuration


def compare(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_manifest, baseline = read_collected(args.baseline)
    candidate_manifest, candidate = read_collected(args.candidate)
    if (
        baseline_manifest.get("worker_policy") != "hardware"
        and not args.allow_nonhardware_baseline
    ):
        raise RuntimeError(
            "baseline worker policy is not hardware; "
            "use --allow-nonhardware-baseline only for a deliberate experiment"
        )
    baseline_config = comparable_configuration(
        baseline_manifest, args.allow_worker_policy_difference
    )
    candidate_config = comparable_configuration(
        candidate_manifest, args.allow_worker_policy_difference
    )
    if baseline_config != candidate_config:
        raise RuntimeError("baseline and candidate study configurations are incompatible")
    baseline_lookup = {
        (row["kind"], int(row["nodes"])): row for row in baseline
    }
    candidate_lookup = {
        (row["kind"], int(row["nodes"])): row for row in candidate
    }
    if set(baseline_lookup) != set(candidate_lookup):
        raise RuntimeError("baseline and candidate do not contain the same scaling points")
    comparisons = []
    failures = []
    for key in sorted(baseline_lookup):
        point_failure_count = len(failures)
        base = baseline_lookup[key]
        cand = candidate_lookup[key]
        if int(float(base["median_unknowns"])) != int(float(cand["median_unknowns"])):
            failures.append(f"{key}: unknown-count mismatch")
        if int(float(base["median_lagged_unknowns"])) != int(
            float(cand["median_lagged_unknowns"])
        ):
            failures.append(f"{key}: lagged-unknown-count mismatch")
        if base["wgs_status"] != cand["wgs_status"]:
            failures.append(f"{key}: WGS-status mismatch")
        if int(base["wgs_iterations"]) != int(cand["wgs_iterations"]):
            failures.append(f"{key}: WGS-iteration mismatch")
        if int(base["trials"]) != int(cand["trials"]):
            failures.append(f"{key}: collected-trial-count mismatch")
        if (
            not args.allow_worker_policy_difference
            and int(base["scheduler_workers"]) != int(cand["scheduler_workers"])
        ):
            failures.append(f"{key}: actual scheduler-worker-count mismatch")
        base_residual = float(base["median_final_residual"])
        cand_residual = float(cand["median_final_residual"])
        if not math.isclose(
            base_residual,
            cand_residual,
            rel_tol=args.residual_rtol,
            abs_tol=args.residual_atol,
        ):
            failures.append(f"{key}: final-residual mismatch")
        scalar_flux_passed = True
        scalar_flux_columns = {}
        for group in SCALAR_FLUX_GROUPS:
            field = f"scalar_flux_max_g{group}"
            base_flux = float(base[field])
            candidate_flux = float(cand[field])
            scalar_flux_columns[f"baseline_{field}"] = base_flux
            scalar_flux_columns[f"candidate_{field}"] = candidate_flux
            if not math.isclose(
                base_flux,
                candidate_flux,
                rel_tol=args.scalar_flux_rtol,
                abs_tol=args.scalar_flux_atol,
            ):
                scalar_flux_passed = False
                failures.append(
                    f"{key}: scalar-flux maximum mismatch for group {group}: "
                    f"{base_flux:.17e} vs {candidate_flux:.17e}"
                )
        base_metric = float(base["metric"])
        candidate_metric = float(cand["metric"])
        ratio = candidate_metric / base_metric
        performance_passed = ratio <= args.max_slowdown
        if not performance_passed:
            failures.append(
                f"{key}: slowdown {ratio:.6f} exceeds {args.max_slowdown:.6f}"
            )
        comparisons.append(
            {
                "kind": key[0],
                "nodes": key[1],
                "baseline": base_metric,
                "candidate": candidate_metric,
                "candidate_over_baseline": ratio,
                "threshold": args.max_slowdown,
                "performance_passed": performance_passed,
                "scalar_flux_passed": scalar_flux_passed,
                "passed": len(failures) == point_failure_count,
                **scalar_flux_columns,
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
    write_rows(output / "comparison.csv", comparisons)
    lines = [
        "# Tuolumne CBCD comparison",
        "",
        f"Baseline: `{baseline_manifest['label']}` (`{baseline_manifest['revision']}`)",
        "",
        f"Candidate: `{candidate_manifest['label']}` (`{candidate_manifest['revision']}`)",
        "",
        (
            "| Kind | Nodes | Baseline | Candidate | Candidate / baseline | "
            "Flux equivalent | Passed |"
        ),
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['kind']} | {row['nodes']} | {row['baseline']:.8g} "
            f"| {row['candidate']:.8g} | {row['candidate_over_baseline']:.6f} "
            f"| {row['scalar_flux_passed']} | {row['passed']} |"
        )
    lines.extend(["", "## Gate failures", ""])
    lines.extend(f"- {failure}" for failure in failures)
    if not failures:
        lines.append("- None")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        output / "comparison.json",
        {
            "generated_at_utc": utc_now(),
            "passed": not failures,
            "failures": failures,
            "allow_worker_policy_difference": args.allow_worker_policy_difference,
            "allow_nonhardware_baseline": args.allow_nonhardware_baseline,
            "scalar_flux_rtol": args.scalar_flux_rtol,
            "scalar_flux_atol": args.scalar_flux_atol,
        },
    )
    if failures:
        raise RuntimeError("comparison gates failed: " + "; ".join(failures))
    print(f"Comparison passed and was written to {output}")


def collect_profile(args):
    study, manifest = load_manifest(args.study)
    if manifest["type"] != "profile":
        raise RuntimeError("collect-profile requires a profile study")
    verify_study_files(study, manifest)
    rows = []
    failures = []
    invalid_attempts = []
    used_attempts = []
    for case in manifest["cases"]:
        attempts = successful_attempts(study, case)
        if not attempts:
            failures.append(f"{case['id']}: no successful attempt")
            continue
        valid_case_attempts = 0
        for attempt in attempts:
            try:
                values = read_attempt(study, manifest, case, attempt)[0]
            except (OSError, RuntimeError, ValueError) as error:
                invalid_attempts.append(f"{case['id']}/{attempt.name}: {error}")
                continue
            valid_case_attempts += 1
            used_attempts.append(attempt)
            rows.append(
                {
                    "profile": case["profile"],
                    "nodes": case["nodes"],
                    "ranks": case["ranks"],
                    "attempt": attempt.name,
                    **values,
                    "result_directory": str(attempt),
                }
            )
        if valid_case_attempts == 0:
            failures.append(f"{case['id']}: no valid successful attempt")
    write_rows(study / "profile-summary.csv", rows)
    artifacts = {"profile-summary.csv": sha256(study / "profile-summary.csv")}
    atomic_json(
        study / "profile-collection.json",
        {
            "collected_at_utc": utc_now(),
            "complete": not failures,
            "failures": failures,
            "invalid_attempts": invalid_attempts,
            "artifacts": artifacts,
            "attempt_artifacts": artifact_hashes(study, used_attempts),
        },
    )
    if failures:
        raise RuntimeError("profile collection failed: " + "; ".join(failures))
    print(f"Collected {len(rows)} profile result(s) in {study}")


def add_common_prepare_arguments(parser, profile=False):
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binary", type=executable, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-cache", type=Path, required=True)
    parser.add_argument("--gmsh", type=executable, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--queue", default="pbatch")
    parser.add_argument("--bank")
    parser.add_argument("--time-limit", default="6h" if profile else "4h")
    parser.add_argument("--gpu-mode", choices=("SPX",), default="SPX")
    parser.add_argument(
        "--worker-policy",
        choices=("hardware", "resource-aware"),
        default="hardware",
    )
    parser.add_argument("--cbcd-workers", type=positive_integer)
    parser.add_argument(
        "--save-angular-flux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain the full angular-flux vector (needed by trunk device CBC)",
    )


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="prepare an immutable scaling study")
    add_common_prepare_arguments(prepare_parser)
    prepare_parser.add_argument("--nodes", type=parse_nodes, default=DEFAULT_NODES)
    prepare_parser.add_argument("--kinds", type=parse_kinds, default=("strong", "weak"))
    prepare_parser.add_argument("--repetitions", type=positive_integer, default=3)
    prepare_parser.add_argument("--strong-divisor", type=positive_integer, default=39)
    prepare_parser.add_argument("--max-iterations", type=positive_integer, default=10)
    prepare_parser.set_defaults(function=prepare)

    profile_parser = commands.add_parser(
        "prepare-profile", help="prepare selected independent profiler jobs"
    )
    add_common_prepare_arguments(profile_parser, profile=True)
    profile_parser.add_argument("--profile-divisor", type=positive_integer, default=15)
    profile_parser.add_argument("--profile-nodes", type=parse_nodes, default=(1, 2, 4))
    profile_parser.add_argument("--profiles", type=parse_profiles, default=DEFAULT_PROFILES)
    profile_parser.add_argument("--max-iterations", type=positive_integer, default=2)
    profile_parser.set_defaults(function=prepare_profile)

    verify_parser = commands.add_parser("verify", help="verify all immutable study hashes")
    verify_parser.add_argument("--study", type=Path, required=True)
    verify_parser.set_defaults(function=verify)

    submit_parser = commands.add_parser("submit", help="idempotently submit selected cases")
    submit_parser.add_argument("--study", type=Path, required=True)
    submit_parser.add_argument("--nodes", type=parse_nodes)
    submit_parser.add_argument("--kinds", type=parse_kinds)
    submit_parser.add_argument("--profiles", type=parse_profiles)
    submit_parser.add_argument("--resubmit", action="store_true")
    submit_parser.set_defaults(function=submit)

    collect_parser = commands.add_parser("collect", help="strictly collect a scaling study")
    collect_parser.add_argument("--study", type=Path, required=True)
    collect_parser.add_argument("--allow-incomplete", action="store_true")
    collect_parser.add_argument("--require-monotonic", action="store_true")
    collect_parser.add_argument(
        "--monotonic-tolerance", type=nonnegative_float, default=0.0
    )
    collect_parser.set_defaults(function=collect)

    compare_parser = commands.add_parser("compare", help="apply performance/numerical gates")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--max-slowdown", type=positive_float, default=1.03)
    compare_parser.add_argument(
        "--monotonic-tolerance", type=nonnegative_float, default=0.0
    )
    compare_parser.add_argument(
        "--residual-rtol", type=nonnegative_float, default=1.0e-6
    )
    compare_parser.add_argument(
        "--residual-atol", type=nonnegative_float, default=1.0e-12
    )
    compare_parser.add_argument(
        "--scalar-flux-rtol", type=nonnegative_float, default=1.0e-10
    )
    compare_parser.add_argument(
        "--scalar-flux-atol", type=nonnegative_float, default=1.0e-12
    )
    compare_parser.add_argument("--allow-worker-policy-difference", action="store_true")
    compare_parser.add_argument("--allow-nonhardware-baseline", action="store_true")
    compare_parser.set_defaults(function=compare)

    profile_collect = commands.add_parser(
        "collect-profile", help="strictly inventory selected profile outputs"
    )
    profile_collect.add_argument("--study", type=Path, required=True)
    profile_collect.set_defaults(function=collect_profile)
    return top


def main():
    args = parser().parse_args()
    if hasattr(args, "monotonic_tolerance") and args.monotonic_tolerance < 0.0:
        raise SystemExit("monotonic tolerance must be nonnegative")
    if hasattr(args, "max_slowdown") and args.max_slowdown <= 0.0:
        raise SystemExit("max slowdown must be positive")
    for name in ("residual_rtol", "residual_atol", "scalar_flux_rtol", "scalar_flux_atol"):
        if hasattr(args, name) and getattr(args, name) < 0.0:
            raise SystemExit(f"{name.replace('_', ' ')} must be nonnegative")
    args.function(args)


if __name__ == "__main__":
    main()
