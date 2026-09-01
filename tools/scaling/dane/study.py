#!/usr/bin/env python3

"""Prepare, submit, and collect reproducible host-CBC studies on Dane."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


NODES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WEAK_DIVISORS = {
    1: 15,
    2: 19,
    4: 24,
    8: 31,
    16: 39,
    32: 49,
    64: 62,
    128: 78,
    256: 98,
}
IMPLEMENTATIONS = ("cycles", "trunk")
KINDS = ("strong", "weak")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
AVG_SWEEP_RE = re.compile(r"avg_sweep_time\s*=\s*([0-9.eE+-]+)\s*s")
GRIND_RE = re.compile(r"sweep_time_per_unknown\s*=\s*([0-9.eE+-]+)\s*ns")
UNKNOWNS_RE = re.compile(r"\bunknowns\s*=\s*([0-9]+)")
LAGGED_RE = re.compile(r"\blagged_unknowns\s*=\s*([0-9]+)")


@dataclass(frozen=True)
class Measurement:
    implementation: str
    kind: str
    nodes: int
    ranks: int
    trial: int
    average_sweep_seconds: float
    sweep_nanoseconds_per_unknown: float
    unknowns: int
    lagged_unknowns: int | None
    stdout: Path


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def parse_nodes(value: str) -> tuple[int, ...]:
    nodes = tuple(int(item) for item in value.split(","))
    if not nodes or any(node not in WEAK_DIVISORS for node in nodes):
        supported = ",".join(str(node) for node in NODES)
        raise argparse.ArgumentTypeError(f"nodes must be selected from {supported}")
    if len(nodes) != len(set(nodes)):
        raise argparse.ArgumentTypeError("node counts must be unique")
    return nodes


def run(command: list[str], *, capture: bool = False) -> str:
    print("+", shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o700)


def make_input(mesh: Path, cross_sections: Path) -> str:
    return f'''#!/usr/bin/env python3

n_g = 64
n_polar = 14
n_azimuthal = 32

meshgen = DistributedMeshGenerator(
    inputs=[FromFileMeshGenerator(filename={str(mesh)!r})]
)
grid = meshgen.Execute()
grid.SetOrthogonalBoundaries()

xs_diag = MultiGroupXS()
xs_diag.LoadFromOpenSn({str(cross_sections)!r})

source = [0.0 for _ in range(n_g)]
source[0] = 1.0

quadrature = GLCProductQuadrature3DXYZ(
    n_polar=n_polar,
    n_azimuthal=n_azimuthal,
    scattering_order=0,
)

problem = DiscreteOrdinatesProblem(
    mesh=grid,
    num_groups=n_g,
    groupsets=[
        {{
            "groups_from_to": (0, n_g - 1),
            "angular_quadrature": quadrature,
            "angle_aggregation_type": "single",
            "inner_linear_method": "petsc_richardson",
            "l_abs_tol": 1.0e-12,
            "l_max_its": 10,
            "allow_cycles": True,
        }},
    ],
    xs_map=[{{"block_ids": [1], "xs": xs_diag}}],
    boundary_conditions=[
        {{"name": "xmin", "type": "isotropic", "group_strength": source}},
    ],
    options={{
        "max_mpi_message_size": 256 * 1024,
        "save_angular_flux": False,
    }},
    sweep_type="CBC",
    use_gpus=False,
)

solver = SteadyStateSourceSolver(problem=problem)
solver.Initialize()
solver.Execute()
'''


def make_build_job(manifest: dict) -> str:
    environment = manifest["environment"]
    source_environment = (
        f"source {shell_quote(environment)}\n" if environment else ""
    )
    configure = (
        "-DCMAKE_BUILD_TYPE=Native "
        "-DOPENSN_WITH_CUDA=OFF -DOPENSN_WITH_HIP=OFF -DOPENSN_WITH_SYCL=OFF"
    )
    sections = []
    for implementation in IMPLEMENTATIONS:
        data = manifest["implementations"][implementation]
        sections.append(
            f'''echo "Configuring {data["label"]} at {data["sha"]}"
cmake -S {shell_quote(data["source"])} -B {shell_quote(data["build"])} {configure}
cmake --build {shell_quote(data["build"])} --parallel "$SLURM_CPUS_PER_TASK"
grep -qx 'CMAKE_BUILD_TYPE:STRING=Native' {shell_quote(Path(data["build"]) / "CMakeCache.txt")}
test -x {shell_quote(data["binary"])}
'''
        )
    return f'''#!/bin/bash -l
#SBATCH --job-name=dane-cbc-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={manifest["build_jobs"]}
#SBATCH --partition=pbatch
#SBATCH --account={manifest["bank"]}
#SBATCH --exclusive
#SBATCH --time={manifest["build_time_limit"]}
#SBATCH --output={shell_quote(Path(manifest["root"]) / "slurm" / "build-%j.out")}
#SBATCH --error={shell_quote(Path(manifest["root"]) / "slurm" / "build-%j.err")}

set -euo pipefail
{source_environment}
{''.join(sections)}
'''


def make_study_job(
    manifest: dict, implementation: str, kind: str, nodes: int
) -> str:
    data = manifest["implementations"][implementation]
    root = Path(manifest["root"])
    result = root / "results" / implementation / kind / f"nodes-{nodes}"
    input_file = root / "inputs" / f"{kind}-{nodes}.py"
    environment = manifest["environment"]
    source_environment = (
        f"source {shell_quote(environment)}\n" if environment else ""
    )
    tasks = nodes * manifest["ranks_per_node"]
    return f'''#!/bin/bash -l
#SBATCH --job-name=cbc-{implementation}-{kind}-{nodes}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={manifest["ranks_per_node"]}
#SBATCH --cpus-per-task=1
#SBATCH --partition=pbatch
#SBATCH --account={manifest["bank"]}
#SBATCH --exclusive
#SBATCH --time={manifest["time_limit"]}
#SBATCH --output={shell_quote(root / "slurm" / f"{implementation}-{kind}-{nodes}-%j.out")}
#SBATCH --error={shell_quote(root / "slurm" / f"{implementation}-{kind}-{nodes}-%j.err")}

set -euo pipefail
{source_environment}
export OPENSN_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

binary={shell_quote(data["binary"])}
result={shell_quote(result)}
mkdir -p "$result"

grep -qx 'CMAKE_BUILD_TYPE:STRING=Native' {shell_quote(Path(data["build"]) / "CMakeCache.txt")}
test -x "$binary"
test "$SLURM_JOB_NUM_NODES" -eq {nodes}

{{
  echo "implementation={implementation}"
  echo "label={data["label"]}"
  echo "revision={data["sha"]}"
  echo "build_type=Native"
  echo "nodes={nodes}"
  echo "ranks={tasks}"
  echo "ranks_per_node={manifest["ranks_per_node"]}"
  echo "slurm_job_id=$SLURM_JOB_ID"
  date --iso-8601=seconds
}} > "$result/job-metadata.txt"

for trial in $(seq 1 {manifest["repetitions"]}); do
  trial_dir="$result/trial-$trial"
  mkdir -p "$trial_dir"
  if [[ -f "$trial_dir/completed" ]]; then
    echo "Skipping completed trial $trial"
    continue
  fi

  echo "Starting {implementation} {kind} nodes={nodes} trial=$trial"
  srun \\
    --nodes={nodes} \\
    --ntasks={tasks} \\
    --ntasks-per-node={manifest["ranks_per_node"]} \\
    --cpus-per-task=1 \\
    --distribution=block \\
    --mpibind=on \\
    "$binary" -i {shell_quote(input_file)} \\
    > "$trial_dir/stdout.txt" \\
    2> "$trial_dir/stderr.txt"
  touch "$trial_dir/completed"
done
'''


def generate_mesh(gmsh: str, geometry: Path, divisor: int, output: Path) -> None:
    if output.exists():
        print(f"Reusing mesh {output}")
        return
    run(
        [
            gmsh,
            "-3",
            "-v",
            "1",
            "-setnumber",
            "divisor",
            str(divisor),
            "-o",
            str(output),
            str(geometry),
        ]
    )
    if not output.is_file():
        raise RuntimeError(f"Gmsh did not create the expected mesh: {output}")


def prepare(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if root.exists():
        raise RuntimeError(f"campaign directory already exists: {root}")
    if args.ranks_per_node <= 0 or args.repetitions <= 0 or args.build_jobs <= 0:
        raise RuntimeError("ranks, repetitions, and build jobs must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.bank):
        raise RuntimeError("bank names may contain only letters, digits, '.', '_', and '-'")
    if "\n" in args.time_limit or "\n" in args.build_time_limit:
        raise RuntimeError("time limits must not contain newlines")
    if args.environment and not args.environment.is_file():
        raise RuntimeError(f"environment setup file does not exist: {args.environment}")

    for directory in ("inputs", "jobs", "meshes", "results", "slurm"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    cross_sections = root / "inputs" / "xs_168g.xs"
    cross_sections.write_bytes(args.cross_sections.read_bytes())

    strong_mesh = root / "meshes" / "strong.msh"
    generate_mesh(args.gmsh, args.geometry, args.strong_divisor, strong_mesh)
    for nodes in args.nodes:
        generate_mesh(
            args.gmsh,
            args.geometry,
            WEAK_DIVISORS[nodes],
            root / "meshes" / f"weak-{nodes}.msh",
        )

    for nodes in args.nodes:
        (root / "inputs" / f"strong-{nodes}.py").write_text(
            make_input(strong_mesh, cross_sections)
        )
        (root / "inputs" / f"weak-{nodes}.py").write_text(
            make_input(root / "meshes" / f"weak-{nodes}.msh", cross_sections)
        )

    implementations = {
        "cycles": {
            "label": "CBC cycles",
            "sha": args.cycles_sha,
            "source": str(args.cycles_source.resolve()),
            "build": str(args.cycles_build.resolve()),
            "binary": str((args.cycles_build.resolve() / "python" / "opensn")),
        },
        "trunk": {
            "label": "Trunk CBC",
            "sha": args.trunk_sha,
            "source": str(args.trunk_source.resolve()),
            "build": str(args.trunk_build.resolve()),
            "binary": str((args.trunk_build.resolve() / "python" / "opensn")),
        },
    }
    manifest = {
        "schema": 1,
        "root": str(root),
        "nodes": list(args.nodes),
        "ranks_per_node": args.ranks_per_node,
        "repetitions": args.repetitions,
        "strong_divisor": args.strong_divisor,
        "weak_divisors": {str(node): WEAK_DIVISORS[node] for node in args.nodes},
        "bank": args.bank,
        "environment": str(args.environment.resolve()) if args.environment else "",
        "time_limit": args.time_limit,
        "build_time_limit": args.build_time_limit,
        "build_jobs": args.build_jobs,
        "implementations": implementations,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_executable(root / "jobs" / "build.sbatch", make_build_job(manifest))
    for implementation in IMPLEMENTATIONS:
        for kind in KINDS:
            for nodes in args.nodes:
                path = root / "jobs" / f"{implementation}-{kind}-{nodes}.sbatch"
                write_executable(
                    path, make_study_job(manifest, implementation, kind, nodes)
                )

    print(
        f"Prepared {len(IMPLEMENTATIONS) * len(KINDS) * len(args.nodes)} "
        f"scaling jobs in {root}"
    )


def load_manifest(root: Path) -> dict:
    path = root.resolve() / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"missing campaign manifest: {path}")
    return json.loads(path.read_text())


def parse_job_id(output: str) -> str:
    job_id = output.split(";", 1)[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"could not parse Slurm job ID from {output!r}")
    return job_id


def submit(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = load_manifest(root)
    ids_file = root / "job-ids.tsv"
    if ids_file.exists():
        raise RuntimeError(f"jobs were already submitted: {ids_file}")

    build_id = parse_job_id(
        run(["sbatch", "--parsable", str(root / "jobs" / "build.sbatch")], capture=True)
    )
    rows = [("build", build_id)]
    ids_file.write_text(f"build\t{build_id}\n")
    print(f"submitted build: {build_id}")

    for implementation in IMPLEMENTATIONS:
        for kind in KINDS:
            for nodes in manifest["nodes"]:
                name = f"{implementation}-{kind}-{nodes}"
                job = root / "jobs" / f"{name}.sbatch"
                job_id = parse_job_id(
                    run(
                        [
                            "sbatch",
                            "--parsable",
                            f"--dependency=afterok:{build_id}",
                            str(job),
                        ],
                        capture=True,
                    )
                )
                rows.append((name, job_id))
                with ids_file.open("a") as stream:
                    stream.write(f"{name}\t{job_id}\n")
                print(f"submitted {name}: {job_id}")

    print(f"Submitted {len(rows)} job(s); IDs are in {ids_file}")


def read_job_ids(root: Path) -> list[tuple[str, str]]:
    path = root.resolve() / "job-ids.tsv"
    if not path.is_file():
        raise RuntimeError(f"missing job ID file: {path}")
    rows = []
    for line in path.read_text().splitlines():
        name, job_id = line.split("\t", 1)
        rows.append((name, job_id))
    return rows


def status(args: argparse.Namespace) -> None:
    rows = read_job_ids(args.root)
    ids = ",".join(job_id for _, job_id in rows)
    print("Active jobs:")
    subprocess.run(
        ["squeue", "--jobs", ids, "--format=%.18i %.32j %.9T %.10M %.6D %R"],
        check=False,
    )
    print("\nAccounting records:")
    subprocess.run(
        [
            "sacct",
            "--jobs",
            ids,
            "--allocations",
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName,State,Elapsed,NNodes",
        ],
        check=False,
    )


def parse_measurement(
    stdout: Path,
    implementation: str,
    kind: str,
    nodes: int,
    ranks_per_node: int,
    trial: int,
) -> Measurement | None:
    if not stdout.is_file() or not (stdout.parent / "completed").is_file():
        return None
    text = ANSI_ESCAPE.sub("", stdout.read_text(errors="replace"))
    average = AVG_SWEEP_RE.findall(text)
    grind = GRIND_RE.findall(text)
    unknowns = UNKNOWNS_RE.findall(text)
    lagged = LAGGED_RE.findall(text)
    if not average or not grind or not unknowns:
        return None
    return Measurement(
        implementation=implementation,
        kind=kind,
        nodes=nodes,
        ranks=nodes * ranks_per_node,
        trial=trial,
        average_sweep_seconds=float(average[-1]),
        sweep_nanoseconds_per_unknown=float(grind[-1]),
        unknowns=int(unknowns[-1]),
        lagged_unknowns=int(lagged[-1]) if lagged else None,
        stdout=stdout,
    )


def median_absolute_deviation(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def interquartile_range(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def summarize(measurements: list[Measurement]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[Measurement]] = {}
    for measurement in measurements:
        key = (measurement.implementation, measurement.kind, measurement.nodes)
        grouped.setdefault(key, []).append(measurement)

    time_medians = {
        key: statistics.median(item.average_sweep_seconds for item in values)
        for key, values in grouped.items()
    }
    grind_medians = {
        key: statistics.median(item.sweep_nanoseconds_per_unknown for item in values)
        for key, values in grouped.items()
    }
    rows = []
    for key in sorted(grouped, key=lambda item: (item[1], item[0], item[2])):
        implementation, kind, nodes = key
        values = grouped[key]
        times = [item.average_sweep_seconds for item in values]
        grinds = [item.sweep_nanoseconds_per_unknown for item in values]
        baseline_key = (implementation, kind, 1)
        baseline = (
            grind_medians.get(baseline_key)
            if kind == "strong"
            else time_medians.get(baseline_key)
        )
        efficiency = None
        if baseline is not None:
            metric = statistics.median(grinds) if kind == "strong" else statistics.median(times)
            denominator = metric * (nodes if kind == "strong" else 1)
            efficiency = 100.0 * baseline / denominator
        lagged = [item.lagged_unknowns for item in values if item.lagged_unknowns is not None]
        rows.append(
            {
                "implementation": implementation,
                "kind": kind,
                "nodes": nodes,
                "ranks": values[0].ranks,
                "trials": len(values),
                "average_sweep_seconds": statistics.median(times),
                "average_sweep_mad_seconds": median_absolute_deviation(times),
                "average_sweep_iqr_seconds": interquartile_range(times),
                "sweep_nanoseconds_per_unknown": statistics.median(grinds),
                "unknowns": int(statistics.median(item.unknowns for item in values)),
                "lagged_unknowns": int(statistics.median(lagged)) if lagged else "",
                "efficiency_percent": efficiency if efficiency is not None else "",
            }
        )
    return rows


def write_raw_csv(path: Path, measurements: list[Measurement]) -> None:
    fields = [field for field in Measurement.__dataclass_fields__ if field != "stdout"] + [
        "stdout"
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in measurements:
            row = {field: getattr(item, field) for field in fields}
            row["stdout"] = str(item.stdout)
            writer.writerow(row)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, manifest: dict, rows: list[dict]) -> None:
    labels = {
        key: value["label"] for key, value in manifest["implementations"].items()
    }
    lines = [
        "# Dane host-CBC scaling results",
        "",
        f"All builds use `CMAKE_BUILD_TYPE=Native`; all runs use "
        f"{manifest['ranks_per_node']} MPI ranks per node.",
        "",
        f"- CBC cycles revision: `{manifest['implementations']['cycles']['sha']}`",
        f"- Trunk CBC revision: `{manifest['implementations']['trunk']['sha']}`",
        "",
    ]
    for kind in KINDS:
        lines.extend(
            [
                f"## {kind.capitalize()} scaling",
                "",
                "| Implementation | Nodes | Ranks | Trials | Sweep (s) | MAD (s) | "
                "IQR (s) | ns/unknown | Unknowns | Lagged unknowns | Efficiency |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["kind"] != kind:
                continue
            efficiency = row["efficiency_percent"]
            efficiency_text = f"{efficiency:.2f}%" if efficiency != "" else "n/a"
            lagged = row["lagged_unknowns"] if row["lagged_unknowns"] != "" else "n/a"
            lines.append(
                f"| {labels[row['implementation']]} | {row['nodes']} | {row['ranks']} | "
                f"{row['trials']} | {row['average_sweep_seconds']:.9g} | "
                f"{row['average_sweep_mad_seconds']:.3g} | "
                f"{row['average_sweep_iqr_seconds']:.3g} | "
                f"{row['sweep_nanoseconds_per_unknown']:.9g} | {row['unknowns']} | "
                f"{lagged} | {efficiency_text} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def collect(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = load_manifest(root)
    measurements = []
    expected = 0
    for implementation in IMPLEMENTATIONS:
        for kind in KINDS:
            for nodes in manifest["nodes"]:
                for trial in range(1, manifest["repetitions"] + 1):
                    expected += 1
                    stdout = (
                        root
                        / "results"
                        / implementation
                        / kind
                        / f"nodes-{nodes}"
                        / f"trial-{trial}"
                        / "stdout.txt"
                    )
                    measurement = parse_measurement(
                        stdout,
                        implementation,
                        kind,
                        nodes,
                        manifest["ranks_per_node"],
                        trial,
                    )
                    if measurement is not None:
                        measurements.append(measurement)

    if not measurements:
        print(f"Collected 0 of {expected} expected measurements.")
        return
    rows = summarize(measurements)
    write_raw_csv(root / "raw-results.csv", measurements)
    write_summary_csv(root / "summary.csv", rows)
    write_summary_markdown(root / "summary.md", manifest, rows)
    print(f"Collected {len(measurements)} of {expected} expected measurements.")
    print(root / "summary.md")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--bank", required=True)
    prepare_parser.add_argument("--cycles-source", type=Path, required=True)
    prepare_parser.add_argument("--cycles-sha", required=True)
    prepare_parser.add_argument("--cycles-build", type=Path, required=True)
    prepare_parser.add_argument("--trunk-source", type=Path, required=True)
    prepare_parser.add_argument("--trunk-sha", required=True)
    prepare_parser.add_argument("--trunk-build", type=Path, required=True)
    prepare_parser.add_argument("--geometry", type=Path, required=True)
    prepare_parser.add_argument("--cross-sections", type=Path, required=True)
    prepare_parser.add_argument("--environment", type=Path)
    prepare_parser.add_argument("--gmsh", default="gmsh")
    prepare_parser.add_argument("--nodes", type=parse_nodes, default=NODES)
    prepare_parser.add_argument("--ranks-per-node", type=int, default=64)
    prepare_parser.add_argument("--repetitions", type=int, default=3)
    prepare_parser.add_argument("--strong-divisor", type=int, default=39)
    prepare_parser.add_argument("--time-limit", default="01:00:00")
    prepare_parser.add_argument("--build-time-limit", default="01:00:00")
    prepare_parser.add_argument("--build-jobs", type=int, default=16)
    prepare_parser.set_defaults(action=prepare)

    for command, action in (("submit", submit), ("status", status), ("collect", collect)):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", type=Path, required=True)
        command_parser.set_defaults(action=action)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        args.action(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
