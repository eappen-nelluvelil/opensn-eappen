#!/usr/bin/env python3

"""Prepare, collect, and profile paired AAH/CBC studies on LLNL Dane."""

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCALING_DIR = HERE.parent
GEOMETRY = SCALING_DIR / "lib/cube.geo"
XS = SCALING_DIR / "lib/xs_168g.xs"
INPUT_TEMPLATE = HERE / "transport.py.in"
DEFAULT_NODES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WEAK_DIVISORS = dict(zip(DEFAULT_NODES, (15, 19, 24, 31, 39, 49, 62, 78, 98)))
FLOAT = r"[0-9.eE+-]+"
SWEEP_TIME_RE = re.compile(rf"avg_sweep_time\s*=\s*({FLOAT})\s*s")
UNKNOWNS_RE = re.compile(rf"(?<!lagged_)\bunknowns\s*=\s*({FLOAT})")
LAGGED_RE = re.compile(rf"\blagged_unknowns\s*=\s*({FLOAT})")
WALL_RE = re.compile(rf"wall_seconds=({FLOAT})")
RSS_RE = re.compile(r"max_rss_kb=(\d+)")


def parse_nodes(value):
    try:
        nodes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("nodes must be comma-separated integers") from error
    if not nodes or nodes[0] <= 0:
        raise argparse.ArgumentTypeError("nodes must be positive")
    unsupported = set(nodes) - WEAK_DIVISORS.keys()
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"no weak-scaling mesh divisor is defined for {sorted(unsupported)}"
        )
    return nodes


def parse_profile_modes(value):
    available = ("baseline", "coarse", "pmpi")
    if value == "all":
        return available
    modes = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = set(modes) - set(available)
    if not modes or invalid:
        raise argparse.ArgumentTypeError(
            "profile modes must be all or a comma-separated subset of baseline,coarse,pmpi"
        )
    return modes


def executable(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"not an executable file: {path}")
    return path


def write_executable(path, content):
    path.write_text(content)
    path.chmod(0o700)


def mesh_for(cache, gmsh, divisor):
    cache.mkdir(parents=True, exist_ok=True)
    mesh = cache / f"cube-d{divisor}.msh"
    if mesh.is_file():
        return mesh
    if mesh.exists():
        raise RuntimeError(f"mesh cache path is not a file: {mesh}")

    temporary = cache / f".{mesh.stem}.{os.getpid()}{mesh.suffix}"
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
            str(GEOMETRY),
        ],
        check=True,
    )
    temporary.replace(mesh)
    return mesh


def write_input(path, mesh, algorithm):
    content = INPUT_TEMPLATE.read_text()
    replacements = {
        "@MESH@": repr(str(mesh)),
        "@XS@": repr(str(XS.resolve())),
        "@SWEEP_TYPE@": repr(algorithm),
    }
    for token, value in replacements.items():
        content = content.replace(token, value)
    path.write_text(content)


def job_script(args, study, kind, nodes, inputs):
    ranks = nodes * 64
    result_dir = study / "results" / kind / f"nodes-{nodes}"
    scheduler_dir = study / "scheduler"
    account = f"#SBATCH --account={args.account}\n" if args.account else ""
    cases = "\n".join(
        f"    {algorithm}) input={shlex.quote(str(path))} ;;"
        for algorithm, path in inputs.items()
    )
    return f"""#!/bin/zsh
#SBATCH --job-name={args.label[:18]}-{kind[0]}-{nodes}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --partition={args.partition}
{account}#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time={args.time_limit}
#SBATCH --output={scheduler_dir}/{kind}-{nodes}-%j.out
#SBATCH --error={scheduler_dir}/{kind}-{nodes}-%j.err

set -euo pipefail

source {shlex.quote(str(args.environment))}
export OPENSN_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=true

binary={shlex.quote(str(args.binary))}

result_dir={shlex.quote(str(result_dir))}
mkdir -p "$result_dir"

for trial in {{1..{args.repetitions}}}; do
  if (( trial % 2 )); then
    algorithms=(AAH CBC)
  else
    algorithms=(CBC AAH)
  fi

  for algorithm in $algorithms; do
    case $algorithm in
{cases}
    esac
    stem="$result_dir/${{algorithm:l}}-${{trial}}"
    /usr/bin/time -f 'wall_seconds=%e max_rss_kb=%M' -o "$stem.time" \\
      srun --nodes={nodes} --ntasks={ranks} --ntasks-per-node=64 \\
        --cpus-per-task=1 --hint=nomultithread --cpu-bind=cores \\
        --distribution=block:cyclic --kill-on-bad-exit=1 \\
        "$binary" -i "$input" > "$stem.out" 2> "$stem.err"
  done
done
"""


def prepare(args):
    study = args.output.expanduser().resolve()
    if any(character.isspace() for character in str(study)):
        raise RuntimeError("study path cannot contain whitespace in Slurm directives")
    if study.exists() and any(study.iterdir()):
        raise RuntimeError(f"study directory is not empty: {study}")
    if not args.environment.is_file():
        raise RuntimeError(f"environment script does not exist: {args.environment}")
    if not args.gmsh.is_file() or not os.access(args.gmsh, os.X_OK):
        raise RuntimeError(f"Gmsh executable does not exist: {args.gmsh}")

    for path in (study / "inputs", study / "jobs", study / "results", study / "scheduler"):
        path.mkdir(parents=True, exist_ok=True)

    divisors = {args.strong_divisor, *(WEAK_DIVISORS[node] for node in args.nodes)}
    meshes = {
        divisor: mesh_for(args.mesh_cache.expanduser().resolve(), args.gmsh, divisor)
        for divisor in sorted(divisors)
    }
    cases = []
    jobs = []

    for kind in ("strong", "weak"):
        for nodes in args.nodes:
            divisor = args.strong_divisor if kind == "strong" else WEAK_DIVISORS[nodes]
            mesh = meshes[divisor]
            inputs = {}
            for algorithm in ("AAH", "CBC"):
                input_path = study / "inputs" / f"{kind}-{nodes}-{algorithm.lower()}.py"
                write_input(input_path, mesh, algorithm)
                inputs[algorithm] = input_path
            job = study / "jobs" / f"{kind}-{nodes}.zsh"
            write_executable(
                job,
                job_script(args, study, kind, nodes, inputs),
            )
            jobs.append(job)
            cases.append(
                {
                    "kind": kind,
                    "nodes": nodes,
                    "ranks": nodes * 64,
                    "divisor": divisor,
                    "mesh": str(mesh),
                    "inputs": {key: str(value) for key, value in inputs.items()},
                    "job": str(job),
                }
            )

    write_executable(
        study / "submit.zsh",
        "#!/bin/zsh\nset -euo pipefail\n\n"
        + "\n".join(f"sbatch {shlex.quote(str(job))}" for job in jobs)
        + "\n",
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "revision": args.revision,
        "binary": str(args.binary),
        "environment": str(args.environment),
        "nodes": args.nodes,
        "ranks_per_node": 64,
        "repetitions": args.repetitions,
        "strong_divisor": args.strong_divisor,
        "weak_divisors": {str(node): WEAK_DIVISORS[node] for node in args.nodes},
        "cases": cases,
    }
    (study / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(jobs)} jobs in {study}")


def read_result(output, timing):
    text = output.read_text(errors="replace")
    sweep_times = SWEEP_TIME_RE.findall(text)
    unknowns = UNKNOWNS_RE.findall(text)
    lagged = LAGGED_RE.findall(text)
    time_text = timing.read_text(errors="replace")
    wall = WALL_RE.search(time_text)
    rss = RSS_RE.search(time_text)
    if not sweep_times or not unknowns or wall is None:
        raise RuntimeError(f"missing metrics in {output}")
    return {
        "avg_sweep_time_s": float(sweep_times[-1]),
        "unknowns": float(unknowns[-1]),
        "lagged_unknowns": float(lagged[-1]) if lagged else 0.0,
        "wall_time_s": float(wall.group(1)),
        "max_rss_kb": int(rss.group(1)) if rss else None,
    }


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def collect(args):
    study = args.study.expanduser().resolve()
    manifest = json.loads((study / "manifest.json").read_text())
    rows = []
    missing = []
    for case in manifest["cases"]:
        for algorithm in ("AAH", "CBC"):
            for trial in range(1, manifest["repetitions"] + 1):
                stem = study / "results" / case["kind"] / f"nodes-{case['nodes']}" / f"{algorithm.lower()}-{trial}"
                if not stem.with_suffix(".out").is_file() or not stem.with_suffix(".time").is_file():
                    missing.append(str(stem))
                    continue
                values = read_result(stem.with_suffix(".out"), stem.with_suffix(".time"))
                rows.append(
                    {
                        "kind": case["kind"],
                        "nodes": case["nodes"],
                        "ranks": case["ranks"],
                        "algorithm": algorithm,
                        "trial": trial,
                        **values,
                    }
                )
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"{len(missing)} result sets are missing; use --allow-incomplete to inspect partial data")
    write_rows(study / "results.csv", rows)

    groups = {}
    for row in rows:
        groups.setdefault((row["kind"], row["algorithm"], row["nodes"]), []).append(row)
    medians = {}
    for key, values in groups.items():
        if key[0] == "strong":
            samples = [value["avg_sweep_time_s"] / value["unknowns"] * 1.0e9 for value in values]
        else:
            samples = [value["avg_sweep_time_s"] for value in values]
        medians[key] = statistics.median(samples)

    summary = []
    for key in sorted(groups):
        kind, algorithm, nodes = key
        values = groups[key]
        metric = medians[key]
        base_nodes = min(node for k, a, node in groups if k == kind and a == algorithm)
        base = medians[(kind, algorithm, base_nodes)]
        efficiency = base / metric * (base_nodes / nodes if kind == "strong" else 1.0)
        summary.append(
            {
                "kind": kind,
                "nodes": nodes,
                "algorithm": algorithm,
                "trials": len(values),
                "metric": metric,
                "metric_unit": "ns/unknown" if kind == "strong" else "s",
                "efficiency_percent": efficiency * 100.0,
                "median_wall_time_s": statistics.median(value["wall_time_s"] for value in values),
                "median_lagged_unknowns": statistics.median(value["lagged_unknowns"] for value in values),
            }
        )
    write_rows(study / "summary.csv", summary)
    write_summary_markdown(study / "summary.md", summary)
    make_plots(study, summary)
    print(f"Collected {len(rows)} runs in {study}")
    if missing:
        print(f"Skipped {len(missing)} incomplete result sets")


def write_summary_markdown(path, rows):
    lines = [
        "# Dane AAH/CBC scaling results",
        "",
        "| Study | Nodes | Algorithm | Trials | Metric | Unit | Efficiency | Wall time (s) | Lagged unknowns |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['kind']} | {row['nodes']} | {row['algorithm']} | {row['trials']} "
            f"| {row['metric']:.8g} | {row['metric_unit']} | {row['efficiency_percent']:.2f}% "
            f"| {row['median_wall_time_s']:.3f} | {row['median_lagged_unknowns']:.8g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def make_plots(study, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV and Markdown summaries were written")
        return

    for kind in ("strong", "weak"):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for algorithm, color in (("AAH", "#1f77b4"), ("CBC", "#d95f02")):
            selected = [row for row in rows if row["kind"] == kind and row["algorithm"] == algorithm]
            if not selected:
                continue
            nodes = [row["nodes"] for row in selected]
            metric = [row["metric"] for row in selected]
            ax.plot(nodes, metric, marker="o", color=color, label=algorithm)
            if kind == "strong":
                ideal = [metric[0] * nodes[0] / node for node in nodes]
                ax.plot(nodes, ideal, linestyle="--", color=color, alpha=0.45)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted({row["nodes"] for row in rows if row["kind"] == kind}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Nodes (64 MPI ranks/node)")
        ax.set_ylabel("Average sweep time per unknown (ns)" if kind == "strong" else "Average sweep time (s)")
        ax.set_title(f"Dane {kind} scaling")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(study / f"{kind}.pdf")
        plt.close(fig)


def profile(args):
    study = args.study.expanduser().resolve()
    manifest = json.loads((study / "manifest.json").read_text())
    allocation_nodes = int(os.environ.get("SLURM_NNODES", os.environ.get("SLURM_JOB_NUM_NODES", "0")))
    if allocation_nodes <= 0:
        raise RuntimeError("profile must run inside a Slurm allocation")
    case = next(
        (
            item
            for item in manifest["cases"]
            if item["kind"] == args.kind and item["nodes"] == allocation_nodes
        ),
        None,
    )
    if case is None:
        raise RuntimeError(
            f"study has no {allocation_nodes}-node {args.kind}-scaling input"
        )

    binary = args.binary or Path(manifest["binary"])
    binary = binary.expanduser().resolve()
    algorithm = args.algorithm.upper()
    input_path = Path(case["inputs"][algorithm])
    output_root = study / "profiles" / args.label / args.kind / f"nodes-{allocation_nodes}"
    output_root.mkdir(parents=True, exist_ok=True)
    modes = args.mode
    ranks = allocation_nodes * 64
    base_command = [
        "srun",
        f"--nodes={allocation_nodes}",
        f"--ntasks={ranks}",
        "--ntasks-per-node=64",
        "--cpus-per-task=1",
        "--hint=nomultithread",
        "--cpu-bind=cores",
        "--distribution=block:cyclic",
        "--kill-on-bad-exit=1",
        str(binary),
        "-i",
        str(input_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OPENSN_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_PLACES": "cores",
            "OMP_PROC_BIND": "true",
        }
    )
    records = []
    for trial in range(1, args.repetitions + 1):
        trial_modes = modes if trial % 2 else tuple(reversed(modes))
        for mode in trial_modes:
            output_dir = output_root / mode / algorithm.lower() / f"trial-{trial}"
            output_dir.mkdir(parents=True, exist_ok=True)
            command = list(base_command)
            if mode == "coarse":
                caliper_output = output_dir / "profile.txt"
                config = (
                    f'runtime-report(output="{caliper_output}",aggregate_across_ranks,'
                    "calc.inclusive,print.metadata,order_by_time,max_column_width=180,"
                    "region.count)"
                )
                command.insert(-2, f"--caliper={config}")
            elif mode == "pmpi":
                caliper_output = output_dir / "mpi.txt"
                command.insert(-2, f'--caliper=mpi-report(output="{caliper_output}")')

            stdout_path = output_dir / "stdout.txt"
            stderr_path = output_dir / "stderr.txt"
            (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            start = time.perf_counter()
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                result = subprocess.run(
                    command,
                    cwd=study,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            wall_seconds = time.perf_counter() - start
            if result.returncode:
                raise RuntimeError(
                    f"{mode} profile failed with status {result.returncode}: {output_dir}"
                )

            sweep_times = SWEEP_TIME_RE.findall(stdout_path.read_text(errors="replace"))
            if not sweep_times:
                raise RuntimeError(f"missing sweep time in {stdout_path}")
            record = {
                "mode": mode,
                "algorithm": algorithm,
                "kind": args.kind,
                "nodes": allocation_nodes,
                "ranks": ranks,
                "trial": trial,
                "binary": str(binary),
                "avg_sweep_time_s": float(sweep_times[-1]),
                "wall_time_s": wall_seconds,
            }
            (output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)

    summaries = []
    baseline = None
    for mode in modes:
        selected = [record for record in records if record["mode"] == mode]
        median_sweep = statistics.median(record["avg_sweep_time_s"] for record in selected)
        median_wall = statistics.median(record["wall_time_s"] for record in selected)
        if mode == "baseline":
            baseline = median_sweep
        summaries.append(
            {
                "mode": mode,
                "trials": len(selected),
                "median_avg_sweep_time_s": median_sweep,
                "median_wall_time_s": median_wall,
            }
        )

    if baseline is not None:
        for summary in summaries:
            summary["sweep_overhead_percent"] = (
                (summary["median_avg_sweep_time_s"] / baseline - 1.0) * 100.0
            )
    summary_path = output_root / f"profile-summary-{algorithm.lower()}.json"
    summary_path.write_text(json.dumps({"runs": records, "summary": summaries}, indent=2) + "\n")
    lines = [
        f"# {algorithm} {args.kind} profiling comparison",
        "",
        "| Mode | Trials | Median sweep (s) | Median wall (s) | Sweep overhead |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        overhead = summary.get("sweep_overhead_percent")
        overhead_text = "n/a" if overhead is None else f"{overhead:.2f}%"
        lines.append(
            f"| {summary['mode']} | {summary['trials']} "
            f"| {summary['median_avg_sweep_time_s']:.8g} "
            f"| {summary['median_wall_time_s']:.3f} | {overhead_text} |"
        )
    (output_root / f"profile-summary-{algorithm.lower()}.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(f"Profiles and overhead summary written to {output_root}")


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="generate paired strong/weak jobs")
    prepare_parser.add_argument("--binary", type=executable, required=True)
    prepare_parser.add_argument("--environment", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--mesh-cache", type=Path, required=True)
    prepare_parser.add_argument("--gmsh", type=Path, default=Path("gmsh"))
    prepare_parser.add_argument("--label", required=True)
    prepare_parser.add_argument("--revision", required=True)
    prepare_parser.add_argument("--nodes", type=parse_nodes, default=DEFAULT_NODES)
    prepare_parser.add_argument("--repetitions", type=int, default=3)
    prepare_parser.add_argument("--strong-divisor", type=int, default=39)
    prepare_parser.add_argument("--partition", default="pbatch")
    prepare_parser.add_argument("--account")
    prepare_parser.add_argument("--time-limit", default="04:00:00")
    prepare_parser.set_defaults(function=prepare)

    collect_parser = commands.add_parser("collect", help="summarize completed jobs")
    collect_parser.add_argument("--study", type=Path, required=True)
    collect_parser.add_argument("--allow-incomplete", action="store_true")
    collect_parser.set_defaults(function=collect)

    profile_parser = commands.add_parser("profile", help="profile one algorithm in the current allocation")
    profile_parser.add_argument("--study", type=Path, required=True)
    profile_parser.add_argument("--algorithm", choices=("AAH", "CBC"), default="CBC")
    profile_parser.add_argument("--kind", choices=("strong", "weak"), default="strong")
    profile_parser.add_argument(
        "--mode",
        type=parse_profile_modes,
        default=parse_profile_modes("all"),
        help="all or a comma-separated subset of baseline,coarse,pmpi",
    )
    profile_parser.add_argument("--label", default="optimized")
    profile_parser.add_argument("--repetitions", type=int, default=3)
    profile_parser.add_argument("--binary", type=executable)
    profile_parser.set_defaults(function=profile)
    return top


def main():
    args = parser().parse_args()
    if getattr(args, "gmsh", None) == Path("gmsh"):
        resolved = subprocess.run(["zsh", "-lc", "command -v gmsh"], capture_output=True, text=True)
        if resolved.returncode:
            raise RuntimeError("gmsh is not on PATH; pass --gmsh /absolute/path/to/gmsh")
        args.gmsh = Path(resolved.stdout.strip()).resolve()
    if getattr(args, "environment", None):
        args.environment = args.environment.expanduser().resolve()
    if getattr(args, "repetitions", 1) <= 0:
        raise RuntimeError("repetitions must be positive")
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
