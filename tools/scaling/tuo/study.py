#!/usr/bin/env python3

"""Prepare and collect device-CBC scaling and profiling studies on Tuolumne."""

import argparse
import csv
import json
import os
import re
import shlex
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
            f"no weak-scaling divisor is defined for {sorted(unsupported)}"
        )
    return nodes


def executable(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"not an executable file: {path}")
    return path


def write_executable(path, content):
    path.write_text(content)
    path.chmod(0o700)


def quote(value):
    return shlex.quote(str(value))


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


def write_input(path, mesh, max_iterations, save_angular_flux):
    content = TEMPLATE.read_text()
    replacements = {
        "@MESH@": repr(str(mesh)),
        "@XS@": repr(str(XS.resolve())),
        "@MAX_ITERATIONS@": str(max_iterations),
        "@SAVE_ANGULAR_FLUX@": repr(save_angular_flux),
    }
    for token, value in replacements.items():
        content = content.replace(token, value)
    path.write_text(content)


def flux_header(label, nodes, tasks, queue, bank, time_limit, stdout, stderr):
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


def runtime_environment(environment):
    return f"""source {quote(environment)}
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
unset OPENSN_NUM_THREADS OPENSN_CBCD_NUM_WORKERS OMP_NUM_THREADS
"""


def scaling_job(args, study, kind, nodes, input_path):
    ranks = nodes * 4
    result_dir = study / "results" / kind / f"nodes-{nodes}"
    scheduler = study / "scheduler"
    header = flux_header(
        f"{args.label}-{kind[0]}-{nodes}",
        nodes,
        ranks,
        args.queue,
        args.bank,
        args.time_limit,
        scheduler / f"{kind}-{nodes}-{{{{id}}}}.out",
        scheduler / f"{kind}-{nodes}-{{{{id}}}}.err",
    )
    return header + runtime_environment(args.environment) + f"""
binary={quote(args.binary)}
input={quote(input_path)}
result_dir={quote(result_dir)}
mkdir -p -- "$result_dir"

for trial in {{1..{args.repetitions}}}; do
  stem="$result_dir/trial-$trial"
  /usr/bin/time -f 'wall_seconds=%e max_rss_kb=%M' -o "$stem.time" \\
    flux run -N {nodes} -n {ranks} --exclusive \\
      "$binary" -i "$input" > "$stem.out" 2> "$stem.err"
done
"""


def prepare(args):
    study = args.output.expanduser().resolve()
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
            input_path = study / "inputs" / f"{kind}-{nodes}.py"
            write_input(
                input_path,
                meshes[divisor],
                args.max_iterations,
                args.save_angular_flux,
            )
            job = study / "jobs" / f"{kind}-{nodes}.zsh"
            write_executable(job, scaling_job(args, study, kind, nodes, input_path))
            jobs.append(job)
            cases.append(
                {
                    "kind": kind,
                    "nodes": nodes,
                    "ranks": nodes * 4,
                    "divisor": divisor,
                    "mesh": str(meshes[divisor]),
                    "input": str(input_path),
                    "job": str(job),
                }
            )

    write_executable(
        study / "submit.zsh",
        "#!/bin/zsh\nset -euo pipefail\n\n"
        + "\n".join(f"flux batch {quote(job)}" for job in jobs)
        + "\n",
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": "tuolumne",
        "label": args.label,
        "revision": args.revision,
        "binary": str(args.binary),
        "environment": str(args.environment),
        "nodes": args.nodes,
        "ranks_per_node": 4,
        "gpus_per_rank": 1,
        "repetitions": args.repetitions,
        "strong_divisor": args.strong_divisor,
        "weak_divisors": {str(node): WEAK_DIVISORS[node] for node in args.nodes},
        "max_iterations": args.max_iterations,
        "save_angular_flux": args.save_angular_flux,
        "cases": cases,
    }
    (study / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(jobs)} Flux jobs in {study}")


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
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


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
        base_nodes = min(node for study_kind, node in groups if study_kind == kind)
        base = medians[(kind, base_nodes)]
        metric = medians[(kind, nodes)]
        efficiency = base / metric * (base_nodes / nodes if kind == "strong" else 1.0)
        summary.append(
            {
                "kind": kind,
                "nodes": nodes,
                "ranks": nodes * 4,
                "trials": len(values),
                "metric": metric,
                "metric_unit": "ns/unknown" if kind == "strong" else "s",
                "efficiency_percent": efficiency * 100.0,
                "median_wall_time_s": statistics.median(value["wall_time_s"] for value in values),
                "median_lagged_unknowns": statistics.median(
                    value["lagged_unknowns"] for value in values
                ),
            }
        )
    return summary


def write_summary(path, label, rows):
    lines = [
        f"# {label} Tuolumne scaling results",
        "",
        "| Study | Nodes | Ranks | Trials | Metric | Unit | Efficiency | Wall time (s) | Lagged unknowns |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['kind']} | {row['nodes']} | {row['ranks']} | {row['trials']} "
            f"| {float(row['metric']):.8g} | {row['metric_unit']} "
            f"| {float(row['efficiency_percent']):.2f}% "
            f"| {float(row['median_wall_time_s']):.3f} "
            f"| {float(row['median_lagged_unknowns']):.8g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_series(output, title_prefix, series):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; summaries were written without plots")
        return

    colors = ("#1f77b4", "#d95f02", "#2a9d8f", "#7f3c8d")
    for kind in ("strong", "weak"):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for (label, rows), color in zip(series, colors):
            selected = [row for row in rows if row["kind"] == kind]
            if not selected:
                continue
            nodes = [int(row["nodes"]) for row in selected]
            metric = [float(row["metric"]) for row in selected]
            ax.plot(nodes, metric, marker="o", color=color, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ticks = sorted({int(row["nodes"]) for _, rows in series for row in rows if row["kind"] == kind})
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Nodes (4 MPI ranks/node, 1 MI300A/rank)")
        ax.set_ylabel(
            "Average sweep time per unknown (ns)" if kind == "strong" else "Average sweep time (s)"
        )
        ax.set_title(f"{title_prefix} {kind} scaling")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{kind}.pdf")
        plt.close(fig)


def collect(args):
    study = args.study.expanduser().resolve()
    manifest = json.loads((study / "manifest.json").read_text())
    rows = []
    missing = []
    failed = []
    for case in manifest["cases"]:
        for trial in range(1, manifest["repetitions"] + 1):
            stem = study / "results" / case["kind"] / f"nodes-{case['nodes']}" / f"trial-{trial}"
            output = stem.with_suffix(".out")
            timing = stem.with_suffix(".time")
            if not output.is_file() or not timing.is_file():
                missing.append(str(stem))
                continue
            try:
                values = read_result(output, timing)
            except RuntimeError:
                failed.append(str(stem))
                continue
            rows.append(
                {
                    "kind": case["kind"],
                    "nodes": case["nodes"],
                    "ranks": case["ranks"],
                    "trial": trial,
                    **values,
                }
            )

    if (missing or failed) and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(missing)} result sets are missing and {len(failed)} failed; "
            "use --allow-incomplete to collect successful runs"
        )
    write_rows(study / "results.csv", rows)
    summary = summarize(rows)
    write_rows(study / "summary.csv", summary)
    write_summary(study / "summary.md", manifest["label"], summary)
    plot_series(study, "Tuolumne device CBC", [(manifest["label"], summary)])
    (study / "collection.json").write_text(
        json.dumps({"missing": missing, "failed": failed}, indent=2) + "\n"
    )
    print(f"Collected {len(rows)} successful runs in {study}")


def read_summary(study):
    study = study.expanduser().resolve()
    manifest = json.loads((study / "manifest.json").read_text())
    with (study / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return manifest, rows


def compare(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_manifest, baseline = read_summary(args.baseline)
    candidate_manifest, candidate = read_summary(args.candidate)
    series = [
        (baseline_manifest["label"], baseline),
        (candidate_manifest["label"], candidate),
    ]
    plot_series(output, "Tuolumne device CBC", series)

    lookup = {
        (row["kind"], int(row["nodes"])): float(row["metric"])
        for row in baseline
    }
    comparisons = []
    for row in candidate:
        key = (row["kind"], int(row["nodes"]))
        if key not in lookup:
            continue
        candidate_metric = float(row["metric"])
        comparisons.append(
            {
                "kind": key[0],
                "nodes": key[1],
                "baseline": lookup[key],
                "candidate": candidate_metric,
                "candidate_over_baseline": candidate_metric / lookup[key],
            }
        )
    write_rows(output / "comparison.csv", comparisons)
    lines = [
        "# Tuolumne device CBC scaling comparison",
        "",
        f"Baseline: `{baseline_manifest['label']}` (`{baseline_manifest['revision']}`)",
        "",
        f"Candidate: `{candidate_manifest['label']}` (`{candidate_manifest['revision']}`)",
        "",
        "| Study | Nodes | Baseline metric | Candidate metric | Candidate / baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['kind']} | {row['nodes']} | {row['baseline']:.8g} "
            f"| {row['candidate']:.8g} | {row['candidate_over_baseline']:.4f} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Comparison written to {output}")


def profile_job(args, profile, nodes, study, input_path):
    tasks = 1 if profile == "omniperf" else nodes * 4
    result = study / "results" / profile / f"nodes-{nodes}"
    scheduler = study / "scheduler"
    header = flux_header(
        f"{args.label}-{profile}-{nodes}",
        nodes,
        tasks,
        args.queue,
        args.bank,
        args.time_limit,
        scheduler / f"{profile}-{nodes}-{{{{id}}}}.out",
        scheduler / f"{profile}-{nodes}-{{{{id}}}}.err",
    )
    common = runtime_environment(args.environment) + f"""
binary={quote(args.binary)}
input={quote(input_path)}
result={quote(result)}
mkdir -p -- "$result"
"""
    if profile == "baseline":
        command = f'flux run -N {nodes} -n {tasks} --exclusive "$binary" -i "$input" > "$result/stdout.txt" 2> "$result/stderr.txt"\n'
    elif profile == "caliper":
        config = f'runtime-report(output="{result}/profile.txt",aggregate_across_ranks,calc.inclusive,print.metadata,order_by_time,max_column_width=180,region.count)'
        command = f'flux run -N {nodes} -n {tasks} --exclusive "$binary" --caliper={quote(config)} -i "$input" > "$result/stdout.txt" 2> "$result/stderr.txt"\n'
    elif profile == "pmpi":
        config = f'mpi-report(output="{result}/mpi.txt")'
        command = f'flux run -N {nodes} -n {tasks} --exclusive "$binary" --caliper={quote(config)} -i "$input" > "$result/stdout.txt" 2> "$result/stderr.txt"\n'
    elif profile == "rocprof":
        command = f"""export OPENSN_PROFILE_MODE=rocprof
export OPENSN_PROFILE_BINARY="$binary"
export OPENSN_PROFILE_INPUT="$input"
export OPENSN_PROFILE_OUTPUT="$result"
flux run -N 1 -n 4 --exclusive {quote(PROFILE_WRAPPER)} > "$result/stdout.txt" 2> "$result/stderr.txt"
"""
    elif profile == "hpctoolkit":
        command = f"""module load hpctoolkit
flux run -N 1 -n 4 --exclusive \\
  hpcrun -o "$result/measurements" -e CPUTIME@5000 -e gpu=rocm \\
  "$binary" -i "$input" > "$result/stdout.txt" 2> "$result/stderr.txt"
"""
    elif profile == "omniperf":
        command = f"""module load omniperf
cd "$result"
flux run -N 1 -n 1 -c 21 -g 1 --exclusive \\
  omniperf profile --name cbcd --no-roof -b SQ TCC TCP -k SweepKernel -- \\
  "$binary" -i "$input" > "$result/stdout.txt" 2> "$result/stderr.txt"
"""
    else:
        raise ValueError(profile)
    return header + common + command


def prepare_profile(args):
    study = args.output.expanduser().resolve()
    if study.exists() and any(study.iterdir()):
        raise RuntimeError(f"profile directory is not empty: {study}")
    if not args.environment.is_file():
        raise RuntimeError(f"environment script does not exist: {args.environment}")
    for path in (study / "inputs", study / "jobs", study / "results", study / "scheduler"):
        path.mkdir(parents=True, exist_ok=True)

    mesh = mesh_for(args.mesh_cache.expanduser().resolve(), args.gmsh, args.profile_divisor)
    input_path = study / "inputs" / "profile.py"
    write_input(input_path, mesh, args.max_iterations, args.save_angular_flux)
    profiles = ("baseline", "caliper", "pmpi", "rocprof", "hpctoolkit", "omniperf")
    jobs = []
    cases = []
    for profile in profiles:
        profile_nodes = (
            args.profile_nodes if profile in ("baseline", "caliper", "pmpi") else (1,)
        )
        for nodes in profile_nodes:
            job = study / "jobs" / f"{profile}-{nodes}.zsh"
            write_executable(job, profile_job(args, profile, nodes, study, input_path))
            jobs.append(job)
            cases.append(
                {
                    "profile": profile,
                    "nodes": nodes,
                    "ranks": 1 if profile == "omniperf" else nodes * 4,
                    "job": str(job),
                }
            )
    write_executable(
        study / "submit.zsh",
        "#!/bin/zsh\nset -euo pipefail\n\n"
        + "\n".join(f"flux batch {quote(job)}" for job in jobs)
        + "\n",
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": "tuolumne",
        "label": args.label,
        "revision": args.revision,
        "binary": str(args.binary),
        "environment": str(args.environment),
        "mesh": str(mesh),
        "profile_divisor": args.profile_divisor,
        "max_iterations": args.max_iterations,
        "save_angular_flux": args.save_angular_flux,
        "profiles": profiles,
        "profile_nodes": args.profile_nodes,
        "cases": cases,
    }
    (study / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(jobs)} independent Flux profile jobs in {study}")


def collect_profile(args):
    study = args.study.expanduser().resolve()
    manifest = json.loads((study / "manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        profile = case["profile"]
        result = study / "results" / profile / f"nodes-{case['nodes']}"
        stdout = result / "stdout.txt"
        sweep = None
        if stdout.is_file():
            matches = SWEEP_TIME_RE.findall(stdout.read_text(errors="replace"))
            if matches:
                sweep = float(matches[-1])
        rows.append(
            {
                "profile": profile,
                "nodes": case["nodes"],
                "ranks": case["ranks"],
                "completed": sweep is not None,
                "avg_sweep_time_s": sweep,
                "result_directory": str(result),
            }
        )
    write_rows(study / "profile-summary.csv", rows)
    lines = [
        f"# {manifest['label']} Tuolumne profile inventory",
        "",
        "| Profile | Nodes | Ranks | Completed | Average sweep time (s) | Result directory |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        sweep = "n/a" if row["avg_sweep_time_s"] is None else f"{row['avg_sweep_time_s']:.8g}"
        lines.append(
            f"| {row['profile']} | {row['nodes']} | {row['ranks']} | {row['completed']} "
            f"| {sweep} | `{row['result_directory']}` |"
        )
    (study / "profile-summary.md").write_text("\n".join(lines) + "\n")
    print(f"Profile inventory written to {study}")


def add_common_prepare_arguments(parser, profile=False):
    parser.add_argument("--binary", type=executable, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-cache", type=Path, required=True)
    parser.add_argument("--gmsh", type=executable, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--queue", default="pbatch")
    parser.add_argument("--bank")
    parser.add_argument("--time-limit", default="6h" if profile else "4h")
    parser.add_argument(
        "--save-angular-flux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain the full angular-flux vector (required by trunk device CBC)",
    )


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="prepare scaling jobs")
    add_common_prepare_arguments(prepare_parser)
    prepare_parser.add_argument("--nodes", type=parse_nodes, default=DEFAULT_NODES)
    prepare_parser.add_argument("--repetitions", type=int, default=3)
    prepare_parser.add_argument("--strong-divisor", type=int, default=39)
    prepare_parser.add_argument("--max-iterations", type=int, default=10)
    prepare_parser.set_defaults(function=prepare)

    collect_parser = commands.add_parser("collect", help="collect one scaling study")
    collect_parser.add_argument("--study", type=Path, required=True)
    collect_parser.add_argument("--allow-incomplete", action="store_true")
    collect_parser.set_defaults(function=collect)

    compare_parser = commands.add_parser("compare", help="compare two collected studies")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(function=compare)

    profile_parser = commands.add_parser("prepare-profile", help="prepare independent profile jobs")
    add_common_prepare_arguments(profile_parser, profile=True)
    profile_parser.add_argument("--profile-divisor", type=int, default=15)
    profile_parser.add_argument("--profile-nodes", type=parse_nodes, default=(1, 2, 4))
    profile_parser.add_argument("--max-iterations", type=int, default=2)
    profile_parser.set_defaults(function=prepare_profile)

    profile_collect = commands.add_parser("collect-profile", help="inventory profile outputs")
    profile_collect.add_argument("--study", type=Path, required=True)
    profile_collect.set_defaults(function=collect_profile)
    return top


def main():
    args = parser().parse_args()
    if hasattr(args, "repetitions") and args.repetitions <= 0:
        raise SystemExit("repetitions must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
