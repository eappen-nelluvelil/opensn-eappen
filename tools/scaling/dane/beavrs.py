#!/usr/bin/env python3

"""Prepare, submit, and collect a host-CBC BEAVRS run on Dane."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = (
    "beavrs_quarter_core_cpu.py",
    "beavrs_quarter_core_partitioned.obj",
    "beavrs_CASMO-70.h5",
)


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"expected one {description} marker, found {text.count(old)}"
        )
    return text.replace(old, new, 1)


def make_input(source: Path) -> str:
    text = source.read_text()
    text = replace_once(
        text,
        '            "angular_quadrature": quadrature,\n',
        '            "angular_quadrature": quadrature,\n'
        '            "angle_aggregation_type": "single",\n'
        '            "allow_cycles": True,\n',
        "groupset quadrature",
    )
    text = replace_once(
        text,
        '            "save_angular_flux": False,\n',
        '            "max_mpi_message_size": 256 * 1024,\n'
        '            "save_angular_flux": False,\n',
        "angular-flux option",
    )
    text = replace_once(
        text,
        "        use_gpus=USE_GPUS,\n",
        '        sweep_type="CBC",\n'
        "        use_gpus=USE_GPUS,\n",
        "problem backend",
    )
    compile(text, str(source), "exec")
    return text


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing manifest: {path}")
    return json.loads(path.read_text())


def make_job(record: dict) -> str:
    output = Path(record["output"])
    scaling = record["scaling_manifest"]
    implementation = scaling["implementations"]["branch"]
    environment = scaling["environment"]
    source_environment = (
        f"source {shell_quote(environment)}\n" if environment else ""
    )
    nodes = record["nodes"]
    ranks_per_node = record["ranks_per_node"]
    ranks = nodes * ranks_per_node
    benchmark = Path(record["benchmark_source"])
    build = Path(implementation["build"])
    binary = Path(implementation["binary"])
    input_file = output / "beavrs_quarter_core_cbc.py"
    result_root = output / "results"
    return f'''#!/bin/bash -l
#SBATCH --job-name=beavrs-cbc-{nodes}n
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ranks_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --partition=pbatch
#SBATCH --account={scaling["bank"]}
#SBATCH --exclusive
#SBATCH --time={record["time_limit"]}
#SBATCH --output={shell_quote(output / "slurm" / "%j.out")}
#SBATCH --error={shell_quote(output / "slurm" / "%j.err")}

set -euo pipefail
{source_environment}export OPENSN_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

binary={shell_quote(binary)}
result={shell_quote(result_root)}/run-$SLURM_JOB_ID-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$result"
ln -s {shell_quote(benchmark / "beavrs_quarter_core_partitioned.obj")} "$result/"
ln -s {shell_quote(benchmark / "beavrs_CASMO-70.h5")} "$result/"
cp {shell_quote(input_file)} "$result/input.py"

grep -qx 'CMAKE_BUILD_TYPE:STRING=Native' {shell_quote(build / "CMakeCache.txt")}
test -x "$binary"
test "$SLURM_JOB_NUM_NODES" -eq {nodes}

{{
  echo "revision={implementation['sha']}"
  echo "build_type=Native"
  echo "nodes={nodes}"
  echo "ranks={ranks}"
  echo "ranks_per_node={ranks_per_node}"
  echo "opensn_num_threads=1"
  echo "slurm_job_id=$SLURM_JOB_ID"
  date --iso-8601=seconds
}} > "$result/job-metadata.txt"

cd "$result"
set +e
/usr/bin/time -f 'wall_seconds=%e launcher_max_rss_kb=%M' -o "$result/time.txt" \
  srun \
    --nodes={nodes} \
    --ntasks={ranks} \
    --ntasks-per-node={ranks_per_node} \
    --cpus-per-task=1 \
    --distribution=block \
    --mpibind=on \
    "$binary" --verbose 1 -i "$result/input.py" \
    > "$result/stdout.txt" \
    2> "$result/stderr.txt"
exit_code=$?
set -e
echo "$exit_code" > "$result/exit_code.txt"
if (( exit_code != 0 )) || ! grep -Fq 'OpenSn finished execution.' "$result/stdout.txt"; then
  touch "$result/FAILED"
  exit $((exit_code == 0 ? 1 : exit_code))
fi
touch "$result/SUCCESS"
'''


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"campaign directory already exists: {output}")
    if args.nodes <= 0 or args.ranks_per_node <= 0:
        raise RuntimeError("nodes and ranks per node must be positive")
    if not re.fullmatch(r"(?:\d+-)?\d{1,2}:\d{2}:\d{2}", args.time_limit):
        raise RuntimeError("time limit must use [days-]hours:minutes:seconds")

    scaling_root = args.scaling_root.resolve()
    scaling = load_json(scaling_root / "manifest.json")
    benchmark = args.benchmark_source.resolve()
    for name in REQUIRED_FILES:
        if not (benchmark / name).is_file():
            raise RuntimeError(f"missing BEAVRS input: {benchmark / name}")

    for directory in (output, output / "slurm", output / "results"):
        directory.mkdir(parents=True, exist_ok=True)
    generated_input = output / "beavrs_quarter_core_cbc.py"
    generated_input.write_text(make_input(benchmark / "beavrs_quarter_core_cpu.py"))

    record = {
        "schema": 1,
        "output": str(output),
        "scaling_root": str(scaling_root),
        "scaling_manifest": scaling,
        "benchmark_source": str(benchmark),
        "nodes": args.nodes,
        "ranks_per_node": args.ranks_per_node,
        "time_limit": args.time_limit,
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    job = output / "beavrs-cbc.sbatch"
    job.write_text(make_job(record))
    job.chmod(0o700)
    print(f"Prepared {job}")


def scaling_build_id(record: dict) -> str:
    path = Path(record["scaling_root"]) / "job-ids.tsv"
    if not path.is_file():
        raise RuntimeError(f"scaling campaign has not been submitted: {path}")
    for line in path.read_text().splitlines():
        name, job_id = line.split("\t", 1)
        if name == "build" and job_id.isdigit():
            return job_id
    raise RuntimeError(f"missing build job ID in {path}")


def submit(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    record = load_json(output / "manifest.json")
    job_id_path = output / "job-id.txt"
    if job_id_path.exists():
        raise RuntimeError(f"BEAVRS job was already submitted: {job_id_path}")
    build_id = scaling_build_id(record)
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{build_id}",
            str(output / "beavrs-cbc.sbatch"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"could not parse Slurm job ID from {result.stdout!r}")
    job_id_path.write_text(job_id + "\n")
    print(f"submitted beavrs-cbc: {job_id} (afterok:{build_id})")


def read_job_id(output: Path) -> str:
    path = output.resolve() / "job-id.txt"
    if not path.is_file():
        raise RuntimeError(f"missing BEAVRS job ID: {path}")
    return path.read_text().strip()


def status(args: argparse.Namespace) -> None:
    job_id = read_job_id(args.output)
    subprocess.run(
        ["squeue", "--jobs", job_id, "--format=%.18i %.32j %.9T %.10M %.6D %R"],
        check=False,
    )
    subprocess.run(
        [
            "sacct",
            "--jobs",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName,State,Elapsed,NNodes,MaxRSS",
        ],
        check=False,
    )


def collect(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    runs = sorted(
        (path for path in (output / "results").glob("run-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise RuntimeError(f"no BEAVRS runs found under {output / 'results'}")
    run = runs[0]
    if (run / "SUCCESS").is_file():
        state = "SUCCESS"
    elif (run / "FAILED").is_file():
        state = "FAILED"
    else:
        state = "INCOMPLETE"
    lines = [
        "# Dane host-CBC BEAVRS result",
        "",
        f"- State: {state}",
        f"- Result: `{run}`",
        "",
        "```text",
    ]
    if (run / "stdout.txt").is_file():
        pattern = re.compile(
            r"k_eff|Identified .* pins|Pin power min/mean/max|avg_sweep_time|"
            r"sweep_time_per_unknown|OpenSn finished"
        )
        lines.extend(
            line.rstrip()
            for line in (run / "stdout.txt").read_text(errors="replace").splitlines()
            if pattern.search(line)
        )
    lines.extend(["```", ""])
    (output / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--scaling-root", type=Path, required=True)
    prepare_command.add_argument("--output", type=Path, required=True)
    prepare_command.add_argument("--benchmark-source", type=Path, required=True)
    prepare_command.add_argument("--nodes", type=int, default=32)
    prepare_command.add_argument("--ranks-per-node", type=int, default=64)
    prepare_command.add_argument("--time-limit", default="24:00:00")
    prepare_command.set_defaults(action=prepare)
    for name, action in (("submit", submit), ("status", status), ("collect", collect)):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.set_defaults(action=action)
    return top


def main() -> int:
    args = parser().parse_args()
    try:
        args.action(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
