#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Prepare and launch paired CBC studies using an existing cluster environment."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid

from study_build import digest, write_json


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def command(parts, **kwargs):
    return subprocess.run(list(map(str, parts)), check=True, **kwargs)


def environment(parts, filename):
    shell = ["zsh", "-c"] if str(filename).endswith(".zsh") else ["bash", "-lc"]
    return [*shell,
            'set -euo pipefail; source "$1"; shift; exec "$@"',
            "cbc-study", str(filename), *map(str, parts)]


def tool(settings, action, case, *parts):
    return [settings["python"], settings["tools"], action, "--root", case, *map(str, parts)]


def case_names(label):
    if label == "trunk-host":
        return ("reference",)
    return ("fusion-off", "fusion-on") if label.endswith("device") else (
        "lagging-off", "lagging-on")


def partition_nodes(config, partition, label):
    return config.get(partition + "_nodes_by_label", {}).get(label, config[partition + "_nodes"])


def prepare(args):
    config = read(args.config)
    site = config["site"]
    if site not in ("dane", "tuolumne"):
        raise ValueError("site must be dane or tuolumne")
    repo = Path(config["repo"]).resolve(strict=True)
    env_file = Path(config["environment"]).resolve(strict=True)
    token = stamp()
    work = Path(config["work_parent"]).resolve() / token
    root = Path(config["results_parent"]).resolve() / f"cbc-scaling-{site}-{token}"
    work.mkdir(parents=True, exist_ok=False)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(env_file, root / "environment.txt")
    python = subprocess.check_output(environment(
        ["python", "-c", "import sys; print(sys.executable)"], env_file), text=True).strip()
    settings = dict(config, work=str(work), root=str(root), python=python,
                    driver=str(work / "cbc_cluster_study.py"),
                    tools=str(work / "tools/cbc_study.py"), cases={}, sources={})
    support = work / "tools"
    support.mkdir()
    for name in ("cbc_cluster_study.py", "cbc_study.py", "cbc_study_overlay.py",
                 "cbc_study_input.py.in", "study_build.py"):
        shutil.copy2(HERE / name, support / name)
    shutil.copy2(HERE / "cbc_cluster_study.py", work / "cbc_cluster_study.py")
    shutil.copytree(support, root / "tools")
    smoke_mesh = str(Path(config["smoke_mesh"]).resolve(strict=True))
    settings["smoke_mesh"] = smoke_mesh
    nodes = set(config["pdebug_nodes"] + config["pbatch_nodes"])
    for partition in ("pdebug", "pbatch"):
        for selected in config.get(partition + "_nodes_by_label", {}).values():
            nodes.update(selected)
    if any(n < 1 or n > 256 for n in nodes):
        raise ValueError("Select node counts between one and 256")
    host_meshes = read(config["host_mesh_manifest"])["meshes"]
    device_meshes = read(config["device_mesh_manifest"])["meshes"] if site == "tuolumne" else None
    default_labels = ["trunk-host", "cycles-host", "compact-host"]
    if site == "tuolumne":
        default_labels.append("compact-device")
    requested_labels = config.get("labels", default_labels)
    if not requested_labels or set(requested_labels) - set(default_labels):
        raise ValueError("Invalid implementation labels for this cluster")
    for impl in ("trunk", "cycles", "compact"):
        if not any(label.startswith(impl + "-") for label in requested_labels):
            continue
        revision = config[impl + "_revision"]
        full = subprocess.check_output(["git", "-C", repo, "rev-parse", revision + "^{commit}"],
                                       text=True).strip()
        source = work / ("source-" + impl)
        command(["git", "-C", repo, "worktree", "add", "--detach", source, full])
        settings["sources"][impl] = dict(path=str(source), revision=full)
        labels = [impl + "-host"]
        if impl == "compact" and site == "tuolumne":
            labels.append("compact-device")
        labels = [label for label in labels if label in requested_labels]
        for label in labels:
            gpu = label.endswith("device")
            rpn = 4 if gpu else config.get("host_ranks_per_node", 64 if site == "dane" else 84)
            threads = 21 if gpu else 1
            meshes = device_meshes if gpu else host_meshes
            for selection in case_names(label):
                key = label + "/" + selection
                for partition in ("pdebug", "pbatch", "smoke"):
                    selected_nodes = ([1] if partition == "smoke" else
                                      partition_nodes(config, partition, label))
                    root_case = root / partition / key
                    manifest = dict(source=str(source), nodes=selected_nodes,
                                    ranks_per_node=(4 if gpu else 2)
                                    if partition == "smoke" else rpn,
                                    threads=threads, trials=1
                                    if partition == "smoke" else config.get("trials", 3),
                                    iterations=16, modes=["baseline", "caliper-mpi", "pmpi"]
                                    + (["rocprof"] if gpu else []),
                                    reuse_build=config["device_reuse_build"
                                                       if gpu else "host_reuse_build"],
                                    build_root=str(work / ("build-" + label)),
                                    use_gpus=gpu, allow_cycles=selection == "lagging-on",
                                    fuse_worker_launches=selection == "fusion-on",
                                    meshes={kind: {str(n): smoke_mesh if partition == "smoke" else
                                            meshes[kind][str(n)] for n in selected_nodes}
                                            for kind in ("strong", "weak")})
                    actual_rpn = manifest["ranks_per_node"]
                    if site == "dane":
                        manifest["launcher"] = [
                            "srun", "--mpi=pmix", "--kill-on-bad-exit=1",
                            "--nodes={nodes}", "--ntasks={ranks}",
                            f"--ntasks-per-node={actual_rpn}", "--cpus-per-task=1",
                            "--distribution=block", "--mpibind=on"]
                        if partition == "smoke":
                            manifest["launcher"].append("--overlap")
                        manifest["cancel"] = ["scancel", "{jobid}"]
                    else:
                        manifest["launcher"] = [
                            "flux", "run", "-N", "{nodes}", "-n", "{ranks}",
                            "--exclusive", "-o", f"mpibind=greedy:0,smt:1,gpu_optim:{int(gpu)}",
                            "-o", "exit-on-error",
                            "-t", "{seconds}s"]
                        manifest["cancel"] = ["flux", "cancel", "{jobid}"]
                        manifest["rocprof_launcher_options"] = ["-o", "exit-timeout=none"]
                    filename = work / (partition + "-" + label + "-" + selection + ".json")
                    write_json(filename, manifest)
                    command(environment(tool(settings, "prepare", root_case,
                                             "--config", filename), env_file))
                    settings["cases"][partition + "/" + key] = str(root_case)
    settings["environment_sha256"] = digest(env_file)
    settings["tool_hashes"] = {str(path): digest(path) for path in support.iterdir()}
    settings["tool_hashes"][settings["driver"]] = digest(settings["driver"])
    write_json(work / "settings.json", settings)
    write_json(root / "settings.json", settings)
    print("Settings:", work / "settings.json", flush=True)
    print("Results:", root, flush=True)


def inventory(settings, directory):
    directory.mkdir(parents=True, exist_ok=False)
    commands = {"cpu.json": ["lscpu", "--json"],
                "numa.txt": ["numactl", "--hardware"],
                "modules.txt": ["bash", "-c", "module list 2>&1"],
                "mpi.txt": ["mpicxx", "--showme:version"
                            if settings["site"] == "dane" else "-show"],
                "python.txt": [
                    settings["python"], "-c",
                    "import sys, mpi4py; mpi4py.rc.initialize=False; mpi4py.rc.finalize=False; "
                    "from mpi4py import MPI; print(sys.executable, sys.version); "
                    "print(mpi4py.__version__); "
                    "print(MPI.Get_library_version())"]}
    for name, parts in commands.items():
        with (directory / name).open("x") as log:
            try:
                result = subprocess.run(parts, stdout=log, stderr=subprocess.STDOUT,
                                        check=False, timeout=60)
                print("Inventory exit:", result.returncode, file=log)
            except (OSError, subprocess.TimeoutExpired) as error:
                print("Inventory unavailable:", error, file=log)
    write_json(directory / "environment.json", {
        k: v for k, v in os.environ.items()
        if k.startswith(("SLURM_", "FLUX_", "MPIBIND", "MPICH_", "OMPI_", "HSA_"))
        or k in ("OMP_NUM_THREADS", "OPENSN_NUM_THREADS", "LD_LIBRARY_PATH", "PATH")})


def attach_builds(settings, label):
    original = Path(settings["cases"]["pdebug/" + label + "/" + case_names(label)[0]])
    for key, value in settings["cases"].items():
        if "/" + label + "/" not in key or Path(value) == original:
            continue
        for variant in ("native", "profile"):
            target = Path(value) / "builds" / variant
            target.mkdir(parents=True, exist_ok=True)
            record = read(original / "builds" / variant / "build.json")
            if (target / "build.json").exists():
                if read(target / "build.json") != record:
                    raise ValueError("Existing binary record differs")
            else:
                write_json(target / "build.json", record)


def allocated_build(settings, label):
    root = Path(settings["root"])
    inventory(settings, root / ("build-environment-" + label + "-" + stamp()))
    case = settings["cases"]["pdebug/" + label + "/" + case_names(label)[0]]
    command(tool(settings, "build", case, "--jobs", settings.get("build_jobs", 16)))
    attach_builds(settings, label)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_CPU_BIND", "SLURM_MEM_BIND"))}
    for selection in case_names(label):
        case = settings["cases"]["smoke/" + label + "/" + selection]
        command(tool(settings, "run", case, "--kind", "strong", "--nodes", 1,
                     "--seconds", 900, "--allocation", os.environ.get("SLURM_JOB_ID", "flux")),
                env=env)
    marker = root / (label + "-BUILD-PASSED.json")
    if not marker.exists():
        write_json(marker, dict(label=label, completed=stamp()))
    print("Build and paired smoke modes passed:", label, flush=True)


def driver_parts(settings, action, label, *parts):
    return environment([settings["python"], settings["driver"], action, label,
                        "--settings", str(Path(settings["work"]) / "settings.json"),
                        *map(str, parts)],
                       settings["environment"])


def build(settings, label):
    parts = driver_parts(settings, "_build", label)
    if settings["site"] == "dane":
        launch = ["salloc", "-p", "pdebug", "-A", settings["account"], "-N", "1", "-n", "1",
                  "-c", "16", "--exclusive", "--mem=0", "-t", "01:00:00", "srun", "--mpi=none",
                  "-N", "1", "-n", "1", "-c", "16", "--mpibind=off", "--cpu-bind=none"]
    else:
        launch = ["flux", "alloc", "-q", "pdebug", "-B", settings["account"], "-N", "1",
                  "--exclusive", "--amd-gpumode=SPX", "-t", "1h", "flux", "run", "-N", "1",
                  "-n", "1", "--exclusive", "-o", "spindle.level=off"]
    log_path = Path(settings["root"]) / ("build-" + label + "-" + stamp() + ".log")
    print("Build log:", log_path, flush=True)
    with log_path.open("x") as log:
        command(launch + parts, stdout=log, stderr=subprocess.STDOUT)


def allocated_run(settings, label, kind, nodes, partition, allocation):
    deadline = time.monotonic() + settings.get("walltime_seconds", 3600) - 90
    inventory(settings, Path(allocation) / "environment")
    selections = case_names(label)
    first = settings["cases"][partition + "/" + label + "/" + selections[0]]
    config = read(Path(first) / "manifest.json")
    for trial in range(1, config["trials"] + 1):
        for mode in config["modes"]:
            for selection in selections[::(-1 if trial % 2 == 0 else 1)]:
                remaining = deadline - time.monotonic()
                if remaining < 150:
                    return 3
                case = settings["cases"][partition + "/" + label + "/" + selection]
                command(tool(settings, "run", case, "--kind", kind, "--nodes", nodes,
                             "--seconds", remaining, "--mode", mode, "--trial", trial,
                             "--allocation", os.environ.get("SLURM_JOB_ID", "flux")))
    return 0


def launch(settings, label, kind, nodes, partition, batch):
    if nodes not in partition_nodes(settings, partition, label):
        raise ValueError("Node count was not prepared for this partition")
    if (settings["site"] == "dane" and label in ("trunk-host", "cycles-host")
            and kind == "strong" and nodes == 1):
        print("Skipping known trunk/cycles strong-one-node capacity failure", flush=True)
        return
    if not (Path(settings["root"]) / (label + "-BUILD-PASSED.json")).exists():
        raise ValueError("Build and paired smoke tests must pass before launch")
    if batch:
        existing = Path(settings["root"]) / f"submitted-{label}-{kind}-{nodes}.json"
        if existing.exists():
            print("Already submitted:", read(existing)["job_id"], flush=True)
            return
    directory = Path(settings["root"]) / "allocations" / (f"{label}-{kind}-{nodes}n-" + stamp())
    directory.mkdir(parents=True, exist_ok=False)
    parts = driver_parts(settings, "_run", label, kind, nodes, partition, directory)
    hours = settings.get("walltime_seconds", 3600) / 3600
    if settings["site"] == "dane":
        script = directory / "job.sh"
        script.write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(parts) + "\n")
        launch_args = ["sbatch" if batch else "salloc", "-p", partition, "-A", settings["account"],
                       "-N", str(nodes),
                       f"--ntasks-per-node={settings.get('host_ranks_per_node', 64)}",
                       "--cpus-per-task=1", "--exclusive", "--mem=0", "-t",
                       f"{int(hours):02d}:{round((hours % 1) * 60):02d}:00"]
        if batch:
            launch_args += ["--parsable", "-J", f"cbc-{label}-{kind}-{nodes}",
                            "-o", str(directory / "driver.log"), "-e",
                            str(directory / "driver.err"), str(script)]
        else:
            launch_args += ["srun", "--mpi=none", "-N", "1", "-n", "1", "-c", "1", "--overlap",
                            "--mpibind=off", "--cpu-bind=none", "bash", str(script)]
    else:
        launch_args = ["flux", "batch" if batch else "alloc", "-q", partition,
                       "-B", settings["account"],
                       "-N", str(nodes), "--exclusive", "--amd-gpumode=SPX", "-t", f"{hours:g}h"]
        if batch:
            script = directory / "job.sh"
            script.write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(parts) + "\n")
            launch_args += ["--output", str(directory / "driver.log"), str(script)]
        else:
            launch_args += parts
    write_json(directory / "launch.json", dict(
        command=launch_args, paired_cases=list(case_names(label))))
    if batch:
        job_id = subprocess.check_output(launch_args, text=True).strip()
        write_json(existing, dict(job_id=job_id, allocation=str(directory)))
        print(label, kind, nodes, "submitted:", job_id, flush=True)
    else:
        print("Interactive log:", directory / "driver.log", flush=True)
        with (directory / "driver.log").open("x") as log:
            result = subprocess.run(launch_args, stdout=log, stderr=subprocess.STDOUT, check=False)
        print("Allocation exit:", result.returncode, flush=True)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, launch_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    for action in ("build", "run", "submit", "status", "_build", "_run"):
        p = sub.add_parser(action)
        p.add_argument("--settings", type=Path, required=True)
        if action == "submit":
            p.add_argument("--kind", choices=("strong", "weak"))
        if action in ("build", "run", "_build", "_run"):
            p.add_argument("label", choices=("trunk-host", "cycles-host", "compact-host",
                                             "compact-device"))
        if action in ("run", "_run"):
            p.add_argument("kind", choices=("strong", "weak"))
            p.add_argument("nodes", type=int)
        if action == "_run":
            p.add_argument("partition", choices=("pdebug", "pbatch"))
            p.add_argument("allocation", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args)
    settings = read(args.settings)
    if digest(settings["environment"]) != settings["environment_sha256"]:
        raise ValueError("The recorded environment file changed")
    for path, expected in settings["tool_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Recorded study helper changed: {path}")
    if args.action in ("build", "_build"):
        return (build if args.action == "build" else allocated_build)(settings, args.label)
    if args.action == "run":
        return launch(settings, args.label, args.kind, args.nodes, "pdebug", False)
    if args.action == "_run":
        return allocated_run(settings, args.label, args.kind, args.nodes,
                             args.partition, args.allocation)
    if args.action == "submit":
        labels = sorted({key.split('/')[1] for key in settings["cases"]})
        for label in labels:
            for kind in ((args.kind,) if args.kind else ("strong", "weak")):
                for nodes in partition_nodes(settings, "pbatch", label):
                    launch(settings, label, kind, nodes, "pbatch", True)
    if args.action == "status":
        for key, value in settings["cases"].items():
            if not key.startswith("smoke/"):
                print(key, flush=True)
                command(environment(tool(settings, "status", value), settings["environment"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
