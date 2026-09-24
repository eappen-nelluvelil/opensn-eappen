#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Resumable host CBC studies with one MPI process lifetime per trial."""

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from study_build import build_variant, cache_values, digest, write_json
from host_cbc_study_overlay import annotate


HERE = Path(__file__).resolve().parent
MODES = ("baseline", "caliper-mpi", "pmpi")
FLOAT = r"[0-9.eE+\-]+"


def unique_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]


def clean_environment(environment):
    prefixes = ("CALI_", "OPENSN_MEMORY_", "OPENSN_CBCD_", "ROCPROF", "ROCTRACER",
                "SCOREP_", "TAU_", "NSYS_", "NCU_")
    excluded = {"LD_PRELOAD", "LD_AUDIT", "HSA_TOOLS_LIB", "HSA_TOOLS_LIB64",
                "ROCP_TOOL_LIBRARIES", "OMP_TOOL_LIBRARIES", "OMP_TOOL",
                "GLIBC_TUNABLES", "MALLOC_ARENA_MAX", "MALLOC_TRIM_THRESHOLD_"}
    env = {key: value for key, value in environment.items()
           if not key.startswith(prefixes) and key not in excluded}
    env.update(CALI_CONFIG="", CALI_SERVICES_ENABLE="", CALI_CONFIG_FILE="/dev/null",
               CALI_USE_OMPT="0", OMP_TOOL="disabled", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    return env


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def locked(path):
    with Path(path).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def revision(source):
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                   text=True).strip()


def prepare(args):
    config = read_json(args.config)
    for key in ("nodes", "ranks_per_node", "threads", "meshes", "launcher",
                "reuse_build", "build_root"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")
    if not config["nodes"] or any(not isinstance(n, int) or n < 1 for n in config["nodes"]):
        raise ValueError("nodes must be positive integers")
    if len(set(config["nodes"])) != len(config["nodes"]):
        raise ValueError("Duplicate node counts")
    for key in ("ranks_per_node", "threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    config["iterations"] = int(config.get("iterations", 16))
    config["trials"] = int(config.get("trials", 3))
    if config["iterations"] < 1 or config["trials"] < 1:
        raise ValueError("iterations and trials must be positive")
    config["modes"] = config.get("modes", list(MODES))
    if not config["modes"] or len(set(config["modes"])) != len(config["modes"]):
        raise ValueError("Select distinct modes")
    if set(config["modes"]) - set(MODES):
        raise ValueError("Unknown profile mode")
    if config["modes"][0] != "baseline":
        raise ValueError("baseline must be the first mode for numerical comparisons")
    if not isinstance(config["launcher"], list) or not config["launcher"]:
        raise ValueError("launcher must be an argument list")
    for part in config["launcher"]:
        part.format(nodes=1, ranks=config["ranks_per_node"], threads=config["threads"], seconds=60)
    source = Path(config["source"]).resolve(strict=True)
    config["source"] = str(source)
    config["revision"] = revision(source)
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD"], check=True)
    config["reuse_build"] = str(Path(config["reuse_build"]).resolve(strict=True))
    config["build_root"] = str(Path(config["build_root"]).resolve())
    if Path(config["build_root"]).exists():
        raise ValueError("Use a new build root")
    cache = cache_values(Path(config["reuse_build"]) / "CMakeCache.txt")
    for backend in ("CUDA", "HIP", "SYCL"):
        if cache.get("OPENSN_WITH_" + backend, (None, "OFF"))[1] not in ("OFF", "FALSE", "0"):
            raise ValueError("Reuse a CPU-only dependency configuration")
    root = args.root.resolve()
    assets = {}
    for kind in ("strong", "weak"):
        for nodes in config["nodes"]:
            mesh = Path(config["meshes"][kind][str(nodes)]).resolve(strict=True)
            config["meshes"][kind][str(nodes)] = str(mesh)
            if str(mesh) not in assets:
                assets[str(mesh)] = digest(mesh)
    xs = source / "tools/scaling/lib/xs_168g.xs"
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    shutil.copy2(xs, root / "inputs/xs_168g.xs")
    template = (HERE / "host_cbc_study_input.py.in").read_text()
    for kind in ("strong", "weak"):
        for nodes in config["nodes"]:
            text = template.replace("@MESH@", repr(config["meshes"][kind][str(nodes)]))
            text = text.replace("@XS@", repr(str(root / "inputs/xs_168g.xs")))
            text = text.replace("@ITERATIONS@", str(config["iterations"]))
            compile(text, "input.py", "exec")
            path = root / "inputs" / f"{kind}-{nodes}.py"
            path.write_text(text)
            assets[str(path)] = digest(path)
    assets[str(root / "inputs/xs_168g.xs")] = digest(root / "inputs/xs_168g.xs")
    config["assets"] = assets
    config["helper_hashes"] = {str(HERE / name): digest(HERE / name) for name in (
        "host_cbc_study.py", "host_cbc_study_overlay.py", "study_build.py",
        "host_cbc_study_input.py.in")}
    config["reuse_cache_sha256"] = digest(Path(config["reuse_build"]) / "CMakeCache.txt")
    write_json(root / "manifest.json", config)
    print(f"Prepared: {root}", flush=True)


def verify_inputs(config):
    if revision(config["source"]) != config["revision"]:
        raise ValueError("Source revision changed. Prepare a new study.")
    subprocess.run(["git", "-C", config["source"], "diff", "--quiet", "HEAD"], check=True)
    for path, expected in (config["assets"] | config["helper_hashes"]).items():
        if digest(path) != expected:
            raise ValueError(f"Study input/helper changed: {path}")


def build(args):
    root = args.root.resolve(strict=True)
    config = read_json(root / "manifest.json")
    verify_inputs(config)
    previous = Path(config["reuse_build"])
    if digest(previous / "CMakeCache.txt") != config["reuse_cache_sha256"]:
        raise ValueError("The reused build configuration changed")
    with locked(root / ".build.lock"):
        for variant in ("native", "profile"):
            source = Path(config["source"])
            if variant == "profile":
                source = annotate(source, Path(config["build_root"]) / "profile-source",
                                  root / "profile-overlay.patch")
            output = root / "builds" / variant
            record = output / "build.json"
            if record.exists():
                verify_fingerprint(read_json(record))
            else:
                destination = Path(config["build_root"]) / variant
                hashes = build_variant(source, previous, destination, output, args.jobs)
                shutil.copy2(destination / "CMakeCache.txt", output / "CMakeCache.txt")
                write_json(record, dict(binary=str(destination / "python/opensn"),
                                        fingerprints=hashes, revision=config["revision"],
                                        configuration="Native", variant=variant))
    print("Unmodified Native baseline and annotated Native profile builds are ready", flush=True)


def verify_fingerprint(record):
    for path, expected in record["fingerprints"].items():
        if digest(path) != expected:
            raise ValueError(f"Recorded binary/library changed: {path}")


def measurements(text, iterations):
    if "OPENSN_STUDY_TRIAL_COMPLETE" not in text or "OpenSn finished execution." not in text:
        raise ValueError("Missing application completion markers")
    samples = re.findall(rf"avg_sweep_time = ({FLOAT}) s, "
                         rf"sweep_time_per_unknown = ({FLOAT}) ns", text)
    counts = re.findall(r"\bunknowns = (\d+), lagged_unknowns = (\d+)", text)
    fluxes = re.findall(rf"OPENSN_STUDY_FLUX_MAX group=(\d+) value=({FLOAT})", text)
    finals = re.findall(r"final, status = (\w+), iterations = (\d+)", text)
    if len(samples) != 1 or len(counts) != 1 or finals != [("iteration_limit", str(iterations))]:
        raise ValueError("Unexpected fixed-work sweep/iteration signature")
    if sorted(group for group, _ in fluxes) != ["0", "63"]:
        raise ValueError("Missing or duplicate scalar-flux checks")
    seconds, grind = map(float, samples[0])
    unknowns, lagged = map(int, counts[0])
    values = [seconds, grind] + [float(value) for _, value in fluxes]
    if not all(math.isfinite(v) and v >= 0 for v in values) or unknowns < 1 or lagged != 0:
        raise ValueError("Invalid physical/timing data")
    if not math.isclose(grind, 1.0e9 * seconds / unknowns, rel_tol=2.0e-6):
        raise ValueError("Inconsistent sweep/grind-time normalization")
    residuals = re.findall(rf"iteration = (\d+), residual = ({FLOAT})", text)
    if [int(i) for i, _ in residuals] != list(range(iterations + 1)):
        raise ValueError("Incomplete residual history")
    if not all(math.isfinite(float(value)) for _, value in residuals):
        raise ValueError("Nonfinite residual")
    return dict(sweep_seconds=seconds, grind_ns=grind, unknowns=unknowns,
                flux_max={group: float(value) for group, value in fluxes},
                residuals=[float(value) for _, value in residuals])


def profile_artifacts(attempt, mode):
    if mode in ("caliper-mpi", "pmpi"):
        path = attempt / ("mpi-regions.txt" if mode == "caliper-mpi" else "mpi.txt")
        if not path.is_file() or "MPI_" not in path.read_text():
            raise ValueError(f"Missing MPI profile: {path}")
    if mode == "caliper-mpi":
        text = (attempt / "mpi-regions.txt").read_text()
        if "HostCBC.Setup.FLUDS" not in text or "HostCBC.Setup.Beta" not in text:
            raise ValueError("Missing host CBC setup profile regions")


def validate_attempt(attempt, mode, config, nodes):
    if (attempt / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("Nonzero application exit")
    result = measurements((attempt / "stdout.txt").read_text(), config["iterations"])
    profile_artifacts(attempt, mode)
    records = read_json(attempt / "placement.json")
    if sorted(r["rank"] for r in records) != list(range(nodes * config["ranks_per_node"])):
        raise ValueError("Missing or duplicate rank placement records")
    hosts = {}
    for record in records:
        hosts[record["hostname"]] = hosts.get(record["hostname"], 0) + 1
    if len(hosts) != nodes or set(hosts.values()) != {config["ranks_per_node"]}:
        raise ValueError("Unexpected rank distribution across nodes")
    memory = read_json(attempt / "memory.json")
    if len(memory) != len(records):
        raise ValueError("Incomplete per-rank memory summary")
    return result


def successes(root, kind, nodes, mode, trial, config):
    directory = root / "results" / kind / f"nodes-{nodes}" / mode / f"trial-{trial}"
    valid = []
    for attempt in sorted(directory.glob("attempt-*")):
        if not (attempt / "SUCCESS").exists():
            continue
        try:
            if read_json(attempt / "SUCCESS")["manifest_sha256"] != digest(root / "manifest.json"):
                continue
            launch = read_json(attempt / "launch.json")
            if digest(attempt / "input.py") != launch["input_sha256"]:
                continue
            result = validate_attempt(attempt, mode, config, nodes)
            recorded = read_json(attempt / "result.json")
            if any(recorded.get(key) != value for key, value in result.items()):
                continue
            valid.append(attempt)
        except (ValueError, KeyError, OSError):
            continue
    return valid


def pending(root, kind, nodes, config):
    return [(mode, trial) for trial in range(1, config["trials"] + 1)
            for mode in config["modes"]
            if not successes(root, kind, nodes, mode, trial, config)]


def status(args):
    config = read_json(args.root / "manifest.json")
    kinds = [args.kind] if args.kind else ["strong", "weak"]
    nodes_list = [args.nodes] if args.nodes else config["nodes"]
    for kind in kinds:
        for nodes in nodes_list:
            missing = pending(args.root, kind, nodes, config)
            total = len(config["modes"]) * config["trials"]
            print(f"{kind} {nodes} nodes: {total - len(missing)}/{total} complete")
            if missing:
                print("  remaining: " + ", ".join(f"{m}/{t}" for m, t in missing))
    if getattr(args, "check_complete", False):
        return 0 if not pending(args.root, args.kind, args.nodes, config) else 3
    return 0


def terminate(proc, attempt, config):
    job = attempt / "step-job-id.txt"
    if job.exists() and config.get("cancel"):
        command = [part.format(jobid=job.read_text().strip()) for part in config["cancel"]]
        try:
            subprocess.run(command, check=False, timeout=15, stdout=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def launch_trial(root, config, kind, nodes, mode, trial, budget, allocation):
    directory = root / "results" / kind / f"nodes-{nodes}" / mode / f"trial-{trial}"
    attempt = directory / ("attempt-" + unique_id())
    attempt.mkdir(parents=True)
    shutil.copy2(root / "inputs" / f"{kind}-{nodes}.py", attempt / "input.py")
    replacements = dict(nodes=nodes, ranks=nodes * config["ranks_per_node"],
                        threads=config["threads"], seconds=max(1, int(budget)))
    command = [part.format(**replacements) for part in config["launcher"]]
    command += [sys.executable, str(HERE / "host_cbc_study.py"), "_rank",
                "--root", str(root), "--attempt", str(attempt), "--mode", mode]
    env = clean_environment(os.environ)
    env.update(OPENSN_NUM_THREADS=str(config["threads"]), OMP_NUM_THREADS=str(config["threads"]))
    write_json(attempt / "launch.json", dict(command=command, allocation=allocation,
               binary=read_json(root / "builds" / build_mode(mode) / "build.json"),
               manifest_sha256=digest(root / "manifest.json"),
               input_sha256=digest(attempt / "input.py"), host=socket.gethostname()))
    print(f"{kind} nodes={nodes} {mode} trial={trial}/{config['trials']}\n  {attempt}", flush=True)
    start = time.monotonic()
    try:
        with (attempt / "stdout.txt").open("x") as out, (attempt / "stderr.txt").open("x") as err:
            proc = subprocess.Popen(command, cwd=attempt, env=env, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=budget)
            except BaseException:
                terminate(proc, attempt, config)
                raise
        (attempt / "exit_code.txt").write_text(str(code) + "\n")
        result = validate_attempt(attempt, mode, config, nodes)
        baseline = successes(root, kind, nodes, "baseline", 1, config)
        if baseline:
            reference = read_json(baseline[0] / "result.json")
            if result["unknowns"] != reference["unknowns"] or any(
                    not math.isclose(result["flux_max"][g], reference["flux_max"][g],
                                     rel_tol=1.0e-10, abs_tol=1.0e-12) for g in ("0", "63")):
                raise ValueError("Scalar-flux signature differs from this case's baseline")
            if any(not math.isclose(value, expected, rel_tol=2.0e-5, abs_tol=1.0e-12)
                   for value, expected in zip(result["residuals"], reference["residuals"])):
                raise ValueError("Residual history differs from this case's baseline")
        result["elapsed_seconds"] = time.monotonic() - start
        write_json(attempt / "result.json", result)
        write_json(attempt / "SUCCESS", dict(manifest_sha256=digest(root / "manifest.json")))
        print(f"  completed in {result['elapsed_seconds']:.1f}s", flush=True)
        return result["elapsed_seconds"]
    except BaseException as error:
        failure = dict(error=repr(error), elapsed_seconds=time.monotonic() - start)
        write_json(attempt / "FAILED", failure)
        raise


def run(args):
    deadline = time.monotonic() + args.seconds
    root = args.root.resolve(strict=True)
    config = read_json(root / "manifest.json")
    if args.nodes not in config["nodes"]:
        raise ValueError("Node count is not in this study")
    verify_inputs(config)
    for variant in set(build_mode(m) for m in config["modes"]):
        verify_fingerprint(read_json(root / "builds" / variant / "build.json"))
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Allocation time remaining must be positive and finite")
    with locked(root / f".run-{args.kind}-{args.nodes}.lock"):
        estimates = {}
        for mode, trial in pending(root, args.kind, args.nodes, config):
            remaining = deadline - time.monotonic() - 60
            needed = max(120, 1.3 * estimates.get(mode, 0))
            if remaining < needed:
                print("Insufficient allocation time. Re-run this case explicitly to resume.",
                      flush=True)
                return 3
            estimates[mode] = launch_trial(root, config, args.kind, args.nodes, mode, trial,
                                           remaining, args.allocation)
    return 0


def build_mode(mode):
    return "profile" if mode == "caliper-mpi" else "native"


def rank_run(args):
    root = args.root.resolve(strict=True)
    config = read_json(root / "manifest.json")
    rank = next((os.environ[key] for key in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK")
                 if key in os.environ), None)
    if rank is None:
        raise ValueError("No MPI rank identity from launcher")
    rank = int(rank)
    env = clean_environment(os.environ)
    env.update(OPENSN_NUM_THREADS=str(config["threads"]), OMP_NUM_THREADS=str(config["threads"]),
               OPENSN_HOST_STUDY_ATTEMPT=str(args.attempt))
    if rank == 0 and "SLURM_STEP_ID" in env:
        (args.attempt / "step-job-id.txt").write_text(
            env["SLURM_JOB_ID"] + "." + env["SLURM_STEP_ID"] + "\n")
    binary = read_json(root / "builds" / build_mode(args.mode) / "build.json")["binary"]
    command = [binary, "--verbose", "1", "-i", str(args.attempt / "input.py")]
    if args.mode == "caliper-mpi":
        env.pop("CALI_SERVICES_ENABLE", None)
        command += ['--caliper=runtime-report(output="' + str(args.attempt / "mpi-regions.txt")
                    + '",aggregate_across_ranks,calc.inclusive,print.metadata,order_by_time,'
                    'max_column_width=180,profile.mpi,region.count,region.stats)']
    elif args.mode == "pmpi":
        env.pop("CALI_SERVICES_ENABLE", None)
        command += ['--caliper=mpi-report(output="' + str(args.attempt / "mpi.txt") + '")']
    os.execvpe(command[0], command, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(action=prepare)
    p = commands.add_parser("build")
    p.add_argument("--jobs", type=int, default=16)
    p.set_defaults(action=build)
    p = commands.add_parser("run")
    p.add_argument("--kind", choices=("strong", "weak"), required=True)
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--allocation", required=True)
    p.set_defaults(action=run)
    p = commands.add_parser("status")
    p.add_argument("--kind", choices=("strong", "weak"))
    p.add_argument("--nodes", type=int)
    p.add_argument("--check-complete", action="store_true")
    p.set_defaults(action=status)
    p = commands.add_parser("_rank")
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.set_defaults(action=rank_run)
    for p in commands.choices.values():
        p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "check_complete", False) and (not args.kind or not args.nodes):
        parser.error("--check-complete requires --kind and --nodes")
    if args.command == "run":
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
    return args.action(args)


if __name__ == "__main__":
    sys.exit(main())
