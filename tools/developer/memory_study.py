#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Prepare repeated CBC memory diagnostics and reuse an existing dependency stack."""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2]
MARKERS = '''
import ctypes
import json
import socket
import time
from pathlib import Path

def read_memory_file(path):
    try:
        return Path(path).read_text()
    except OSError as error:
        return {"error": str(error)}

def cgroup_memory_snapshot(membership, root=Path("/sys/fs/cgroup")):
    result = {}
    for line in membership.splitlines():
        if not line.startswith("0::"):
            continue
        directory = root / line[3:].lstrip("/")
        while directory == root or root in directory.parents:
            result[str(directory)] = {
                name: read_memory_file(directory / name)
                for name in ("memory.current", "memory.peak", "memory.max",
                             "memory.high", "memory.events", "memory.stat",
                             "memory.pressure", "memory.swap.current")}
            if directory == root:
                break
            directory = directory.parent
    return result

def memory_marker(trial, stage):
    trace_dir = os.environ.get("OPENSN_MEMORY_TRACE_DIR")
    if not trace_dir:
        return
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(trial=trial, stage=stage, time=time.time(), pid=os.getpid(),
                  host=socket.gethostname(),
                  status=Path("/proc/self/status").read_text())
    record["node_memory"] = Path("/proc/meminfo").read_text()
    record["cgroup"] = Path("/proc/self/cgroup").read_text()
    record["smaps_rollup"] = read_memory_file("/proc/self/smaps_rollup")
    record["numa_maps"] = read_memory_file("/proc/self/numa_maps")
    record["schedstat"] = read_memory_file("/proc/self/schedstat")
    if os.environ.get("OPENSN_MEMORY_SMAPS") == "1":
        record["smaps"] = read_memory_file("/proc/self/smaps")
    record["maps"] = read_memory_file("/proc/self/maps")
    record["memory_pressure"] = read_memory_file("/proc/pressure/memory")
    record["cgroup_memory"] = cgroup_memory_snapshot(record["cgroup"])
    filename = "python-" + socket.gethostname() + "-" + str(os.getpid()) + ".jsonl"
    with (directory / filename).open("a") as f:
        f.write(json.dumps(record) + "\\n")

def trim_memory():
    if os.environ.get("OPENSN_MEMORY_TRIM") == "1":
        libc = ctypes.CDLL(None)
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        libc.malloc_trim(0)
'''


def render(mesh, gpu, repetitions, iterations, allow_cycles=None):
    template = (SOURCE / "tools/scaling/lib/unstructured.py").read_text()
    values = dict(mesh_file=str(mesh), use_gpus=str(gpu),
                  outer_repetition=str(repetitions), wgs_iteration=str(iterations))
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    tree = ast.parse(template)
    loops = [n for n in tree.body if isinstance(n, ast.For)]
    if len(loops) != 1:
        raise ValueError("Expected one top-level repetition loop")
    loop = loops[0]
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "DiscreteOrdinatesProblem"]
    if len(calls) != 1:
        raise ValueError("Expected one problem constructor")
    call = calls[0]
    call.keywords.append(ast.keyword(arg="sweep_type", value=ast.Constant("CBC")))
    for keyword in call.keywords:
        if keyword.arg == "options":
            keyword.value.keys.append(ast.Constant("save_angular_flux"))
            keyword.value.values.append(ast.Constant(False))
        if keyword.arg == "groupsets":
            for groupset in keyword.value.elts:
                groupset.keys.append(ast.Constant("allow_cycles"))
                cycles = not gpu if allow_cycles is None else allow_cycles
                groupset.values.append(ast.Constant(cycles))
    function = ast.parse("def run_trial():\n    pass\n").body[0]
    function.body = loop.body
    index = tree.body.index(loop)
    tree.body[index:] = ast.parse(MARKERS).body + [function] + ast.parse(f'''
for trial in range({repetitions}):
    if rank == 0:
        print(f"MEMORY_TRIAL_BEGIN {{trial + 1}}/{repetitions}", flush=True)
    memory_marker(trial, "before_trial")
    run_trial()
    memory_marker(trial, "after_trial")
    trim_memory()
    if os.environ.get("OPENSN_MEMORY_TRIM") == "1":
        memory_marker(trial, "after_trim")
    if rank == 0:
        print(f"MEMORY_TRIAL_END {{trial + 1}}/{repetitions}", flush=True)
''').body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def prepare(args):
    mesh = args.mesh.resolve(strict=True)
    if '"' in str(mesh) or "\n" in str(mesh):
        raise ValueError("Unsupported mesh path")
    output = args.output.resolve()
    allow_cycles = getattr(args, "allow_cycles", None)
    text = render(mesh, args.gpu, args.repetitions, args.iterations, allow_cycles)
    output.mkdir(parents=True, exist_ok=False)
    (output / "trace").mkdir()
    (output / "input.py").write_text(text)
    shutil.copy2(SOURCE / "tools/scaling/lib/xs_168g.xs", output / "xs_168g.xs")
    record = dict(mesh=str(mesh), gpu=args.gpu, repetitions=args.repetitions,
                  iterations=args.iterations, allow_cycles=allow_cycles, source=str(SOURCE),
                  revision=subprocess.check_output(
                      ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip())
    (output / "input.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output)


def cache_values(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, kind = key_type.split(":", 1)
        result[key] = (kind, value)
    return result


def build(args):
    previous = args.reuse_build.resolve(strict=True)
    destination = args.build.resolve()
    if destination.exists():
        raise ValueError("Use a new build directory")
    cache = cache_values(previous / "CMakeCache.txt")
    keys = {
        "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER",
        "CMAKE_HIP_COMPILER", "CMAKE_CUDA_HOST_COMPILER", "CMAKE_HIP_PLATFORM",
        "CMAKE_CUDA_ARCHITECTURES", "CMAKE_HIP_ARCHITECTURES", "CMAKE_PREFIX_PATH",
        "Python3_EXECUTABLE", "Python_EXECUTABLE",
        "MPI_C_COMPILER", "MPI_CXX_COMPILER", "PETSC_DIR", "PETSC_ARCH",
        "PETSC_INCLUDE_DIR", "PETSC_LIBRARY", "MPICPP_LITE_INCLUDE_DIR",
        "Boost_DIR", "HDF5_DIR", "VTK_DIR", "caliper_DIR", "pybind11_DIR",
        "mpicpp-lite_DIR", "GTest_DIR", "OPENSN_WITH_CUDA", "OPENSN_WITH_HIP",
        "OPENSN_WITH_SYCL", "CUDAToolkit_ROOT", "ROCM_ROOT",
        "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_CUDA_FLAGS", "CMAKE_HIP_FLAGS",
    }
    for language in ("C", "CXX", "CUDA", "HIP"):
        keys.add(f"CMAKE_{language}_FLAGS_NATIVE")
    command = ["cmake", "-S", str(SOURCE), "-B", str(destination),
               "-G", cache["CMAKE_GENERATOR"][1]]
    for key in sorted(keys & cache.keys()):
        kind, value = cache[key]
        if value and not value.endswith("-NOTFOUND"):
            command.append(f"-D{key}:{kind}={value}")
    for package in ("googletest", "mpicpp-lite"):
        candidate = previous / "_deps" / (package + "-src")
        if candidate.is_dir():
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_{package.upper()}={candidate}")
    if "googletest-distribution_SOURCE_DIR" in cache:
        candidate = Path(cache["googletest-distribution_SOURCE_DIR"][1])
        if candidate.is_dir():
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={candidate}")
    command += ["-DCMAKE_BUILD_TYPE=Native", "-DOPENSN_WITH_PYTHON=ON",
                "-DOPENSN_WITH_PYTHON_MODULE=OFF"]
    destination.mkdir(parents=True)
    shutil.copy2(previous / "CMakeCache.txt", destination / "reused-CMakeCache.txt")
    (destination / "configure-command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(destination), "--target", "opensn",
                    "--parallel", str(args.jobs)], check=True)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def rocprof_rank(args):
    rank = next((os.environ[key] for key in
                 ("FLUX_TASK_RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK") if key in os.environ), None)
    if rank is None or not rank.isdecimal():
        raise ValueError("Cannot determine the MPI rank for rocprofv3")
    command = [str(args.binary), "-i", "input.py"]
    if int(rank) == 0:
        profiler = shutil.which("rocprofv3")
        if profiler is None:
            raise ValueError("rocprofv3 is not on PATH")
        output = args.output / "rank-0"
        output.mkdir(parents=True, exist_ok=False)
        record = dict(host=socket.gethostname(), rank=rank,
                      affinity=sorted(os.sched_getaffinity(0)),
                      environment={key: value for key, value in os.environ.items()
                                   if key.startswith(("ROCR_", "HIP_", "OMP_", "OPENSN_"))})
        (output / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        with (output / "version.txt").open("w") as stream:
            subprocess.run([profiler, "--version"], stdout=stream, stderr=subprocess.STDOUT,
                           check=True)
        command = [profiler, "--hip-runtime-trace", "--kernel-trace", "--memory-copy-trace",
                   "--memory-allocation-trace", "--stats", "--output-format", "csv",
                   "--output-directory", str(output), "--"] + command
    os.execvp(command[0], command)


def run_case(args):
    directory = args.case.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    launcher = args.launcher
    if launcher[:1] == ["--"]:
        launcher = launcher[1:]
    if not launcher:
        raise ValueError("Supply the MPI launcher after --")
    command = launcher + [str(binary), "-i", "input.py"]
    mode = getattr(args, "mode", "memory")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CALI_", "OPENSN_MEMORY_"))}
    env.update(CALI_CONFIG="", CALI_SERVICES_ENABLE="", CALI_CONFIG_FILE="/dev/null",
               CALI_USE_OMPT="0", OMP_TOOL="disabled")
    if mode == "memory":
        env.update(OPENSN_MEMORY_TRACE_DIR=str(directory / "trace"),
                   OPENSN_MEMORY_ALLOCATOR="1", OPENSN_MEMORY_TRIM="1" if args.trim else "0",
                   OPENSN_MEMORY_SMAPS="1" if args.smaps else "0")
        if args.sample_seconds:
            env["OPENSN_MEMORY_SAMPLE_SECONDS"] = str(args.sample_seconds)
    elif args.trim or args.smaps or args.sample_seconds:
        raise ValueError("Memory diagnostics require --mode memory")
    if mode == "caliper":
        env["CALI_CONFIG"] = "runtime-report(profile.mpi,calc.inclusive=true,output=caliper.txt)"
        env["CALI_LOG_VERBOSITY"] = "2"
    if mode == "rocprof":
        if not json.loads((directory / "input.json").read_text()).get("gpu"):
            raise ValueError("rocprofv3 requires a HIP GPU input and build")
        if shutil.which("rocprofv3") is None:
            raise ValueError("rocprofv3 is not on PATH")
        command = launcher + [sys.executable, str(Path(__file__).resolve()), "rocprof-rank",
                              "--binary", str(binary), "--output", str(directory / "rocprof")]
    prefixes = ("OPENSN_", "OMP_", "SLURM_", "FLUX_", "MPI", "OMPI_", "PMIX_",
                "CUDA_", "HIP_", "ROCR_", "GLIBC_", "MALLOC_", "CALI_")
    settings = {k: v for k, v in env.items() if k.startswith(prefixes)}
    record = dict(command=command, mode=mode, environment=settings, hashes={})
    for path in (binary, directory / "input.py", directory / "xs_168g.xs"):
        with path.open("rb") as stream:
            record["hashes"][str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    with (directory / "launch.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(f"Diagnostics: {directory}", flush=True)
    with (directory / "stdout.txt").open("x") as out, (directory / "stderr.txt").open("x") as err:
        result = subprocess.run(command, cwd=directory, env=env, stdout=out, stderr=err)
    (directory / "exit_code.txt").write_text(str(result.returncode) + "\n")
    print(f"Exit code: {result.returncode}", flush=True)
    if result.returncode:
        sys.exit(1)
    expected = json.loads((directory / "input.json").read_text())["repetitions"]
    output = (directory / "stdout.txt").read_text()
    completed = re.findall(r"MEMORY_TRIAL_END (\d+)/(\d+)", output)
    if completed != [(str(i), str(expected)) for i in range(1, expected + 1)]:
        raise RuntimeError("Incomplete trial sequence. Inspect stdout and allocation accounting")
    times = re.findall(r"avg_sweep_time = ([\deE+.-]+) s, sweep_time_per_unknown = ([\deE+.-]+) ns",
                       output)
    if len(times) != expected:
        raise RuntimeError("Missing sweep timings")
    if mode == "caliper" and (not (directory / "caliper.txt").is_file()
                              or (directory / "caliper.txt").stat().st_size == 0):
        raise RuntimeError("Caliper did not produce its requested report")
    if mode == "rocprof":
        for name in ("hip_api", "kernel", "memory_copy", "memory_allocation"):
            if not any(path.stat().st_size for path in
                       (directory / "rocprof" / "rank-0").glob(f"*{name}_trace.csv")):
                raise RuntimeError(f"rocprofv3 did not produce its {name} trace")
    (directory / "trials.json").write_text(json.dumps([
        dict(trial=i, avg_sweep_time_s=float(sweep), grind_ns=float(grind))
        for i, (sweep, grind) in enumerate(times, 1)], indent=2) + "\n")
    (directory / "SUCCESS").touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--allow-cycles", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--repetitions", type=positive, default=17)
    p.add_argument("--iterations", type=positive, default=16)
    p.set_defaults(action=prepare)
    p = commands.add_parser("build")
    p.add_argument("--reuse-build", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--jobs", type=positive, default=16)
    p.set_defaults(action=build)
    p = commands.add_parser("run")
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--mode", choices=("baseline", "memory", "caliper", "rocprof"), default="memory")
    p.add_argument("--trim", action="store_true")
    p.add_argument("--smaps", action="store_true",
                   help="Capture resident memory by mapping at each trial boundary")
    p.add_argument("--sample-seconds", type=positive,
                   help="Sample host memory in one diagnostic thread per rank during C++ calls")
    p.add_argument("launcher", nargs=argparse.REMAINDER)
    p.set_defaults(action=run_case)
    p = commands.add_parser("rocprof-rank")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(action=rocprof_rank)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
