#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Build and run repeated CBC baselines without profiling services."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


SOURCE = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    cache = cache_values(previous / "CMakeCache.txt")
    keys = {
        "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER",
        "CMAKE_HIP_COMPILER", "CMAKE_CUDA_HOST_COMPILER", "CMAKE_HIP_PLATFORM",
        "CMAKE_CUDA_ARCHITECTURES", "CMAKE_HIP_ARCHITECTURES", "CMAKE_PREFIX_PATH",
        "Python3_EXECUTABLE", "Python_EXECUTABLE", "MPI_C_COMPILER", "MPI_CXX_COMPILER",
        "PETSC_DIR", "PETSC_ARCH", "PETSC_INCLUDE_DIR", "PETSC_LIBRARY",
        "MPICPP_LITE_INCLUDE_DIR", "Boost_DIR", "HDF5_DIR", "VTK_DIR", "caliper_DIR",
        "pybind11_DIR", "mpicpp-lite_DIR", "GTest_DIR", "OPENSN_WITH_CUDA",
        "OPENSN_WITH_HIP", "OPENSN_WITH_SYCL", "CUDAToolkit_ROOT", "ROCM_ROOT",
    }
    for language in ("C", "CXX", "CUDA", "HIP"):
        keys.update((f"CMAKE_{language}_FLAGS", f"CMAKE_{language}_FLAGS_NATIVE"))
    command = ["cmake", "-S", str(SOURCE), "-B", str(destination),
               "-G", cache["CMAKE_GENERATOR"][1]]
    for key in sorted(keys & cache.keys()):
        kind, value = cache[key]
        if value and not value.endswith("-NOTFOUND"):
            if "FLAGS" in key and any(s in value for s in (
                    "sanitize", "coverage", "-pg", "finstrument", "profile-generate")):
                raise ValueError(f"Instrumented build flags in {key}: {value}")
            command.append(f"-D{key}:{kind}={value}")
    for package in ("googletest", "mpicpp-lite"):
        candidate = previous / "_deps" / (package + "-src")
        if package == "googletest" and "googletest-distribution_SOURCE_DIR" in cache:
            candidate = Path(cache["googletest-distribution_SOURCE_DIR"][1])
        if candidate.is_dir():
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_{package.upper()}={candidate}")
    command += ["-DCMAKE_BUILD_TYPE=Native", "-DOPENSN_WITH_PYTHON=ON",
                "-DOPENSN_WITH_PYTHON_MODULE=OFF"]
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(previous / "CMakeCache.txt", destination / "reused-CMakeCache.txt")
    write_json(destination / "configure-command.json", command)
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(destination), "--target", "opensn",
                    "--parallel", str(args.jobs)], check=True)


def render(mesh, gpu, repetitions, iterations):
    template = (SOURCE / "tools/scaling/lib/unstructured.py").read_text()
    values = dict(mesh_file=str(mesh), use_gpus=str(gpu),
                  outer_repetition=str(repetitions), wgs_iteration=str(iterations))
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    tree = ast.parse(template)
    loops = [node for node in tree.body if isinstance(node, ast.For)]
    if not loops:
        starts = [i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                  and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id == "DiscreteOrdinatesProblem"]
        if len(starts) != 1 or len(tree.body[starts[0]:]) != 4:
            raise ValueError("Expected one solver setup and execution block")
        loop = ast.parse("for i in range(1):\n    pass\n").body[0]
        loop.body = tree.body[starts[0]:]
        tree.body[starts[0]:] = [loop]
        loops = [loop]
    if len(loops) != 1:
        raise ValueError("Expected one top-level repetition loop")
    loop = loops[0]
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "DiscreteOrdinatesProblem"]
    if len(calls) != 1:
        raise ValueError("Expected one problem constructor")
    calls[0].keywords.append(ast.keyword(arg="sweep_type", value=ast.Constant("CBC")))
    for keyword in calls[0].keywords:
        if keyword.arg == "options":
            keyword.value.keys.append(ast.Constant("save_angular_flux"))
            keyword.value.values.append(ast.Constant(False))
        if keyword.arg == "groupsets":
            for groupset in keyword.value.elts:
                groupset.keys.append(ast.Constant("allow_cycles"))
                groupset.values.append(ast.Constant(False))
                for i, key in enumerate(groupset.keys):
                    if ast.literal_eval(key) == "l_max_its":
                        groupset.values[i] = ast.Constant(iterations)
                    if ast.literal_eval(key) == "l_abs_tol":
                        groupset.values[i] = ast.Constant(1.0e-18)
    function = ast.parse("def run_trial():\n    pass\n").body[0]
    function.body = loop.body
    index = tree.body.index(loop)
    tree.body[index:index + 1] = [function] + ast.parse(f'''
for trial in range({repetitions}):
    if rank == 0:
        print(f"BASELINE_TRIAL_BEGIN {{trial + 1}}/{repetitions}", flush=True)
    run_trial()
    if rank == 0:
        print(f"BASELINE_TRIAL_END {{trial + 1}}/{repetitions}", flush=True)
''').body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def prepare(args):
    if args.fuse_worker_launches and not args.gpu:
        raise ValueError("Combined worker launches require --gpu")
    mesh = args.mesh.resolve(strict=True)
    if any(c in str(mesh) for c in ('"', "\n", "\\")):
        raise ValueError("Unsupported mesh path")
    text = render(mesh, args.gpu, args.repetitions, args.iterations)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.py").write_text(text)
    xs = output / "xs_168g.xs"
    shutil.copy2(SOURCE / "tools/scaling/lib/xs_168g.xs", xs)
    write_json(output / "input.json", dict(
        mesh=str(mesh), mesh_sha256=digest(mesh), xs_sha256=digest(xs),
        gpu=args.gpu, repetitions=args.repetitions, iterations=args.iterations,
        allow_cycles=False, fuse_worker_launches=args.fuse_worker_launches,
        source=str(SOURCE), revision=subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()))


def baseline_environment(environment):
    prefixes = ("CALI_", "OPENSN_MEMORY_", "OPENSN_CBCD_", "ROCPROF", "ROCTRACER",
                "SCOREP_", "TAU_", "NSYS_", "NCU_")
    removed = {"LD_PRELOAD", "LD_AUDIT", "HSA_TOOLS_LIB", "HSA_TOOLS_LIB64",
               "ROCP_TOOL_LIBRARIES", "OMP_TOOL_LIBRARIES", "OMP_TOOL",
               "GLIBC_TUNABLES", "MALLOC_ARENA_MAX", "MALLOC_TRIM_THRESHOLD_"}
    env = {key: value for key, value in environment.items()
           if not key.startswith(prefixes) and key not in removed}
    env.update(CALI_CONFIG="", CALI_SERVICES_ENABLE="", CALI_CONFIG_FILE="/dev/null",
               CALI_USE_OMPT="0", OMP_TOOL="disabled")
    return env


def completed_trials(text):
    begin = None
    samples = []
    for line in text.splitlines():
        marker = re.search(r"BASELINE_TRIAL_BEGIN (\d+)/(\d+)", line)
        if marker:
            begin = dict(trial=int(marker[1]), measurements=[])
        timing = re.search(r"avg_sweep_time = ([\deE+.-]+) s, "
                           r"sweep_time_per_unknown = ([\deE+.-]+) ns", line)
        if begin is not None and timing:
            begin["measurements"].append(dict(sweep_seconds=float(timing[1]),
                                              grind_ns=float(timing[2])))
        end = re.search(r"BASELINE_TRIAL_END (\d+)/(\d+)", line)
        if end and begin is not None and int(end[1]) == begin["trial"]:
            if len(begin["measurements"]) == 1:
                samples.append(dict(trial=begin["trial"], **begin["measurements"][0]))
            begin = None
    return samples


def run_case(args):
    case = args.case.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    launcher = args.launcher[1:] if args.launcher[:1] == ["--"] else args.launcher
    if not launcher:
        raise ValueError("Supply the MPI launcher after --")
    env = baseline_environment(os.environ)
    config = json.loads((case / "input.json").read_text())
    env["OPENSN_CBCD_FUSE_WORKER_LAUNCHES"] = str(config["fuse_worker_launches"])
    command = launcher + [str(binary), "-i", "input.py"]
    prefixes = ("OPENSN_", "OMP_", "SLURM_", "FLUX_", "MPI", "OMPI_", "PMIX_",
                "CUDA_", "HIP_", "ROCR_", "CALI_", "HSA_", "PYTHON", "VIRTUAL_ENV")
    with (case / "launch.json").open("x") as stream:
        json.dump(dict(command=command, binary_sha256=digest(binary),
                       input_sha256=digest(case / "input.py"),
                       environment={k: v for k, v in env.items() if k.startswith(prefixes)}),
                  stream, indent=2)
    with (case / "stdout.txt").open("x") as out, (case / "stderr.txt").open("x") as err:
        result = subprocess.run(command, cwd=case, env=env, stdout=out, stderr=err)
    (case / "exit_code.txt").write_text(str(result.returncode) + "\n")
    samples = completed_trials((case / "stdout.txt").read_text())
    write_json(case / "trials.json", samples)
    expected = config["repetitions"]
    complete = [s["trial"] for s in samples] == list(range(1, expected + 1))
    if result.returncode != 0 or not complete:
        raise SystemExit(f"Incomplete baseline: exit={result.returncode}, "
                         f"complete trials={len(samples)}/{expected}. Logs: {case}")
    (case / "SUCCESS").touch()
    print(f"Completed {expected} trials: {case}", flush=True)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build")
    p.add_argument("--reuse-build", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--jobs", type=positive, default=16)
    p.set_defaults(action=build)
    p = commands.add_parser("prepare")
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--fuse-worker-launches", type=int, choices=(0, 1), default=0)
    p.add_argument("--repetitions", type=positive, default=17)
    p.add_argument("--iterations", type=positive, default=16)
    p.set_defaults(action=prepare)
    p = commands.add_parser("run")
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("launcher", nargs=argparse.REMAINDER)
    p.set_defaults(action=run_case)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
