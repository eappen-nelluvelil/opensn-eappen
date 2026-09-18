# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Reuse a configured dependency stack for isolated Native study builds."""

import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def cache_values(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, kind = key_type.split(":", 1)
        values[key] = (kind, value)
    return values


def configure_command(source, previous, destination):
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
    command = ["cmake", "-S", str(source), "-B", str(destination),
               "-G", cache["CMAKE_GENERATOR"][1]]
    for key in sorted(keys & cache.keys()):
        kind, value = cache[key]
        if value and not value.endswith("-NOTFOUND"):
            if "FLAGS" in key and any(flag in value for flag in (
                    "sanitize", "coverage", "-pg", "finstrument", "profile-generate")):
                raise ValueError(f"Instrumented dependency build flags in {key}: {value}")
            command.append(f"-D{key}:{kind}={value}")
    for package in ("googletest", "mpicpp-lite"):
        candidate = previous / "_deps" / (package + "-src")
        if package == "googletest" and "googletest-distribution_SOURCE_DIR" in cache:
            candidate = Path(cache["googletest-distribution_SOURCE_DIR"][1])
        if candidate.is_dir():
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_{package.upper()}={candidate}")
    command += ["-DCMAKE_BUILD_TYPE=Native", "-DOPENSN_WITH_PYTHON=ON",
                "-DOPENSN_WITH_PYTHON_MODULE=OFF", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON"]
    return command


def fingerprint(build):
    paths = {str((build / "python/opensn").resolve(strict=True))}
    paths.update(str(path.resolve()) for path in build.glob("libopensn.so*"))
    if len(paths) < 2:
        raise ValueError(f"Missing shared OpenSn library in {build}")
    return {path: digest(path) for path in sorted(paths)}


def build_variant(source, previous, destination, record_dir, jobs):
    record_dir.mkdir(parents=True, exist_ok=True)
    command = configure_command(source, previous, destination)
    plan = record_dir / "configure.json"
    if plan.exists():
        if json.loads(plan.read_text()) != command:
            raise ValueError("Build configuration changed. Prepare a new study.")
    else:
        if destination.exists():
            raise ValueError(f"Refusing to reuse an unrecorded build: {destination}")
        write_json(plan, command)
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(destination), "--target", "opensn",
                    "--parallel", str(jobs)], check=True)
    return fingerprint(destination)
