#!/usr/bin/env python3
"""Reproducible local CBCD benchmark and profiling driver.

The local 1/2/4-rank studies deliberately put every MPI rank on one GPU.  They
are contention diagnostics, not accelerator strong-scaling measurements.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_INPUT = HERE / "inputs/transport_3d_ortho_cbcd.py"
DEFAULT_RESULTS = HERE / "results"
EXPECTED_MAX_GROUPS = {0, 19}
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SIGNATURE_SCHEMA_VERSION = 2
DEFAULT_WORKLOAD_ASSETS = (REPO_ROOT / "test/assets/xs/xs_graphite_pure.xs",)

CONFIG_RE = re.compile(
    r"^CBCD_PROFILE_CONFIG ranks=(\d+) cells_per_axis=(\d+) cells=(\d+) "
    r"groups=(\d+) directions=(\d+) save_angular_flux=(true|false)$",
    re.MULTILINE,
)
MAX_RE = re.compile(rf"^CBCD_PROFILE_MAX group=(\d+) value=({FLOAT})$", re.MULTILINE)
WGS_RE = re.compile(
    rf"^\[0\][^\n]*\bWGS groups \[(\d+)-(\d+)\] iteration = (\d+), residual = ({FLOAT})"
    r"(?:, status = ([A-Za-z_]+))?",
    re.MULTILINE,
)
TIMING_RE = re.compile(
    rf"^\[0\][^\n]*\bWGS groups \[(\d+)-(\d+)\] avg_sweep_time = ({FLOAT}) s, "
    rf"sweep_time_per_unknown = ({FLOAT}) ns",
    re.MULTILINE,
)
UNKNOWNS_RE = re.compile(
    rf"^\[0\][^\n]*\bWGS groups \[(\d+)-(\d+)\] unknowns = (\d+), "
    rf"lagged_unknowns = (\d+), lagged_pct = ({FLOAT})",
    re.MULTILINE,
)
WORKER_RE = re.compile(
    r"^\[0\][^\n]*\bCBCD scheduler: policy=(hardware|resource-aware), workers=(\d+), "
    r"communicator_threads=(\d+), reserved_communicator_threads=(\d+)",
    re.MULTILINE,
)
COMPLETION_TEXT = "OpenSn finished execution."
NSYS_REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_kern_exec_sum",
    "mpi_event_sum",
    "mpi_msg_size_sum",
)
NSYS_NVTX_REPORTS = ("nvtx_sum", "nvtx_gpu_proj_sum")


class StudyError(RuntimeError):
    """A study could not be completed or validated."""


@dataclasses.dataclass
class CommandResult:
    """Recorded outcome for one external command."""

    command_id: int
    kind: str
    argv: list[str]
    cwd: str
    environment: dict[str, str]
    environment_unset: list[str]
    log: str
    status: str
    exit_code: int | None = None
    timed_out: bool = False
    wall_seconds: float | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    validation: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RankSignature:
    """Numerical invariants for one fixed MPI rank count."""

    maxima: dict[int, float]
    wgs_final_iteration: int
    wgs_iteration_count: int
    unknowns: int
    lagged_unknowns: int
    groups: tuple[int, int] = (0, 20)

    def __post_init__(self) -> None:
        if set(self.maxima) != EXPECTED_MAX_GROUPS:
            raise StudyError(f"signature maxima must contain exactly groups {sorted(EXPECTED_MAX_GROUPS)}")
        if any(not math.isfinite(value) for value in self.maxima.values()):
            raise StudyError("signature maxima must be finite")
        if self.groups != (0, 20):
            raise StudyError(f"signature groups must be (0, 20), found {self.groups}")
        if self.wgs_final_iteration < 0 or self.wgs_iteration_count != self.wgs_final_iteration + 1:
            raise StudyError("signature WGS count must equal the zero-based final iteration plus one")
        if self.unknowns <= 0 or self.lagged_unknowns < 0 or self.lagged_unknowns > self.unknowns:
            raise StudyError("signature unknown counts are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "maxima": {str(group): value for group, value in sorted(self.maxima.items())},
            "wgs_final_iteration": self.wgs_final_iteration,
            "wgs_iteration_count": self.wgs_iteration_count,
            "unknowns": self.unknowns,
            "lagged_unknowns": self.lagged_unknowns,
            "groups": list(self.groups),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RankSignature":
        try:
            required = {
                "maxima",
                "wgs_final_iteration",
                "wgs_iteration_count",
                "unknowns",
                "lagged_unknowns",
                "groups",
            }
            if set(value) != required:
                raise StudyError(
                    f"rank signature fields must be exactly {sorted(required)}, found {sorted(value)}"
                )
            maxima = {int(group): float(number) for group, number in value["maxima"].items()}
            groups = tuple(int(item) for item in value["groups"])
            if len(groups) != 2:
                raise ValueError("groups must contain two entries")
            return cls(
                maxima=maxima,
                wgs_final_iteration=int(value["wgs_final_iteration"]),
                wgs_iteration_count=int(value["wgs_iteration_count"]),
                unknowns=int(value["unknowns"]),
                lagged_unknowns=int(value["lagged_unknowns"]),
                groups=(groups[0], groups[1]),
            )
        except (KeyError, TypeError, ValueError, StudyError) as error:
            raise StudyError(f"invalid signature data: {error}") from error


@dataclasses.dataclass
class SignatureSet:
    """Schema-v2 workload and rank-indexed numerical signatures."""

    workload: dict[str, Any]
    by_ranks: dict[int, RankSignature]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SIGNATURE_SCHEMA_VERSION,
            "workload": self.workload,
            "by_ranks": {
                str(ranks): signature.as_dict()
                for ranks, signature in sorted(self.by_ranks.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SignatureSet":
        try:
            if set(value) != {"schema_version", "workload", "by_ranks"}:
                raise StudyError(
                    "signature-set fields must be exactly schema_version, workload, and by_ranks"
                )
            if value.get("schema_version") != SIGNATURE_SCHEMA_VERSION:
                raise StudyError(
                    f"signature schema_version must be {SIGNATURE_SCHEMA_VERSION}; "
                    "rank-global schema-v1 references are unsafe"
                )
            workload = value["workload"]
            raw_by_ranks = value["by_ranks"]
            if not isinstance(workload, dict) or not isinstance(raw_by_ranks, dict) or not raw_by_ranks:
                raise StudyError("signature workload and non-empty by_ranks map are required")
            by_ranks: dict[int, RankSignature] = {}
            for ranks_text, raw_signature in raw_by_ranks.items():
                ranks = int(ranks_text)
                if ranks <= 0 or ranks in by_ranks or not isinstance(raw_signature, dict):
                    raise StudyError("signature rank keys must be unique positive integers")
                by_ranks[ranks] = RankSignature.from_dict(raw_signature)
            return cls(workload=workload, by_ranks=by_ranks)
        except (KeyError, TypeError, ValueError, StudyError) as error:
            if isinstance(error, StudyError) and str(error).startswith("signature schema_version"):
                raise
            raise StudyError(f"invalid rank-indexed signature data: {error}") from error

    def require_rank(self, ranks: int) -> RankSignature:
        try:
            return self.by_ranks[ranks]
        except KeyError as error:
            raise StudyError(f"signature reference has no entry for requested MPI rank count {ranks}") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: pathlib.Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def command_text(argv: Sequence[str]) -> str:
    return shlex.join(str(item) for item in argv)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("entries must be unique positive integers")
    return parsed


def parse_policies(value: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in value.split(",") if item.strip())
    valid = {"hardware", "resource-aware"}
    if not policies or len(set(policies)) != len(policies) or set(policies) - valid:
        raise argparse.ArgumentTypeError("use a unique comma-separated subset of hardware,resource-aware")
    return policies


def parse_pe_map(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    try:
        for entry in value.split(","):
            rank_text, pe_text = entry.split("=", 1)
            ranks = int(rank_text)
            pe = int(pe_text)
            if ranks <= 0 or pe <= 0 or ranks in result:
                raise ValueError
            result[ranks] = pe
    except ValueError as error:
        raise argparse.ArgumentTypeError("use unique positive RANKS=PE entries") from error
    if not result:
        raise argparse.ArgumentTypeError("PE map cannot be empty")
    return result


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def descriptive_statistics(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("statistics require at least one value")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        "count": len(values),
        "median": median,
        "mad": statistics.median(deviations),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "minimum": min(values),
        "maximum": max(values),
    }


def values_close(actual: float, expected: float, absolute: float, relative: float) -> bool:
    return abs(actual - expected) <= absolute + relative * abs(expected)


def validate_output(
    text: str,
    *,
    ranks: int,
    cells_per_axis: int,
    expected_policy: str,
    expected_worker_override: int | None,
    expected_signature: RankSignature | None,
    max_absolute_tolerance: float,
    max_relative_tolerance: float,
) -> dict[str, Any]:
    """Strictly validate and parse one OpenSn CBCD run."""

    errors: list[str] = []
    completion_count = text.count(COMPLETION_TEXT)
    if completion_count != 1:
        errors.append(f"expected one completion marker, found {completion_count}")

    worker_records = WORKER_RE.findall(text)
    if len(worker_records) != 1:
        errors.append(f"expected one verbose CBCD scheduler worker record, found {len(worker_records)}")
        worker_record = None
    else:
        raw_worker = worker_records[0]
        worker_record = {
            "policy": raw_worker[0],
            "workers": int(raw_worker[1]),
            "communicator_threads": int(raw_worker[2]),
            "reserved_communicator_threads": int(raw_worker[3]),
        }
        if worker_record["policy"] != expected_policy:
            errors.append(
                f"worker-policy mismatch: expected {expected_policy}, found {worker_record['policy']}"
            )
        if worker_record["workers"] <= 0 or worker_record["communicator_threads"] != 1:
            errors.append(f"invalid CBCD scheduler worker record: {worker_record}")
        if worker_record["reserved_communicator_threads"] not in (0, 1):
            errors.append(f"invalid reserved communicator count: {worker_record}")
        if expected_policy == "hardware" and worker_record["reserved_communicator_threads"] != 0:
            errors.append(f"hardware policy unexpectedly reserved a worker: {worker_record}")
        if expected_worker_override is not None:
            expected_workers = min(32, expected_worker_override)
            if worker_record["workers"] != expected_workers:
                errors.append(
                    f"fixed-worker mismatch: requested {expected_worker_override}, "
                    f"expected {expected_workers} angle-set workers, found {worker_record['workers']}"
                )

    configs = CONFIG_RE.findall(text)
    if len(configs) != 1:
        errors.append(f"expected one CBCD_PROFILE_CONFIG record, found {len(configs)}")
        config = None
    else:
        raw = configs[0]
        config = {
            "ranks": int(raw[0]),
            "cells_per_axis": int(raw[1]),
            "cells": int(raw[2]),
            "groups": int(raw[3]),
            "directions": int(raw[4]),
            "save_angular_flux": raw[5] == "true",
        }
        expected_config = {
            "ranks": ranks,
            "cells_per_axis": cells_per_axis,
            "cells": cells_per_axis**3,
            "groups": 21,
            "directions": 32,
            "save_angular_flux": False,
        }
        if config != expected_config:
            errors.append(f"configuration mismatch: expected {expected_config}, found {config}")

    maximum_records = MAX_RE.findall(text)
    maxima: dict[int, float] = {}
    for group_text, value_text in maximum_records:
        group = int(group_text)
        value = float(value_text)
        if group in maxima:
            errors.append(f"duplicate maximum record for group {group}")
        maxima[group] = value
        if not math.isfinite(value):
            errors.append(f"non-finite maximum for group {group}")
    if set(maxima) != EXPECTED_MAX_GROUPS:
        errors.append(f"expected maximum groups {sorted(EXPECTED_MAX_GROUPS)}, found {sorted(maxima)}")

    wgs_records = WGS_RE.findall(text)
    wgs: list[dict[str, Any]] = []
    for lo_text, hi_text, iteration_text, residual_text, status in wgs_records:
        record = {
            "groups": [int(lo_text), int(hi_text)],
            "iteration": int(iteration_text),
            "residual": float(residual_text),
            "status": status or None,
        }
        wgs.append(record)
        if not math.isfinite(record["residual"]) or record["residual"] < 0.0:
            errors.append(f"invalid WGS residual at iteration {record['iteration']}")
    if not wgs:
        errors.append("no WGS iteration records")
    else:
        iterations = [record["iteration"] for record in wgs]
        if iterations != list(range(iterations[-1] + 1)):
            errors.append(f"WGS iterations are not exactly contiguous from zero: {iterations}")
        if any(record["groups"] != [0, 20] for record in wgs):
            errors.append("unexpected WGS group range")
        converged = [record for record in wgs if record["status"] == "converged"]
        if len(converged) != 1 or converged[0] is not wgs[-1]:
            errors.append("expected exactly the final WGS record to have status=converged")
        unexpected_statuses = [record["status"] for record in wgs[:-1] if record["status"] is not None]
        if unexpected_statuses:
            errors.append(f"unexpected non-final WGS statuses: {unexpected_statuses}")

    timing_records = TIMING_RE.findall(text)
    if len(timing_records) != 1:
        errors.append(f"expected one WGS timing record, found {len(timing_records)}")
        timing = None
    else:
        raw = timing_records[0]
        timing = {
            "groups": [int(raw[0]), int(raw[1])],
            "avg_sweep_seconds": float(raw[2]),
            "sweep_nanoseconds_per_unknown": float(raw[3]),
        }
        if timing["groups"] != [0, 20]:
            errors.append("unexpected timing group range")
        if not all(math.isfinite(timing[key]) and timing[key] > 0.0 for key in timing if key != "groups"):
            errors.append(f"invalid WGS timing values: {timing}")

    unknown_records = UNKNOWNS_RE.findall(text)
    if len(unknown_records) != 1:
        errors.append(f"expected one WGS unknown-count record, found {len(unknown_records)}")
        unknowns = None
    else:
        raw = unknown_records[0]
        unknowns = {
            "groups": [int(raw[0]), int(raw[1])],
            "unknowns": int(raw[2]),
            "lagged_unknowns": int(raw[3]),
            "lagged_percent": float(raw[4]),
        }
        if unknowns["groups"] != [0, 20] or unknowns["unknowns"] <= 0.0:
            errors.append(f"invalid WGS unknown counts: {unknowns}")
        if (
            unknowns["lagged_unknowns"] < 0
            or unknowns["lagged_unknowns"] > unknowns["unknowns"]
            or not math.isfinite(unknowns["lagged_percent"])
            or not (0.0 <= unknowns["lagged_percent"] <= 100.0)
        ):
            errors.append(f"invalid lagged unknown counts: {unknowns}")
        expected_percent = 100.0 * unknowns["lagged_unknowns"] / unknowns["unknowns"]
        if abs(unknowns["lagged_percent"] - expected_percent) > 0.0050001:
            errors.append(
                "lagged percentage is inconsistent with exact counts: "
                f"reported {unknowns['lagged_percent']}, computed {expected_percent}"
            )

    actual_signature = None
    if wgs and set(maxima) == EXPECTED_MAX_GROUPS and unknowns is not None:
        actual_signature = RankSignature(
            maxima=maxima,
            wgs_final_iteration=wgs[-1]["iteration"],
            wgs_iteration_count=len(wgs),
            unknowns=unknowns["unknowns"],
            lagged_unknowns=unknowns["lagged_unknowns"],
            groups=(wgs[-1]["groups"][0], wgs[-1]["groups"][1]),
        )
        if expected_signature is not None:
            if actual_signature.groups != expected_signature.groups:
                errors.append(
                    f"WGS group signature mismatch: expected {expected_signature.groups}, "
                    f"found {actual_signature.groups}"
                )
            if actual_signature.wgs_final_iteration != expected_signature.wgs_final_iteration:
                errors.append(
                    "WGS final iteration mismatch: "
                    f"expected {expected_signature.wgs_final_iteration}, "
                    f"found {actual_signature.wgs_final_iteration}"
                )
            if actual_signature.wgs_iteration_count != expected_signature.wgs_iteration_count:
                errors.append(
                    "WGS iteration-count mismatch: "
                    f"expected {expected_signature.wgs_iteration_count}, "
                    f"found {actual_signature.wgs_iteration_count}"
                )
            if actual_signature.unknowns != expected_signature.unknowns:
                errors.append(
                    f"unknown-count mismatch: expected {expected_signature.unknowns}, "
                    f"found {actual_signature.unknowns}"
                )
            if actual_signature.lagged_unknowns != expected_signature.lagged_unknowns:
                errors.append(
                    f"lagged-unknown-count mismatch: expected {expected_signature.lagged_unknowns}, "
                    f"found {actual_signature.lagged_unknowns}"
                )
            for group, expected in expected_signature.maxima.items():
                actual = maxima.get(group)
                if actual is None or not values_close(
                    actual, expected, max_absolute_tolerance, max_relative_tolerance
                ):
                    errors.append(
                        f"maximum mismatch for group {group}: expected {expected:.12e}, "
                        f"found {actual!r}, atol={max_absolute_tolerance}, "
                        f"rtol={max_relative_tolerance}"
                    )

    if errors:
        raise StudyError("output validation failed:\n- " + "\n- ".join(errors))

    assert (
        worker_record is not None
        and config is not None
        and timing is not None
        and unknowns is not None
        and actual_signature is not None
    )
    return {
        "completion_count": completion_count,
        "config": config,
        "scheduler": worker_record,
        "maxima": {str(group): value for group, value in sorted(maxima.items())},
        "wgs": wgs,
        "wgs_final_iteration": actual_signature.wgs_final_iteration,
        "wgs_iteration_count": actual_signature.wgs_iteration_count,
        "wgs_final_residual": wgs[-1]["residual"],
        "timing": timing,
        "unknowns": unknowns,
        "signature": actual_signature.as_dict(),
    }


def run_capture(argv: Sequence[str], cwd: pathlib.Path, timeout: float = 10.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(item) for item in argv],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        return {
            "argv": [str(item) for item in argv],
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": [str(item) for item in argv], "error": str(error)}


def executable_info(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "executable": os.access(resolved, os.X_OK),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def file_info(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def workload_path_label(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def workload_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    """Return the physics/input identity that a numerical reference applies to."""

    return {
        "config": {
            "cells_per_axis": args.cells_per_axis,
            "cells": args.cells_per_axis**3,
            "groups": 21,
            "directions": 32,
            "save_angular_flux": False,
        },
        "input": {
            "path": workload_path_label(args.input_resolved),
            "sha256": sha256_file(args.input_resolved),
        },
        "assets": [
            {"path": workload_path_label(path), "sha256": sha256_file(path)}
            for path in args.workload_assets_resolved
        ],
    }


def git_provenance(repo: pathlib.Path) -> dict[str, Any]:
    revision = run_capture(["git", "rev-parse", "HEAD"], repo)
    status = run_capture(["git", "status", "--porcelain=v1", "--untracked-files=all"], repo)
    unstaged = run_capture(["git", "diff", "--binary", "HEAD"], repo)
    staged = run_capture(["git", "diff", "--binary", "--cached", "HEAD"], repo)
    revision_text = revision.get("stdout", "").strip()
    status_text = status.get("stdout", "")
    diff_bytes = (staged.get("stdout", "") + unstaged.get("stdout", "")).encode()

    untracked: list[dict[str, Any]] = []
    for line in status_text.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        candidate = repo / relative
        if candidate.is_file():
            untracked.append({"path": relative, "sha256": sha256_file(candidate)})
        else:
            untracked.append({"path": relative, "sha256": None})
    dirty_digest = hashlib.sha256()
    dirty_digest.update(status_text.encode())
    dirty_digest.update(diff_bytes)
    for entry in untracked:
        dirty_digest.update(entry["path"].encode())
        dirty_digest.update((entry["sha256"] or "").encode())

    def command_summary(command: dict[str, Any]) -> dict[str, Any]:
        stdout = command.get("stdout", "")
        return {
            "argv": command.get("argv"),
            "exit_code": command.get("exit_code"),
            "error": command.get("error"),
            "stderr": command.get("stderr", ""),
            "stdout_sha256": sha256_bytes(stdout.encode()),
            "stdout_bytes": len(stdout.encode()),
        }

    return {
        "revision": revision_text or None,
        "revision_is_exact": bool(re.fullmatch(r"[0-9a-f]{40}", revision_text)),
        "dirty": bool(status_text.strip()),
        "status": status_text.splitlines(),
        "dirty_diff_sha256": sha256_bytes(diff_bytes),
        "dirty_state_sha256": dirty_digest.hexdigest(),
        "untracked": untracked,
        "commands": {
            "revision": command_summary(revision),
            "status": command_summary(status),
            "unstaged": command_summary(unstaged),
            "staged": command_summary(staged),
        },
    }


def build_cache_provenance(binary: pathlib.Path) -> dict[str, Any]:
    build_dir = binary.expanduser().resolve().parent.parent
    cache = build_dir / "CMakeCache.txt"
    result = {"build_directory": str(build_dir), "cache": file_info(cache)}
    if not cache.is_file():
        return result
    keys = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_FLAGS",
        "CMAKE_CUDA_ARCHITECTURES",
        "OPENSN_WITH_CUDA",
        "caliper_DIR",
    }
    selected: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        left, value = line.split("=", 1)
        key = left.split(":", 1)[0]
        if key in keys:
            selected[key] = value
    result["selected_values"] = selected
    return result


FRESH_PREFIX_LIBRARY_PREFIXES = (
    "libcaliper",
    "libpetsc",
    "libhdf5",
    "libvtk",
    "libHYPRE",
    "libsuperlu",
    "libptscotch",
    "libscotch",
    "libparmetis",
    "libmetis",
)


def elf_identity(path: pathlib.Path) -> dict[str, Any]:
    """Hash one resolved ELF and record its build ID and runtime search path."""

    resolved = path.resolve()
    result = file_info(resolved)
    notes = run_capture(["readelf", "-n", str(resolved)], REPO_ROOT, timeout=10.0)
    dynamic = run_capture(["readelf", "-d", str(resolved)], REPO_ROOT, timeout=10.0)
    note_text = notes.get("stdout", "") + notes.get("stderr", "")
    dynamic_text = dynamic.get("stdout", "") + dynamic.get("stderr", "")
    build_id = re.search(r"Build ID:\s*([0-9a-fA-F]+)", note_text)
    search_paths = [
        {"kind": kind, "value": value}
        for kind, value in re.findall(r"\((RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic_text)
    ]
    result.update(
        {
            "build_id": build_id.group(1).lower() if build_id else None,
            "search_paths": search_paths,
            "readelf_notes_exit_code": notes.get("exit_code"),
            "readelf_dynamic_exit_code": dynamic.get("exit_code"),
        }
    )
    return result


def dynamic_link_provenance(binary: pathlib.Path) -> dict[str, Any]:
    """Record and validate the full dynamic-loader closure selected for OpenSn."""

    resolved_binary = binary.expanduser().resolve()
    linkage = run_capture(["ldd", str(resolved_binary)], REPO_ROOT, timeout=30.0)
    text = linkage.get("stdout", "") + linkage.get("stderr", "")
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("linux-vdso"):
            continue
        missing_match = re.match(r"(\S+)\s+=>\s+not found$", line)
        if missing_match:
            missing.append(missing_match.group(1))
            continue
        mapped = re.match(r"(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-fA-F]+\)$", line)
        loader = re.match(r"(/\S+)\s+\(0x[0-9a-fA-F]+\)$", line)
        if mapped:
            soname, path_text = mapped.groups()
        elif loader:
            path_text = loader.group(1)
            soname = pathlib.Path(path_text).name
        else:
            continue
        requested_path = pathlib.Path(path_text)
        identity = elf_identity(requested_path)
        entries.append(
            {
                "soname": soname,
                "loader_path": str(requested_path),
                "realpath": str(requested_path.resolve()),
                **identity,
            }
        )

    entries.sort(key=lambda item: (item["soname"], item["realpath"]))
    build_directory = resolved_binary.parent.parent
    caliper_entries = [item for item in entries if item["soname"].startswith("libcaliper")]
    opensn_entries = [item for item in entries if item["soname"].startswith("libopensn")]
    structural_errors: list[str] = []
    if len(opensn_entries) != 1:
        structural_errors.append(f"expected one libopensn entry, found {len(opensn_entries)}")
    if len(caliper_entries) != 1:
        structural_errors.append(f"expected one libcaliper entry, found {len(caliper_entries)}")
    dependency_prefix = (
        pathlib.Path(caliper_entries[0]["realpath"]).parent.parent if len(caliper_entries) == 1 else None
    )
    prefix_violations: list[dict[str, str]] = []
    for entry in entries:
        soname = entry["soname"]
        path = pathlib.Path(entry["realpath"])
        expected_root: pathlib.Path | None = None
        if soname.startswith("libopensn"):
            expected_root = build_directory
        elif dependency_prefix is not None and soname.startswith(FRESH_PREFIX_LIBRARY_PREFIXES):
            expected_root = dependency_prefix
        if expected_root is not None:
            try:
                path.relative_to(expected_root.resolve())
            except ValueError:
                prefix_violations.append(
                    {"soname": soname, "path": str(path), "expected_root": str(expected_root.resolve())}
                )
    identity_errors = [
        f"{entry['soname']} at {entry['realpath']} could not be hashed"
        for entry in entries
        if not entry.get("sha256")
    ]

    digest_payload = [
        {
            "soname": entry["soname"],
            "realpath": entry["realpath"],
            "sha256": entry.get("sha256"),
            "build_id": entry.get("build_id"),
            "search_paths": entry.get("search_paths"),
        }
        for entry in entries
    ]
    valid = (
        linkage.get("exit_code") == 0
        and bool(entries)
        and not missing
        and not prefix_violations
        and not structural_errors
        and not identity_errors
    )
    return {
        "ldd": linkage,
        "binary": elf_identity(resolved_binary),
        "entries": entries,
        "entry_count": len(entries),
        "missing": missing,
        "structural_errors": structural_errors,
        "dependency_prefix": str(dependency_prefix) if dependency_prefix is not None else None,
        "prefix_violations": prefix_violations,
        "identity_errors": identity_errors,
        "closure_sha256": sha256_bytes(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
        ),
        "valid": valid,
    }


def linked_caliper_provenance(binary: pathlib.Path) -> dict[str, Any]:
    """Inspect the Caliper library selected by the OpenSn binary's dynamic linkage."""

    resolved = binary.expanduser().resolve()
    result: dict[str, Any] = {
        "binary": str(resolved),
        "linked": False,
        "mpi": False,
        "nvtx": False,
        "cupti": False,
    }
    if not resolved.is_file():
        result["reason"] = "binary does not exist"
        return result

    linkage = run_capture(["ldd", str(resolved)], REPO_ROOT, timeout=10.0)
    result["ldd"] = linkage
    combined = linkage.get("stdout", "") + linkage.get("stderr", "")
    match = re.search(r"\blibcaliper\.so(?:\.\S+)?\s+=>\s+(\S+)", combined)
    if not match:
        result["reason"] = "ldd did not resolve libcaliper"
        return result

    library = pathlib.Path(match.group(1)).resolve()
    prefix = library.parent.parent
    header = prefix / "include/caliper/caliper-config.h"
    query = prefix / "bin/cali-query"
    result.update(
        {
            "linked": True,
            "library": file_info(library),
            "prefix": str(prefix),
            "config_header": file_info(header),
            "cali_query": str(query) if query.is_file() else None,
        }
    )

    header_text = header.read_text(encoding="utf-8", errors="replace") if header.is_file() else ""
    version_match = re.search(r'^\s*#define\s+CALIPER_VERSION\s+"([^"]+)"', header_text, re.MULTILINE)
    result["version"] = version_match.group(1) if version_match else None
    header_nvtx = bool(re.search(r"^\s*#define\s+CALIPER_HAVE_NVTX(?:\s+1)?\s*$", header_text, re.MULTILINE))
    header_cupti = bool(re.search(r"^\s*#define\s+CALIPER_HAVE_CUPTI(?:\s+1)?\s*$", header_text, re.MULTILINE))
    header_mpi = bool(re.search(r"^\s*#define\s+CALIPER_HAVE_MPI(?:\s+1)?\s*$", header_text, re.MULTILINE))
    result["header_features"] = {"mpi": header_mpi, "nvtx": header_nvtx, "cupti": header_cupti}

    if query.is_file() and os.access(query, os.X_OK):
        services = run_capture([str(query), "--help", "services"], REPO_ROOT, timeout=10.0)
        configs = run_capture([str(query), "--help", "configs"], REPO_ROOT, timeout=10.0)
        result["services_probe"] = services
        result["configs_probe"] = configs
        service_text = services.get("stdout", "") + services.get("stderr", "")
        config_text = configs.get("stdout", "") + configs.get("stderr", "")
        result["nvtx_service_listed"] = bool(re.search(r"^\s*nvtx\s", service_text, re.MULTILINE))
        result["mpi_services_listed"] = all(
            re.search(rf"^\s*{name}\s", service_text, re.MULTILINE) for name in ("mpi", "mpireport")
        )
        result["cupti_services_listed"] = all(
            re.search(rf"^\s*{name}\s", service_text, re.MULTILINE) for name in ("cupti", "cuptitrace")
        )
        result["mpi_report_recipe_listed"] = "mpi-report" in config_text
        result["cuda_activity_recipe_listed"] = "cuda-activity-report" in config_text
        services_succeeded = services.get("exit_code") == 0
        configs_succeeded = configs.get("exit_code") == 0
        result["probes_succeeded"] = services_succeeded and configs_succeeded
        result["mpi"] = bool(
            header_mpi
            and services_succeeded
            and configs_succeeded
            and result["mpi_services_listed"]
            and result["mpi_report_recipe_listed"]
        )
        result["nvtx"] = bool(header_nvtx and services_succeeded and result["nvtx_service_listed"])
        result["cupti"] = bool(
            header_cupti
            and services_succeeded
            and configs_succeeded
            and result["cupti_services_listed"]
            and result["cuda_activity_recipe_listed"]
        )
    else:
        result["reason"] = "matching Caliper prefix has no executable cali-query capability probe"
    return result


def physical_cores_in_affinity() -> tuple[int, list[int]]:
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = list(range(os.cpu_count() or 1))
    cores: set[tuple[str, str]] = set()
    for cpu in cpus:
        topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
            cores.add((package, core))
        except OSError:
            return len(cpus), cpus
    return max(1, len(cores)), cpus


def tool_provenance(cwd: pathlib.Path, dry_run: bool) -> dict[str, Any]:
    tools = {
        "python": [sys.executable, "--version"],
        "mpirun": ["mpirun", "--version"],
        "nsys": ["nsys", "--version"],
        "ncu": ["ncu", "--version"],
        "compute-sanitizer": ["compute-sanitizer", "--version"],
        "perf": ["perf", "--version"],
        "nvcc": ["nvcc", "--version"],
        "readelf": ["readelf", "--version"],
        "ldd": ["ldd", "--version"],
    }
    result: dict[str, Any] = {}
    for name, command in tools.items():
        executable = shutil.which(command[0]) if command[0] != sys.executable else sys.executable
        result[name] = {"path": executable}
        if executable:
            probe = [executable, *command[1:]]
            result[name]["version_probe"] = run_capture(probe, cwd, timeout=5.0)
    if dry_run:
        result["gpu"] = {"probe_skipped": True, "reason": "dry-run does not access a GPU"}
    else:
        nvidia_smi = shutil.which("nvidia-smi")
        result["gpu"] = {
            "path": nvidia_smi,
            "probe": run_capture(
                [
                    nvidia_smi or "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,compute_cap,pci.bus_id",
                    "--format=csv,noheader",
                ],
                cwd,
                timeout=5.0,
            ),
            "initial_state_probe": run_capture(
                [
                    nvidia_smi or "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu",
                    "--format=csv,noheader",
                ],
                cwd,
                timeout=5.0,
            ),
        }
    return result


def collect_provenance(
    binary: pathlib.Path,
    input_path: pathlib.Path,
    workload_assets: Sequence[pathlib.Path],
    dry_run: bool,
) -> dict[str, Any]:
    physical_cores, affinity = physical_cores_in_affinity()
    relevant_environment = {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith(("CALI", "CUDA", "NVIDIA", "OMP", "OMPI", "PMI", "MPI", "OPENSN"))
        or key in {"PATH", "LD_LIBRARY_PATH"}
    }
    dynamic_link = dynamic_link_provenance(binary)
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_provenance(REPO_ROOT),
        "binary": executable_info(binary),
        "input": file_info(input_path),
        "workload_assets": [file_info(path) for path in workload_assets],
        "build": build_cache_provenance(binary),
        "caliper": linked_caliper_provenance(binary),
        "dynamic_link": dynamic_link,
        "tools": tool_provenance(REPO_ROOT, dry_run),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "physical_cores_in_affinity": physical_cores,
            "affinity_cpus": affinity,
        },
        "environment": relevant_environment,
    }


def gpu_state_probe(device: str, cwd: pathlib.Path) -> dict[str, Any]:
    """Capture selected-device telemetry and active CUDA processes around one run."""

    executable = shutil.which("nvidia-smi") or "nvidia-smi"
    state = run_capture(
        [
            executable,
            f"--id={device}",
            "--query-gpu=timestamp,uuid,pstate,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu",
            "--format=csv,noheader",
        ],
        cwd,
        timeout=5.0,
    )
    processes = run_capture(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        cwd,
        timeout=5.0,
    )
    return {"device": device, "state": state, "compute_processes": processes}


def selected_tool_provenance(args: argparse.Namespace) -> dict[str, Any]:
    selected = {"mpirun": args.mpirun_resolved, "python": sys.executable}
    if args.command == "nsys":
        selected["nsys"] = args.nsys_resolved
    elif args.command == "ncu":
        selected["ncu"] = args.ncu_resolved
    elif args.command == "sanitizer":
        selected["compute-sanitizer"] = args.compute_sanitizer_resolved
    result: dict[str, Any] = {}
    for name, executable in selected.items():
        version_argument = "--version"
        result[name] = {
            **executable_info(pathlib.Path(executable)),
            "version_probe": run_capture([executable, version_argument], REPO_ROOT, timeout=5.0),
        }
    return result


class Study:
    """Atomic study directory with durable state transitions."""

    def __init__(self, root: pathlib.Path, mode: str, label: str, dry_run: bool):
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or mode
        stem = f"{timestamp}-{safe_label}-{mode}"
        final = root / stem
        suffix = 1
        while final.exists() or (root / f".{final.name}.tmp-{os.getpid()}").exists():
            final = root / f"{stem}-{suffix}"
            suffix += 1
        self.final = final
        self.working = root / f".{final.name}.tmp-{os.getpid()}"
        self.working.mkdir(mode=0o700)
        (self.working / "logs").mkdir()
        (self.working / "artifacts").mkdir()
        self.dry_run = dry_run
        self.commands: list[CommandResult] = []
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "mode": mode,
            "label": safe_label,
            "same_gpu_contention_diagnostic": True,
            "scaling_disclaimer": (
                "All MPI ranks share one GPU; rank trends are contention diagnostics, not accelerator strong scaling."
            ),
            "dry_run": dry_run,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "initializing",
            "commands": [],
        }
        self._write_state("initializing")

    def relative(self, path: pathlib.Path) -> str:
        return str(path.relative_to(self.working))

    def _write_state(self, status: str, **extra: Any) -> None:
        state = {
            "status": status,
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "completed_commands": len([item for item in self.commands if item.status == "completed"]),
            "total_commands": len(self.commands),
            **extra,
        }
        atomic_write_json(self.working / "state.json", state)

    def write_manifest(self) -> None:
        self.manifest["commands"] = [result.as_dict() for result in self.commands]
        atomic_write_json(self.working / "manifest.json", self.manifest)
        rows: list[dict[str, Any]] = []
        for result in self.commands:
            row = result.as_dict()
            row["argv"] = json.dumps(row["argv"], separators=(",", ":"))
            row["environment"] = json.dumps(row["environment"], sort_keys=True, separators=(",", ":"))
            row["environment_unset"] = json.dumps(row["environment_unset"], separators=(",", ":"))
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True, separators=(",", ":"))
            row["validation"] = json.dumps(row["validation"], sort_keys=True, separators=(",", ":"))
            rows.append(row)
        atomic_write_csv(
            self.working / "runs.csv",
            rows,
            (
                "command_id",
                "kind",
                "status",
                "exit_code",
                "timed_out",
                "wall_seconds",
                "cwd",
                "log",
                "argv",
                "environment",
                "environment_unset",
                "metadata",
                "validation",
            ),
        )

    def execute(
        self,
        *,
        kind: str,
        argv: Sequence[str],
        cwd: pathlib.Path,
        environment: dict[str, str],
        timeout: float,
        log_name: str,
        metadata: dict[str, Any] | None = None,
        stdout_name: str | None = None,
    ) -> CommandResult:
        command_id = len(self.commands) + 1
        log_path = self.working / "logs" / log_name
        stdout_path = self.working / "artifacts" / stdout_name if stdout_name else None
        environment_unset = sorted(
            {key for key in os.environ if key.startswith("CALI_")}
            | {"CALI_CONFIG", "CALI_CONFIG_FILE", "CALI_SERVICES_ENABLE", "OPENSN_CBCD_NUM_WORKERS"}
        )
        record = CommandResult(
            command_id=command_id,
            kind=kind,
            argv=[str(item) for item in argv],
            cwd=str(cwd),
            environment=dict(sorted(environment.items())),
            environment_unset=environment_unset,
            log=self.relative(log_path),
            status="planned" if self.dry_run else "running",
            metadata=dict(metadata or {}),
        )
        self.commands.append(record)
        self._write_state(record.status, current_command=record.as_dict())
        self.write_manifest()
        if self.dry_run:
            print(f"DRY-RUN [{kind}] {command_text(record.argv)}")
            return record

        full_environment = os.environ.copy()
        for key in environment_unset:
            full_environment.pop(key, None)
        full_environment.update(environment)
        gpu_device = environment.get("CUDA_VISIBLE_DEVICES")
        if gpu_device is not None:
            record.metadata["gpu_state_before"] = gpu_state_probe(gpu_device, cwd)
            self.write_manifest()
        start = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        with log_path.open("wb") as log_stream:
            stdout_stream = stdout_path.open("wb") if stdout_path else log_stream
            try:
                process = subprocess.Popen(
                    record.argv,
                    cwd=cwd,
                    env=full_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=log_stream,
                    start_new_session=True,
                )
                try:
                    record.exit_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    record.timed_out = True
                    record.exit_code = self._terminate(process, timeout)
            except OSError as error:
                log_stream.write(f"unable to launch command: {error}\n".encode())
                record.exit_code = 127
            except BaseException:
                if process is not None and process.poll() is None:
                    self._terminate(process, timeout)
                raise
            finally:
                if stdout_path:
                    stdout_stream.close()
        record.wall_seconds = time.monotonic() - start
        if gpu_device is not None:
            record.metadata["gpu_state_after"] = gpu_state_probe(gpu_device, cwd)
        record.status = "completed" if record.exit_code == 0 and not record.timed_out else "failed"
        self._write_state(record.status, current_command=record.as_dict())
        self.write_manifest()
        return record

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes], timeout: float) -> int:
        if process.poll() is not None:
            return process.returncode
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=min(5.0, max(1.0, timeout / 10.0)))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.wait()

    def finish(self, status: str, error: str | None = None) -> pathlib.Path:
        self.manifest["status"] = status
        self.manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if error:
            self.manifest["error"] = error
        self._write_state(status, error=error)
        self.write_manifest()
        os.replace(self.working, self.final)
        return self.final


def read_log(study: Study, result: CommandResult) -> str:
    return (study.working / result.log).read_text(encoding="utf-8", errors="replace")


def require_success(result: CommandResult) -> None:
    if result.timed_out:
        raise StudyError(f"{result.kind} timed out after {result.wall_seconds:.3f} seconds")
    if result.exit_code != 0:
        raise StudyError(f"{result.kind} exited with status {result.exit_code}; see {result.log}")


def validate_profiler_csv(path: pathlib.Path, description: str) -> dict[str, Any]:
    """Require a real CSV header and at least one profiler data row.

    Nsight Systems 2025.3 writes SQLite export/progress messages to stdout
    before the requested CSV report.  Treat leading one-column records as a
    preamble instead of assuming that the first non-empty record is the
    header.  A skipped or empty report still fails because it has no
    multi-column header followed by a width-matched data record.
    """

    if not path.is_file() or path.stat().st_size == 0:
        raise StudyError(f"{description} did not create a non-empty CSV artifact: {path}")
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        rows = [row for row in csv.reader(stream) if any(cell.strip() for cell in row)]
    header_index = next(
        (
            index
            for index, row in enumerate(rows[:-1])
            if len(row) >= 2 and any(len(candidate) == len(row) for candidate in rows[index + 1 :])
        ),
        None,
    )
    if header_index is None:
        raise StudyError(
            f"{description} CSV has no parseable header/data rows (profiler may have skipped the report): {path}"
        )
    columns = rows[header_index]
    data_rows = [row for row in rows[header_index + 1 :] if len(row) == len(columns)]
    return {
        "path": str(path),
        "columns": columns,
        "data_rows": len(data_rows),
        "preamble_rows": header_index,
    }


def resolve_signature_set(args: argparse.Namespace) -> SignatureSet | None:
    if args.reference:
        path = pathlib.Path(args.reference).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        signatures = SignatureSet.from_dict(data)
        current_workload = workload_descriptor(args)
        if signatures.workload != current_workload:
            raise StudyError(
                "signature workload does not match the current input/config/assets:\n"
                f"expected {json.dumps(signatures.workload, sort_keys=True)}\n"
                f"current  {json.dumps(current_workload, sort_keys=True)}"
            )
        return signatures
    return None


def validate_result(
    study: Study,
    result: CommandResult,
    args: argparse.Namespace,
    ranks: int,
    signature: RankSignature | None,
) -> tuple[dict[str, Any], RankSignature]:
    require_success(result)
    try:
        parsed = validate_output(
            read_log(study, result),
            ranks=ranks,
            cells_per_axis=args.cells_per_axis,
            expected_policy=str(result.metadata.get("policy", args.policy)),
            expected_worker_override=args.workers,
            expected_signature=signature,
            max_absolute_tolerance=args.max_atol,
            max_relative_tolerance=args.max_rtol,
        )
    except StudyError:
        result.status = "validation-failed"
        study._write_state("validation-failed", current_command=result.as_dict())
        study.write_manifest()
        raise
    result.validation = parsed
    study.write_manifest()
    return parsed, RankSignature.from_dict(parsed["signature"])


def resolve_program(value: str, dry_run: bool) -> str:
    candidate = pathlib.Path(value).expanduser()
    if candidate.parent != pathlib.Path(".") or candidate.is_absolute():
        resolved = candidate.resolve()
        if not dry_run and (not resolved.is_file() or not os.access(resolved, os.X_OK)):
            raise StudyError(f"not an executable file: {resolved}")
        return str(resolved)
    found = shutil.which(value)
    if found:
        return found
    if dry_run:
        return value
    raise StudyError(f"executable is not available on PATH: {value}")


def rank_pe(args: argparse.Namespace, ranks: int, physical_cores: int) -> int:
    if args.pe_map and ranks in args.pe_map:
        pe = args.pe_map[ranks]
    elif args.pe_per_rank:
        pe = args.pe_per_rank
    else:
        pe = max(1, physical_cores // ranks)
    if ranks * pe > physical_cores and not args.allow_cpu_oversubscribe:
        raise StudyError(
            f"{ranks} ranks x {pe} PE exceeds {physical_cores} physical cores in the affinity mask; "
            "adjust --pe-map/--pe-per-rank or pass --allow-cpu-oversubscribe"
        )
    return pe


def mpi_prefix(args: argparse.Namespace, ranks: int, pe: int) -> list[str]:
    if args.map_by == "none" and args.bind_to != "none" and pe > 1:
        raise StudyError("--map-by none cannot express a multi-PE binding; select slot/core/node or use PE=1")
    command = [args.mpirun_resolved, "--np", str(ranks)]
    if args.bind_to != "none":
        command.extend(["--bind-to", args.bind_to])
    if args.map_by != "none":
        command.extend(["--map-by", f"{args.map_by}:PE={pe}"])
    if args.report_bindings:
        command.append("--report-bindings")
    return command


def opensn_command(args: argparse.Namespace) -> list[str]:
    return [
        args.binary_resolved,
        "-c",
        "--verbose",
        "1",
        "--py",
        f"profile_cells_per_axis = {args.cells_per_axis}",
        "--py",
        f"profile_max_iterations = {args.max_iterations}",
        "-i",
        str(args.input_resolved),
    ]


def run_environment(args: argparse.Namespace, policy: str, pe: int) -> dict[str, str]:
    environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "OPENSN_CBCD_WORKER_POLICY": policy,
        "OPENSN_NUM_THREADS": str(pe),
    }
    if args.workers is not None:
        environment["OPENSN_CBCD_NUM_WORKERS"] = str(args.workers)
    return environment


def initialize_study(args: argparse.Namespace, study: Study) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    study.manifest["arguments"] = {
        key: json_safe(value)
        for key, value in vars(args).items()
        if not key.endswith("_resolved") and key != "handler"
    }
    study.manifest["provenance"] = collect_provenance(
        pathlib.Path(args.binary_resolved),
        args.input_resolved,
        args.workload_assets_resolved,
        args.dry_run,
    )
    study.manifest["provenance"]["selected_tools"] = selected_tool_provenance(args)
    if not study.manifest["provenance"]["git"]["revision_is_exact"]:
        raise StudyError("unable to record an exact 40-character Git revision")
    dynamic_link = study.manifest["provenance"]["dynamic_link"]
    if not args.dry_run and not dynamic_link.get("valid", False):
        raise StudyError(
            "dynamic-link closure is incomplete or escaped the selected build/dependency prefix: "
            f"missing={dynamic_link.get('missing')}, structural={dynamic_link.get('structural_errors')}, "
            f"identity={dynamic_link.get('identity_errors')}, "
            f"violations={dynamic_link.get('prefix_violations')}"
        )
    if not args.dry_run:
        gpu_probe = study.manifest["provenance"]["tools"]["gpu"].get("probe", {})
        if gpu_probe.get("exit_code") != 0 or not gpu_probe.get("stdout", "").strip():
            raise StudyError(
                "nvidia-smi could not provide mandatory GPU provenance: "
                f"{gpu_probe.get('error') or gpu_probe.get('stderr') or 'empty output'}"
            )
    if args.reference:
        study.manifest["reference"] = file_info(args.reference)
    study.manifest["status"] = "dry-run" if args.dry_run else "running"
    study.write_manifest()


def benchmark_condition(policy: str, workers: int | None) -> str:
    return f"fixed-workers-{workers}" if workers is not None else policy


def build_benchmark_schedule(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Create a reproducible, balanced block schedule with adjacent A/B pairs."""

    rng = random.Random(args.schedule_seed)
    policies = list(args.policies)
    initial_flip = {ranks: rng.randrange(2) for ranks in args.ranks}
    schedule: list[dict[str, Any]] = []

    for warmup_block in range(args.warmups):
        cases = [(ranks, policy) for ranks in args.ranks for policy in policies]
        rng.shuffle(cases)
        for ranks, policy in cases:
            schedule.append(
                {
                    "policy": policy,
                    "condition": benchmark_condition(policy, args.workers),
                    "ranks": ranks,
                    "warmup": True,
                    "repetition": warmup_block,
                    "block": warmup_block,
                    "pair_id": f"warmup-{warmup_block}-np{ranks}",
                    "pair_position": None,
                }
            )

    for block in range(args.trials):
        rank_order = list(args.ranks)
        rng.shuffle(rank_order)
        for ranks in rank_order:
            policy_order = list(policies)
            if len(policy_order) == 2 and (initial_flip[ranks] + block) % 2 == 1:
                policy_order.reverse()
            for pair_position, policy in enumerate(policy_order):
                schedule.append(
                    {
                        "policy": policy,
                        "condition": benchmark_condition(policy, args.workers),
                        "ranks": ranks,
                        "warmup": False,
                        "repetition": block,
                        "block": block,
                        "pair_id": f"trial-{block}-np{ranks}",
                        "pair_position": pair_position if len(policy_order) == 2 else None,
                    }
                )

    for order, item in enumerate(schedule, start=1):
        item["execution_order"] = order
    return schedule


def run_benchmark(args: argparse.Namespace, study: Study) -> None:
    physical_cores = study.manifest["provenance"]["host"]["physical_cores_in_affinity"]
    reference = resolve_signature_set(args)
    signatures = reference or SignatureSet(workload=workload_descriptor(args), by_ranks={})
    if reference is not None:
        for ranks in args.ranks:
            reference.require_rank(ranks)

    schedule = build_benchmark_schedule(args)
    study.manifest["benchmark_schedule"] = {
        "seed": args.schedule_seed,
        "design": "randomized rank blocks with adjacent policy pairs and alternating AB/BA order",
        "baseline_policy": args.policies[0],
        "candidate_policy": args.policies[1] if len(args.policies) == 2 else None,
        "commands": schedule,
    }
    study.write_manifest()

    rows: list[dict[str, Any]] = []
    completed = 0
    for planned in schedule:
        metadata = dict(planned)
        policy = str(planned["policy"])
        ranks = int(planned["ranks"])
        pe = rank_pe(args, ranks, physical_cores)
        metadata.update({"pe_per_rank": pe, "worker_override": args.workers})
        label = "warmup" if planned["warmup"] else "trial"
        result = study.execute(
            kind="benchmark",
            argv=[*mpi_prefix(args, ranks, pe), *opensn_command(args)],
            cwd=args.input_resolved.parent,
            environment=run_environment(args, policy, pe),
            timeout=args.timeout,
            log_name=(
                f"benchmark-{int(planned['execution_order']):03d}-{planned['condition']}-"
                f"np{ranks}-{label}{planned['repetition']}.log"
            ),
            metadata=metadata,
        )
        if study.dry_run:
            continue
        expected = signatures.by_ranks.get(ranks)
        parsed, actual_signature = validate_result(study, result, args, ranks, expected)
        if expected is None:
            signatures.by_ranks[ranks] = actual_signature
            atomic_write_json(study.working / "signature.json", signatures.as_dict())
        completed += 1
        timing = parsed["timing"]
        unknowns = parsed["unknowns"]
        row = {
            **metadata,
            "actual_workers": parsed["scheduler"]["workers"],
            "reserved_communicator_threads": parsed["scheduler"]["reserved_communicator_threads"],
            "wall_seconds": result.wall_seconds,
            "avg_sweep_seconds": timing["avg_sweep_seconds"],
            "sweep_nanoseconds_per_unknown": timing["sweep_nanoseconds_per_unknown"],
            "unknowns": unknowns["unknowns"],
            "lagged_unknowns": unknowns["lagged_unknowns"],
            "wgs_final_iteration": parsed["wgs_final_iteration"],
            "wgs_final_residual": parsed["wgs_final_residual"],
            "max_group_0": parsed["maxima"]["0"],
            "max_group_19": parsed["maxima"]["19"],
            "log": result.log,
        }
        rows.append(row)
        atomic_write_csv(study.working / "benchmark-runs.csv", rows, tuple(row))
        study._write_state("running", benchmark_completed=completed, benchmark_total=len(schedule))

    if study.dry_run:
        return
    missing_ranks = set(args.ranks) - set(signatures.by_ranks)
    if missing_ranks:
        raise StudyError(
            f"benchmark signatures are missing requested ranks {sorted(missing_ranks)}; "
            f"available {sorted(signatures.by_ranks)}"
        )
    atomic_write_json(study.working / "signature.json", signatures.as_dict())

    summary_rows: list[dict[str, Any]] = []
    for policy in args.policies:
        for ranks in args.ranks:
            selected = [
                row
                for row in rows
                if row["policy"] == policy and row["ranks"] == ranks and not row["warmup"]
            ]
            worker_counts = {int(row["actual_workers"]) for row in selected}
            reserved_counts = {int(row["reserved_communicator_threads"]) for row in selected}
            if len(worker_counts) != 1 or len(reserved_counts) != 1:
                raise StudyError(
                    f"scheduler resources changed within condition {policy} at np={ranks}: "
                    f"workers={sorted(worker_counts)}, reserved={sorted(reserved_counts)}"
                )
            summary: dict[str, Any] = {
                "condition": benchmark_condition(policy, args.workers),
                "policy": policy,
                "ranks": ranks,
                "pe_per_rank": selected[0]["pe_per_rank"],
                "actual_workers": next(iter(worker_counts)),
                "reserved_communicator_threads": next(iter(reserved_counts)),
                "worker_override": args.workers,
                "same_gpu_contention_diagnostic": True,
            }
            for field in ("avg_sweep_seconds", "sweep_nanoseconds_per_unknown", "wall_seconds"):
                stats = descriptive_statistics([float(row[field]) for row in selected])
                for name, value in stats.items():
                    summary[f"{field}_{name}"] = value
            summary_rows.append(summary)

    for policy in args.policies:
        baseline = next(
            row
            for row in summary_rows
            if row["policy"] == policy and row["ranks"] == args.baseline_ranks
        )
        baseline_time = baseline["avg_sweep_seconds_median"]
        for row in summary_rows:
            if row["policy"] != policy:
                continue
            speedup = baseline_time / row["avg_sweep_seconds_median"]
            row["same_gpu_speedup"] = speedup
            row["same_gpu_rank_efficiency"] = speedup / (row["ranks"] / args.baseline_ranks)

    paired_rows: list[dict[str, Any]] = []
    if len(args.policies) == 2:
        baseline_policy, candidate_policy = args.policies
        for ranks in args.ranks:
            for block in range(args.trials):
                pair = [
                    row
                    for row in rows
                    if not row["warmup"] and row["ranks"] == ranks and row["block"] == block
                ]
                by_policy = {str(row["policy"]): row for row in pair}
                baseline_row = by_policy[baseline_policy]
                candidate_row = by_policy[candidate_policy]
                paired_rows.append(
                    {
                        "pair_id": baseline_row["pair_id"],
                        "block": block,
                        "ranks": ranks,
                        "baseline_policy": baseline_policy,
                        "candidate_policy": candidate_policy,
                        "baseline_execution_order": baseline_row["execution_order"],
                        "candidate_execution_order": candidate_row["execution_order"],
                        "baseline_avg_sweep_seconds": baseline_row["avg_sweep_seconds"],
                        "candidate_avg_sweep_seconds": candidate_row["avg_sweep_seconds"],
                        "candidate_over_baseline": (
                            candidate_row["avg_sweep_seconds"] / baseline_row["avg_sweep_seconds"]
                        ),
                        "baseline_over_candidate_speedup": (
                            baseline_row["avg_sweep_seconds"] / candidate_row["avg_sweep_seconds"]
                        ),
                    }
                )
        atomic_write_csv(study.working / "paired-policy-comparison.csv", paired_rows, tuple(paired_rows[0]))
        atomic_write_json(study.working / "paired-policy-comparison.json", paired_rows)

    fields: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    atomic_write_csv(study.working / "summary.csv", summary_rows, fields)
    atomic_write_json(study.working / "summary.json", summary_rows)
    study.manifest["signature"] = signatures.as_dict()
    study.manifest["summary"] = summary_rows
    study.manifest["paired_policy_comparison"] = paired_rows


def require_profile_reference(
    args: argparse.Namespace, signatures: SignatureSet | None, ranks: Sequence[int]
) -> SignatureSet | None:
    if args.dry_run:
        return signatures
    if signatures is None:
        raise StudyError("profile runs require --reference to a validated schema-v2 benchmark signature.json")
    for rank_count in ranks:
        signatures.require_rank(rank_count)
    return signatures


def run_caliper(args: argparse.Namespace, study: Study) -> None:
    signatures = require_profile_reference(args, resolve_signature_set(args), args.ranks)
    physical_cores = study.manifest["provenance"]["host"]["physical_cores_in_affinity"]
    caliper = study.manifest["provenance"]["caliper"]
    if not caliper.get("linked", False) or not caliper.get("mpi", False):
        raise StudyError(
            "Caliper profiles require the selected OpenSn binary to link a Caliper with verified "
            f"MPI/mpireport support: {caliper.get('library', caliper.get('reason'))}"
        )
    if args.mode == "auto":
        modes = ("runtime", "mpi", "cuda") if caliper.get("cupti", False) else ("runtime", "mpi")
        if not caliper.get("cupti", False):
            study.manifest.setdefault("degraded_capabilities", []).append(
                "Caliper auto mode omitted CUDA activity because linked Caliper lacks CUPTI support."
            )
    elif args.mode == "both":
        modes = ("runtime", "mpi")
    elif args.mode == "all":
        modes = ("runtime", "mpi", "cuda")
    else:
        modes = (args.mode,)
    if "cuda" in modes and not caliper.get("cupti", False):
        raise StudyError(
            "Caliper CUDA activity mode was requested, but the libcaliper linked by OpenSn does not "
            f"advertise CUPTI/cuda-activity-report support: {caliper.get('library', caliper.get('reason'))}"
        )
    for ranks in args.ranks:
        pe = rank_pe(args, ranks, physical_cores)
        for mode in modes:
            output = study.working / "artifacts" / f"caliper-{mode}-np{ranks}.txt"
            if mode == "runtime":
                config = (
                    f'runtime-report(output="{output}",aggregate_across_ranks,calc.inclusive,'
                    "print.metadata,order_by_time,max_column_width=180,region.count,region.stats)"
                )
            elif mode == "mpi":
                config = f'mpi-report(output="{output}")'
            else:
                config = (
                    f'cuda-activity-report(output="{output}",show_kernels=true,'
                    "aggregate_across_ranks=true),cuda.memcpy"
                )
            target = opensn_command(args)
            target.insert(2, f"--caliper={config}")
            result = study.execute(
                kind=f"caliper-{mode}",
                argv=[*mpi_prefix(args, ranks, pe), *target],
                cwd=args.input_resolved.parent,
                environment=run_environment(args, args.policy, pe),
                timeout=args.timeout,
                log_name=f"caliper-{mode}-np{ranks}.log",
                metadata={"mode": mode, "ranks": ranks, "pe_per_rank": pe},
            )
            if study.dry_run:
                continue
            expected = signatures.require_rank(ranks) if signatures is not None else None
            validate_result(study, result, args, ranks, expected)
            if not output.is_file() or output.stat().st_size == 0:
                raise StudyError(f"Caliper did not create a non-empty report: {output}")


def build_nsys_profile_command(
    *,
    nsys: str,
    output: pathlib.Path,
    trace: str,
    mpi_impl: str,
    gpu_metrics: bool,
    gpu_metrics_collector: bool,
    gpu_device: str,
    gpu_metrics_set: str,
    gpu_metrics_frequency: int,
    target: Sequence[str],
) -> list[str]:
    command = [
        nsys,
        "profile",
        "--sample=none",
        "--cpuctxsw=none",
        f"--trace={trace}",
        f"--mpi-impl={mpi_impl}",
        "--cuda-event-trace=false",
        "--cuda-memory-usage=false",
        "--stats=false",
        "--force-overwrite=true",
        f"--output={output}",
    ]
    if gpu_metrics and gpu_metrics_collector:
        command.extend(
            [
                f"--gpu-metrics-devices={gpu_device}",
                f"--gpu-metrics-set={gpu_metrics_set}",
                f"--gpu-metrics-frequency={gpu_metrics_frequency}",
            ]
        )
    else:
        command.append("--gpu-metrics-devices=none")
    command.extend(target)
    return command


def nsys_wrapper_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank-mode", choices=("rank0", "all"), required=True)
    parser.add_argument("--nsys", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--mpi-impl", choices=("openmpi", "mpich"), required=True)
    parser.add_argument("--gpu-metrics", action="store_true")
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--gpu-metrics-set", default="gb20x")
    parser.add_argument("--gpu-metrics-frequency", type=positive_int, default=1000)
    options, target = parser.parse_known_args(argv)
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("target command is required after --")

    global_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
    local_rank = int(
        os.environ.get(
            "OMPI_COMM_WORLD_LOCAL_RANK",
            os.environ.get("MPI_LOCALRANKID", os.environ.get("SLURM_LOCALID", str(global_rank))),
        )
    )
    if options.rank_mode == "rank0" and global_rank != 0:
        os.execvpe(target[0], target, os.environ.copy())

    output = pathlib.Path(options.output_directory) / f"{options.output_prefix}-rank{global_rank}"
    command = build_nsys_profile_command(
        nsys=options.nsys,
        output=output,
        trace=options.trace,
        mpi_impl=options.mpi_impl,
        gpu_metrics=options.gpu_metrics,
        gpu_metrics_collector=local_rank == 0,
        gpu_device=options.gpu_device,
        gpu_metrics_set=options.gpu_metrics_set,
        gpu_metrics_frequency=options.gpu_metrics_frequency,
        target=target,
    )
    environment = os.environ.copy()
    if options.enable_nvtx:
        environment["CALI_SERVICES_ENABLE"] = "nvtx"
    if options.rank_mode == "rank0":
        environment["NSYS_MPI_STORE_TEAMS_PER_RANK"] = "1"
    os.execvpe(options.nsys, command, environment)
    return 127


def run_nsys(args: argparse.Namespace, study: Study) -> None:
    signatures = require_profile_reference(args, resolve_signature_set(args), args.ranks)
    physical_cores = study.manifest["provenance"]["host"]["physical_cores_in_affinity"]
    caliper = study.manifest["provenance"]["caliper"]
    nvtx_available = bool(caliper.get("nvtx", False))
    if args.nvtx is True and not nvtx_available:
        raise StudyError(
            "--nvtx was forced, but the libcaliper linked by OpenSn does not advertise the NVTX service: "
            f"{caliper.get('library', caliper.get('reason'))}"
        )
    use_nvtx = nvtx_available if args.nvtx is None else args.nvtx
    trace = "cuda,mpi,nvtx" if use_nvtx else "cuda,mpi"
    study.manifest["nsys_capabilities"] = {
        "nvtx_requested": args.nvtx,
        "nvtx_detected_in_linked_caliper": nvtx_available,
        "nvtx_enabled": use_nvtx,
        "linked_caliper_prefix": caliper.get("prefix"),
    }
    if args.nvtx is None and not nvtx_available:
        study.manifest.setdefault("degraded_capabilities", []).append(
            "Nsight Systems will trace CUDA+MPI without NVTX because linked Caliper lacks NVTX support."
        )
    if use_nvtx:
        study.manifest["nvtx_requirement"] = (
            "The rank-selecting wrapper enables CALI_SERVICES_ENABLE=nvtx only in profiled ranks, "
            "based on the feature probe of OpenSn's linked Caliper."
        )
    for ranks in args.ranks:
        pe = rank_pe(args, ranks, physical_cores)
        prefix = f"nsys-{args.rank_mode}-np{ranks}"
        wrapper = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "_nsys-wrapper",
            "--rank-mode",
            args.rank_mode,
            "--nsys",
            args.nsys_resolved,
            "--output-directory",
            str(study.working / "artifacts"),
            "--output-prefix",
            prefix,
            "--trace",
            trace,
            "--mpi-impl",
            args.mpi_impl,
            "--gpu-device",
            args.gpu,
            "--gpu-metrics-set",
            args.gpu_metrics_set,
            "--gpu-metrics-frequency",
            str(args.gpu_metrics_frequency),
        ]
        if args.gpu_metrics:
            wrapper.append("--gpu-metrics")
        if use_nvtx:
            wrapper.append("--enable-nvtx")
        wrapper.extend(["--", *opensn_command(args)])
        environment = run_environment(args, args.policy, pe)
        environment["NSYS_SYSTEM_ID"] = f"cbcd-local-{os.getpid()}-np{ranks}"
        result = study.execute(
            kind=f"nsys-{args.rank_mode}",
            argv=[*mpi_prefix(args, ranks, pe), *wrapper],
            cwd=args.input_resolved.parent,
            environment=environment,
            timeout=args.timeout,
            log_name=f"{prefix}.log",
            metadata={
                "rank_mode": args.rank_mode,
                "ranks": ranks,
                "pe_per_rank": pe,
                "gpu_metrics": args.gpu_metrics,
                "single_gpu_metrics_collector": args.gpu_metrics,
                "trace": trace,
            },
        )
        if study.dry_run:
            continue
        expected = signatures.require_rank(ranks) if signatures is not None else None
        validate_result(study, result, args, ranks, expected)
        reports = sorted((study.working / "artifacts").glob(f"{prefix}-rank*.nsys-rep"))
        expected_reports = 1 if args.rank_mode == "rank0" else ranks
        if len(reports) != expected_reports:
            raise StudyError(f"expected {expected_reports} nsys reports for np={ranks}, found {len(reports)}")
        empty_reports = [str(report) for report in reports if report.stat().st_size == 0]
        if empty_reports:
            raise StudyError(f"Nsight Systems created empty reports: {empty_reports}")
        if args.no_stats:
            continue
        base_reports = tuple(
            report_name
            for report_name in NSYS_REPORTS
            if ranks > 1 or report_name != "mpi_msg_size_sum"
        )
        stats_reports = (*base_reports, *NSYS_NVTX_REPORTS) if use_nvtx else base_reports
        for report in reports:
            for report_index, report_name in enumerate(stats_reports):
                stats_argv = [
                    args.nsys_resolved,
                    "stats",
                    "--timeunit",
                    "ns",
                    "--report",
                    report_name,
                    "--format",
                    "csv",
                    "--output",
                    "-",
                    str(report),
                ]
                if report_index == 0:
                    stats_argv.insert(2, "--force-export=true")
                artifact_name = f"{report.stem}-{report_name}.csv"
                stats = study.execute(
                    kind="nsys-stats",
                    argv=stats_argv,
                    cwd=args.input_resolved.parent,
                    environment={},
                    timeout=args.stats_timeout,
                    log_name=f"{report.stem}-{report_name}-stats.log",
                    stdout_name=artifact_name,
                    metadata={"source_report": study.relative(report), "report": report_name},
                )
                require_success(stats)
                validation = validate_profiler_csv(
                    study.working / "artifacts" / artifact_name,
                    f"Nsight Systems {report_name}",
                )
                stats.metadata["artifact_validation"] = validation
                study.write_manifest()


def run_ncu(args: argparse.Namespace, study: Study) -> None:
    signatures = require_profile_reference(args, resolve_signature_set(args), (1,))
    physical_cores = study.manifest["provenance"]["host"]["physical_cores_in_affinity"]
    pe = rank_pe(args, 1, physical_cores)
    output = study.working / "artifacts" / "cbcd-1rank"
    profiler = [
        args.ncu_resolved,
        "--target-processes",
        "application-only",
        "--set",
        args.set,
        "--replay-mode",
        "kernel",
        "--kernel-name",
        args.kernel_name,
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--force-overwrite",
        "--export",
        str(output),
    ]
    result = study.execute(
        kind="ncu",
        argv=[*mpi_prefix(args, 1, pe), *profiler, *opensn_command(args)],
        cwd=args.input_resolved.parent,
        environment=run_environment(args, args.policy, pe),
        timeout=args.timeout,
        log_name="ncu-1rank.log",
        metadata={
            "ranks": 1,
            "pe_per_rank": pe,
            "launch_skip": args.launch_skip,
            "launch_count": args.launch_count,
            "end_to_end_timing_valid": False,
        },
    )
    if study.dry_run:
        return
    expected = signatures.require_rank(1) if signatures is not None else None
    validate_result(study, result, args, 1, expected)
    report = output.with_suffix(".ncu-rep")
    if not report.is_file() or report.stat().st_size == 0:
        raise StudyError(f"Nsight Compute did not create {report}")
    imported = study.execute(
        kind="ncu-import",
        argv=[args.ncu_resolved, "--import", str(report), "--page", "details", "--csv"],
        cwd=args.input_resolved.parent,
        environment={},
        timeout=args.import_timeout,
        log_name="ncu-import.log",
        stdout_name="cbcd-1rank-details.csv",
        metadata={"source_report": study.relative(report)},
    )
    require_success(imported)
    imported.metadata["artifact_validation"] = validate_profiler_csv(
        study.working / "artifacts" / "cbcd-1rank-details.csv",
        "Nsight Compute details import",
    )
    study.write_manifest()


def run_sanitizer(args: argparse.Namespace, study: Study) -> None:
    signatures = require_profile_reference(args, resolve_signature_set(args), (1,))
    physical_cores = study.manifest["provenance"]["host"]["physical_cores_in_affinity"]
    pe = rank_pe(args, 1, physical_cores)
    log_pattern = study.working / "artifacts" / f"compute-sanitizer-{args.tool}-%p.log"
    sanitizer = [
        args.compute_sanitizer_resolved,
        "--tool",
        args.tool,
        "--target-processes",
        "application-only",
        "--kernel-name",
        f"kernel_substring:{args.kernel_substring}",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--error-exitcode",
        str(args.error_exit_code),
        "--log-file",
        str(log_pattern),
    ]
    if args.tool == "memcheck":
        sanitizer.extend(["--leak-check", "no"])
    result = study.execute(
        kind=f"compute-sanitizer-{args.tool}",
        argv=[*mpi_prefix(args, 1, pe), *sanitizer, *opensn_command(args)],
        cwd=args.input_resolved.parent,
        environment=run_environment(args, args.policy, pe),
        timeout=args.timeout,
        log_name=f"compute-sanitizer-{args.tool}.log",
        metadata={
            "ranks": 1,
            "pe_per_rank": pe,
            "launch_skip": args.launch_skip,
            "launch_count": args.launch_count,
            "performance_timing_valid": False,
        },
    )
    if study.dry_run:
        return
    expected = signatures.require_rank(1) if signatures is not None else None
    validate_result(study, result, args, 1, expected)
    reports = list((study.working / "artifacts").glob(f"compute-sanitizer-{args.tool}-*.log"))
    if not reports:
        raise StudyError("Compute Sanitizer did not create its diagnostic log")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--workload-asset",
        type=pathlib.Path,
        action="append",
        default=[],
        help="repeat for external physics/data files; the default input automatically records its XS file",
    )
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--label", default="local")
    parser.add_argument("--dry-run", action="store_true", help="record and print commands without launching them")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--gpu", default="0", help="single CUDA device identifier; every rank shares it")
    parser.add_argument("--policy", choices=("hardware", "resource-aware"), default="resource-aware")
    parser.add_argument("--workers", type=positive_int, help="positive OPENSN_CBCD_NUM_WORKERS override")
    parser.add_argument("--cells-per-axis", type=positive_int, default=18)
    parser.add_argument("--max-iterations", type=positive_int, default=100)
    parser.add_argument("--timeout", type=positive_float, default=120.0)
    parser.add_argument("--bind-to", choices=("core", "hwthread", "none"), default="core")
    parser.add_argument("--map-by", choices=("slot", "core", "node", "none"), default="slot")
    parser.add_argument("--pe-per-rank", type=positive_int)
    parser.add_argument("--pe-map", type=parse_pe_map, help="rank-specific PE allocation, e.g. 1=8,2=4,4=2")
    parser.add_argument("--allow-cpu-oversubscribe", action="store_true")
    parser.add_argument("--report-bindings", action="store_true")
    parser.add_argument("--reference", type=pathlib.Path, help="signature.json from a validated benchmark")
    parser.add_argument("--max-atol", type=float, default=1.0e-10)
    parser.add_argument("--max-rtol", type=float, default=1.0e-6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run local CBCD benchmarks and profiles. All MPI ranks share one GPU; results are contention "
            "diagnostics, not accelerator strong scaling."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="sequential uninstrumented warmups and trials")
    add_common_arguments(benchmark)
    benchmark.add_argument("--ranks", type=parse_csv_ints, default=(1, 2, 4))
    benchmark.add_argument("--policies", type=parse_policies, default=("resource-aware",))
    benchmark.add_argument("--warmups", type=nonnegative_int, default=1)
    benchmark.add_argument("--trials", type=positive_int, default=5)
    benchmark.add_argument(
        "--schedule-seed",
        type=nonnegative_int,
        help="seed for randomized rank blocks and balanced A/B policy order (generated and recorded by default)",
    )
    benchmark.add_argument("--baseline-ranks", type=positive_int, default=1)
    benchmark.set_defaults(handler=run_benchmark)

    caliper = subparsers.add_parser("caliper", help="Caliper runtime-region and/or PMPI report")
    add_common_arguments(caliper)
    caliper.add_argument("--ranks", type=parse_csv_ints, default=(1, 2, 4))
    caliper.add_argument(
        "--mode",
        choices=("auto", "runtime", "mpi", "cuda", "both", "all"),
        default="auto",
        help="auto adds CUDA activity when CUPTI is detected in OpenSn's linked Caliper",
    )
    caliper.set_defaults(handler=run_caliper)

    nsys = subparsers.add_parser("nsys", help="Nsight Systems rank-filtered timeline")
    add_common_arguments(nsys)
    nsys.add_argument("--ranks", type=parse_csv_ints, default=(1, 2, 4))
    nsys.add_argument("--rank-mode", choices=("rank0", "all"), default="rank0")
    nsys.add_argument("--nsys", default="nsys")
    nsys.add_argument("--mpi-impl", choices=("openmpi", "mpich"), default="openmpi")
    nsys.add_argument("--gpu-metrics", action="store_true")
    nsys.add_argument("--gpu-metrics-set", default="gb20x")
    nsys.add_argument("--gpu-metrics-frequency", type=positive_int, default=1000)
    nvtx = nsys.add_mutually_exclusive_group()
    nvtx.add_argument(
        "--nvtx",
        dest="nvtx",
        action="store_true",
        help="require NVTX from the Caliper library linked by OpenSn",
    )
    nvtx.add_argument("--no-nvtx", dest="nvtx", action="store_false", help="disable automatic NVTX forwarding")
    nsys.set_defaults(nvtx=None)
    nsys.add_argument("--no-stats", action="store_true")
    nsys.add_argument("--stats-timeout", type=positive_float, default=120.0)
    nsys.set_defaults(handler=run_nsys)

    ncu = subparsers.add_parser("ncu", help="one-rank, one-kernel Nsight Compute microprofile")
    add_common_arguments(ncu)
    ncu.add_argument("--ncu", default="ncu")
    ncu.add_argument("--set", default="basic")
    ncu.add_argument("--kernel-name", default="regex:.*SweepKernel.*")
    ncu.add_argument("--launch-skip", type=nonnegative_int, default=32)
    ncu.add_argument("--launch-count", type=positive_int, default=1)
    ncu.add_argument("--import-timeout", type=positive_float, default=120.0)
    ncu.set_defaults(handler=run_ncu)

    sanitizer = subparsers.add_parser("sanitizer", help="bounded one-rank Compute Sanitizer diagnostic")
    add_common_arguments(sanitizer)
    sanitizer.add_argument("--compute-sanitizer", default="compute-sanitizer")
    sanitizer.add_argument("--tool", choices=("memcheck", "racecheck", "synccheck", "initcheck"), default="memcheck")
    sanitizer.add_argument("--kernel-substring", default="SweepKernel")
    sanitizer.add_argument("--launch-skip", type=nonnegative_int, default=32)
    sanitizer.add_argument("--launch-count", type=positive_int, default=4)
    sanitizer.add_argument("--error-exit-code", type=positive_int, default=99)
    sanitizer.set_defaults(handler=run_sanitizer)
    return parser


def prepare_arguments(args: argparse.Namespace) -> None:
    args.input_resolved = args.input.expanduser().resolve()
    if not args.input_resolved.is_file():
        raise StudyError(f"input file does not exist: {args.input_resolved}")
    args.binary_resolved = resolve_program(str(args.binary), args.dry_run)
    args.mpirun_resolved = resolve_program(args.mpirun, args.dry_run)
    raw_assets = list(args.workload_asset)
    if args.input_resolved == DEFAULT_INPUT.resolve():
        raw_assets.extend(DEFAULT_WORKLOAD_ASSETS)
    unique_assets: dict[pathlib.Path, None] = {}
    for asset in raw_assets:
        resolved_asset = asset.expanduser().resolve()
        if not resolved_asset.is_file():
            raise StudyError(f"workload asset does not exist: {resolved_asset}")
        unique_assets[resolved_asset] = None
    args.workload_assets_resolved = tuple(sorted(unique_assets, key=str))
    if args.max_atol < 0.0 or args.max_rtol < 0.0 or not math.isfinite(args.max_atol + args.max_rtol):
        raise StudyError("maximum tolerances must be finite and nonnegative")
    if args.reference:
        args.reference = args.reference.expanduser().resolve()
        if not args.reference.is_file():
            raise StudyError(f"reference signature does not exist: {args.reference}")
    requested_ranks = args.ranks if hasattr(args, "ranks") else (1,)
    if tuple(requested_ranks) != tuple(sorted(requested_ranks)):
        raise StudyError("--ranks must be in strictly increasing order")
    if args.pe_map and args.pe_per_rank:
        raise StudyError("use either --pe-map or --pe-per-rank, not both")
    if args.pe_map and set(args.pe_map) != set(requested_ranks):
        raise StudyError(
            f"--pe-map must contain exactly the requested ranks {list(requested_ranks)}, "
            f"found {sorted(args.pe_map)}"
        )
    if "," in args.gpu:
        raise StudyError("--gpu must identify exactly one CUDA device")
    if args.command == "benchmark":
        if args.baseline_ranks not in args.ranks:
            raise StudyError("--baseline-ranks must be one of --ranks")
        if args.workers is not None and len(args.policies) != 1:
            raise StudyError(
                "--workers fixes the worker count and therefore requires exactly one --policies entry; "
                "it is reported as a fixed-workers condition, not a policy A/B comparison"
            )
        if args.schedule_seed is None:
            args.schedule_seed = random.SystemRandom().randrange(0, 2**63)
    if args.command == "nsys":
        args.nsys_resolved = resolve_program(args.nsys, args.dry_run)
    elif args.command == "ncu":
        args.ncu_resolved = resolve_program(args.ncu, args.dry_run)
    elif args.command == "sanitizer":
        args.compute_sanitizer_resolved = resolve_program(args.compute_sanitizer, args.dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_nsys-wrapper":
        return nsys_wrapper_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    study: Study | None = None
    try:
        prepare_arguments(args)
        study = Study(args.output_root, args.command, args.label, args.dry_run)
        initialize_study(args, study)
        args.handler(args, study)
        final_status = "dry-run" if args.dry_run else "complete"
        result_directory = study.finish(final_status)
        print(result_directory)
        return 0
    except (StudyError, OSError, json.JSONDecodeError) as error:
        if study is not None:
            result_directory = study.finish("failed", str(error))
            print(f"failed study preserved at {result_directory}", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if study is not None:
            result_directory = study.finish("interrupted", "keyboard interrupt")
            print(f"interrupted study preserved at {result_directory}", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
