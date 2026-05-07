#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from pathlib import Path
from lib import generate_strong_scaling, generate_weak_scaling

base_dir = Path(__file__).resolve().parent

environment = ""
"""Command to activate environement script."""

geo_filename = base_dir / "lib/cube.geo"
"""Name of the Gmsh geometry file."""

gmsh_binary = "gmsh"
"""Path to the Gmsh binary. Default is 'gmsh' assuming it's in the system PATH."""

opensn_binary = base_dir.parents[1] / "build/python/opensn"
"""Path to the OpenSn binary. Default is '../../../build/python/opensn'."""

strong_divisor = 39
"""Divisor for Gmsh to control mesh resolution for strong scaling study (default: 39)."""

weak_divisor_reference_tasks = 64
"""Tasks per node for the configured weak-scaling reference divisors."""

# 64 tasks per node with 256 cells per task
weak_divisors = [15, 19, 24, 31, 39, 49, 62, 78, 98, 124]
# 96 tasks per node with 256 cell per task
# weak_divisors = [17, 22, 28, 35, 44, 56, 71, 89, 113, 137]
# 64 tasks per node with 2048 cells per task
# weak_divisors = [30, 39, 48, 62, 78, 98, 124]

strong_nodes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
"""List of node counts to generate strong-scaling job files for."""

weak_nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
"""List of node counts to generate weak-scaling job files for."""

ncores = 96
"""
Number of CPU cores available per node.

If this value is not evenly divisible by the number of GPUs, round it down.
"""

partition = ""
"""Name of the partition/queue to submit jobs to."""

ngpus = 0
"""Number of GPUs per node (default: 0 for CPU-only)."""

gpu_config = {
    "slurm": "#SBATCH --gpus-per-task=1",
    "lc-flux": "#flux: --setattr=gpumode=SPX",
    "alcf-pbs": ""
}
"""GPU configurations."""

user_defined_config = ""
"""
Custom user-defined argument for launching with GPU.

If this variable is defined, it will overwrite default options in ``gpu_config``.
"""

extra_data = {
    "name": "label_name",
    "description": "Description of the cluster and scaling study."
}


def parse_int_list(value):
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("Node counts must be positive integers.")
    return values


def weak_divisors_for_tasks(nodes, tasks_per_node):
    divisor_by_node = dict(zip(weak_nodes, weak_divisors))
    missing_nodes = [node for node in nodes if node not in divisor_by_node]
    if missing_nodes:
        raise ValueError(
            "No weak-scaling mesh divisors are configured for nodes: "
            + ",".join(map(str, missing_nodes))
        )

    scale = (tasks_per_node / weak_divisor_reference_tasks) ** (1.0 / 3.0)
    return [max(1, round(divisor_by_node[node] * scale)) for node in nodes]


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Generate files for strong or weak scaling studies with OpenSn.")
    parser.add_argument(
        "--type",
        type=str,
        choices=["strong", "weak"],
        required=True,
        help="Type of scaling test to generate files for."
    )
    parser.add_argument(
        "--sweep-type",
        type=str.upper,
        choices=["AAH", "CBC"],
        default="AAH",
        help="Sweep algorithm to use in the generated OpenSn input. Defaults to AAH."
    )
    processor_group = parser.add_mutually_exclusive_group()
    processor_group.add_argument(
        "--use-gpus",
        action="store_true",
        help="Enable GPU sweep execution in the generated OpenSn input and launch scripts."
    )
    processor_group.add_argument(
        "--processor",
        type=str,
        choices=["cpu", "gpu"],
        default=None,
        help=(
            "Processor target for the generated study files. Defaults to CPU. "
            "This is retained for compatibility; prefer --use-gpus for GPU studies."
        )
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["slurm", "lc-flux", "alcf-pbs"],
        default="slurm",
        help="Job submitting system. Defaults to slurm."
    )
    parser.add_argument(
        "--opensn-binary",
        type=Path,
        default=opensn_binary,
        help=f"Path to the OpenSn executable. Defaults to {opensn_binary}."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="",
        help=(
            "Optional study identifier appended to output directories and job names, "
            "for example a branch or binary label."
        )
    )
    parser.add_argument(
        "--nodes",
        type=parse_int_list,
        default=None,
        help=(
            "Comma-separated node counts for this study. Defaults to the script's "
            "strong or weak node list."
        )
    )
    parser.add_argument(
        "--cores-per-node",
        type=int,
        default=ncores,
        help=f"CPU cores per node. Defaults to {ncores}."
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=ngpus,
        help=f"GPUs per node for GPU studies. Defaults to {ngpus}."
    )
    parser.add_argument(
        "--strong-divisor",
        type=int,
        default=strong_divisor,
        help=f"Gmsh divisor for strong scaling. Defaults to {strong_divisor}."
    )
    args = parser.parse_args()

    processor = args.processor or ("gpu" if args.use_gpus else "cpu")

    if processor == "gpu" and args.gpus_per_node == 0:
        raise ValueError("Please specify the number of GPUs per node.")
    if processor == "gpu" and args.cores_per_node < args.gpus_per_node:
        raise ValueError("GPU studies require at least one CPU core per GPU rank.")

    if user_defined_config:
        gpu_option = user_defined_config
    else:
        gpu_option = gpu_config[args.engine]

    nodes = args.nodes or (strong_nodes if args.type == "strong" else weak_nodes)
    tasks_per_node = args.gpus_per_node if processor == "gpu" else args.cores_per_node

    inputs = {
        "opensn_binary": args.opensn_binary,
        "gmsh_binary": gmsh_binary,
        "geo_filename": geo_filename,
        "ncores": args.cores_per_node,
        "partition": partition,
        "processor": processor,
        "sweep_type": args.sweep_type,
        "ngpus": args.gpus_per_node,
        "engine": args.engine,
        "environment": environment,
        "gpu_config": gpu_option,
        "study_name": args.study_name
    }
    if args.type == "strong":
        generate_strong_scaling(nodes, divisor=args.strong_divisor, **inputs)
    else:
        generate_weak_scaling(nodes, weak_divisors_for_tasks(nodes, tasks_per_node), **inputs)
