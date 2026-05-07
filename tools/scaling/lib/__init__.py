#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate GMSH input files for scaling test with unstructured mesh.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

base_dir = Path(__file__).resolve().parent
env = Environment(
    loader=FileSystemLoader(base_dir),
    autoescape=False
)
templates = {
    "slurm": env.get_template("slurm_template.txt"),
    "lc-flux": env.get_template("flux_template.txt"),
    "alcf-pbs": env.get_template("pbs_template.txt")
}
commands = {
    "slurm": "sbatch",
    "lc-flux": "flux batch",
    "alcf-pbs": "qsub"
}
py_template = env.get_template("unstructured.py")


def make_study_tag(sweep_type, processor, study_name=""):
    """Create a filesystem- and scheduler-friendly study tag."""

    tag = f"{sweep_type.lower()}_{processor}"
    if study_name:
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", study_name.strip()).strip("-")
        if suffix:
            tag = f"{tag}_{suffix}"
    return tag


def run_gmsh(gmsh_binary, input_geo, divisor, output_msh):
    """Run Gmsh on a .geo file to generate a mesh."""

    if output_msh.exists():
        print(f"Reusing mesh {output_msh}")
        return

    output_msh.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        gmsh_binary,
        "-3",
        "-v", "0",
        "-setnumber", "divisor", str(divisor),
        "-o", str(output_msh),
        str(input_geo),
    ]
    print("Generating mesh with command: {}".format(" ".join(cmd)))
    subprocess.run(cmd, check=True)


def make_mesh_tag(prefix, n_tasks, items):
    """Create a compact mesh-cache tag for a ranks-per-node mesh configuration."""

    items = list(items)
    config = ",".join(f"{node}:{divisor}" for node, divisor in items)
    digest = hashlib.sha1(config.encode("utf-8")).hexdigest()[:10]
    if prefix == "strong" and len(items) == 1:
        return f"{prefix}_rpn{n_tasks}_div{items[0][1]}"
    nodes = "-".join(str(node) for node, _ in items)
    if len(nodes) > 48:
        nodes = f"{items[0][0]}-to-{items[-1][0]}-{len(items)}nodes"
    return f"{prefix}_rpn{n_tasks}_nodes{nodes}_{digest}"


def make_mesh_dir(prefix, n_tasks, items):
    """Return the shared mesh-cache directory for a scaling mesh configuration."""

    return base_dir.parent / "output" / "meshes" / make_mesh_tag(prefix, n_tasks, items)


def make_launch_script_keys(
    base_name,
    outdir,
    opensn_binary,
    processor,
    sweep_type,
    n_cores,
    partition,
    ngpus,
    environment,
    gpu_config
):
    """Create a dictionary of keys for job file generation."""
    keys = {
        "base_name": base_name,
        "outdir": str(outdir),
        "opensn_binary": opensn_binary,
        "environment": environment,
        "partition": partition,
        "processor": processor,
        "sweep_type": sweep_type
    }
    if processor == "gpu":
        keys["use_gpus"] = True
        keys["gpu_options"] = gpu_config
        keys["n_tasks"] = ngpus
        keys["n_cores"] = n_cores // ngpus
    else:
        keys["use_gpus"] = False
        keys["gpu_options"] = ""
        keys["n_tasks"] = n_cores
        keys["n_cores"] = 1
    return keys


def task_count_per_node(processor, n_cores, ngpus):
    """Return the MPI task count per node for the selected processor target."""

    return ngpus if processor == "gpu" else n_cores


def create_job_script(engine, keys, n_nodes, input_script, prefix, output_dir):
    """Create a job file to run OpenSn."""

    content = templates[engine].render(**keys, n_nodes=n_nodes, input_script=input_script)
    fname = output_dir / f"{prefix}_{n_nodes}.sh"
    print(f"Generating job file {fname}")
    with open(fname, "w") as job_file:
        job_file.write(content)
    return fname


def generate_strong_scaling(nodes, **kwargs):
    """Generate files for strong scaling study."""

    # copy necessary files to output directory
    processor = kwargs["processor"]
    sweep_type = kwargs["sweep_type"]
    study_tag = make_study_tag(sweep_type, processor, kwargs.get("study_name", ""))
    engine = kwargs["engine"]
    out_dir = base_dir.parent / "output" / f"strong_{study_tag}"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(
        base_dir / "xs_168g.xs",
        out_dir / "xs_168g.xs"
    )

    n_tasks = task_count_per_node(processor, kwargs["ncores"], kwargs["ngpus"])
    mesh_dir = make_mesh_dir("strong", n_tasks, [(0, kwargs["divisor"])])
    mesh_file = mesh_dir / "strong_scaling.msh"

    # generate or reuse the shared mesh
    run_gmsh(
        kwargs["gmsh_binary"],
        kwargs["geo_filename"],
        kwargs["divisor"],
        output_msh=mesh_file
    )

    # create job files
    keys = make_launch_script_keys(
        base_name=f"strong_{study_tag}",
        outdir=out_dir,
        opensn_binary=kwargs["opensn_binary"],
        processor=processor,
        sweep_type=sweep_type,
        n_cores=kwargs["ncores"],
        partition=kwargs["partition"],
        ngpus=kwargs["ngpus"],
        environment=kwargs["environment"],
        gpu_config=kwargs["gpu_config"]
    )
    script_name = out_dir / "strong_scaling.py"
    scripts = []
    for n_nodes in nodes:
        script = create_job_script(
            engine=engine,
            keys=keys,
            n_nodes=n_nodes,
            input_script=script_name,
            prefix="strong",
            output_dir=out_dir
        )
        scripts.append(script)
    with open(out_dir / "submit_jobs.sh", "w") as launch_file:
        launch_file.write("#!/bin/bash\n\n")
        for script in scripts:
            launch_file.write(f"{commands[engine]} {script.name}\n")

    # generate the Python script
    print("Generating strong scaling Python script.")
    script_content = py_template.render(**keys, mesh_file=mesh_file)
    with open(script_name, "w") as script_file:
        script_file.write(script_content)


def generate_weak_scaling(nodes, divisors, **kwargs):
    """Generate files for weak scaling study."""

    if len(nodes) != len(divisors):
        raise ValueError("Weak scaling requires one mesh divisor per node count.")

    # copy necessary files to output directory
    processor = kwargs["processor"]
    sweep_type = kwargs["sweep_type"]
    study_tag = make_study_tag(sweep_type, processor, kwargs.get("study_name", ""))
    engine = kwargs["engine"]
    out_dir = base_dir.parent / "output" / f"weak_{study_tag}"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(
        base_dir / "xs_168g.xs",
        out_dir / "xs_168g.xs"
    )

    n_tasks = task_count_per_node(processor, kwargs["ncores"], kwargs["ngpus"])
    mesh_dir = make_mesh_dir("weak", n_tasks, zip(nodes, divisors))

    # generate or reuse the shared meshes
    mesh_files = {}
    for n_nodes, divisor in zip(nodes, divisors):
        mesh_file = mesh_dir / f"weak_scaling_{n_nodes}.msh"
        run_gmsh(
            kwargs["gmsh_binary"],
            kwargs["geo_filename"],
            divisor,
            output_msh=mesh_file
        )
        mesh_files[n_nodes] = mesh_file

    # create job files
    keys = make_launch_script_keys(
        base_name=f"weak_{study_tag}",
        outdir=out_dir,
        opensn_binary=kwargs["opensn_binary"],
        processor=processor,
        sweep_type=sweep_type,
        n_cores=kwargs["ncores"],
        partition=kwargs["partition"],
        ngpus=kwargs["ngpus"],
        environment=kwargs["environment"],
        gpu_config=kwargs["gpu_config"]
    )
    scripts = []
    for n_nodes in nodes:
        script = create_job_script(
            engine=engine,
            keys=keys,
            n_nodes=n_nodes,
            input_script=out_dir / f"weak_scaling_{n_nodes}.py",
            prefix="weak",
            output_dir=out_dir
        )
        scripts.append(script)
    with open(out_dir / "submit_jobs.sh", "w") as launch_file:
        launch_file.write("#!/bin/bash\n\n")
        for script in scripts:
            launch_file.write(f"{commands[engine]} {script.name}\n")

    # generate the Python script
    print("Generating weak scaling Python script.")
    for n_nodes in nodes:
        script_content = py_template.render(**keys, mesh_file=mesh_files[n_nodes])
        script_name = out_dir / f"weak_scaling_{n_nodes}.py"
        with open(script_name, "w") as script_file:
            script_file.write(script_content)
