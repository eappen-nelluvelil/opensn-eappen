#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Submit independent-process baseline cases to Flux without duplicate jobs."""

import argparse
from pathlib import Path
import subprocess

import cbcd_study as study
from study_build import digest, write_json


def previous_submissions(root, kind, nodes):
    return sorted((root / "allocations").glob(f"{kind}-{nodes}n-*/submission.json"))


def submit(args):
    root = args.root.resolve(strict=True)
    config = study.read_json(root / "manifest.json")
    batch = study.read_json(root / "batch.json")
    if config["modes"] != ["baseline"]:
        raise ValueError("Batch submission requires a baseline-only campaign")
    kinds = [args.kind] if args.kind else ["strong", "weak"]
    limit = batch["max_nodes"]
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("max_nodes must be the site's approved positive node limit")
    nodes_list = args.nodes or [n for n in config["nodes"] if n <= limit]
    if args.nodes is None:
        excluded = [n for n in config["nodes"] if n > limit]
        if excluded:
            print(f"Not submitting nodes above the approved limit {limit}: {excluded}", flush=True)
    if not nodes_list or len(set(nodes_list)) != len(nodes_list) or any(
            n not in config["nodes"] or n > limit for n in nodes_list):
        raise ValueError("Requested nodes are absent from the study or exceed the approved limit")
    if not isinstance(batch["submit"], list) or batch["submit"][:2] != ["flux", "batch"]:
        raise ValueError("submit must be a flux batch argument list")
    study.verify_inputs(config)
    study.verify_fingerprint(study.read_json(root / "builds/native/build.json"))
    for path, expected in batch["assets"].items():
        if digest(path) != expected:
            raise ValueError(f"Batch launcher/environment changed: {path}")
    with study.locked(root / ".submit.lock"):
        for kind in kinds:
            for nodes in nodes_list:
                if not study.pending(root, kind, nodes, config):
                    print(f"{kind} {nodes}: complete, not submitted", flush=True)
                    continue
                records = previous_submissions(root, kind, nodes)
                if records:
                    for record in records:
                        job_file = record.parent / "job-id.txt"
                        if not job_file.exists():
                            raise ValueError(f"Unresolved submission. Inspect {record.parent}")
                    if not args.retry:
                        print(f"{kind} {nodes}: already submitted. "
                              "Use --retry after inspecting logs.",
                              flush=True)
                        continue
                    for record in records:
                        job_id = (record.parent / "job-id.txt").read_text().strip()
                        state = subprocess.check_output(
                            ["flux", "jobs", "--no-header", "--format={state}", job_id],
                            text=True).strip()
                        if state != "INACTIVE":
                            raise ValueError(f"Job {job_id} is not confirmed inactive: {state}")
                allocation = root / "allocations" / f"{kind}-{nodes}n-{study.unique_id()}"
                allocation.mkdir(parents=True)
                replacements = dict(root=str(root), kind=kind, nodes=nodes,
                                    ranks=nodes * config["ranks_per_node"],
                                    allocation=str(allocation))
                command = [part.format(**replacements) for part in batch["submit"]]
                write_json(allocation / "submission.json", dict(
                    command=command, manifest_sha256=digest(root / "manifest.json"),
                    batch_sha256=digest(root / "batch.json")))
                with (allocation / "submit.stdout").open("x") as out, \
                        (allocation / "submit.stderr").open("x") as err:
                    subprocess.run(command, check=True, stdout=out, stderr=err)
                job_id = (allocation / "submit.stdout").read_text().strip()
                if not job_id or len(job_id.split()) != 1:
                    raise ValueError(f"Cannot determine submitted job ID. Inspect {allocation}")
                (allocation / "job-id.txt").write_text(job_id + "\n")
                print(f"{kind} {nodes}: submitted {job_id}\n  {allocation}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("strong", "weak"))
    parser.add_argument("--nodes", type=int, nargs="+")
    parser.add_argument("--retry", action="store_true")
    submit(parser.parse_args())


if __name__ == "__main__":
    main()
