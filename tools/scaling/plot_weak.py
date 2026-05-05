#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob
import warnings
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from generate_scaling_study import extra_data

try:
    import yaml
except ImportError:
    yaml = None


def study_label(sweep_type, processor):
    """Return a history label that includes algorithm and execution target."""
    return f"{extra_data['name']}_{sweep_type.lower()}_{processor}_weak_scaling"


def extract_data(filename):
    """Extract n, average sweep time, and number of unknowns from a file."""

    match = re.search(r'_(\d+)\.out$', filename)
    if not match:
        return None
    n = int(match.group(1))

    avg_time = None

    avg_time_re = re.compile(r'avg_sweep_time\s*=\s*([0-9.eE+-]+)\s*s')

    with open(filename, 'r') as f:
        for line in f:
            avg_match = avg_time_re.search(line)
            if avg_match:
                avg_time = float(avg_match.group(1))

    if avg_time is None:
        return None

    metric = avg_time
    return n, metric


def plot_data(data, output_file, with_history, sweep_type, processor):
    """Plot the data and save to a file."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    n_nodes = [d[0] for d in data]
    sweep_time = [d[1] for d in data]
    efficiency = [sweep_time[0] * 100.0 / t for t in sweep_time]

    history = {}
    history_file = Path(__file__).resolve().parent / "history.yaml"
    if with_history and yaml is None:
        warnings.warn("PyYAML is not installed. Plotting without history.")
    elif with_history and history_file.exists():
        with open(history_file, "r") as f:
            history_dict = yaml.safe_load(f)
        history_label = study_label(sweep_type, processor)
        if history_dict is not None and history_label in history_dict:
            history_data = history_dict[history_label]
            history["nodes"] = history_data["nodes"]
            history["efficiency"] = history_data["efficiency"]

    fig, ax = plt.subplots()
    ax.plot(n_nodes, efficiency, marker='o', color='xkcd:cerulean', label='efficiency')
    xticks = n_nodes.copy()
    if history:
        ax.plot(history["nodes"], history["efficiency"], marker='o',
                color='xkcd:coral', label='history')
        xticks = sorted(set(n_nodes) | set(history["nodes"]))
    elif with_history:
        warnings.warn(
            "History file not found or history label not in file. "
            "Plotting without history."
        )
    ax.set_xlabel("Number of nodes")
    ax.set_xscale('log')
    ax.set_xticks(xticks, xticks)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0.0, top=max(efficiency) + 10.0)
    ax.set_ylabel("Efficiency (%)")
    ax.set_title(f"Node-to-node weak scaling ({sweep_type}, {processor})")
    ax.grid(True, which='both')
    ax.legend()
    fig.savefig(output_file)
    plt.show()


def export_data(data, output_file, sweep_type, processor):
    """Export data to a YAML file."""

    if yaml is None:
        raise ImportError("Saving history requires PyYAML.")

    sweep_time = [d[1] for d in data]
    efficiency = [sweep_time[0] * 100.0 / t for t in sweep_time]

    label = study_label(sweep_type, processor)
    export_dict = None
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            export_dict = yaml.safe_load(f)
    if export_dict is None:
        export_dict = {}
    export_dict[label] = {
        "description": extra_data['description'],
        "time": datetime.now().isoformat(),
        "nodes": [d[0] for d in data],
        "efficiency": efficiency
    }
    with open(output_file, "w") as f:
        yaml.dump(export_dict, f)


if __name__ == "__main__":

    # read command-line arguments
    parser = ArgumentParser(description="Plot scaling data from output files.")
    parser.add_argument(
        "--output",
        type=str,
        default="weak_scaling_plot.pdf",
        help="Filename for the output plot (default: weak_scaling_plot.pdf)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=(
            "Folder to find weak scaling results. Defaults to "
            "output/weak_{sweep_type}_{cpu/gpu}."
        )
    )
    parser.add_argument(
        "--sweep-type",
        type=str.upper,
        choices=["AAH", "CBC"],
        default="AAH",
        help="Sweep algorithm associated with the results. Defaults to AAH."
    )
    processor_group = parser.add_mutually_exclusive_group()
    processor_group.add_argument(
        "--use-gpus",
        action="store_true",
        help="Select GPU result defaults and history label."
    )
    processor_group.add_argument(
        "--processor",
        type=str,
        choices=["cpu", "gpu"],
        default=None,
        help="Processor target associated with the results. Defaults to CPU."
    )
    parser.add_argument(
        "--history",
        type=str,
        choices=["none", "comp", "save"],
        default="none",
        help=(
            "History mode for the plot: "
            "none (only plot current data), "
            "comp (compare with history in the same plot without saving), "
            "or save (plot and overwrite current history value). "
            "(default: none)"
        ),
    )
    args = parser.parse_args()

    processor = args.processor or ("gpu" if args.use_gpus else "cpu")
    input_dir_arg = args.dir or f"output/weak_{args.sweep_type.lower()}_{processor}"

    # get files matching the prefix in the input directory
    input_dir = Path(__file__).resolve().parent / input_dir_arg
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    files = glob.glob(f"{input_dir}/weak_*.out")
    if not files:
        raise FileNotFoundError(f"No files found matching weak_*.out in {input_dir}")

    # extract sweep time
    data = []
    for f in files:
        result = extract_data(f)
        if result:
            data.append(result)
    if not data:
        raise ValueError("No valid data found.")
    data.sort(key=lambda x: x[0])

    # plot
    with_history = (args.history == "comp")
    plot_data(data, args.output, with_history, args.sweep_type, processor)

    # export data to YAML
    if args.history == "save":
        export_data(data, Path(__file__).resolve().parent / "history.yaml", args.sweep_type, processor)
