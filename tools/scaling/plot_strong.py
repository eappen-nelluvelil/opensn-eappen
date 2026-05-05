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
    return f"{extra_data['name']}_{sweep_type.lower()}_{processor}_strong_scaling"


def extract_data(filename):
    """Extract n, average sweep time, and number of unknowns from a file."""

    match = re.search(r'_(\d+)\.out$', filename)
    if not match:
        return None
    n = int(match.group(1))

    avg_time = None
    num_unknowns = None

    avg_time_re = re.compile(r'avg_sweep_time\s*=\s*([0-9.eE+-]+)\s*s')
    unknowns_re = re.compile(r'\bunknowns\s*=\s*([0-9.eE+-]+)')

    with open(filename, 'r') as f:
        for line in f:
            avg_match = avg_time_re.search(line)
            if avg_match:
                avg_time = float(avg_match.group(1))

            unknowns_match = unknowns_re.search(line)
            if unknowns_match:
                num_unknowns = float(unknowns_match.group(1))

    if avg_time is None or num_unknowns is None:
        return None

    metric = avg_time / num_unknowns
    return n, metric


def plot_data(data, output_file, with_history, sweep_type, processor):
    """Plot the data and save to a file."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, NullLocator

    n_nodes = [d[0] for d in data]
    sweep_time = [d[1] * 1e9 for d in data]
    ideal = [sweep_time[0] / n for n in n_nodes]

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
            history["sweep_time"] = [t * 1e9 for t in history_data["sweep_time"]]

    fig, ax = plt.subplots()
    ax.plot(n_nodes, ideal, linestyle='--', color='xkcd:sky blue', label='ideal')
    ax.plot(n_nodes, sweep_time, marker='o', color='xkcd:cerulean', label='sweep time')
    xticks = n_nodes.copy()
    if history:
        ax.plot(history["nodes"], history["sweep_time"], marker='o',
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
    ax.set_ylabel("Average sweep time per unknown (ns)")
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_title(f"Node-to-node strong scaling ({sweep_type}, {processor})")
    ax.grid(True, which='both')
    ax.legend()
    fig.savefig(output_file)
    plt.show()


def export_data(data, output_file, sweep_type, processor):
    """Export data to a YAML file."""

    if yaml is None:
        raise ImportError("Saving history requires PyYAML.")

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
        "sweep_time": [d[1] for d in data]
    }
    with open(output_file, "w") as f:
        yaml.dump(export_dict, f)


if __name__ == "__main__":

    # read command-line arguments
    parser = ArgumentParser(description="Plot scaling data from output files.")
    parser.add_argument(
        "--output",
        type=str,
        default="strong_scaling_plot.pdf",
        help="Filename for the output plot (default: strong_scaling_plot.pdf)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=(
            "Folder to find strong scaling results. Defaults to "
            "output/strong_{sweep_type}_{cpu/gpu}."
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
    input_dir_arg = args.dir or f"output/strong_{args.sweep_type.lower()}_{processor}"

    # get files matching the prefix in the input directory
    input_dir = Path(__file__).resolve().parent / input_dir_arg
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    files = glob.glob(f"{input_dir}/strong_*.out")
    if not files:
        raise FileNotFoundError(f"No files found matching strong_*.out in {input_dir}")

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
