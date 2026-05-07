#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob
import warnings
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from generate_scaling_study import extra_data
from lib import make_study_tag

try:
    import yaml
except ImportError:
    yaml = None


def study_label(sweep_type, processor, study_name=""):
    """Return a history label that includes algorithm and execution target."""
    return f"{extra_data['name']}_{make_study_tag(sweep_type, processor, study_name)}_weak_scaling"


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


def load_data(input_dir):
    """Load weak-scaling data from an output directory."""

    files = glob.glob(f"{input_dir}/weak_*.out")
    if not files:
        raise FileNotFoundError(f"No files found matching weak_*.out in {input_dir}")

    data = []
    for f in files:
        result = extract_data(f)
        if result:
            data.append(result)
    if not data:
        raise ValueError(f"No valid data found in {input_dir}.")
    data.sort(key=lambda x: x[0])
    return data


def compute_efficiency(data):
    """Compute weak-scaling efficiency from average sweep times."""

    sweep_time = [d[1] for d in data]
    return [sweep_time[0] * 100.0 / t for t in sweep_time]


def plot_data(series, output_file, with_history, sweep_type, processor, study_name):
    """Plot the data and save to a file."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    history = {}
    history_file = Path(__file__).resolve().parent / "history.yaml"
    if with_history and yaml is None:
        warnings.warn("PyYAML is not installed. Plotting without history.")
    elif with_history and history_file.exists():
        with open(history_file, "r") as f:
            history_dict = yaml.safe_load(f)
        history_label = study_label(sweep_type, processor, study_name)
        if history_dict is not None and history_label in history_dict:
            history_data = history_dict[history_label]
            history["nodes"] = history_data["nodes"]
            history["efficiency"] = history_data["efficiency"]

    fig, ax = plt.subplots()
    xticks = []
    max_efficiency = 100.0
    for label, data in series:
        n_nodes = [d[0] for d in data]
        efficiency = compute_efficiency(data)
        ax.plot(n_nodes, efficiency, marker='o', label=label)
        xticks = sorted(set(xticks) | set(n_nodes))
        max_efficiency = max(max_efficiency, max(efficiency))
    if history:
        ax.plot(history["nodes"], history["efficiency"], marker='o',
                color='xkcd:coral', label='history')
        xticks = sorted(set(xticks) | set(history["nodes"]))
        max_efficiency = max(max_efficiency, max(history["efficiency"]))
    elif with_history:
        warnings.warn(
            "History file not found or history label not in file. "
            "Plotting without history."
        )
    ax.set_xlabel("Number of nodes")
    ax.set_xscale('log')
    ax.set_xticks(xticks, xticks)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0.0, top=max_efficiency + 10.0)
    ax.set_ylabel("Efficiency (%)")
    ax.set_title(f"Node-to-node weak scaling ({sweep_type}, {processor})")
    ax.grid(True, which='both')
    ax.legend()
    fig.savefig(output_file)
    if plt.get_backend().lower() != "agg":
        plt.show()


def export_data(data, output_file, sweep_type, processor, study_name):
    """Export data to a YAML file."""

    if yaml is None:
        raise ImportError("Saving history requires PyYAML.")

    efficiency = compute_efficiency(data)

    label = study_label(sweep_type, processor, study_name)
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
        action="append",
        default=None,
        help=(
            "Folder to find weak scaling results. Defaults to "
            "output/weak_{sweep_type}_{cpu/gpu}[_study_name]. Can be specified multiple times."
        )
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=None,
        help="Label for a plotted result directory. Can be specified once per --dir."
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
    parser.add_argument(
        "--study-name",
        type=str,
        default="",
        help="Optional study identifier used for the default result directory and history label."
    )
    args = parser.parse_args()

    processor = args.processor or ("gpu" if args.use_gpus else "cpu")
    default_dir = f"output/weak_{make_study_tag(args.sweep_type, processor, args.study_name)}"
    input_dir_args = args.dir or [default_dir]
    if args.label is not None and len(args.label) != len(input_dir_args):
        raise ValueError("Specify either no --label values or exactly one --label per --dir.")
    labels = args.label or [Path(d).name.removeprefix("weak_") for d in input_dir_args]

    series = []
    for label, input_dir_arg in zip(labels, input_dir_args):
        input_dir = Path(__file__).resolve().parent / input_dir_arg
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
        series.append((label, load_data(input_dir)))

    # plot
    with_history = (args.history == "comp")
    plot_data(series, args.output, with_history, args.sweep_type, processor, args.study_name)

    # export data to YAML
    if args.history == "save":
        if len(series) != 1:
            raise ValueError("History save is only supported for a single result directory.")
        export_data(series[0][1],
                    Path(__file__).resolve().parent / "history.yaml",
                    args.sweep_type,
                    processor,
                    args.study_name)
