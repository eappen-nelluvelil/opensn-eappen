#!/usr/bin/env python3

"""Create a CBCD input from the maintained BEAVRS GPU benchmark."""

import argparse
from pathlib import Path


def replace_once(text, old, new, description):
    if text.count(old) != 1:
        raise RuntimeError(f"expected one {description} marker, found {text.count(old)}")
    return text.replace(old, new, 1)


def prepare(source, output):
    text = source.read_text()
    text = replace_once(
        text,
        '            "angular_quadrature": quadrature,\n',
        '            "angular_quadrature": quadrature,\n'
        '            "angle_aggregation_type": "single",\n'
        '            "allow_cycles": False,\n',
        "groupset quadrature",
    )
    text = replace_once(
        text,
        '            "save_angular_flux": False,\n',
        '            "max_mpi_message_size": 256 * 1024,\n'
        '            "save_angular_flux": False,\n',
        "angular-flux option",
    )
    text = replace_once(
        text,
        "        use_gpus=USE_GPUS,\n",
        '        sweep_type="CBC",\n'
        "        use_gpus=USE_GPUS,\n",
        "device-problem backend",
    )
    compile(text, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
