#!/usr/bin/env python3
"""Prepare an isolated, diagnostic BEAVRS retry using an explicitly selected build."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


def quote(value):
    return shlex.quote(str(value))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one input marker: {old!r}")
    return text.replace(old, new, 1)


def adapt_input(text, device):
    if 'sweep_type=' in text:
        raise ValueError("use the original benchmark input, not an already adapted copy")
    text = replace_once(
        text, '            "angular_quadrature": quadrature,\n',
        '            "angular_quadrature": quadrature,\n'
        '            "angle_aggregation_type": "single",\n'
        f'            "allow_cycles": {not device},\n',
    )
    text = replace_once(
        text, '            "save_angular_flux": False,\n',
        '            "max_mpi_message_size": 256 * 1024,\n'
        '            "save_angular_flux": False,\n',
    )
    text = replace_once(text, "        use_gpus=USE_GPUS,\n",
                        '        sweep_type="CBC",\n        use_gpus=USE_GPUS,\n')
    text = replace_once(text, f"USE_GPUS = {device}\n", f"USE_GPUS = {device}\n")
    start = text.find("def gather_fuel_cells(")
    end = text.find("def identify_fuel_pins(")
    if start < 0 or end <= start:
        raise ValueError("missing fuel-cell gathering function")
    # Preserve the same globally deduplicated records without a padded all-reduce
    # or dependence on the console's std::vector Python converter.
    text = text[:start] + '''def gather_fuel_cells(local_cells):
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI

    records = MPI.COMM_WORLD.allgather(local_cells)
    return sorted({record for part in records for record in part},
                  key=lambda record: (record[1], record[0], record[2]))


''' + text[end:]
    # Keep physical parameters and convergence tolerances unchanged.
    text = replace_once(text, "\nmain()\n", '''
import functools
import time


def _beavrs_stage(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        start = time.monotonic()
        if rank == 0:
            print(f"BEAVRS_STAGE {function.__name__} begin", flush=True)
        result = function(*args, **kwargs)
        if rank == 0:
            print(f"BEAVRS_STAGE {function.__name__} end seconds={time.monotonic()-start:.3f}",
                  flush=True)
            if function.__name__ == "solve_eigenvalue":
                print(f"BEAVRS_EIGENVALUE {result[1]:.10g}", flush=True)
            if function.__name__ == "identify_fuel_pins":
                print(f"BEAVRS_PIN_CLUSTERS {sum(map(len, result.values()))}", flush=True)
        return result
    return wrapped


for _name in ("build_mesh", "load_cross_sections", "solve_eigenvalue",
              "gather_fuel_cells", "identify_fuel_pins", "calculate_pin_powers"):
    globals()[_name] = _beavrs_stage(globals()[_name])
main()
''')
    compile(text, "benchmark.py", "exec")
    return text


def wrapper():
    return '''import faulthandler
import os
import sys
import traceback
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(1800, repeat=True)
try:
    print(f"BEAVRS_START rank={globals().get('rank')} host={os.uname().nodename} "
          f"python={sys.version.split()[0]}", flush=True)
    exec(compile(Path("benchmark.py").read_text(), "benchmark.py", "exec"), globals())
except BaseException:
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
'''


def preflight():
    return '''import os
import numpy
import scipy
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
if not MPI.Is_initialized():
    raise RuntimeError("mpi4py did not find OpenSn's initialized MPI runtime")
comm = MPI.COMM_WORLD
if comm.size != int(os.environ["BEAVRS_EXPECTED_RANKS"]):
    raise RuntimeError("Unexpected MPI rank count")
if comm.allreduce(1) != comm.size:
    raise RuntimeError("MPI collective preflight failed")
print(f"BEAVRS_PREFLIGHT rank={comm.rank} size={comm.size} "
      f"host={os.uname().nodename} scipy={scipy.__version__} "
      f"HIP_VISIBLE_DEVICES={os.getenv('HIP_VISIBLE_DEVICES')} "
      f"ROCR_VISIBLE_DEVICES={os.getenv('ROCR_VISIBLE_DEVICES')}", flush=True)
'''


def job_script(args, sha):
    root = args.output
    binary = args.build / "python/opensn"
    ranks_per_node = 4 if args.cluster == "tuo" else 64
    ranks = args.nodes * ranks_per_node
    if args.cluster == "dane":
        header = f'''#!/bin/bash
#SBATCH --job-name=beavrs-cbc-{args.nodes}n
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --partition=pbatch
#SBATCH --account={args.bank}
#SBATCH --time={args.time_limit}
#SBATCH --output={quote(root / 'scheduler/%j.out')}
#SBATCH --error={quote(root / 'scheduler/%j.err')}
'''
        launcher = (f'srun --mpi=pmix --kill-on-bad-exit=1 --nodes={args.nodes} '
                    f'--ntasks={ranks} --ntasks-per-node=64 --cpus-per-task=1 '
                    '--distribution=block --mpibind=on '
                    '--error="$run/rank-%t.stderr"')
        thread_count = 1
        gpu_env = "unset MPICH_GPU_SUPPORT_ENABLED MPICH_SMP_SINGLE_COPY_MODE"
        job_id = "${SLURM_JOB_ID:-manual}"
    else:
        header = f'''#!/bin/zsh
#flux: --job-name=beavrs-cbcd-{args.nodes}n
#flux: -N {args.nodes}
#flux: -n {ranks}
#flux: -c 21
#flux: -g 1
#flux: --exclusive
#flux: -q pbatch
#flux: -B {args.bank}
#flux: -t {args.time_limit}
#flux: --output={root / 'scheduler/{{id}}.out'}
#flux: --error={root / 'scheduler/{{id}}.err'}
'''
        launcher = (f'flux run -N {args.nodes} -n {ranks} -c 21 -g 1 --exclusive '
                    '-o exit-on-error --label-io')
        thread_count = 21
        gpu_env = ('export MPICH_GPU_SUPPORT_ENABLED=1\n'
                   'export MPICH_SMP_SINGLE_COPY_MODE=XPMEM\n'
                   'unset OPENSN_CBCD_NUM_WORKERS')
        job_id = "${FLUX_JOB_ID:-manual}"
    return header + f'''
set -euo pipefail
source {quote(args.environment)}
export OPENSN_NUM_THREADS={thread_count} OMP_NUM_THREADS={thread_count}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export BEAVRS_QC_N_POLAR=4 BEAVRS_QC_N_AZIMUTHAL=32
export BEAVRS_QC_SCATTERING_ORDER=1 BEAVRS_QC_USE_CMFD=False
export BEAVRS_EXPECTED_RANKS={ranks}
{gpu_env}
unset CALI_CONFIG CALI_SERVICES_ENABLE
run={quote(root / 'results')}/run-{job_id}-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$run"
cp {quote(root / 'input.py')} {quote(root / 'benchmark.py')} "$run/"
cp {quote(root / 'preflight.py')} "$run/"
ln -s {quote(args.benchmark / 'beavrs_quarter_core_partitioned.obj')} "$run/"
ln -s {quote(args.benchmark / 'beavrs_CASMO-70.h5')} "$run/"
cd "$run"
finish_run()
{{
  local code=$?
  if [[ ! -f SUCCESS ]]; then
    echo "$code" > exit_code.txt
    touch FAILED
  fi
}}
trap finish_run EXIT
{{
  date -u
  echo 'source_revision={sha}'
  echo 'ranks={ranks}'
  module list
  command -v python
  ldd {quote(binary)}
  sha256sum {quote(binary)} input.py benchmark.py beavrs_CASMO-70.h5
}} > metadata.txt 2>&1
grep -qx 'CMAKE_BUILD_TYPE:STRING=Native' {quote(args.build / 'CMakeCache.txt')}
# This must succeed before launching the full benchmark.
python -c '
import numpy, scipy
import mpi4py
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
print(numpy.__version__, scipy.__version__)
' > imports.txt 2>&1
{launcher} {quote(binary)} --suppress-color -i preflight.py > preflight.txt 2> preflight.stderr
set +e
{launcher} {quote(binary)} --suppress-color -i input.py > stdout.txt 2> stderr.txt
rc=$?
set -e
echo "$rc" > exit_code.txt
if (( rc != 0 )) || ! grep -q 'OpenSn finished execution.' stdout.txt; then
  touch FAILED
  exit $((rc == 0 ? 1 : rc))
fi
touch SUCCESS
'''


def prepare(args):
    for name in ("source", "build", "environment", "benchmark", "output"):
        setattr(args, name, getattr(args, name).resolve())
    minimum = 16 if args.cluster == "tuo" else 32
    if args.nodes < minimum:
        raise ValueError(f"this campaign requires at least {minimum} nodes")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.bank):
        raise ValueError("invalid bank")
    if not re.fullmatch(r"[0-9:dhms-]+", args.time_limit):
        raise ValueError("invalid time limit")
    if args.output.exists():
        raise ValueError(f"result directory already exists: {args.output}")
    backend = 'gpu' if args.cluster == 'tuo' else 'cpu'
    original = args.benchmark / f"beavrs_quarter_core_{backend}.py"
    for path in (original, args.environment, args.build / "python/opensn",
                 args.benchmark / "beavrs_quarter_core_partitioned.obj",
                 args.benchmark / "beavrs_CASMO-70.h5"):
        if not path.is_file():
            raise ValueError(f"missing file: {path}")
    cache = (args.build / "CMakeCache.txt").read_text()
    if "CMAKE_BUILD_TYPE:STRING=Native\n" not in cache:
        raise ValueError("build must be Native")
    source_entry = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache, re.M)
    if not source_entry or Path(source_entry[1]).resolve() != args.source:
        raise ValueError("CMake source does not match --source")
    dirty = subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True,
    )
    if dirty.strip():
        raise ValueError("source worktree must be clean")
    sha = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"],
                                  text=True).strip()
    stamp = args.build / "source-revision.txt"
    if not stamp.is_file() or stamp.read_text().strip() != sha:
        raise ValueError("Native build has not finished for this revision; wait for the build job")
    text = adapt_input(original.read_text(), args.cluster == "tuo")
    args.output.mkdir(parents=True)
    for folder in ("scheduler", "results"):
        (args.output / folder).mkdir()
    (args.output / "benchmark.py").write_text(text)
    (args.output / "input.py").write_text(wrapper())
    (args.output / "preflight.py").write_text(preflight())
    record = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    record["sha"] = sha
    record["original_input_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    job = args.output / ("beavrs.sbatch" if args.cluster == "dane" else "beavrs.flux")
    job.write_text(job_script(args, sha))
    subprocess.run(["zsh" if args.cluster == "tuo" else "bash", "-n", str(job)], check=True)
    print(f"Prepared {job}")
    if args.submit:
        command = ["sbatch", "--parsable"] if args.cluster == "dane" else ["flux", "batch"]
        job_id = subprocess.check_output([*command, str(job)], text=True).strip()
        (args.output / "job-id.txt").write_text(job_id + "\n")
        print(f"Submitted {job_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=("dane", "tuo"), required=True)
    for name in ("source", "build", "environment", "benchmark", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--time-limit", required=True)
    parser.add_argument("--bank", default="cbronze")
    parser.add_argument("--submit", action="store_true")
    try:
        prepare(parser.parse_args())
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
