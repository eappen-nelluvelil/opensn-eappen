#!/usr/bin/env python3

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "prepare_beavrs_cbcd.py"
SPEC = importlib.util.spec_from_file_location("prepare_beavrs_cbcd", MODULE_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class BeavrsCBCDTest(unittest.TestCase):
    def test_input_conversion_selects_noncycle_cbcd_and_minimal_flux_storage(self):
        source_text = '''group_sets = [
        {
            "angular_quadrature": quadrature,
        }
    ]
problem = DiscreteOrdinatesProblem(
    options={
            "save_angular_flux": False,
    },
        use_gpus=USE_GPUS,
)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            output = root / "output.py"
            source.write_text(source_text)
            PREPARE.prepare(source, output)
            text = output.read_text()
            compile(text, str(output), "exec")
            self.assertIn('"angle_aggregation_type": "single"', text)
            self.assertIn('"allow_cycles": False', text)
            self.assertIn('sweep_type="CBC"', text)
            self.assertIn('"max_mpi_message_size": 256 * 1024', text)
            self.assertIn('"save_angular_flux": False', text)

    def test_runner_is_valid_zsh(self):
        result = subprocess.run(
            ["zsh", "-n", str(SCRIPT_DIR / "run_beavrs_cbcd.zsh")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runner_prepares_a_valid_sixteen_node_job(self):
        source_text = '''group_sets = [
        {
            "angular_quadrature": quadrature,
        }
    ]
problem = DiscreteOrdinatesProblem(
    options={
            "save_angular_flux": False,
    },
        use_gpus=USE_GPUS,
)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            build = work / "build-opensn"
            binary = build / "python/opensn"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o700)
            (build / "CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Native\n"
            )
            work.mkdir(exist_ok=True)
            (work / "env.zsh").write_text(":\n")

            benchmark = root / "benchmark"
            benchmark.mkdir()
            (benchmark / "beavrs_quarter_core_gpu.py").write_text(source_text)
            (benchmark / "beavrs_quarter_core_partitioned.obj").write_text("mesh\n")
            (benchmark / "beavrs_CASMO-70.h5").write_text("xs\n")

            environment = os.environ.copy()
            environment.update(
                {
                    "OPENSN_SOURCE": str(SCRIPT_DIR.parents[2]),
                    "OPENSN_TUO_ROOT": str(work),
                    "OPENSN_TUO_BUILD": str(build),
                    "OPENSN_TUO_RESULTS": str(root / "results"),
                    "OPENSN_TUO_BEAVRS_SOURCE": str(benchmark),
                    "OPENSN_TUO_BEAVRS_NODES": "16",
                    "OPENSN_TUO_BEAVRS_TIME_LIMIT": "4h",
                    "OPENSN_TUO_NUM_THREADS": "21",
                    "OPENSN_TUO_BANK": "bank",
                }
            )
            runner = SCRIPT_DIR / "run_beavrs_cbcd.zsh"
            result = subprocess.run(
                ["zsh", str(runner), "prepare", "beavrs-test"],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            job = root / "results/beavrs-test-beavrs/beavrs-cbcd.zsh"
            text = job.read_text()
            self.assertIn("#flux: -N 16", text)
            self.assertIn("#flux: -n 64", text)
            self.assertIn("#flux: -t 4h", text)
            self.assertIn("${FLUX_JOB_ID:-allocation}", text)
            self.assertIn('mkdir -p -- "$run"', text)
            self.assertIn("export BEAVRS_QC_N_POLAR=4", text)
            syntax = subprocess.run(
                ["zsh", "-n", str(job)], capture_output=True, text=True, check=False
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)


if __name__ == "__main__":
    unittest.main()
