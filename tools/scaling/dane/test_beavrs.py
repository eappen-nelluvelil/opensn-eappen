#!/usr/bin/env python3

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dane_beavrs", SCRIPT_DIR / "beavrs.py")
BEAVRS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BEAVRS)


SOURCE = '''problem = DiscreteOrdinatesProblem(
    groupsets=[
        {
            "angular_quadrature": quadrature,
        }
    ],
    options={
            "save_angular_flux": False,
    },
        use_gpus=USE_GPUS,
)
'''


class DaneBeavrsTest(unittest.TestCase):
    def test_conversion_selects_cycle_capable_host_cbc(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.py"
            source.write_text(SOURCE)
            text = BEAVRS.make_input(source)
            compile(text, "beavrs.py", "exec")
            self.assertIn('"angle_aggregation_type": "single"', text)
            self.assertIn('"allow_cycles": True', text)
            self.assertIn('"max_mpi_message_size": 256 * 1024', text)
            self.assertIn('sweep_type="CBC"', text)
            self.assertIn('use_gpus=USE_GPUS', text)

    def test_generated_job_uses_requested_native_host_layout(self):
        record = {
            "output": "/results/beavrs",
            "scaling_root": "/results/scaling",
            "scaling_manifest": {
                "bank": "bank",
                "environment": "/deps/env.sh",
                "implementations": {
                    "branch": {
                        "sha": "a" * 40,
                        "build": "/build",
                        "binary": "/build/python/opensn",
                    }
                },
            },
            "benchmark_source": "/inputs/beavrs",
            "nodes": 32,
            "ranks_per_node": 64,
            "time_limit": "24:00:00",
        }
        job = BEAVRS.make_job(record)
        self.assertIn("#SBATCH --nodes=32", job)
        self.assertIn("#SBATCH --ntasks-per-node=64", job)
        self.assertIn("#SBATCH --time=24:00:00", job)
        self.assertIn("--ntasks=2048", job)
        self.assertIn("--mpibind=on", job)
        self.assertIn("export OPENSN_NUM_THREADS=1", job)
        self.assertIn("CMAKE_BUILD_TYPE:STRING=Native", job)
        self.assertIn('cd "$result"', job)
        result = subprocess.run(["bash", "-n"], input=job, text=True, check=False)
        self.assertEqual(result.returncode, 0)

    def test_shell_runner_is_valid(self):
        result = subprocess.run(
            ["zsh", "-n", str(SCRIPT_DIR / "run_beavrs_cbc.zsh")], check=False
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
