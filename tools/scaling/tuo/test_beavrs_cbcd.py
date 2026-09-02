#!/usr/bin/env python3

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
