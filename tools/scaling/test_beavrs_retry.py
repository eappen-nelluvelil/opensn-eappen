import argparse
import importlib.util
import subprocess
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "beavrs_retry", Path(__file__).with_name("beavrs_retry.py")
)
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)


class RetryTest(unittest.TestCase):
    def test_both_scheduler_scripts(self):
        for cluster, nodes, shell in (("dane", 32, "bash"), ("tuo", 16, "zsh")):
            args = argparse.Namespace(
                cluster=cluster, nodes=nodes, bank="cbronze",
                time_limit="24:00:00" if cluster == "dane" else "12h",
                output=Path("/tmp/result"), build=Path("/tmp/build"),
                environment=Path("/tmp/env"), benchmark=Path("/tmp/input"),
            )
            job = retry.job_script(args, "a" * 40)
            self.assertEqual(subprocess.run([shell, "-n"], input=job, text=True).returncode, 0)
            self.assertIn("import numpy, scipy", job)
            self.assertIn("PYTHONUNBUFFERED=1", job)
            self.assertIn("touch SUCCESS", job)
            self.assertIn("-i preflight.py", job)
            if cluster == "dane":
                self.assertIn("--ntasks=2048", job)
                self.assertIn("--kill-on-bad-exit=1", job)
            else:
                self.assertIn("-n 64 -c 21 -g 1", job)
                self.assertTrue(job.startswith("#!/bin/zsh"))

    def test_input_checks_and_wrapper(self):
        source = '''USE_GPUS = False
def gather_fuel_cells(local_cells):
    pass
def identify_fuel_pins(fuel_cells):
    pass
def main():
    problem = Problem(
        groupsets=[{
            "angular_quadrature": quadrature,
            "save_angular_flux": False,
        }],
        use_gpus=USE_GPUS,
    )
main()
'''
        generated = retry.adapt_input(source, False)
        compile(generated, "benchmark.py", "exec")
        compile(retry.wrapper(), "input.py", "exec")
        compile(retry.preflight(), "preflight.py", "exec")
        self.assertIn('"allow_cycles": True', generated)
        self.assertIn("BEAVRS_STAGE", generated)
        with self.assertRaises(ValueError):
            retry.adapt_input(generated, False)


if __name__ == "__main__":
    unittest.main()
