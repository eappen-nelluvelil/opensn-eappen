import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class CampaignTest(unittest.TestCase):
    def test_wrapper_dispatch_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "zsh"
            capture = root / "capture.json"
            fake.write_text('''#!/usr/bin/python3
import json
import os
import sys
with open(os.environ['CAPTURE'], 'w') as stream:
    json.dump({'args': sys.argv[1:], 'env': dict(os.environ)}, stream)
''')
            fake.chmod(0o700)
            env = dict(os.environ, PATH=f"{root}:/usr/bin:/bin", CAPTURE=str(capture),
                       OPENSN_TUO_STUDY_ROOT=str(root / "study"),
                       OPENSN_TUO_ROOT="stale-build", OPENSN_CBCD_NUM_WORKERS="1000")
            for cluster, command in (("dane", "launch"), ("tuo", "submit-scaling-metrics")):
                subprocess.run(["/usr/bin/zsh", str(HERE / "run_minfluds.zsh"),
                                cluster, "scaling", "campaign-1"], env=env, check=True)
                data = json.loads(capture.read_text())
                self.assertEqual(data['args'][-2:], [command, "campaign-1"])
                if cluster == "dane":
                    self.assertEqual(data['env']['OPENSN_DANE_TIME_LIMIT'], "01:00:00")
                else:
                    self.assertEqual(data['env']['OPENSN_TUO_NUM_THREADS'], "21")
                    self.assertEqual(data['env']['OPENSN_TUO_NODES'],
                                     "1,2,4,8,16,32,64,128,256")
                    self.assertNotIn("OPENSN_CBCD_NUM_WORKERS", data['env'])
                    self.assertNotEqual(data['env']['OPENSN_TUO_ROOT'], "stale-build")

    def test_adapts_supplied_input_without_changing_physics(self):
        import beavrs_retry
        fixture = Path(os.environ.get("OPENSN_BEAVRS_TEST_SOURCE", ""))
        if not (fixture / "beavrs_quarter_core_cpu.py").is_file():
            self.skipTest("Set OPENSN_BEAVRS_TEST_SOURCE to the original input directory")
        for backend, device in (("cpu", False), ("gpu", True)):
            original = (fixture / f"beavrs_quarter_core_{backend}.py").read_text()
            adapted = beavrs_retry.adapt_input(original, device)
            compile(adapted, "benchmark.py", "exec")
            for line in original.splitlines():
                if line.startswith(("K_TOLERANCE =", "NUM_GROUPS =", "PIN_PITCH =",
                                    "PIN_CLUSTER_RADIUS =", "Z_THICKNESS =")):
                    self.assertIn(line, adapted)


if __name__ == "__main__":
    unittest.main()
