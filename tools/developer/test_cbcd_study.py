# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cbcd_study as study
from study_build import configure_command, digest, write_json


OUTPUT = """iteration = 0, residual = 1.000000e+00
iteration = 1, residual = 1.000000e-01
final, status = iteration_limit, iterations = 1
avg_sweep_time = 2.000000e+00 s, sweep_time_per_unknown = 1.000000e+00 ns
unknowns = 2000000000, lagged_unknowns = 0
OPENSN_STUDY_FLUX_MAX group=0 value=0.5
OPENSN_STUDY_FLUX_MAX group=63 value=1e-8
OPENSN_STUDY_TRIAL_COMPLETE
OpenSn finished execution.
"""


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(nodes=[1], ranks_per_node=1, threads=2, iterations=1,
                           trials=3, modes=["baseline"],
                           launcher=[sys.executable, str(self.root / "launcher.py")])
        write_json(self.root / "manifest.json", self.config)
        (self.root / "inputs").mkdir()
        (self.root / "inputs/strong-1.py").write_text("pass\n")
        (self.root / "inputs/weak-1.py").write_text("pass\n")
        (self.root / "builds/native").mkdir(parents=True)
        write_json(self.root / "builds/native/build.json", dict(fingerprints={}))
        (self.root / "launcher.py").write_text('''
import json
import os
from pathlib import Path
import sys
attempt = Path(sys.argv[sys.argv.index('--attempt') + 1])
(attempt / 'placement/rank-0.json').write_text(json.dumps(
    dict(rank=0, hostname='node', pid=os.getpid())))
if os.environ.get('FAIL_STUDY'):
    sys.exit(5)
print(''' + repr(OUTPUT) + ''')
''')

    def launch(self, trial, kind="strong"):
        with redirect_stdout(io.StringIO()):
            return study.launch_trial(self.root, self.config, kind, 1, "baseline", trial,
                                      10, "test-allocation")

    def test_fixed_work_signature(self):
        result = study.measurements(OUTPUT, 1)
        self.assertEqual(result["unknowns"], 2000000000)
        for text in (OUTPUT.replace("OPENSN_STUDY_TRIAL_COMPLETE", ""),
                     OUTPUT.replace("iteration_limit", "converged"),
                     OUTPUT.replace("lagged_unknowns = 0", "lagged_unknowns = 1"),
                     OUTPUT.replace("1.000000e+00 ns", "2.000000e+00 ns"),
                     OUTPUT.replace("value=0.5", "value=nan"), OUTPUT + OUTPUT):
            with self.subTest(text=text), self.assertRaises(ValueError):
                study.measurements(text, 1)

    def test_fresh_processes_and_resume(self):
        pids = []
        for trial in range(1, 4):
            self.launch(trial)
            attempts = study.successes(self.root, "strong", 1, "baseline", trial, self.config)
            self.assertEqual(len(attempts), 1)
            pids.append(study.read_json(attempts[0] / "placement/rank-0.json")["pid"])
        self.assertEqual(len(set(pids)), 3)
        self.assertEqual(study.pending(self.root, "strong", 1, self.config), [])
        self.assertEqual(len(study.pending(self.root, "weak", 1, self.config)), 3)

    def test_failure_retained_on_retry(self):
        with patch.dict(os.environ, FAIL_STUDY="1"), self.assertRaises(ValueError):
            self.launch(1)
        failed = list(self.root.rglob("FAILED"))
        self.assertEqual(len(failed), 1)
        before = failed[0].read_bytes()
        self.assertEqual(len(study.pending(self.root, "strong", 1, self.config)), 3)
        self.launch(1)
        self.assertEqual(failed[0].read_bytes(), before)
        self.assertEqual(len(list(self.root.rglob("attempt-*"))), 2)
        self.assertEqual(len(study.pending(self.root, "strong", 1, self.config)), 2)

    def test_corrupt_completion_is_not_skipped(self):
        self.launch(1)
        attempt = study.successes(self.root, "strong", 1, "baseline", 1, self.config)[0]
        (attempt / "stdout.txt").write_text("incomplete\n")
        self.assertEqual(study.successes(self.root, "strong", 1, "baseline", 1, self.config), [])

    def test_rank_count_and_numerical_comparison(self):
        self.launch(1)
        launcher = self.root / "launcher.py"
        launcher.write_text(launcher.read_text().replace("value=0.5", "value=0.6"))
        with self.assertRaisesRegex(ValueError, "Scalar-flux"):
            self.launch(2)
        self.config["ranks_per_node"] = 2
        with self.assertRaisesRegex(ValueError, "placement"):
            self.launch(3)

    def test_no_launch_when_allocation_nearly_expired(self):
        args = SimpleNamespace(root=self.root, kind="strong", nodes=1, seconds=70,
                               allocation="test")
        with patch.object(study, "verify_inputs"), redirect_stdout(io.StringIO()):
            self.assertEqual(study.run(args), 3)
        self.assertFalse((self.root / "results").exists())

    def test_timeout_reaps_process_and_retains_failure(self):
        (self.root / "launcher.py").write_text("import time\ntime.sleep(60)\n")
        with self.assertRaises(subprocess.TimeoutExpired), redirect_stdout(io.StringIO()):
            study.launch_trial(self.root, self.config, "strong", 1, "baseline", 1, .1, "test")
        self.assertEqual(len(list(self.root.rglob("FAILED"))), 1)
        self.assertEqual(len(list(self.root.rglob("SUCCESS"))), 0)

    def test_lock_released_after_exception(self):
        path = self.root / "lock"
        with study.locked(path):
            with self.assertRaises(BlockingIOError), study.locked(path):
                pass
        with study.locked(path):
            pass

    def test_trace_search_is_recursive_and_rejects_empty_trace(self):
        directory = self.root / "rocprof/rank-0/hostname"
        directory.mkdir(parents=True)
        for kind in ("hip_api", "kernel", "memory_copy", "memory_allocation"):
            (directory / f"pid_{kind}_trace.csv").write_text("header\nvalue\n")
        study.profile_artifacts(self.root, "rocprof")
        (directory / "pid_hip_api_trace.csv").write_text("header\n")
        with self.assertRaises(ValueError):
            study.profile_artifacts(self.root, "rocprof")

    def test_clean_environment_preserves_binding_and_mpi(self):
        env = study.clean_environment(dict(CALI_CONFIG="bad", CALI_SERVICES_ENABLE="mpi",
                                           OPENSN_CBCD_FUSE_WORKER_LAUNCHES="1",
                                           OPENSN_MEMORY_TRACE="1", LD_PRELOAD="bad.so",
                                           ROCR_VISIBLE_DEVICES="2",
                                           MPICH_GPU_SUPPORT_ENABLED="1"))
        self.assertEqual(env["CALI_CONFIG"], "")
        self.assertEqual(env["CALI_SERVICES_ENABLE"], "")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["MPICH_GPU_SUPPORT_ENABLED"], "1")
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("OPENSN_MEMORY_TRACE", env)
        self.assertNotIn("OPENSN_CBCD_FUSE_WORKER_LAUNCHES", env)

    def test_input_has_one_problem_and_one_solve(self):
        text = (study.HERE / "cbcd_study_input.py.in").read_text()
        text = text.replace("@MESH@", repr("mesh.msh")).replace("@XS@", repr("xs.xs"))
        tree = ast.parse(text.replace("@ITERATIONS@", "16"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        constructors = [n for n in calls if isinstance(n.func, ast.Name)
                        and n.func.id == "DiscreteOrdinatesProblem"]
        self.assertEqual(len(constructors), 1)
        solves = [n for n in calls if isinstance(n.func, ast.Attribute)
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == "solver"
                  and n.func.attr == "Execute"]
        self.assertEqual(len(solves), 1)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                self.assertNotIn(solves[0], list(ast.walk(node)))
        options = {kw.arg: kw.value for kw in constructors[0].keywords}
        self.assertTrue(ast.literal_eval(options["use_gpus"]))
        self.assertEqual(ast.literal_eval(options["num_groups"]), 64)

    def test_build_reuses_stack_but_not_instrumentation(self):
        cache = self.root / "CMakeCache.txt"
        cache.write_text("CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_BUILD_TYPE:STRING=Debug\n"
                         "CMAKE_CXX_COMPILER:FILEPATH=/bin/clang++\n"
                         "CMAKE_PREFIX_PATH:PATH=/deps\nOPENSN_WITH_HIP:BOOL=ON\n"
                         "CMAKE_HIP_ARCHITECTURES:STRING=gfx942\n")
        command = configure_command(Path("source"), self.root, Path("build"))
        self.assertIn("-DCMAKE_BUILD_TYPE=Native", command)
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES:STRING=gfx942", command)
        self.assertIn("-DCMAKE_PREFIX_PATH:PATH=/deps", command)
        cache.write_text(cache.read_text() + "CMAKE_CXX_FLAGS:STRING=-fsanitize=address\n")
        with self.assertRaises(ValueError):
            configure_command(Path("source"), self.root, Path("build"))

    def test_binary_change_is_rejected(self):
        path = self.root / "binary"
        path.write_text("original")
        record = dict(fingerprints={str(path): digest(path)})
        study.verify_fingerprint(record)
        path.write_text("changed")
        with self.assertRaises(ValueError):
            study.verify_fingerprint(record)

    def test_rank_zero_only_rocprof_wrapper(self):
        record = self.root / "builds/native/build.json"
        record.write_text('{"binary": "/test/opensn"}')
        for rank in (0, 1):
            attempt = self.root / f"rank-attempt-{rank}"
            (attempt / "placement").mkdir(parents=True)
            args = SimpleNamespace(root=self.root, attempt=attempt, mode="rocprof")
            with patch.dict(os.environ, FLUX_TASK_RANK=str(rank)), \
                    patch.object(study.os, "execvpe") as execute:
                study.rank_run(args)
            program, command, env = execute.call_args.args
            self.assertEqual(program, "rocprofv3" if rank == 0 else "/test/opensn")
            self.assertEqual(env["CALI_SERVICES_ENABLE"], "")
            if rank == 0:
                self.assertIn("--memory-allocation-trace", command)
            self.assertTrue((attempt / f"placement/rank-{rank}.json").is_file())


if __name__ == "__main__":
    unittest.main()
