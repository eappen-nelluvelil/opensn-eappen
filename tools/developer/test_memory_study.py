# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

import ast
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import memory_study


class MemoryStudyTest(unittest.TestCase):
    @staticmethod
    def complete_run(*args, **kwargs):
        kwargs["stdout"].write("avg_sweep_time = 1.0 s, sweep_time_per_unknown = 2.0 ns\n"
                               "MEMORY_TRIAL_END 1/1\n")
        return SimpleNamespace(returncode=0)

    def test_optional_smaps(self):
        namespace = {"os": os}
        exec(memory_study.MARKERS, namespace)
        for enabled in ("0", "1"):
            with tempfile.TemporaryDirectory() as directory:
                with patch.dict(os.environ, OPENSN_MEMORY_TRACE_DIR=directory,
                                OPENSN_MEMORY_SMAPS=enabled):
                    namespace["memory_marker"](0, "after_trial")
                record = json.loads(next(Path(directory).glob("python-*.jsonl")).read_text())
                self.assertEqual("smaps" in record, enabled == "1")
                if enabled == "1":
                    self.assertIsInstance(record["smaps"], str)
                    self.assertIn("Rss:", record["smaps"])

    def test_cgroup_snapshots(self):
        namespace = {}
        exec(memory_study.MARKERS, namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "job" / "step" / "task"
            leaf.mkdir(parents=True)
            (root / "job" / "memory.max").write_text("1024\n")
            (leaf / "memory.current").write_text("512\n")
            snapshot = namespace["cgroup_memory_snapshot"]("0::/job/step/task\n", root)
            self.assertEqual(set(snapshot),
                             {str(leaf), str(leaf.parent), str(root / "job"), str(root)})
            self.assertEqual(snapshot[str(leaf)]["memory.current"], "512\n")
            self.assertEqual(snapshot[str(root / "job")]["memory.max"], "1024\n")
            self.assertIn("error", snapshot[str(leaf)]["memory.peak"])
            self.assertEqual(namespace["cgroup_memory_snapshot"]("1:memory:/job\n", root), {})

    def test_scoped_trials(self):
        for gpu in (False, True):
            tree = ast.parse(memory_study.render(Path("/tmp/mesh.msh"), gpu, 17, 16))
            scope = next(n for n in tree.body
                         if isinstance(n, ast.FunctionDef) and n.name == "run_trial")
            self.assertEqual([n.targets[0].id for n in scope.body
                              if isinstance(n, ast.Assign)], ["phys", "ss_solver"])
            calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)]
            problem = next(n for n in calls if n.func.id == "DiscreteOrdinatesProblem")
            values = {k.arg: k.value for k in problem.keywords}
            self.assertEqual(ast.literal_eval(values["use_gpus"]), gpu)
            self.assertEqual(ast.literal_eval(values["sweep_type"]), "CBC")
            options = {ast.literal_eval(k): v for k, v in
                       zip(values["options"].keys, values["options"].values)}
            self.assertFalse(ast.literal_eval(options["save_angular_flux"]))
            groupset = values["groupsets"].elts[0]
            groupset = {ast.literal_eval(k): v for k, v in zip(groupset.keys, groupset.values)}
            self.assertEqual(ast.literal_eval(groupset["allow_cycles"]), not gpu)
            self.assertEqual(ast.literal_eval(groupset["l_max_its"]), 16)
            loop = tree.body[-1]
            self.assertEqual(ast.literal_eval(loop.iter.args[0]), 17)
            self.assertEqual(loop.body[2].value.func.id, "run_trial")

    def test_cache_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CMakeCache.txt"
            path.write_text("// header\n# header\nX:STRING=a=b\nY:BOOL=OFF\n")
            self.assertEqual(memory_study.cache_values(path),
                             {"X": ("STRING", "a=b"), "Y": ("BOOL", "OFF")})

    def test_positive(self):
        self.assertEqual(memory_study.positive("17"), 17)
        with self.assertRaises(memory_study.argparse.ArgumentTypeError):
            memory_study.positive("0")

    def test_reuse_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            previous.mkdir()
            (previous / "CMakeCache.txt").write_text(
                "CMAKE_GENERATOR:INTERNAL=Ninja\n"
                "CMAKE_CXX_COMPILER:FILEPATH=/stack/clang++\n"
                "CMAKE_CXX_FLAGS:STRING=-I/stack/include\n"
                "OPENSN_WITH_HIP:BOOL=ON\n"
                "Python3_EXECUTABLE:FILEPATH=/stack/venv/bin/python\n"
                "CMAKE_HOME_DIRECTORY:INTERNAL=/old/source\n"
                "caliper_DIR:PATH=caliper_DIR-NOTFOUND\n")
            args = memory_study.argparse.Namespace(
                reuse_build=previous, build=root / "new", jobs=4)
            with patch.object(memory_study.subprocess, "run") as run:
                memory_study.build(args)
            configure = run.call_args_list[0].args[0]
            self.assertIn("-DPython3_EXECUTABLE:FILEPATH=/stack/venv/bin/python", configure)
            self.assertIn("-DCMAKE_CXX_FLAGS:STRING=-I/stack/include", configure)
            self.assertIn("-DOPENSN_WITH_HIP:BOOL=ON", configure)
            self.assertIn("-DCMAKE_BUILD_TYPE=Native", configure)
            self.assertFalse(any("NOTFOUND" in arg or "HOME_DIRECTORY" in arg
                                 for arg in configure))
            self.assertEqual(run.call_args_list[1].args[0][-2:], ["--parallel", "4"])
            with self.assertRaises(ValueError):
                memory_study.build(args)

    def test_run_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("opensn", "input.py", "xs_168g.xs"):
                (root / filename).write_text("test\n")
            (root / "input.json").write_text('{"repetitions": 1}\n')
            args = memory_study.argparse.Namespace(
                case=root, binary=root / "opensn", trim=False, smaps=True, sample_seconds=15,
                launcher=["--", "mpirun", "--np", "2"])
            with patch.object(memory_study.subprocess, "run") as run:
                run.side_effect = self.complete_run
                memory_study.run_case(args)
            self.assertEqual(run.call_args.args[0][:3], ["mpirun", "--np", "2"])
            self.assertEqual(run.call_args.kwargs["env"]["OPENSN_MEMORY_TRIM"], "0")
            self.assertEqual(run.call_args.kwargs["env"]["OPENSN_MEMORY_SMAPS"], "1")
            self.assertEqual(run.call_args.kwargs["env"]["OPENSN_MEMORY_SAMPLE_SECONDS"], "15")
            self.assertEqual((root / "exit_code.txt").read_text(), "0\n")
            with self.assertRaises(FileExistsError):
                memory_study.run_case(args)

    def test_sampling_requires_explicit_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("opensn", "input.py", "xs_168g.xs"):
                (root / filename).write_text("test\n")
            (root / "input.json").write_text('{"repetitions": 1}\n')
            args = memory_study.argparse.Namespace(
                case=root, binary=root / "opensn", trim=False, smaps=False, sample_seconds=None,
                launcher=["mpirun", "--np", "2"])
            with patch.dict(os.environ, OPENSN_MEMORY_SAMPLE_SECONDS="1"):
                with patch.object(memory_study.subprocess, "run") as run:
                    run.side_effect = self.complete_run
                    memory_study.run_case(args)
            self.assertNotIn("OPENSN_MEMORY_SAMPLE_SECONDS", run.call_args.kwargs["env"])

    def test_profile_modes_are_isolated(self):
        for mode in ("baseline", "caliper"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for filename in ("opensn", "input.py", "xs_168g.xs", "caliper.txt"):
                    (root / filename).write_text("test\n")
                (root / "input.json").write_text('{"repetitions": 1}\n')
                args = memory_study.argparse.Namespace(
                    case=root, binary=root / "opensn", mode=mode, trim=False,
                    smaps=False, sample_seconds=None, launcher=["mpirun", "--np", "2"])
                with patch.dict(os.environ, OPENSN_MEMORY_TRACE_DIR="stale", CALI_CONFIG="stale"):
                    with patch.object(memory_study.subprocess, "run") as run:
                        run.side_effect = self.complete_run
                        memory_study.run_case(args)
                env = run.call_args.kwargs["env"]
                self.assertFalse(any(k.startswith("OPENSN_MEMORY_") for k in env))
                self.assertEqual(bool(env["CALI_CONFIG"]), mode == "caliper")
                self.assertTrue((root / "SUCCESS").is_file())

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("opensn", "input.py", "xs_168g.xs"):
                (root / filename).write_text("test\n")
            (root / "input.json").write_text('{"repetitions": 17}\n')
            args = memory_study.argparse.Namespace(
                case=root, binary=root / "opensn", trim=False,
                smaps=False, sample_seconds=None, launcher=["mpirun", "--np", "2"])
            with patch.object(memory_study.subprocess, "run") as run:
                run.side_effect = self.complete_run
                with self.assertRaisesRegex(RuntimeError, "Incomplete trial"):
                    memory_study.run_case(args)
            self.assertFalse((root / "SUCCESS").exists())

    def test_rocprof_rank_selection(self):
        for rank in ("0", "1"):
            with tempfile.TemporaryDirectory() as directory:
                args = memory_study.argparse.Namespace(
                    binary=Path("/build/opensn"), output=Path(directory))
                with patch.dict(os.environ, {"FLUX_TASK_RANK": rank}, clear=True), \
                        patch.object(memory_study.shutil, "which", return_value="/bin/rocprofv3"), \
                        patch.object(memory_study.subprocess, "run") as run, \
                        patch.object(memory_study.os, "execvp") as execute:
                    memory_study.rocprof_rank(args)
                command = execute.call_args.args[1]
                if rank == "0":
                    self.assertEqual(command[0], "/bin/rocprofv3")
                    self.assertIn("--memory-allocation-trace", command)
                    self.assertEqual(command[-4:], ["--", "/build/opensn", "-i", "input.py"])
                    self.assertEqual(run.call_args.args[0], ["/bin/rocprofv3", "--version"])
                else:
                    self.assertEqual(command, ["/build/opensn", "-i", "input.py"])
                    run.assert_not_called()

    def test_rocprof_requires_trace_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("opensn", "input.py", "xs_168g.xs"):
                (root / filename).write_text("test\n")
            (root / "input.json").write_text('{"repetitions": 1, "gpu": true}\n')
            args = memory_study.argparse.Namespace(
                case=root, binary=root / "opensn", mode="rocprof", trim=False,
                smaps=False, sample_seconds=None, launcher=["mpirun", "--np", "2"])
            with patch.object(memory_study.shutil, "which", return_value="/bin/rocprofv3"), \
                    patch.object(memory_study.subprocess, "run") as run:
                run.side_effect = self.complete_run
                with self.assertRaisesRegex(RuntimeError, "did not produce"):
                    memory_study.run_case(args)
                self.assertIn("rocprof-rank", run.call_args.args[0])
            self.assertFalse((root / "SUCCESS").exists())

    def test_rocprof_requires_rank(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Cannot determine"):
                memory_study.rocprof_rank(SimpleNamespace())

    def test_rocprof_complete_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("opensn", "input.py", "xs_168g.xs"):
                (root / filename).write_text("test\n")
            (root / "input.json").write_text('{"repetitions": 1, "gpu": true}\n')
            args = memory_study.argparse.Namespace(
                case=root, binary=root / "opensn", mode="rocprof", trim=False,
                smaps=False, sample_seconds=None, launcher=["mpirun", "--np", "2"])

            def capture(*args, **kwargs):
                output = root / "rocprof" / "rank-0"
                output.mkdir(parents=True)
                for name in ("hip_api", "kernel", "memory_copy", "memory_allocation"):
                    (output / f"123_{name}_trace.csv").write_text("header\nrecord\n")
                return self.complete_run(*args, **kwargs)

            with patch.object(memory_study.shutil, "which", return_value="/bin/rocprofv3"), \
                    patch.object(memory_study.subprocess, "run", side_effect=capture):
                memory_study.run_case(args)
            self.assertTrue((root / "SUCCESS").exists())

    def test_launcher_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("opensn", "input.py", "xs_168g.xs"):
                (root / filename).write_text("test\n")
            args = memory_study.argparse.Namespace(
                case=root, binary=root / "opensn", mode="baseline", trim=False,
                smaps=False, sample_seconds=None, launcher=["mpirun", "--np", "2"])
            with patch.object(memory_study.subprocess, "run", return_value=SimpleNamespace(
                    returncode=99)):
                with self.assertRaises(SystemExit):
                    memory_study.run_case(args)
            self.assertEqual((root / "exit_code.txt").read_text(), "99\n")
            self.assertFalse((root / "SUCCESS").exists())


if __name__ == "__main__":
    unittest.main()
