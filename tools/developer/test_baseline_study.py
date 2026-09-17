# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

import contextlib
import gc
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import baseline_study as study


class BaselineStudyTest(unittest.TestCase):
    def test_problem_lifetimes_and_settings(self):
        for gpu in (False, True):
            with self.subTest(gpu=gpu):
                references = []
                arguments = []
                executions = []

                class Mesh:
                    def __init__(self, **kwargs):
                        pass

                    def Execute(self):
                        return self

                    def SetOrthogonalBoundaries(self):
                        pass

                    def LoadFromOpenSn(self, path):
                        pass

                class Problem:
                    def __init__(self, **kwargs):
                        self.arguments = kwargs
                        self_check = all(ref() is None for ref in references)
                        if not self_check:
                            raise RuntimeError("Overlapping problem lifetimes")
                        arguments.append(kwargs)
                        references.append(weakref.ref(self))

                class Solver:
                    def __init__(self, problem):
                        self.problem = problem

                    def Initialize(self):
                        pass

                    def Execute(self):
                        executions.append(True)

                scope = dict(opensn_console=True, rank=0, DistributedMeshGenerator=Mesh,
                             FromFileMeshGenerator=Mesh, MultiGroupXS=Mesh,
                             GLCProductQuadrature3DXYZ=Mesh, DiscreteOrdinatesProblem=Problem,
                             SteadyStateSourceSolver=Solver)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(study.render(Path("cube.msh"), gpu, 17, 16), scope)
                gc.collect()
                self.assertEqual(len(executions), 17)
                self.assertTrue(all(ref() is None for ref in references))
                self.assertEqual(output.getvalue().count("BASELINE_TRIAL_END"), 17)
                for args in arguments:
                    self.assertEqual(args["sweep_type"], "CBC")
                    self.assertEqual(args["use_gpus"], gpu)
                    self.assertFalse(args["options"]["save_angular_flux"])
                    group = args["groupsets"][0]
                    self.assertFalse(group["allow_cycles"])
                    self.assertEqual(group["l_max_its"], 16)
                    self.assertEqual(group["angle_aggregation_type"], "single")

    def test_environment(self):
        env = study.baseline_environment(dict(
            CALI_CONFIG="runtime-report", CALI_MEMORY_POOL_SIZE="1",
            LD_PRELOAD="libprofiler.so", OPENSN_MEMORY_TRACE_DIR="trace",
            OPENSN_CBCD_NUM_WORKERS="8", HSA_TOOLS_LIB="profiler",
            OPENSN_NUM_THREADS="21", MPICH_GPU_SUPPORT_ENABLED="1",
            HIP_VISIBLE_DEVICES="2", PATH="/bin"))
        self.assertEqual(env["CALI_CONFIG"], "")
        self.assertEqual(env["CALI_SERVICES_ENABLE"], "")
        self.assertEqual(env["CALI_CONFIG_FILE"], "/dev/null")
        for key in ("CALI_MEMORY_POOL_SIZE", "LD_PRELOAD", "OPENSN_MEMORY_TRACE_DIR",
                    "OPENSN_CBCD_NUM_WORKERS", "HSA_TOOLS_LIB"):
            self.assertNotIn(key, env)
        self.assertEqual(env["OPENSN_NUM_THREADS"], "21")
        self.assertEqual(env["MPICH_GPU_SUPPORT_ENABLED"], "1")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "2")

    def test_incomplete_trial_is_excluded(self):
        timing = "[0] avg_sweep_time = 2.0e+00 s, sweep_time_per_unknown = 5.0e-02 ns\n"
        text = "BASELINE_TRIAL_BEGIN 1/17\n" + timing + "BASELINE_TRIAL_END 1/17\n"
        text += "BASELINE_TRIAL_BEGIN 2/17\n" + timing
        self.assertEqual(study.completed_trials(text), [
            dict(trial=1, sweep_seconds=2.0, grind_ns=0.05)])
        self.assertEqual(study.completed_trials("BASELINE_TRIAL_END 1/17\n"), [])

    def test_build_reuses_stack_and_forces_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            (old / "CMakeCache.txt").write_text(
                "CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_BUILD_TYPE:STRING=Debug\n"
                "CMAKE_CXX_COMPILER:FILEPATH=/stack/clang++\n"
                "CMAKE_PREFIX_PATH:STRING=/stack/deps\n"
                "Python3_EXECUTABLE:FILEPATH=/existing/venv/bin/python\n"
                "CMAKE_HIP_ARCHITECTURES:STRING=gfx942\n"
                "OPENSN_WITH_HIP:BOOL=ON\n")
            args = SimpleNamespace(reuse_build=old, build=root / "new", jobs=4)
            with patch.object(study.subprocess, "run") as run:
                study.build(args)
            command = run.call_args_list[0].args[0]
            self.assertIn("-DCMAKE_BUILD_TYPE=Native", command)
            self.assertIn("-DPython3_EXECUTABLE:FILEPATH=/existing/venv/bin/python", command)
            self.assertIn("-DOPENSN_WITH_HIP:BOOL=ON", command)
            self.assertEqual(run.call_count, 2)
            with self.assertRaises(FileExistsError):
                study.build(args)

    def test_reject_instrumented_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeCache.txt").write_text(
                "CMAKE_GENERATOR:INTERNAL=Ninja\n"
                "CMAKE_CXX_FLAGS:STRING=-fsanitize=address\n")
            with self.assertRaises(ValueError):
                study.build(SimpleNamespace(reuse_build=root, build=root / "new", jobs=4))


if __name__ == "__main__":
    unittest.main()
