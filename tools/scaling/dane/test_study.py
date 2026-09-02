#!/usr/bin/env python3

import os
import tempfile
import unittest
from pathlib import Path
from subprocess import run
from unittest import mock

import study


SCRIPT_DIR = Path(__file__).resolve().parent


class StudyTest(unittest.TestCase):
    def test_prepare_writes_complete_study_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "campaign"
            geometry = base / "cube.geo"
            cross_sections = base / "xs.xs"
            geometry.write_text("geometry")
            cross_sections.write_text("cross sections")
            args = study.make_parser().parse_args(
                [
                    "prepare",
                    "--root",
                    str(root),
                    "--bank",
                    "example",
                    "--source",
                    str(base / "source"),
                    "--sha",
                    "a" * 40,
                    "--build",
                    str(base / "build"),
                    "--geometry",
                    str(geometry),
                    "--cross-sections",
                    str(cross_sections),
                    "--nodes",
                    "1,2",
                ]
            )

            def fake_gmsh(command, *, capture=False):
                del capture
                Path(command[command.index("-o") + 1]).touch()
                return ""

            with mock.patch.object(study, "run", side_effect=fake_gmsh):
                args.action(args)

            jobs = sorted((root / "jobs").glob("*.sbatch"))
            self.assertEqual(len(jobs), 5)
            self.assertTrue((root / "inputs" / "strong-1.py").is_file())
            self.assertTrue((root / "inputs" / "weak-2.py").is_file())
            for job in jobs:
                self.assertEqual(
                    run(["bash", "-n", str(job)], capture_output=True).returncode,
                    0,
                )

    def test_parse_measurement_uses_final_timing_record(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            (trial / "completed").touch()
            stdout = trial / "stdout.txt"
            stdout.write_text(
                "unknowns=100 lagged_unknowns=3\n"
                "avg_sweep_time=2.5e-1 s sweep_time_per_unknown=2.5e+6 ns\n"
                "unknowns=200 lagged_unknowns=4\n"
                "avg_sweep_time=1.0e-1 s sweep_time_per_unknown=5.0e+5 ns\n"
            )
            result = study.parse_measurement(stdout, "branch", "strong", 2, 64, 1)
            self.assertIsNotNone(result)
            self.assertEqual(result.ranks, 128)
            self.assertEqual(result.unknowns, 200)
            self.assertEqual(result.lagged_unknowns, 4)
            self.assertEqual(result.average_sweep_seconds, 0.1)
            self.assertEqual(result.sweep_nanoseconds_per_unknown, 5.0e5)

    def test_strong_and_weak_efficiencies_have_distinct_ideals(self):
        measurements = []
        for kind, node_one, node_two in (
            ("strong", 4.0, 2.0),
            ("weak", 4.0, 5.0),
        ):
            for nodes, value in ((1, node_one), (2, node_two)):
                measurements.append(
                    study.Measurement(
                        implementation="branch",
                        kind=kind,
                        nodes=nodes,
                        ranks=64 * nodes,
                        trial=1,
                        average_sweep_seconds=value,
                        sweep_nanoseconds_per_unknown=value,
                        unknowns=100 * nodes,
                        lagged_unknowns=0,
                        stdout=Path("stdout.txt"),
                    )
                )
        rows = study.summarize(measurements)
        values = {(row["kind"], row["nodes"]): row for row in rows}
        self.assertEqual(values[("strong", 2)]["efficiency_percent"], 100.0)
        self.assertEqual(values[("weak", 2)]["efficiency_percent"], 80.0)

    def test_generated_input_selects_host_cbc_and_fixed_iterations(self):
        text = study.make_input(Path("mesh.msh"), Path("xs.xs"))
        compile(text, "generated-input.py", "exec")
        self.assertIn('sweep_type="CBC"', text)
        self.assertIn('"allow_cycles": True', text)
        self.assertIn('"save_angular_flux": False', text)
        self.assertIn('"l_abs_tol": 1.0e-12', text)
        self.assertIn('"l_max_its": 10', text)

    def test_generated_jobs_are_native_and_use_64_ranks_per_node(self):
        manifest = {
            "root": "/tmp/campaign",
            "environment": "",
            "build_jobs": 16,
            "bank": "example",
            "build_time_limit": "01:00:00",
            "time_limit": "01:00:00",
            "ranks_per_node": 64,
            "repetitions": 3,
            "implementations": {
                implementation: {
                    "label": implementation,
                    "sha": implementation * 8,
                    "source": f"/tmp/{implementation}-source",
                    "build": f"/tmp/{implementation}-build",
                    "binary": f"/tmp/{implementation}-build/python/opensn",
                }
                for implementation in study.IMPLEMENTATIONS
            },
        }
        build = study.make_build_job(manifest)
        job = study.make_study_job(manifest, "branch", "strong", 4)
        self.assertIn("-DCMAKE_BUILD_TYPE=Native", build)
        self.assertIn("-DOPENSN_WITH_CUDA=OFF", build)
        self.assertIn("#SBATCH --ntasks-per-node=64", job)
        self.assertIn("--ntasks=256", job)
        self.assertIn("--mpibind=on", job)
        self.assertIn("export OPENSN_NUM_THREADS=1", job)
        self.assertEqual(run(["bash", "-n"], input=build, text=True).returncode, 0)
        self.assertEqual(run(["bash", "-n"], input=job, text=True).returncode, 0)

    def test_dane_shell_entrypoints_are_valid(self):
        for script in ("bootstrap_opensn.zsh", "run_cbc_scaling.zsh"):
            result = run(
                ["zsh", "-n", str(SCRIPT_DIR / script)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_bootstrap_forces_private_dependencies_and_bounded_builds(self):
        text = (SCRIPT_DIR / "bootstrap_opensn.zsh").read_text()
        packages = ("mpicpp-lite", "Boost", "PETSc", "HDF5", "VTK", "caliper")
        for package in packages:
            self.assertIn(f"-DCMAKE_DISABLE_FIND_PACKAGE_{package}=TRUE", text)
        self.assertIn('cmake --build "$OPENSN_DANE_DEPS_BUILD" --parallel 1', text)
        self.assertIn("-DCMAKE_BUILD_TYPE=Native", text)
        self.assertIn("--partition=pdebug", text)
        self.assertIn("--exclusive", text)
        self.assertIn("cmake/3.30.5", text)
        self.assertIn("export CC=$mpi_cc", text)
        self.assertIn("export CXX=$mpi_cxx", text)

        dependency_recipe = SCRIPT_DIR.parents[1] / "dependencies" / "CMakeLists.txt"
        self.assertNotIn("--download-cmake=yes", dependency_recipe.read_text())

    def test_runner_default_environment_path_has_no_whitespace(self):
        environment = os.environ.copy()
        environment.pop("OPENSN_DANE_ENVIRONMENT", None)
        environment["OPENSN_DANE_WORK_ROOT"] = "/tmp/dane-work"
        result = run(
            [
                "zsh",
                str(SCRIPT_DIR / "run_cbc_scaling.zsh"),
                "paths",
                "test-label",
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "environment=/tmp/dane-work/toolchains/isolated-1/opensn-dane-env.sh",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
