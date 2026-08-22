"""Tests for the deliberately simple Tuolumne CBCD study workflow."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).with_name("study.py")
SPEC = importlib.util.spec_from_file_location("tuo_study", MODULE_PATH)
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def make_executable(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o700)
    return path


def prepare_args(root, **updates):
    binary = make_executable(root / "opensn")
    environment = root / "env.zsh"
    environment.write_text("export TEST_ENV=1\n")
    values = {
        "binary": binary,
        "environment": environment,
        "output": root / "study",
        "mesh_dir": root / "meshes",
        "label": "update-3",
        "nodes": (1, 2, 4),
        "kinds": ("strong",),
        "repetitions": 1,
        "queue": "pdebug",
        "bank": "bank",
        "time_limit": "60m",
        "worker_policy": "hardware",
        "cbcd_workers": None,
        "strong_divisor": 39,
        "max_iterations": 2,
        "save_angular_flux": False,
        "refresh": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def create_meshes(directory, divisors):
    directory.mkdir(parents=True, exist_ok=True)
    for divisor in divisors:
        (directory / f"cube-d{divisor}.msh").write_text(f"mesh {divisor}\n")


def result_values(nodes=1, iterations=8, workers=192, sweep=None):
    return {
        "kind": "strong",
        "nodes": nodes,
        "avg_sweep_time_s": sweep if sweep is not None else 1.0 / nodes,
        "unknowns": 1024,
        "lagged_unknowns": nodes,
        "wgs_status": "iteration_limit",
        "wgs_iterations": iterations,
        "scheduler_workers": workers,
        "final_residual": 1.0e-4,
        "wall_time_s": 2.0,
        "launcher_max_rss_kb": 4096,
        "scalar_flux_max_g0": 0.50758,
        "scalar_flux_max_g63": 2.52527e-4,
    }


class MeshTests(unittest.TestCase):
    def test_existing_plain_meshes_are_reused_without_running_a_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_meshes(root, (6, 39))
            with mock.patch.object(subprocess, "run") as run:
                meshes = STUDY.required_meshes(root, {39, 6})
            run.assert_not_called()
            self.assertEqual(meshes[6], (root / "cube-d6.msh").resolve())
            self.assertEqual(meshes[39], (root / "cube-d39.msh").resolve())

    def test_all_missing_meshes_are_reported_before_study_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_meshes(root / "meshes", (6,))
            args = prepare_args(
                root,
                nodes=(1, 2),
                kinds=("strong", "weak"),
                mesh_dir=root / "meshes",
            )
            with self.assertRaisesRegex(RuntimeError, "cube-d8.msh") as caught:
                STUDY.prepare(args)
            self.assertIn("cube-d39.msh", str(caught.exception))
            self.assertFalse(args.output.exists())


class PreparationTests(unittest.TestCase):
    def test_interactive_strong_study_uses_one_mesh_and_modern_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prepare_args(root)
            create_meshes(args.mesh_dir, (39,))
            STUDY.prepare(args)

            record = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(record["nodes"], [1, 2, 4])
            self.assertEqual(record["kinds"], ["strong"])
            self.assertEqual(record["worker_policy"], "hardware")
            self.assertEqual(len(record["cases"]), 3)
            job = (args.output / "jobs/strong-2.zsh").read_text()
            self.assertIn("#flux: -N 2", job)
            self.assertIn("#flux: -n 8", job)
            self.assertIn("flux run -N 2 -n 8 --exclusive -o exit-on-error", job)
            self.assertIn("export OPENSN_CBCD_WORKER_POLICY=hardware", job)
            self.assertIn("unset OPENSN_CBCD_NUM_WORKERS", job)
            self.assertIn('result="$result_root/run-$job_tag-$started-$$"', job)
            self.assertIn("job_tag=${FLUX_JOB_ID:-allocation}", job)
            self.assertIn('flux_job_id=${FLUX_JOB_ID:-unset}', job)
            self.assertNotIn("FLUX_JOB_ID:?", job)
            self.assertNotRegex(job, r"(?m)^(?:\s*local\s+)?status=")
            self.assertNotIn("amd-gpumode", job)
            self.assertNotIn("setattr=gpumode", job)
            self.assertNotRegex(job, r"(?:^|\s)-[cg](?:\s|=|[0-9])")
            compile(
                (args.output / "inputs/strong-2.py").read_text(),
                "strong-2.py",
                "exec",
            )
            syntax = subprocess.run(
                ["zsh", "-n", str(args.output / "jobs/strong-2.zsh")],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_resource_aware_fixed_worker_count_is_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prepare_args(
                root,
                worker_policy="resource-aware",
                cbcd_workers=20,
            )
            create_meshes(args.mesh_dir, (39,))
            STUDY.prepare(args)
            job = (args.output / "jobs/strong-1.zsh").read_text()
            self.assertIn("export OPENSN_CBCD_WORKER_POLICY=resource-aware", job)
            self.assertIn("export OPENSN_CBCD_NUM_WORKERS=20", job)

    def test_refresh_replaces_jobs_and_preserves_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prepare_args(root)
            create_meshes(args.mesh_dir, (39,))
            STUDY.prepare(args)
            old_job = args.output / "jobs/strong-1.zsh"
            old_job.write_text("obsolete\n")
            preserved = args.output / "results/strong/nodes-1/previous-result.txt"
            preserved.parent.mkdir(parents=True)
            preserved.write_text("keep\n")

            args.refresh = True
            STUDY.prepare(args)

            self.assertNotEqual(old_job.read_text(), "obsolete\n")
            self.assertEqual(preserved.read_text(), "keep\n")

    def test_production_defaults_cover_eighteen_jobs(self):
        defaults = STUDY.parser().parse_args(
            [
                "prepare",
                "--binary",
                "/bin/true",
                "--environment",
                "/env",
                "--output",
                "/study",
                "--mesh-dir",
                "/meshes",
                "--label",
                "test",
            ]
        )
        self.assertEqual(defaults.nodes, STUDY.DEFAULT_NODES)
        self.assertEqual(defaults.kinds, ("strong", "weak"))
        self.assertEqual(len(defaults.nodes) * len(defaults.kinds), 18)
        self.assertEqual(defaults.repetitions, 3)
        self.assertEqual(defaults.max_iterations, 10)

    def test_profile_jobs_are_valid_zsh_and_keep_default_launch_unmodified(self):
        args = SimpleNamespace(
            label="profile",
            queue="pbatch",
            bank=None,
            time_limit="6h",
            environment=Path("/stack/env.zsh"),
            binary=Path("/build/python/opensn"),
            worker_policy="resource-aware",
            cbcd_workers=None,
        )
        for profile in STUDY.PROFILE_NAMES:
            with self.subTest(profile=profile):
                job = STUDY.profile_job(
                    args,
                    Path("/study"),
                    profile,
                    1,
                    Path("/study/inputs/profile.py"),
                )
                syntax = subprocess.run(
                    ["zsh", "-n"],
                    input=job,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
                self.assertNotIn("amd-gpumode", job)
                self.assertNotRegex(job, r"(?m)^(?:\s*local\s+)?status=")
                self.assertNotRegex(job, r"(?:^|\s)-[cg](?:\s|=|[0-9])")
                if profile == "caliper-mpi":
                    self.assertIn("profile.mpi", job)
                    self.assertNotIn("mpi.message.count", job)
                    self.assertNotIn("mpi.message.size", job)
                    self.assertNotIn("comm.stats", job)
                if profile == "rocprof":
                    self.assertIn('*/rank-*/*.csv', job)

    def test_multinode_profilers_preserve_layout_except_omniperf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prepare_args(
                root,
                output=root / "profile-study",
                queue="pbatch",
                worker_policy="resource-aware",
            )
            args.profile_nodes = (1, 2, 4)
            args.profile_divisor = 39
            args.profiles = ("rocprof", "hpctoolkit", "omniperf")
            create_meshes(args.mesh_dir, (39,))

            STUDY.prepare_profile(args)

            record = json.loads((args.output / "manifest.json").read_text())
            cases = {
                (case["profile"], case["nodes"], case["ranks"])
                for case in record["cases"]
            }
            self.assertIn(("rocprof", 4, 16), cases)
            self.assertIn(("hpctoolkit", 4, 16), cases)
            self.assertEqual(
                {case for case in cases if case[0] == "omniperf"},
                {("omniperf", 1, 1)},
            )

    def test_failure_trap_executes_under_zsh_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            result_root = Path(directory) / "results"
            script = (
                "set -euo pipefail\n"
                + STUDY.run_directory_setup(result_root, ("case=test",))
                + "false\n"
            )
            completed = subprocess.run(
                ["zsh"],
                input=script,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            runs = list(result_root.glob("run-*"))
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "FAILED").is_file())
            self.assertEqual((runs[0] / "job_exit_code.txt").read_text(), "1\n")

    def test_scaling_job_executes_successfully_under_zsh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prepare_args(root, nodes=(1,))
            create_meshes(args.mesh_dir, (39,))
            tools = root / "tools"
            tools.mkdir()
            flux = tools / "flux"
            flux.write_text(
                "#!/bin/zsh\n"
                "print -- 'WGS groups [0-63] final, status = iteration_limit, iterations = 2'\n"
                "print -- 'CBCD scheduler: policy=hardware, workers=192, communicator_threads=1'\n"
                "print -- 'unknowns = 1024, lagged_unknowns = 7, avg_sweep_time = 0.1 s'\n"
                "print -- 'OPENSN_TUO_SCALAR_FLUX_MAX group=0 value=0.5'\n"
                "print -- 'OPENSN_TUO_SCALAR_FLUX_MAX group=63 value=0.0002'\n"
                "print -- 'OpenSn finished execution.'\n"
            )
            flux.chmod(0o700)
            args.environment.write_text(f"export PATH={tools}:$PATH\n")
            STUDY.prepare(args)

            completed = subprocess.run(
                ["zsh", str(args.output / "jobs/strong-1.zsh")],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            runs = list((args.output / "results/strong/nodes-1").glob("run-*"))
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "SUCCESS").is_file())
            self.assertTrue((runs[0] / "trial-1/SUCCESS").is_file())


class ResultTests(unittest.TestCase):
    def test_result_requires_wgs_workers_and_scalar_flux_observables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.txt"
            stdout.write_text(
                "WGS groups [0-63] iteration = 8, residual = 1.0e-4\n"
                "WGS groups [0-63] final, status = iteration_limit, iterations = 8\n"
                "CBCD scheduler: policy=hardware, workers=192, communicator_threads=1\n"
                "unknowns = 1024, lagged_unknowns = 7, avg_sweep_time = 2.5e-2 s\n"
                "OPENSN_TUO_SCALAR_FLUX_MAX group=0 value=5.0758e-1\n"
                "OPENSN_TUO_SCALAR_FLUX_MAX group=63 value=2.52527e-4\n"
                "OpenSn finished execution.\n"
            )
            (root / "time.txt").write_text(
                "wall_seconds=1.5 launcher_max_rss_kb=4096\n"
            )
            (root / "exit_code.txt").write_text("0\n")
            (root / "SUCCESS").touch()
            result = STUDY.read_result(
                stdout,
                root / "time.txt",
                root / "exit_code.txt",
                root / "SUCCESS",
            )
            self.assertEqual(result["wgs_iterations"], 8)
            self.assertEqual(result["scheduler_workers"], 192)
            self.assertEqual(result["lagged_unknowns"], 7)

            stdout.write_text(stdout.read_text().replace("group=63", "group=1"))
            with self.assertRaisesRegex(RuntimeError, "required CBCD metrics"):
                STUDY.read_result(
                    stdout,
                    root / "time.txt",
                    root / "exit_code.txt",
                    root / "SUCCESS",
                )

    def test_iterations_may_change_across_nodes_but_not_within_one_point(self):
        rows = [
            result_values(1, iterations=8),
            result_values(1, iterations=8),
            result_values(2, iterations=11),
            result_values(2, iterations=11),
        ]
        summary = STUDY.summarize(rows)
        self.assertEqual([row["wgs_iterations"] for row in summary], [8, 11])
        rows[1]["wgs_iterations"] = 9
        with self.assertRaisesRegex(RuntimeError, "inconsistent numerical signature"):
            STUDY.summarize(rows)

    def test_roundoff_scale_flux_variation_is_measured_without_a_magic_tolerance(self):
        rows = [result_values(1) for _ in range(3)]
        flux_values = (
            5.73598477615347879e-1,
            5.73598477615347768e-1,
            5.73598477615347435e-1,
        )
        for row, value in zip(rows, flux_values):
            row["scalar_flux_max_g0"] = value

        summary = STUDY.summarize(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["scalar_flux_max_g0"], sorted(flux_values)[1])
        self.assertEqual(summary[0]["scalar_flux_max_g0_min"], min(flux_values))
        self.assertEqual(summary[0]["scalar_flux_max_g0_max"], max(flux_values))
        self.assertEqual(summary[0]["scalar_flux_max_g0_ulp_span"], 4)
        self.assertEqual(summary[0]["scalar_flux_max_g63_ulp_span"], 0)

    def test_collection_uses_all_successful_repeat_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            run_roots = [
                study / "results/strong/nodes-1/run-first",
                study / "results/strong/nodes-1/run-second",
            ]
            study.mkdir()
            (study / "manifest.json").write_text(
                json.dumps(
                    {
                        "type": "scaling",
                        "label": "test",
                        "worker_policy": "hardware",
                        "repetitions": 1,
                        "cases": [
                            {
                                "id": "strong-1",
                                "kind": "strong",
                                "nodes": 1,
                                "ranks": 4,
                            }
                        ],
                    }
                )
            )
            for index, run in enumerate(run_roots, start=1):
                trial = run / "trial-1"
                trial.mkdir(parents=True)
                run.joinpath("SUCCESS").touch()
                trial.joinpath("SUCCESS").touch()
                trial.joinpath("exit_code.txt").write_text("0\n")
                trial.joinpath("time.txt").write_text("wall_seconds=1\n")
                trial.joinpath("stdout.txt").write_text(
                    "WGS groups [0-63] iteration = 8, residual = 1e-4\n"
                    "WGS groups [0-63] final, status = iteration_limit, iterations = 8\n"
                    "CBCD scheduler: workers=192\n"
                    f"unknowns = 1024, lagged_unknowns = 7, avg_sweep_time = {index}.0 s\n"
                    "OPENSN_TUO_SCALAR_FLUX_MAX group=0 value=0.5\n"
                    "OPENSN_TUO_SCALAR_FLUX_MAX group=63 value=2e-4\n"
                    "OpenSn finished execution.\n"
                )
            STUDY.collect(
                SimpleNamespace(
                    study=study,
                    allow_incomplete=False,
                    require_monotonic=False,
                    monotonic_tolerance=0.0,
                )
            )
            with (study / "results.csv").open() as stream:
                self.assertEqual(sum(1 for _ in stream), 3)

    def test_profile_collection_reports_cases_without_a_successful_run(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "manifest.json").write_text(
                json.dumps(
                    {
                        "type": "profile",
                        "label": "test-profile",
                        "cases": [
                            {
                                "id": "pmpi-4",
                                "profile": "pmpi",
                                "nodes": 4,
                                "ranks": 16,
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "pmpi-4"):
                STUDY.collect_profile(SimpleNamespace(study=study))
            self.assertTrue((study / "profile-summary.md").is_file())


class SubmissionAndPolicyComparisonTests(unittest.TestCase):
    def test_submit_filters_scaling_jobs_without_tracking_scheduler_state(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            record = {
                "type": "scaling",
                "queue": "pbatch",
                "cases": [
                    {
                        "id": "strong-1",
                        "kind": "strong",
                        "nodes": 1,
                        "job": "/jobs/strong-1.zsh",
                    },
                    {
                        "id": "weak-2",
                        "kind": "weak",
                        "nodes": 2,
                        "job": "/jobs/weak-2.zsh",
                    },
                ],
            }
            (study / "manifest.json").write_text(json.dumps(record))
            args = SimpleNamespace(
                study=study,
                nodes=(1,),
                kinds=("strong",),
                profiles=None,
            )
            completed = subprocess.CompletedProcess([], 0, stdout="job-id\n")
            with mock.patch.object(subprocess, "run", return_value=completed) as run:
                STUDY.submit(args)
            run.assert_called_once_with(
                ["flux", "batch", "/jobs/strong-1.zsh"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

    def test_submit_filters_profile_jobs_by_nodes_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            record = {
                "type": "profile",
                "queue": "pbatch",
                "cases": [
                    {
                        "id": "pmpi-2",
                        "profile": "pmpi",
                        "nodes": 2,
                        "job": "/jobs/pmpi-2.zsh",
                    },
                    {
                        "id": "pmpi-4",
                        "profile": "pmpi",
                        "nodes": 4,
                        "job": "/jobs/pmpi-4.zsh",
                    },
                    {
                        "id": "caliper-4",
                        "profile": "caliper",
                        "nodes": 4,
                        "job": "/jobs/caliper-4.zsh",
                    },
                ],
            }
            (study / "manifest.json").write_text(json.dumps(record))
            args = SimpleNamespace(
                study=study,
                nodes=(4,),
                kinds=None,
                profiles=("pmpi",),
            )
            completed = subprocess.CompletedProcess([], 0, stdout="job-id\n")
            with mock.patch.object(subprocess, "run", return_value=completed) as run:
                STUDY.submit(args)
            run.assert_called_once_with(
                ["flux", "batch", "/jobs/pmpi-4.zsh"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

    def test_policy_compare_requires_same_point_numerics(self):
        row = STUDY.summarize([result_values(1)])[0]
        row = {name: str(value) for name, value in row.items()}
        baseline_record = {
            "label": "hardware",
            "worker_policy": "hardware",
            "nodes": [1],
            "kinds": ["strong"],
            "ranks_per_node": 4,
            "gpus_per_rank": 1,
            "gpu_mode": "SPX",
            "repetitions": 1,
            "strong_divisor": 39,
            "weak_divisors": {},
            "max_iterations": 10,
            "save_angular_flux": False,
        }
        candidate_record = dict(
            baseline_record,
            label="resource-aware",
            worker_policy="resource-aware",
        )
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                baseline=Path("/baseline"),
                candidate=Path("/candidate"),
                output=Path(directory) / "comparison",
                max_slowdown=1.03,
                monotonic_tolerance=0.0,
                residual_rtol=1.0e-6,
                residual_atol=1.0e-12,
                scalar_flux_rtol=1.0e-10,
                scalar_flux_atol=1.0e-12,
            )
            with mock.patch.object(
                STUDY,
                "read_summary",
                side_effect=[
                    (baseline_record, [row]),
                    (candidate_record, [dict(row)]),
                ],
            ), mock.patch.object(STUDY, "plot_series"):
                STUDY.compare(args)

            mismatched = dict(row, wgs_iterations="9")
            args.output = Path(directory) / "mismatch"
            with mock.patch.object(
                STUDY,
                "read_summary",
                side_effect=[
                    (baseline_record, [row]),
                    (candidate_record, [mismatched]),
                ],
            ), mock.patch.object(STUDY, "plot_series"):
                with self.assertRaisesRegex(RuntimeError, "wgs_iterations differs"):
                    STUDY.compare(args)


class SimplicityTests(unittest.TestCase):
    def test_provenance_and_content_addressing_terms_are_absent(self):
        files = (
            MODULE_PATH,
            MODULE_PATH.with_name("bootstrap.zsh"),
            MODULE_PATH.with_name("interactive_cbcd.zsh"),
            MODULE_PATH.with_name("run_cbcd_validation.zsh"),
            MODULE_PATH.with_name("README.md"),
        )
        terms = ("hash" + "lib", "sha" + "256", "check" + "sum", "finger" + "print")
        for path in files:
            source = path.read_text().lower()
            for term in terms:
                with self.subTest(path=path.name, term=term):
                    self.assertNotIn(term, source)
        source = MODULE_PATH.read_text().lower()
        for removed_option in ("--source", "--revision", "--gmsh"):
            with self.subTest(option=removed_option):
                self.assertNotIn(removed_option, source)

    def test_shell_helpers_are_valid_and_policy_order_alternates(self):
        for name in (
            "bootstrap.zsh",
            "interactive_cbcd.zsh",
            "run_cbcd_validation.zsh",
        ):
            path = MODULE_PATH.with_name(name)
            result = subprocess.run(
                ["zsh", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        helper = MODULE_PATH.with_name("interactive_cbcd.zsh").read_text()
        sequence = (
            "run_here hardware 1",
            "run_here resource-aware 1",
            "run_here resource-aware 2",
            "run_here hardware 2",
            "run_here hardware 4",
            "run_here resource-aware 4",
        )
        positions = [helper.index(command) for command in sequence]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("amd-gpumode", helper)
        self.assertNotIn("FLUX_JOB_ID:?", helper)
        self.assertNotRegex(helper, r"(?:^|\s)-[cg](?:\s|=|[0-9])")
        self.assertIn('run_here "$1" 1', helper)
        self.assertIn('prepare_one batch "$1"', helper)
        self.assertIn("prepare-profile", helper)
        self.assertIn("run-profile-interactive", helper)
        self.assertIn("resume-profile-interactive", helper)
        self.assertIn("profile_case_complete", helper)
        self.assertIn('run-profile-interactive-here "$profile" "$selected_nodes"', helper)
        self.assertIn("monitor_generated_job", helper)
        self.assertIn("rebuild-here", helper)

        runner = MODULE_PATH.with_name("run_cbcd_validation.zsh").read_text()
        self.assertIn("smoke-profile", runner)
        self.assertIn("submit-scaling", runner)
        self.assertIn("collect", runner)
        self.assertIn("run-interactive resource-aware", runner)


if __name__ == "__main__":
    unittest.main()
