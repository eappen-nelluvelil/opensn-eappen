"""Focused unit and golden tests for the Tuolumne study generator."""

import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stdout


MODULE_PATH = Path(__file__).with_name("study.py")
SPEC = importlib.util.spec_from_file_location("tuo_study", MODULE_PATH)
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def arguments():
    return SimpleNamespace(
        label="golden",
        queue="pdebug",
        bank="bank",
        time_limit="60m",
        gpu_mode="SPX",
        environment=Path("/stack/env.zsh"),
        binary=Path("/build/python/opensn"),
        revision="a" * 40,
        worker_policy="hardware",
        cbcd_workers=None,
        repetitions=3,
    )


def hashes():
    return {
        "binary": "b" * 64,
        "environment": "e" * 64,
        "input": "i" * 64,
        "mesh": "m" * 64,
        "mesh_path": Path("/study/meshes/cube.msh"),
        "xs": "x" * 64,
        "xs_path": Path("/study/assets/xs.xs"),
        "build_manifest": "j" * 64,
        "build_manifest_path": Path("/study/assets/tuo-build-manifest.json"),
        "build_provenance": [],
    }


class ParserTests(unittest.TestCase):
    def test_parse_nodes_deduplicates_and_sorts(self):
        self.assertEqual(STUDY.parse_nodes("4,1,2,4"), (1, 2, 4))

    def test_spread(self):
        self.assertEqual(STUDY.spread([1.0, 2.0, 3.0]), (2.0, 1.0, 1.0))

    def test_comparison_default_rejects_over_three_percent(self):
        args = STUDY.parser().parse_args(
            [
                "compare",
                "--baseline",
                "/baseline",
                "--candidate",
                "/candidate",
                "--output",
                "/comparison",
            ]
        )
        self.assertEqual(args.max_slowdown, 1.03)


class GoldenJobTests(unittest.TestCase):
    def test_flux_directive_tokens_reject_shell_syntax(self):
        with self.assertRaisesRegex(RuntimeError, "unsafe Flux directive"):
            STUDY.flux_header(
                "bad$(id)", 1, 4, "pbatch", "bank", "1h", "/tmp/o", "/tmp/e", "SPX"
            )

    def test_scaling_job_has_uniform_flux_and_atomic_markers(self):
        job = STUDY.scaling_job(
            arguments(),
            Path("/study"),
            "strong",
            2,
            Path("/study/inputs/strong-2.py"),
            hashes(),
        )
        self.assertIn("#flux: -N 2", job)
        self.assertIn("#flux: -n 8", job)
        self.assertNotIn("#flux: -g", job)
        self.assertIn("#flux: --exclusive", job)
        self.assertIn("#flux: --amd-gpumode=SPX", job)
        self.assertGreaterEqual(job.count("-N 2 -n 8 --exclusive -o exit-on-error"), 2)
        self.assertNotIn("flux run -N 2 -n 8 -g1", job)
        self.assertIn("export OPENSN_CBCD_WORKER_POLICY=hardware", job)
        self.assertIn('"$binary" --verbose 1 -i "$input"', job)
        self.assertIn("runtime dynamic-library closure differs", job)
        self.assertIn('touch "$result/RUNNING"', job)
        self.assertIn('mv -- "$result/RUNNING" "$result/SUCCESS"', job)
        self.assertIn('mv -- "$result/RUNNING" "$result/FAILED"', job)
        self.assertIn("claim_cpus($1, $5) != 21", job)

    def test_profile_commands_expand_result_at_runtime(self):
        setup, command = STUDY.profile_command(
            "caliper", 4, 16, Path("/study/assets/profile_rank.zsh")
        )
        self.assertEqual(setup, "")
        self.assertIn("$result/profile.txt", command)
        self.assertIn("-N 4 -n 16 --exclusive -o exit-on-error", command)
        self.assertNotIn(" -g1 ", command)
        setup, command = STUDY.profile_command(
            "rocprof", 2, 8, Path("/study/assets/profile_rank.zsh")
        )
        self.assertIn('OPENSN_PROFILE_OUTPUT="$result"', setup)
        self.assertIn("/study/assets/profile_rank.zsh", command)

    def test_every_generated_profile_job_is_valid_zsh(self):
        for profile in STUDY.PROFILE_NAMES:
            with self.subTest(profile=profile):
                job = STUDY.profile_job(
                    arguments(),
                    profile,
                    1,
                    Path("/study"),
                    Path("/study/inputs/profile.py"),
                    hashes(),
                )
                result = subprocess.run(
                    ["zsh", "-n"],
                    input=job,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class ResultTests(unittest.TestCase):
    def test_strict_result_and_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.txt"
            stdout.write_text(
                "WGS groups [0-63] iteration = 10, residual = 1.0e-4\n"
                "WGS groups [0-63] final, status = iteration_limit, iterations = 10\n"
                "CBCD scheduler: policy=hardware, workers=192, communicator_threads=1, "
                "reserved_communicator_threads=0, hardware_threads=192, "
                "affinity_threads=42, affinity_cores=21, requested_threads=0, "
                "available_threads=21.\n"
                "unknowns = 1024, lagged_unknowns = 8, avg_sweep_time = 2.5e-2 s\n"
                "OPENSN_TUO_SCALAR_FLUX_MAX group=0 value=5.07580000000000031e-01\n"
                "OPENSN_TUO_SCALAR_FLUX_MAX group=63 value=2.52527000000000010e-04\n"
                "OpenSn finished execution.\n"
            )
            (root / "time.txt").write_text(
                "wall_seconds=1.5 launcher_max_rss_kb=4096\n"
            )
            (root / "exit.txt").write_text("0\n")
            (root / "SUCCESS").touch()
            result = STUDY.read_result(
                stdout,
                root / "time.txt",
                root / "exit.txt",
                root / "SUCCESS",
            )
            self.assertEqual(result["unknowns"], 1024)
            self.assertEqual(result["wgs_iterations"], 10)
            self.assertEqual(result["scheduler_workers"], 192)
            self.assertEqual(result["scalar_flux_max_g0"], 0.50758)

            lines = []
            for host_index in range(2):
                for local_rank in range(4):
                    rank = host_index * 4 + local_rank
                    first_cpu = local_rank * 21
                    cpus = f"{first_cpu}-{first_cpu + 20}"
                    lines.append(
                        f"host{host_index} {rank} {local_rank} "
                        f"{local_rank} {cpus} 21"
                    )
            binding = root / "binding.txt"
            binding.write_text("\n".join(lines) + "\n")
            STUDY.validate_binding(binding, 2, 8)

            binding.write_text("host0 0 0 3 42-62 21\n")
            STUDY.validate_binding(binding, 1, 1)

            binding.write_text(
                "host0 0 0 0 0-20 21\n"
                "host0 1 1 1 20-40 21\n"
            )
            with self.assertRaisesRegex(RuntimeError, "overlapping per-node CPU"):
                STUDY.validate_binding(binding, 1, 2)

            stdout.write_text(
                stdout.read_text().replace(
                    "OpenSn finished execution.\n",
                    "OPENSN_TUO_SCALAR_FLUX_MAX group=1 value=1.0e-3\n"
                    "OpenSn finished execution.\n",
                )
            )
            with self.assertRaisesRegex(RuntimeError, "exactly groups"):
                STUDY.read_result(
                    stdout,
                    root / "time.txt",
                    root / "exit.txt",
                    root / "SUCCESS",
                )

    def test_missing_finished_marker_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stdout.txt").write_text(
                "WGS groups [0] iteration = 1, residual = 1e-2\n"
                "WGS groups [0] final, status = iteration_limit, iterations = 1\n"
                "unknowns = 1, avg_sweep_time = 1 s\n"
            )
            (root / "time.txt").write_text("wall_seconds=1\n")
            (root / "exit.txt").write_text("0\n")
            (root / "SUCCESS").touch()
            with self.assertRaises(RuntimeError):
                STUDY.read_result(
                    root / "stdout.txt",
                    root / "time.txt",
                    root / "exit.txt",
                    root / "SUCCESS",
                )

    def test_summary_statistics_and_monotonic_gate(self):
        rows = []
        for nodes, samples in ((1, (2.0, 4.0, 6.0)), (2, (1.0, 2.0, 3.0))):
            for sample in samples:
                rows.append(
                    {
                        "kind": "strong",
                        "nodes": nodes,
                        "avg_sweep_time_s": sample,
                        "unknowns": 100,
                        "lagged_unknowns": 7,
                        "wgs_status": "iteration_limit",
                        "wgs_iterations": 10,
                        "scheduler_workers": 96,
                        "final_residual": 1.0e-4,
                        "wall_time_s": sample + 1.0,
                        "scalar_flux_max_g0": 0.50758,
                        "scalar_flux_max_g63": 2.52527e-4,
                    }
                )
        summary = STUDY.summarize(rows)
        self.assertEqual(summary[0]["median_avg_sweep_time_s"], 4.0)
        self.assertEqual(summary[0]["avg_sweep_time_mad_s"], 2.0)
        self.assertEqual(summary[0]["avg_sweep_time_iqr_s"], 2.0)
        self.assertEqual(summary[0]["median_lagged_unknowns"], 7)
        self.assertEqual(STUDY.monotonic_failures(summary, 0.0), [])

        summary[1]["median_avg_sweep_time_s"] = 5.0
        self.assertEqual(len(STUDY.monotonic_failures(summary, 0.0)), 1)

    def test_iterations_may_differ_across_node_counts_only(self):
        rows = []
        for nodes, iterations in ((1, 8), (2, 11)):
            for _ in range(2):
                rows.append(
                    {
                        "kind": "strong",
                        "nodes": nodes,
                        "avg_sweep_time_s": 1.0 / nodes,
                        "unknowns": 100,
                        "lagged_unknowns": nodes,
                        "wgs_status": "converged",
                        "wgs_iterations": iterations,
                        "scheduler_workers": 96,
                        "final_residual": 1.0e-12,
                        "wall_time_s": 2.0,
                        "scalar_flux_max_g0": 0.5,
                        "scalar_flux_max_g63": 2.0e-4,
                    }
                )
        summary = STUDY.summarize(rows)
        self.assertEqual([row["wgs_iterations"] for row in summary], [8, 11])

        rows[1]["wgs_iterations"] = 9
        with self.assertRaisesRegex(RuntimeError, "inconsistent numerical signature"):
            STUDY.summarize(rows)

    def test_scalar_flux_mismatch_fails_comparison(self):
        row = {
            "kind": "strong",
            "nodes": "1",
            "trials": "3",
            "metric": "1.0",
            "median_avg_sweep_time_s": "1.0",
            "median_unknowns": "100",
            "median_lagged_unknowns": "2",
            "wgs_status": "converged",
            "wgs_iterations": "8",
            "scheduler_workers": "96",
            "median_final_residual": "1e-12",
            "scalar_flux_max_g0": "0.5",
            "scalar_flux_max_g63": "2e-4",
        }
        candidate = dict(row, scalar_flux_max_g63="3e-4")
        manifest = {
            "label": "test",
            "revision": "a" * 40,
            "worker_policy": "hardware",
            "compatibility": {"same": True},
        }
        args = SimpleNamespace(
            output=None,
            baseline=Path("/baseline"),
            candidate=Path("/candidate"),
            max_slowdown=1.03,
            monotonic_tolerance=0.0,
            residual_rtol=1.0e-6,
            residual_atol=1.0e-12,
            scalar_flux_rtol=1.0e-10,
            scalar_flux_atol=1.0e-12,
            allow_worker_policy_difference=False,
            allow_nonhardware_baseline=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "comparison"
            with mock.patch.object(
                STUDY,
                "read_collected",
                side_effect=[(manifest, [row]), (manifest, [candidate])],
            ):
                with self.assertRaisesRegex(RuntimeError, "scalar-flux maximum"):
                    STUDY.compare(args)


class SubmissionTests(unittest.TestCase):
    def test_incompatible_filters_and_pdebug_are_rejected(self):
        args = SimpleNamespace(
            study=Path("/study"),
            nodes=None,
            kinds=None,
            profiles=("omniperf",),
            resubmit=False,
        )
        scaling = {"type": "scaling", "queue": "pbatch"}
        with mock.patch.object(STUDY, "load_manifest", return_value=(args.study, scaling)):
            with mock.patch.object(STUDY, "verify_study_files"):
                with self.assertRaisesRegex(RuntimeError, "--profiles"):
                    STUDY.submit(args)

        args.profiles = None
        args.kinds = ("strong",)
        profile = {"type": "profile", "queue": "pbatch"}
        with mock.patch.object(STUDY, "load_manifest", return_value=(args.study, profile)):
            with mock.patch.object(STUDY, "verify_study_files"):
                with self.assertRaisesRegex(RuntimeError, "--kinds"):
                    STUDY.submit(args)

        args.kinds = None
        scaling["queue"] = "pdebug"
        with mock.patch.object(STUDY, "load_manifest", return_value=(args.study, scaling)):
            with mock.patch.object(STUDY, "verify_study_files"):
                with self.assertRaisesRegex(RuntimeError, "interactive-only"):
                    STUDY.submit(args)

    def test_pdebug_submit_wrapper_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            STUDY.write_submit_wrapper(stage, Path("/study"), "pdebug")
            wrapper = (stage / "submit.zsh").read_text()
            self.assertIn("interactive-only", wrapper)
            self.assertNotIn("exec python", wrapper)

    def test_resubmit_replaces_an_invalid_success_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            attempt = study / "results/strong/nodes-1/job-old"
            attempt.mkdir(parents=True)
            (attempt / "SUCCESS").touch()
            job = study / "jobs/strong-1.zsh"
            job.parent.mkdir()
            job.write_text("#!/bin/zsh\n")
            manifest = {
                "type": "scaling",
                "queue": "pbatch",
                "repetitions": 1,
                "cases": [
                    {
                        "id": "strong-1",
                        "category": "scaling",
                        "kind": "strong",
                        "nodes": 1,
                        "ranks": 4,
                        "job": str(job),
                    }
                ],
            }
            args = SimpleNamespace(
                study=study,
                nodes=None,
                kinds=None,
                profiles=None,
                resubmit=False,
            )
            with mock.patch.object(STUDY, "load_manifest", return_value=(study, manifest)):
                with mock.patch.object(STUDY, "verify_study_files"):
                    with self.assertRaisesRegex(RuntimeError, "invalid SUCCESS"):
                        STUDY.submit(args)

            args.resubmit = True
            with mock.patch.object(STUDY, "load_manifest", return_value=(study, manifest)):
                with mock.patch.object(STUDY, "verify_study_files"):
                    with mock.patch.object(
                        STUDY, "command_output", return_value="ƒABC123"
                    ) as command:
                        with redirect_stdout(io.StringIO()):
                            STUDY.submit(args)
            command.assert_called_once_with(["flux", "batch", str(job)])

    def test_collected_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "manifest.json").write_text(
                '{"schema_version": 2, "files": {}}\n'
            )
            for name in ("results.csv", "summary.csv", "summary.md"):
                (study / name).write_text("original\n")
            attempt_file = study / "results/strong/nodes-1/job-1/stdout.txt"
            attempt_file.parent.mkdir(parents=True)
            attempt_file.write_text("output\n")
            collection = {
                "complete": True,
                "artifacts": {
                    name: STUDY.sha256(study / name)
                    for name in ("results.csv", "summary.csv", "summary.md")
                },
                "attempt_artifacts": {
                    str(attempt_file.relative_to(study)): STUDY.sha256(attempt_file)
                },
            }
            STUDY.atomic_json(study / "collection.json", collection)
            (study / "summary.csv").write_text("tampered\n")
            with mock.patch.object(STUDY, "verify_study_files"):
                with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                    STUDY.read_collected(study)


class TemplateTests(unittest.TestCase):
    def test_transport_template_is_syntactically_valid_after_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "transport.py"
            STUDY.write_input(
                generated,
                Path(__file__).with_name("transport.py.in"),
                Path("/mesh.msh"),
                Path("/xs.xs"),
                10,
                False,
            )
            compile(generated.read_text(), str(generated), "exec")
            self.assertIn("OPENSN_TUO_SCALAR_FLUX_MAX", generated.read_text())

    def test_embedded_python_is_syntactically_valid(self):
        scripts = ("bootstrap.zsh", "interactive_cbcd.zsh")
        total = 0
        for name in scripts:
            script = Path(__file__).with_name(name)
            programs = []
            active = None
            for line in script.read_text().splitlines():
                if active is None and "<<'PY'" in line:
                    active = []
                elif active is not None and line == "PY":
                    programs.append("\n".join(active) + "\n")
                    active = None
                elif active is not None:
                    active.append(line)
            self.assertIsNone(active)
            total += len(programs)
            for index, program in enumerate(programs):
                compile(program, f"{script}:heredoc-{index}", "exec")
        self.assertGreaterEqual(total, 7)


if __name__ == "__main__":
    unittest.main()
