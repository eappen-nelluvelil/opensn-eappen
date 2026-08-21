import contextlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


RUNNER_PATH = pathlib.Path(__file__).resolve().parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("cbcd_profile_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def valid_output(
    ranks=4,
    maximum_0=0.5,
    maximum_19=8.0e-4,
    policy="hardware",
    workers=8,
    reserved=0,
    unknowns=1000,
    lagged_unknowns=25,
    lagged_percent=2.5,
):
    config = (
        f"CBCD_PROFILE_CONFIG ranks={ranks} cells_per_axis=18 cells=5832 "
        "groups=21 directions=32 save_angular_flux=false"
    )
    scheduler = (
        f"[0] 00:00:00.1 CBCD scheduler: policy={policy}, workers={workers}, "
        f"communicator_threads=1, reserved_communicator_threads={reserved}."
    )
    return f"""{config}
{scheduler}
[0] WGS groups [0-20] iteration = 0, residual = 1.000000e+00
[0] WGS groups [0-20] iteration = 1, residual = 1.000000e-03
[0] WGS groups [0-20] iteration = 2, residual = 1.000000e-07, status = converged
[0] WGS groups [0-20] avg_sweep_time = 5.000000e-02 s, sweep_time_per_unknown = 1.250000e+01 ns
[0] WGS groups [0-20] unknowns = {unknowns}, lagged_unknowns = {lagged_unknowns}, lagged_pct = {lagged_percent}
CBCD_PROFILE_MAX group=0 value={maximum_0:.12e}
CBCD_PROFILE_MAX group=19 value={maximum_19:.12e}
2026-08-21 OpenSn finished execution.
"""


def validate(
    text, *, ranks=4, policy="hardware", expected_worker_override=None, expected_signature=None
):
    return RUNNER.validate_output(
        text,
        ranks=ranks,
        cells_per_axis=18,
        expected_policy=policy,
        expected_worker_override=expected_worker_override,
        expected_signature=expected_signature,
        max_absolute_tolerance=1.0e-10,
        max_relative_tolerance=1.0e-6,
    )


class ParserAndSignatureTests(unittest.TestCase):
    def test_valid_output_returns_rank_signature_and_actual_workers(self):
        parsed = validate(valid_output())
        self.assertEqual(parsed["wgs_final_iteration"], 2)
        self.assertEqual(parsed["wgs_iteration_count"], 3)
        self.assertEqual(parsed["maxima"], {"0": 0.5, "19": 8.0e-4})
        self.assertEqual(parsed["config"]["ranks"], 4)
        self.assertEqual(parsed["scheduler"]["workers"], 8)
        self.assertEqual(parsed["signature"]["unknowns"], 1000)
        self.assertEqual(parsed["signature"]["lagged_unknowns"], 25)

    def test_same_rank_signature_checks_maxima_iterations_and_unknown_counts(self):
        signature = RUNNER.RankSignature(
            maxima={0: 0.5, 19: 8.0e-4},
            wgs_final_iteration=2,
            wgs_iteration_count=3,
            unknowns=1000,
            lagged_unknowns=25,
        )
        with self.assertRaisesRegex(RUNNER.StudyError, "maximum mismatch for group 19"):
            validate(valid_output(maximum_19=2.0e-3), expected_signature=signature)
        with self.assertRaisesRegex(RUNNER.StudyError, "lagged-unknown-count mismatch"):
            validate(valid_output(lagged_unknowns=26, lagged_percent=2.6), expected_signature=signature)

    def test_rank_indexed_schema_allows_different_rank_invariants(self):
        one = RUNNER.RankSignature(
            maxima={0: 0.5, 19: 8.0e-4},
            wgs_final_iteration=5,
            wgs_iteration_count=6,
            unknowns=31352832,
            lagged_unknowns=0,
        )
        four = RUNNER.RankSignature(
            maxima={0: 0.5, 19: 8.0e-4},
            wgs_final_iteration=7,
            wgs_iteration_count=8,
            unknowns=31352832,
            lagged_unknowns=65856,
        )
        signatures = RUNNER.SignatureSet(workload={"id": "fixture"}, by_ranks={1: one, 4: four})
        restored = RUNNER.SignatureSet.from_dict(signatures.as_dict())
        self.assertEqual(restored.require_rank(1).wgs_iteration_count, 6)
        self.assertEqual(restored.require_rank(4).wgs_iteration_count, 8)
        with self.assertRaisesRegex(RUNNER.StudyError, "no entry.*2"):
            restored.require_rank(2)
        with self.assertRaisesRegex(RUNNER.StudyError, "schema_version"):
            RUNNER.SignatureSet.from_dict({"maxima": {}})

    def test_strict_signature_invariants_reject_incomplete_or_impossible_values(self):
        with self.assertRaises(RUNNER.StudyError):
            RUNNER.RankSignature(
                maxima={},
                wgs_final_iteration=2,
                wgs_iteration_count=3,
                unknowns=1000,
                lagged_unknowns=0,
            )
        with self.assertRaises(RUNNER.StudyError):
            RUNNER.RankSignature(
                maxima={0: 0.5, 19: 8.0e-4},
                wgs_final_iteration=2,
                wgs_iteration_count=2,
                unknowns=1000,
                lagged_unknowns=1001,
            )

    def test_missing_completion_noncontiguous_wgs_and_bad_percent_are_rejected(self):
        text = valid_output().replace("iteration = 1", "iteration = 3").replace(
            "2026-08-21 OpenSn finished execution.\n", ""
        )
        with self.assertRaisesRegex(RUNNER.StudyError, "completion marker"):
            validate(text)
        with self.assertRaisesRegex(RUNNER.StudyError, "lagged percentage"):
            validate(valid_output(lagged_percent=9.0))

    def test_runtime_configuration_policy_and_worker_marker_are_required(self):
        with self.assertRaisesRegex(RUNNER.StudyError, "configuration mismatch"):
            validate(valid_output(ranks=2), ranks=4)
        with self.assertRaisesRegex(RUNNER.StudyError, "worker-policy mismatch"):
            validate(valid_output(policy="resource-aware", reserved=1), policy="hardware")
        with self.assertRaisesRegex(RUNNER.StudyError, "fixed-worker mismatch"):
            validate(valid_output(workers=8), expected_worker_override=4)
        without_worker = "\n".join(
            line for line in valid_output().splitlines() if "CBCD scheduler:" not in line
        )
        with self.assertRaisesRegex(RUNNER.StudyError, "worker record"):
            validate(without_worker)


class StatisticsScheduleAndArgumentsTests(unittest.TestCase):
    def test_statistics_include_median_mad_and_interpolated_iqr(self):
        result = RUNNER.descriptive_statistics([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(result["median"], 3.0)
        self.assertEqual(result["mad"], 1.0)
        self.assertEqual(result["q1"], 2.0)
        self.assertEqual(result["q3"], 4.0)
        self.assertEqual(result["iqr"], 2.0)

    def test_pe_map_and_worker_override_require_positive_values(self):
        self.assertEqual(RUNNER.parse_pe_map("1=8,2=4,4=2"), {1: 8, 2: 4, 4: 2})
        with self.assertRaises(Exception):
            RUNNER.parse_pe_map("4=0")
        with self.assertRaises(Exception):
            RUNNER.positive_int("0")

    def test_ranks_baseline_and_pe_map_define_one_complete_ordered_design(self):
        parser = RUNNER.build_parser()

        def arguments(*extra):
            return parser.parse_args(
                [
                    "benchmark",
                    "--dry-run",
                    "--binary",
                    "/bin/true",
                    *extra,
                ]
            )

        with self.assertRaisesRegex(RUNNER.StudyError, "strictly increasing"):
            RUNNER.prepare_arguments(arguments("--ranks", "2,1"))
        with self.assertRaisesRegex(RUNNER.StudyError, "baseline-ranks"):
            RUNNER.prepare_arguments(arguments("--ranks", "2,4", "--baseline-ranks", "1"))
        with self.assertRaisesRegex(RUNNER.StudyError, "exactly the requested ranks"):
            RUNNER.prepare_arguments(
                arguments("--ranks", "1,2,4", "--pe-map", "1=8,2=4")
            )
        valid = arguments(
            "--ranks",
            "1,2,4",
            "--baseline-ranks",
            "1",
            "--pe-map",
            "1=8,2=4,4=2",
        )
        RUNNER.prepare_arguments(valid)
        self.assertEqual(valid.ranks, (1, 2, 4))
        self.assertEqual(valid.pe_map, {1: 8, 2: 4, 4: 2})

    def test_balanced_schedule_is_seeded_paired_and_alternates_ab_ba(self):
        args = types.SimpleNamespace(
            policies=("hardware", "resource-aware"),
            ranks=(1, 2, 4),
            warmups=1,
            trials=4,
            schedule_seed=12345,
            workers=None,
        )
        schedule = RUNNER.build_benchmark_schedule(args)
        self.assertEqual(schedule, RUNNER.build_benchmark_schedule(args))
        trials = [item for item in schedule if not item["warmup"]]
        for ranks in args.ranks:
            orders = []
            for block in range(args.trials):
                pair = [item for item in trials if item["ranks"] == ranks and item["block"] == block]
                self.assertEqual(len(pair), 2)
                self.assertEqual(pair[1]["execution_order"], pair[0]["execution_order"] + 1)
                self.assertEqual(pair[0]["pair_id"], pair[1]["pair_id"])
                orders.append(tuple(item["policy"] for item in pair))
            self.assertEqual(orders[0], orders[2])
            self.assertEqual(orders[1], orders[3])
            self.assertEqual(orders[1], tuple(reversed(orders[0])))

        fixed_args = types.SimpleNamespace(
            policies=("hardware",),
            ranks=(1,),
            warmups=0,
            trials=1,
            schedule_seed=1,
            workers=4,
        )
        self.assertEqual(
            RUNNER.build_benchmark_schedule(fixed_args)[0]["condition"], "fixed-workers-4"
        )

    def test_nsys_command_disables_cpu_sampling_and_uses_one_metrics_collector(self):
        collector = RUNNER.build_nsys_profile_command(
            nsys="/opt/nsys",
            output=pathlib.Path("/tmp/rank0"),
            trace="cuda,mpi,nvtx",
            mpi_impl="openmpi",
            gpu_metrics=True,
            gpu_metrics_collector=True,
            gpu_device="0",
            gpu_metrics_set="gb20x",
            gpu_metrics_frequency=1000,
            target=["/tmp/opensn", "-i", "/tmp/input.py"],
        )
        peer = RUNNER.build_nsys_profile_command(
            nsys="/opt/nsys",
            output=pathlib.Path("/tmp/rank1"),
            trace="cuda,mpi,nvtx",
            mpi_impl="openmpi",
            gpu_metrics=True,
            gpu_metrics_collector=False,
            gpu_device="0",
            gpu_metrics_set="gb20x",
            gpu_metrics_frequency=1000,
            target=["/tmp/opensn", "-i", "/tmp/input.py"],
        )
        self.assertIn("--sample=none", collector)
        self.assertIn("--cpuctxsw=none", collector)
        self.assertIn("--trace=cuda,mpi,nvtx", collector)
        self.assertIn("--gpu-metrics-devices=0", collector)
        self.assertIn("--gpu-metrics-devices=none", peer)
        self.assertFalse(any(item.startswith("--gpu-metrics-set=") for item in peer))

    def test_rank0_nsys_wrapper_injects_nvtx_only_into_profiled_rank(self):
        argv = [
            "--rank-mode",
            "rank0",
            "--nsys",
            "/opt/nsys",
            "--output-directory",
            "/tmp",
            "--output-prefix",
            "fixture",
            "--trace",
            "cuda,mpi,nvtx",
            "--mpi-impl",
            "openmpi",
            "--enable-nvtx",
            "--",
            "/bin/true",
        ]
        with mock.patch.dict(
            os.environ,
            {"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_LOCAL_RANK": "1"},
            clear=True,
        ), mock.patch.object(RUNNER.os, "execvpe", side_effect=RuntimeError("exec")) as execute:
            with self.assertRaisesRegex(RuntimeError, "exec"):
                RUNNER.nsys_wrapper_main(argv)
            self.assertEqual(execute.call_args.args[0], "/bin/true")
            self.assertNotIn("CALI_SERVICES_ENABLE", execute.call_args.args[2])

        with mock.patch.dict(
            os.environ,
            {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_LOCAL_RANK": "0"},
            clear=True,
        ), mock.patch.object(RUNNER.os, "execvpe", side_effect=RuntimeError("exec")) as execute:
            with self.assertRaisesRegex(RuntimeError, "exec"):
                RUNNER.nsys_wrapper_main(argv)
            self.assertEqual(execute.call_args.args[0], "/opt/nsys")
            self.assertEqual(execute.call_args.args[2]["CALI_SERVICES_ENABLE"], "nvtx")


class ProvenanceAndEnvironmentTests(unittest.TestCase):
    def test_execute_scrubs_ambient_caliper_and_worker_settings_before_injection(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {
                "CALI_CONFIG": "cuda-activity-report",
                "CALI_SERVICES_ENABLE": "cupti,cuptitrace",
                "OPENSN_CBCD_NUM_WORKERS": "99",
            },
            clear=False,
        ):
            study = RUNNER.Study(pathlib.Path(temporary), "environment-test", "unit", False)
            code = (
                "import json,os; print(json.dumps({k: os.environ.get(k) for k in "
                "['CALI_CONFIG','CALI_SERVICES_ENABLE','OPENSN_CBCD_NUM_WORKERS']}))"
            )
            metadata = {"fixture": True}
            with mock.patch.object(RUNNER, "gpu_state_probe", return_value={"fixture": "gpu"}):
                result = study.execute(
                    kind="environment",
                    argv=[sys.executable, "-c", code],
                    cwd=pathlib.Path(temporary),
                    environment={"CALI_SERVICES_ENABLE": "nvtx", "CUDA_VISIBLE_DEVICES": "0"},
                    timeout=5.0,
                    log_name="environment.log",
                    metadata=metadata,
                )
            self.assertEqual(result.exit_code, 0)
            values = json.loads(RUNNER.read_log(study, result))
            self.assertIsNone(values["CALI_CONFIG"])
            self.assertEqual(values["CALI_SERVICES_ENABLE"], "nvtx")
            self.assertIsNone(values["OPENSN_CBCD_NUM_WORKERS"])
            self.assertIn("CALI_CONFIG", result.environment_unset)
            self.assertEqual(metadata, {"fixture": True})
            self.assertIn("gpu_state_before", result.metadata)
            study.finish("complete")

    def test_dynamic_link_closure_hashes_resolved_elfs_and_rejects_non_opensn_binary(self):
        provenance = RUNNER.dynamic_link_provenance(pathlib.Path("/bin/true"))
        self.assertFalse(provenance["valid"])
        self.assertIn("libopensn", " ".join(provenance["structural_errors"]))
        self.assertTrue(provenance["entries"])
        self.assertRegex(provenance["closure_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(all(entry.get("sha256") for entry in provenance["entries"]))
        self.assertIn("search_paths", provenance["binary"])

    def test_failed_matching_caliper_query_never_falls_back_to_header_true(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            binary = root / "bin/opensn"
            library = root / "prefix/lib/libcaliper.so.2"
            header = root / "prefix/include/caliper/caliper-config.h"
            query = root / "prefix/bin/cali-query"
            for path in (binary, library, header, query):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            header.write_text(
                '#define CALIPER_VERSION "2.13.0"\n#define CALIPER_HAVE_MPI\n'
                "#define CALIPER_HAVE_NVTX\n#define CALIPER_HAVE_CUPTI\n"
            )
            query.chmod(0o755)

            def fake_capture(argv, _cwd, timeout=10.0):
                if argv[0] == "ldd":
                    return {
                        "argv": argv,
                        "exit_code": 0,
                        "stdout": f"libcaliper.so.2 => {library} (0x1234)\n",
                        "stderr": "",
                    }
                return {"argv": argv, "exit_code": 1, "stdout": "", "stderr": "probe failed"}

            with mock.patch.object(RUNNER, "run_capture", side_effect=fake_capture):
                result = RUNNER.linked_caliper_provenance(binary)
            self.assertTrue(result["linked"])
            self.assertFalse(result["mpi"])
            self.assertFalse(result["nvtx"])
            self.assertFalse(result["cupti"])

    def test_profiler_csv_requires_header_and_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "report.csv"
            path.write_text("Name,Time\nSweepKernel,12\n")
            self.assertEqual(RUNNER.validate_profiler_csv(path, "fixture")["data_rows"], 1)
            path.write_text(
                "Generating SQLite file /tmp/report.sqlite from /tmp/report.nsys-rep\n"
                "Processing [/tmp/report.sqlite] with [cuda_gpu_kern_sum.py]...\n"
                "Name,Time,Instances\n"
                '"SweepKernel<int, 1>",12,4\n'
            )
            validation = RUNNER.validate_profiler_csv(path, "fixture")
            self.assertEqual(validation["preamble_rows"], 2)
            self.assertEqual(validation["columns"], ["Name", "Time", "Instances"])
            self.assertEqual(validation["data_rows"], 1)
            path.write_text("Name,Time\n")
            with self.assertRaises(RUNNER.StudyError):
                RUNNER.validate_profiler_csv(path, "fixture")


class ProcessControlTests(unittest.TestCase):
    def test_timeout_kills_process_group_and_publishes_terminal_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = RUNNER.Study(pathlib.Path(temporary), "timeout-test", "unit", False)
            result = study.execute(
                kind="sleep",
                argv=[sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=pathlib.Path(temporary),
                environment={},
                timeout=0.05,
                log_name="sleep.log",
            )
            self.assertTrue(result.timed_out)
            self.assertEqual(result.status, "failed")
            final = study.finish("failed", "expected unit-test timeout")
            state = json.loads((final / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(len(list(pathlib.Path(temporary).glob(".*.tmp-*"))), 0)


class DryRunTests(unittest.TestCase):
    def test_benchmark_dry_run_records_balanced_commands_without_launching_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "results"
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = RUNNER.main(
                    [
                        "benchmark",
                        "--dry-run",
                        "--binary",
                        "/bin/true",
                        "--input",
                        str(RUNNER.DEFAULT_INPUT),
                        "--output-root",
                        str(output),
                        "--ranks",
                        "1,2,4",
                        "--policies",
                        "hardware,resource-aware",
                        "--warmups",
                        "1",
                        "--trials",
                        "1",
                        "--schedule-seed",
                        "7",
                        "--pe-map",
                        "1=8,2=4,4=2",
                    ]
                )
            self.assertEqual(exit_code, 0)
            studies = list(output.iterdir())
            self.assertEqual(len(studies), 1)
            manifest = json.loads((studies[0] / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry-run")
            self.assertTrue(manifest["same_gpu_contention_diagnostic"])
            self.assertEqual(len(manifest["commands"]), 12)
            self.assertTrue(all(command["status"] == "planned" for command in manifest["commands"]))
            self.assertEqual(manifest["benchmark_schedule"]["seed"], 7)
            self.assertTrue(all("--verbose" in command["argv"] for command in manifest["commands"]))
            self.assertTrue(manifest["provenance"]["tools"]["gpu"]["probe_skipped"])
            self.assertTrue(manifest["provenance"]["workload_assets"][0]["sha256"])

    def test_fixed_workers_rejects_fake_two_policy_comparison(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()):
            exit_code = RUNNER.main(
                [
                    "benchmark",
                    "--dry-run",
                    "--binary",
                    "/bin/true",
                    "--output-root",
                    temporary,
                    "--ranks",
                    "1",
                    "--policies",
                    "hardware,resource-aware",
                    "--workers",
                    "2",
                ]
            )
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
