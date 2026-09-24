# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Focused harness tests; these do not establish transport-solver correctness."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import host_cbc_study as study
from host_cbc_study_overlay import SCOPES, annotated


def log_text():
    return "\n".join([
        "avg_sweep_time = 1.0 s, sweep_time_per_unknown = 1.0 ns",
        "unknowns = 1000000000, lagged_unknowns = 0",
        "iteration = 0, residual = 1.0", "iteration = 1, residual = 0.1",
        "final, status = iteration_limit, iterations = 1",
        "OPENSN_STUDY_FLUX_MAX group=0 value=0.9",
        "OPENSN_STUDY_FLUX_MAX group=63 value=0.01",
        "OPENSN_STUDY_TRIAL_COMPLETE", "OpenSn finished execution.",
    ])


class HostStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(iterations=1, trials=2, modes=list(study.MODES), ranks_per_node=2)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def evidence(self, attempt):
        attempt.mkdir(parents=True)
        (attempt / "exit_code.txt").write_text("0\n")
        (attempt / "stdout.txt").write_text(log_text())
        self.write(attempt / "placement.json", [{"rank": r, "hostname": "node"} for r in (0, 1)])
        self.write(attempt / "memory.json", [{"rank": r, "max_rss_kib": 10} for r in (0, 1)])

    def test_finite_fixed_work(self):
        self.assertEqual(study.measurements(log_text(), 1)["sweep_seconds"], 1)
        for old, new in (("lagged_unknowns = 0", "lagged_unknowns = 1"),
                         ("final, status = iteration_limit", "final, status = converged"),
                         ("sweep_time_per_unknown = 1.0", "sweep_time_per_unknown = 2.0"),
                         ("iteration = 1, residual = 0.1", ""),
                         ("OPENSN_STUDY_TRIAL_COMPLETE", ""),
                         ("value=0.9", "value=nan")):
            with self.subTest(old=old), self.assertRaises(ValueError):
                study.measurements(log_text().replace(old, new), 1)

    def test_attempt_requires_exit_placement_memory(self):
        attempt = self.root / "attempt"
        self.evidence(attempt)
        study.validate_attempt(attempt, "baseline", self.config, 1)
        (attempt / "exit_code.txt").write_text("1\n")
        with self.assertRaisesRegex(ValueError, "Nonzero"):
            study.validate_attempt(attempt, "baseline", self.config, 1)
        (attempt / "exit_code.txt").write_text("0\n")
        self.write(attempt / "placement.json", [{"rank": 0, "hostname": "node"}] * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            study.validate_attempt(attempt, "baseline", self.config, 1)

    def test_profile_output_required(self):
        with self.assertRaises(ValueError):
            study.profile_artifacts(self.root, "caliper-mpi")
        (self.root / "mpi-regions.txt").write_text("MPI_Recv")
        with self.assertRaisesRegex(ValueError, "setup"):
            study.profile_artifacts(self.root, "caliper-mpi")
        (self.root / "mpi-regions.txt").write_text(
            "MPI_Recv HostCBC.Setup.FLUDS HostCBC.Setup.Beta")
        study.profile_artifacts(self.root, "caliper-mpi")
        (self.root / "mpi.txt").write_text("MPI_Allreduce")
        study.profile_artifacts(self.root, "pmpi")

    def test_pending_order_and_modes(self):
        with patch.object(study, "successes", return_value=[]):
            self.assertEqual(study.pending(self.root, "weak", 2, self.config),
                             [(m, i) for i in (1, 2) for m in study.MODES])
        self.assertEqual(study.build_mode("baseline"), "native")
        self.assertEqual(study.build_mode("pmpi"), "native")
        self.assertEqual(study.build_mode("caliper-mpi"), "profile")

    def test_resume_revalidates_evidence(self):
        self.write(self.root / "manifest.json", self.config)
        attempt = self.root / "results/weak/nodes-1/baseline/trial-1/attempt-first"
        self.evidence(attempt)
        (attempt / "input.py").write_text("pass\n")
        self.write(attempt / "launch.json", {"input_sha256": study.digest(attempt / "input.py")})
        self.write(attempt / "result.json", study.validate_attempt(
            attempt, "baseline", self.config, 1))
        manifest_hash = study.digest(self.root / "manifest.json")
        self.write(attempt / "SUCCESS", {"manifest_sha256": manifest_hash})
        self.assertEqual(study.successes(self.root, "weak", 1, "baseline", 1, self.config),
                         [attempt])
        (attempt / "input.py").write_text("changed\n")
        self.assertFalse(study.successes(self.root, "weak", 1, "baseline", 1, self.config))

    def test_environment(self):
        env = study.clean_environment(dict(CALI_CONFIG="runtime-report", LD_PRELOAD="bad.so",
                                           GLIBC_TUNABLES="tuning", PATH="/bin",
                                           SLURM_JOB_ID="1"))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("GLIBC_TUNABLES", env)
        self.assertEqual(env["CALI_CONFIG"], "")
        self.assertEqual(env["SLURM_JOB_ID"], "1")

    def test_overlay_rejects_missing_and_duplicate_functions(self):
        text = '#include "header.h"\nvoid\nSomeFunction()\n{\n}\n'
        result = annotated(text, {"SomeFunction": "Alpha"})
        self.assertEqual(result.count('CALI_CXX_MARK_SCOPE("HostCBC.Setup.Alpha")'), 1)
        for broken in ("", text + text):
            with self.assertRaises(ValueError):
                annotated(broken, {"SomeFunction": "Alpha"})

    def test_source_annotation_anchors(self):
        # Exercise the checked-out source; cross-revision checks belong to the campaign preflight.
        source = Path(__file__).resolve().parents[2]
        for name, scopes in SCOPES.items():
            content = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{name}"],
                                              text=True)
            self.assertEqual(annotated(content, scopes).count("HostCBC.Setup."), len(scopes))

    def test_prepare_all_inputs(self):
        source = Path(__file__).resolve().parents[2]
        previous = self.root / "previous"
        previous.mkdir()
        (previous / "CMakeCache.txt").write_text("OPENSN_WITH_CUDA:BOOL=OFF\n")
        mesh = self.root / "mesh.msh"
        mesh.write_text("mesh fixture")
        config = dict(source=str(source), nodes=[1, 2], ranks_per_node=2, threads=1,
                      meshes={k: {str(n): str(mesh) for n in (1, 2)} for k in ("strong", "weak")},
                      launcher=["mpirun", "-n", "{ranks}"], reuse_build=str(previous),
                      build_root=str(self.root / "build"))
        conf = self.root / "config.json"
        self.write(conf, config)
        root = self.root / "campaign"
        study.prepare(argparse.Namespace(config=conf, root=root))
        manifest = study.read_json(root / "manifest.json")
        for kind in ("strong", "weak"):
            for n in (1, 2):
                path = root / "inputs" / f"{kind}-{n}.py"
                compile(path.read_text(), str(path), "exec")
                self.assertEqual(study.digest(path), manifest["assets"][str(path)])
                self.assertIn("use_gpus=False", path.read_text())
        with self.assertRaises(FileExistsError):
            study.prepare(argparse.Namespace(config=conf, root=root))


if __name__ == "__main__":
    unittest.main()
