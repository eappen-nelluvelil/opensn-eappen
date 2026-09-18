# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cbcd_study_batch as batch
from study_build import write_json


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_json(self.root / "manifest.json", dict(
            nodes=[1, 2, 256, 512], ranks_per_node=4, trials=17, modes=["baseline"]))
        self.batch = dict(max_nodes=256, assets={}, submit=[
            "flux", "batch", "-N", "{nodes}", "-n", "{ranks}", "-t", "1h",
            "--output={allocation}/stdout.txt", "--wrap", "launcher", "{kind}", "{root}"])
        write_json(self.root / "batch.json", self.batch)
        (self.root / "builds/native").mkdir(parents=True)
        write_json(self.root / "builds/native/build.json", dict(fingerprints={}))
        self.args = SimpleNamespace(root=self.root, kind=None, nodes=None, retry=False)
        for name in ("verify_inputs", "pending"):
            mock = patch.object(batch.study, name)
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)
        self.pending.return_value = [("baseline", 1)]
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    @staticmethod
    def accepted(command, **kwargs):
        kwargs["stdout"].write("fABC123\n")

    def test_default_submits_both_kinds_within_limit(self):
        with patch.object(batch.subprocess, "run", side_effect=self.accepted) as run:
            batch.submit(self.args)
        self.assertEqual(run.call_count, 6)
        for call in run.call_args_list:
            command = call.args[0]
            nodes = int(command[command.index("-N") + 1])
            self.assertEqual(int(command[command.index("-n") + 1]), 4 * nodes)
            self.assertIn("1h", command)
            self.assertLessEqual(nodes, 256)
        self.assertEqual(len(list(self.root.rglob("job-id.txt"))), 6)

    def test_no_duplicate_submission(self):
        with patch.object(batch.subprocess, "run", side_effect=self.accepted) as run:
            batch.submit(self.args)
            batch.submit(self.args)
        self.assertEqual(run.call_count, 6)

    def test_out_of_policy_nodes_fail_before_submission(self):
        self.args.nodes = [512]
        with patch.object(batch.subprocess, "run") as run, self.assertRaises(ValueError):
            batch.submit(self.args)
        run.assert_not_called()

    def test_complete_cases_skipped(self):
        self.pending.return_value = []
        with patch.object(batch.subprocess, "run") as run:
            batch.submit(self.args)
        run.assert_not_called()

    def test_retry_requires_confirmed_inactive_job(self):
        self.args.kind, self.args.nodes = "weak", [2]
        with patch.object(batch.subprocess, "run", side_effect=self.accepted) as run:
            batch.submit(self.args)
            self.args.retry = True
            for state in ("RUN\n", "SCHED\n", "CLEANUP\n", ""):
                with patch.object(batch.subprocess, "check_output", return_value=state), \
                        self.assertRaises(ValueError):
                    batch.submit(self.args)
            self.assertEqual(run.call_count, 1)
            with patch.object(batch.subprocess, "check_output", return_value="INACTIVE\n"):
                batch.submit(self.args)
            self.assertEqual(run.call_count, 2)
        self.assertEqual(len(list(self.root.rglob("submission.json"))), 2)

    def test_ambiguous_submission_blocks_retry(self):
        self.args.kind, self.args.nodes = "strong", [1]
        with patch.object(batch.subprocess, "run", side_effect=OSError("disconnected")), \
                self.assertRaises(OSError):
            batch.submit(self.args)
        self.args.retry = True
        with patch.object(batch.subprocess, "run") as run, self.assertRaisesRegex(
                ValueError, "Unresolved submission"):
            batch.submit(self.args)
        run.assert_not_called()

    def test_changed_launcher_rejected(self):
        self.batch["assets"] = {str(self.root / "launcher"): "incorrect"}
        (self.root / "launcher").write_text("original")
        (self.root / "batch.json").write_text(batch.study.json.dumps(self.batch))
        with patch.object(batch.subprocess, "run") as run, self.assertRaisesRegex(
                ValueError, "changed"):
            batch.submit(self.args)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
