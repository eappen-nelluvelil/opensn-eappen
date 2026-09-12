"""Check repeated interactive cases without allocating cluster resources."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HELPER = Path(__file__).with_name("interactive_cbcd.zsh").read_text()


def function(name, following):
    return HELPER[HELPER.index(name + "()\n"):HELPER.index(following + "()\n")]


class RepetitionTest(unittest.TestCase):
    def test_three_trials_stay_in_one_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs").mkdir()
            for kind in ("strong", "weak"):
                for nodes in (1, 2, 4, 8):
                    job = root / "jobs" / f"baseline-{kind}-{nodes}.zsh"
                    job.touch()
                    job.chmod(0o700)
            script = '''set -eu
profile_root=$TEST_ROOT
profile_nodes=1,2,4,8
profile_kinds=strong,weak
profile_repetitions=3
check_profile() { :; }
check_profile_nodes() { :; }
flux() { print allocation-1; }
run_generated_job() {
  print -- "$1 group=$OPENSN_TUO_PROFILE_TRIAL_GROUP pin=$OPENSN_TUO_PIN_PROFILE_NODES"
}
'''
            script += function("run_profile_interactive_here", "profile_case_complete")
            script += '\nrun_profile_interactive_here baseline\n'
            result = subprocess.run(["zsh", "-c", script], text=True, capture_output=True,
                                    env=dict(os.environ, TEST_ROOT=directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = result.stdout.splitlines()
            self.assertEqual(len(rows), 24)
            self.assertEqual(len({row.split("group=")[1] for row in rows}), 1)
            for i in range(0, 24, 3):
                self.assertIn("trial=1/3", rows[i])
                self.assertIn("trial=2/3", rows[i + 1])
                self.assertIn("trial=3/3", rows[i + 2])

    def test_resume_requires_three_successes_in_one_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results/baseline/strong/nodes-1"
            root.mkdir(parents=True)
            for i, group in enumerate(("first", "first", "second")):
                run = root / f"run-{i}"
                run.mkdir()
                (run / "SUCCESS").touch()
                (run / "exit_code.txt").write_text("0\n")
                (run / "metadata.txt").write_text(f"trial_group={group}\n")
            script = 'set -eu\nprofile_root=$TEST_ROOT\nprofile_repetitions=3\n'
            script += function("profile_case_complete", "run_profile_interactive")
            script += '\nprofile_case_complete baseline strong 1\n'
            def check():
                return subprocess.run(["zsh", "-c", script], text=True, capture_output=True,
                                      env=dict(os.environ, TEST_ROOT=directory))
            self.assertEqual(check().returncode, 1)
            (root / "run-2/metadata.txt").write_text("trial_group=first\n")
            result = check()
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
