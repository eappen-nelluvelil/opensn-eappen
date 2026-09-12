"""Test the outgoing-copy campaign driver without a cluster allocation."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


DRIVER = Path(__file__).with_name("run_outgoing_copy_profile.zsh")


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.txt"
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}:{os.environ['PATH']}",
            TEST_ROOT=str(self.root),
            TEST_LOG=str(self.log),
            OPENSN_TUO_STUDY_ROOT=str(self.root / "study"),
            OPENSN_TUO_RESULTS=str(self.root / "results"),
        )
        git = self.bin / "git"
        git.write_text(
            '#!/bin/sh\ncase "$*" in\n'
            '*--show-toplevel*) echo "$TEST_ROOT" ;;\n'
            '*"rev-parse HEAD"*) echo 123456789abcdef ;;\n'
            'esac\n'
        )
        git.chmod(0o700)
        helper = self.root / "tools/scaling/tuo/interactive_cbcd.zsh"
        helper.parent.mkdir(parents=True)
        helper.write_text('''set -eu
print -r -- "$*" >> "$TEST_LOG"
[[ $OPENSN_TUO_PROFILE_NODES == 1,2,4,8 ]]
[[ $OPENSN_TUO_PROFILE_KINDS == strong,weak ]]
[[ $OPENSN_TUO_NUM_THREADS == 21 ]]
[[ $OPENSN_TUO_TIME_LIMIT == 60m ]]
if [[ $1 == build ]]; then
  mkdir -p "$OPENSN_TUO_BUILD"
  print -- "${TEST_BUILD_SHA:-123456789abcdef}" > "$OPENSN_TUO_BUILD/source-revision.txt"
fi
if [[ ${2:-} == ${TEST_FAIL_PROFILE:-never} ]]; then
  exit 1
fi
''')

    def run_driver(self, *args):
        return subprocess.run(
            ["zsh", str(DRIVER), *args], env=self.env, text=True, capture_output=True
        )

    def test_profiles_continue_after_failure_and_collect(self):
        self.env["TEST_FAIL_PROFILE"] = "caliper"
        result = self.run_driver("run", "test-1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(), [
            "build",
            "run-profile-interactive baseline",
            "run-profile-interactive cbcd-metrics",
            "run-profile-interactive caliper",
            "run-profile-interactive caliper-mpi",
            "run-profile-interactive pmpi",
            "run-profile-interactive rocprof",
            "collect-profile",
        ])

    def test_existing_results_require_resume(self):
        campaign = self.root / "results/test-1-profile/resource-aware"
        campaign.mkdir(parents=True)
        self.assertEqual(self.run_driver("run", "test-1").returncode, 2)
        self.assertFalse(self.log.exists())
        (campaign / "manifest.json").write_text("{}")
        result = self.run_driver("resume", "test-1", "baseline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(), [
            "build", "resume-profile-interactive baseline", "collect-profile"
        ])

    def test_wrong_build_revision_stops_before_launch(self):
        self.env["TEST_BUILD_SHA"] = "wrong-revision"
        self.assertEqual(self.run_driver("run", "test-1").returncode, 2)
        self.assertEqual(self.log.read_text().splitlines(), ["build"])

    def test_invalid_label_is_rejected(self):
        self.assertEqual(self.run_driver("run", "../other").returncode, 2)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
