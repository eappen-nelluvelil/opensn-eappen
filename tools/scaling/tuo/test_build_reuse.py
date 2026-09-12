"""Check Python environment reuse without installing packages or building OpenSn."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("build_reuse.zsh")


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.venv = self.root / "previous/venv"
        (self.venv / "bin").mkdir(parents=True)
        self.deps = self.root / "dependencies"
        (self.deps / "deps/lib64/cmake/vtk-9.3").mkdir(parents=True)
        (self.deps / "env.zsh").write_text(f'export PATH="{self.bin}:$PATH"\n')
        (self.venv / "bin/activate").write_text(f'export PATH="{self.venv}/bin:$PATH"\n')
        self.executable(self.bin / "flux", "exit 0")
        self.executable(self.bin / "git", 'case "$*" in *HEAD*) echo revision ;; esac')
        self.executable(self.bin / "cmake", 'mkdir -p "$TEST_ROOT/new/build-opensn"')
        python = '''echo "$*" >> "$TEST_ROOT/python-calls"
case "$*" in
  *"pip install"*|*"-m venv"*) exit 99 ;;
  *SOABI*) echo compatible ;;
esac
'''
        self.executable(self.bin / "python", python)
        self.executable(self.venv / "bin/python", python)
        self.env = dict(os.environ, TEST_ROOT=str(self.root),
                        PATH=f"{self.bin}:{os.environ['PATH']}",
                        OPENSN_TUO_REUSE_VENV=str(self.venv))

    def executable(self, path, body):
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o700)

    def run_build(self):
        return subprocess.run(["zsh", str(SCRIPT), str(self.root), str(self.deps),
                               str(self.root / "new")], env=self.env,
                              text=True, capture_output=True)

    def test_reuse_does_not_install_or_create_environment(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = (self.root / "python-calls").read_text()
        self.assertNotIn("pip install", calls)
        self.assertNotIn("-m venv", calls)
        self.assertIn("pip check", calls)
        self.assertIn(str(self.venv), (self.root / "new/env.zsh").read_text())
        self.assertFalse((self.root / "new/venv").exists())

    def test_missing_environment_fails_without_installing(self):
        self.env["OPENSN_TUO_REUSE_VENV"] = str(self.root / "missing")
        result = self.run_build()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("environment is missing", result.stderr)

    def test_incompatible_abi_fails_without_installing(self):
        self.executable(self.venv / "bin/python", "echo incompatible")
        result = self.run_build()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Python ABI", result.stderr)


if __name__ == "__main__":
    unittest.main()
