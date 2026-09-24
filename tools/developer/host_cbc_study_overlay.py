# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Create a separate, auditable setup-annotation worktree for Native profiling."""

from pathlib import Path
import re
import subprocess


SWEEP = "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/"
SCOPES = {
    SWEEP + "sweep_runtime_builder.cc": {
        "BuildSweepRuntime": "Runtime", "BuildCBCSPDS": "SPDS",
        "BuildCBCCPUFludsCommonData": "FLUDS",
    },
    SWEEP + "fluds/cbc_fluds_common_data.cc": {
        "CBC_FLUDSCommonData::MakeAlpha": "Alpha",
        "CBC_FLUDSCommonData::FinalizeBeta": "Beta",
    },
}


def annotated(text, scopes):
    if "#include <caliper/cali.h>" not in text:
        text = text.replace("\n#include ", "\n#include <caliper/cali.h>\n#include ", 1)
    for function, label in scopes.items():
        pattern = r"(^" + re.escape(function) + r"\([^{}]*?\n\{)"
        insertion = '\n  CALI_CXX_MARK_SCOPE("HostCBC.Setup.' + label + '");'
        text, count = re.subn(pattern, lambda m: m[0] + insertion, text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected one definition of {function}; found {count}")
    return text


def annotate(source, destination, patch_path):
    source, destination, patch_path = map(Path, (source, destination, patch_path))
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       text=True).strip()
    expected = {}
    for name, scopes in SCOPES.items():
        expected[name] = annotated((source / name).read_text(), scopes)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach",
                        str(destination), revision], check=True)
    if subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"],
                               text=True).strip() != revision:
        raise ValueError("Profiling source revision changed")
    for name, text in expected.items():
        path = destination / name
        if path.read_text() not in (text, (source / name).read_text()):
            raise ValueError(f"Unexpected profiling source changes: {path}")
        path.write_text(text)
    changed = subprocess.check_output(["git", "-C", str(destination), "diff", "--name-only"],
                                      text=True).splitlines()
    if set(changed) != set(expected):
        raise ValueError("Unexpected files changed in profiling worktree")
    patch = subprocess.check_output(["git", "-C", str(destination), "diff", "--binary"], text=True)
    if patch_path.exists():
        if patch_path.read_text() != patch:
            raise ValueError("Profiling overlay changed")
    else:
        with patch_path.open("x") as stream:
            stream.write(patch)
    return destination
