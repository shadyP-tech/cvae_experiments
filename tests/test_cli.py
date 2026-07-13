from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from midogpp_thesis.cli import COMMANDS, command_help, main


def test_root_cli_lists_only_canonical_command_groups(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "cvae-preservation" in output
    assert "real-feature-classifier" in output
    assert "workspace" in output
    assert "cvae-rebuild" not in output
    assert set(command_help()) == set(COMMANDS)


def test_root_cli_import_is_lazy_in_fresh_interpreter() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    code = "\n".join(
        (
            "import sys",
            "import midogpp_thesis.cli",
            "forbidden = ('midogpp_thesis.cvae.preservation.sanity', "
            "'midogpp_thesis.real_features.sail.features', "
            "'midogpp_thesis.real_features.classifier_reference.classifiers', "
            "'midogpp_thesis.workspace.runtime')",
            "assert not any(name in sys.modules for name in forbidden)",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
