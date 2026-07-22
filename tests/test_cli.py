from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from midogpp_thesis.cli import COMMANDS, command_help, main
from midogpp_thesis.cvae.preservation.cli import build_parser as build_preservation_parser
from midogpp_thesis.real_features.classifier_reference.cli import (
    build_parser as build_classifier_parser,
)


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


def test_preservation_cli_exposes_separate_lock_study_and_outer_commands() -> None:
    parser = build_preservation_parser()

    source_inner = parser.parse_args(["source-inner-prior-recovery", "--config", "source.yaml"])
    prior_study = parser.parse_args(
        ["source-inner-learned-conditional-prior-study", "--config", "prior-v2.yaml"]
    )
    fisher_study = parser.parse_args(
        ["source-inner-task-fisher-shrinkage-study", "--config", "fisher-v2.yaml"]
    )
    outer = parser.parse_args(["prior-recovery-outer", "--config", "outer.yaml"])

    assert source_inner.surface == "source-inner-prior-recovery"
    assert prior_study.surface == "source-inner-learned-conditional-prior-study"
    assert fisher_study.surface == "source-inner-task-fisher-shrinkage-study"
    assert outer.surface == "prior-recovery-outer"


def test_classifier_cli_exposes_conditional_logit_alignment_command() -> None:
    args = build_classifier_parser().parse_args(
        [
            "conditional-logit-alignment",
            "--config",
            "cla.yaml",
        ]
    )

    assert args.surface == "conditional-logit-alignment"
    assert args.config == "cla.yaml"
    assert not hasattr(args, "artifact_root")
