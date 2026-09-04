from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14 import (
    activation_supersession,
)


def test_parser_registers_v14_activation_supersession() -> None:
    args = cli.build_parser().parse_args(
        [
            "supersede-fixed-bank-harp-router-v14-activation",
            "--repository-root",
            "/tmp/repository",
        ]
    )

    assert args.surface == "supersede-fixed-bank-harp-router-v14-activation"
    assert args.repository_root == "/tmp/repository"
    assert args.confirm is None


def test_v14_supersession_cli_plans_without_confirmation_and_executes_via_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[object, ...]] = []
    plan = SimpleNamespace(
        to_payload=lambda: {
            "status": "READY_TO_ARCHIVE_ACTIVE_UNCONSUMED_PRELEASE_ACTIVATION",
            "filesystem_mutations": 0,
        }
    )
    receipt = SimpleNamespace(
        to_payload=lambda: {
            "status": (
                "ACTIVE_UNCONSUMED_PRELEASE_ACTIVATION_ARCHIVED_AND_SUPERSEDED"
            ),
        }
    )

    def fake_plan(repository_root: str | Path) -> object:
        calls.append(("plan", repository_root))
        return plan

    def fake_supersede(candidate: object, *, confirmation: str) -> object:
        calls.append(("supersede", candidate, confirmation))
        return receipt

    monkeypatch.setattr(
        activation_supersession,
        "plan_harp_v14_active_activation_supersession",
        fake_plan,
    )
    monkeypatch.setattr(
        activation_supersession,
        "supersede_harp_v14_active_activation",
        fake_supersede,
    )

    repository_root = str(tmp_path / "repository")
    assert (
        cli.main(
            [
                "supersede-fixed-bank-harp-router-v14-activation",
                "--repository-root",
                repository_root,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "filesystem_mutations": 0,
        "status": "READY_TO_ARCHIVE_ACTIVE_UNCONSUMED_PRELEASE_ACTIVATION",
    }
    assert calls == [("plan", repository_root)]

    assert (
        cli.main(
            [
                "supersede-fixed-bank-harp-router-v14-activation",
                "--repository-root",
                repository_root,
                "--confirm",
                activation_supersession.ACTIVE_SUPERSESSION_CONFIRMATION,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "ACTIVE_UNCONSUMED_PRELEASE_ACTIVATION_ARCHIVED_AND_SUPERSEDED",
    }
    assert calls == [
        ("plan", repository_root),
        ("plan", repository_root),
        (
            "supersede",
            plan,
            activation_supersession.ACTIVE_SUPERSESSION_CONFIRMATION,
        ),
    ]
