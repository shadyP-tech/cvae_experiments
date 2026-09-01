from __future__ import annotations

from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.input_surfaces import (
    CACHE_INDEX,
    CONTENT_INDEX,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.preparation import (
    CASE_PARTITION,
    LABEL_FREE_BARRIER,
    LABEL_FREE_CONTENT_INDEX,
    PREPARATION_RECEIPT,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V3_EXECUTION_AMENDMENT_GATE,
    HARP_V3_EXPERIMENT_ID,
    HARP_V3_RUN_CONFIRMATION_TOKEN,
    KNOWN_PREPARATION_AUTHORITY_GATES,
    PreparationAuthorityError,
    preparation_authority_registration_error,
    validate_preparation_authority_extra_args,
)
from midogpp_thesis.workspace import runtime as workspace_runtime
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / authorization.WORKSPACE_CONFIG_RELATIVE_PATH


def test_checked_in_v3_is_planned_fenced_and_workspace_valid() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.experiments[EXPERIMENT_ID]
    config = load_config(CONFIG)

    assert HARP_V3_EXPERIMENT_ID == EXPERIMENT_ID
    assert experiment.status == "planned"
    assert config.execution_authorized is False
    assert config.protocol["utility_kind"] == (
        "downstream_classifier_utility_not_NELBO"
    )
    assert config.protocol["routing_stage_compatibility_estimated"] is False
    assert config.protocol["generative_expert_compatibility_claimed"] is False
    assert all(
        config.expected_hashes[role] is None
        for role in (
            "test_cache_content_sha256",
            "development_manifest_sha256",
            "evaluation_manifest_sha256",
            "parent_ledger_sha256",
            "execution_amendment_sha256",
        )
    )
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.preparation_authority_gate == (
        HARP_V3_EXECUTION_AMENDMENT_GATE
    )
    assert tuple(experiment.runner_argv) == authorization.WORKSPACE_RUNNER_ARGV
    assert dict(experiment.runner_env) == dict(authorization.WORKSPACE_RUNNER_ENV)

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.run(EXPERIMENT_ID)


def test_v3_gate_is_closed_world_and_bound_only_to_v3() -> None:
    assert HARP_V3_EXECUTION_AMENDMENT_GATE == (
        "harp_v3_consumed_test_execution_amendment_v1"
    )
    assert HARP_V3_EXECUTION_AMENDMENT_GATE in KNOWN_PREPARATION_AUTHORITY_GATES
    assert preparation_authority_registration_error(
        None,
        experiment_id=EXPERIMENT_ID,
    ) == (
        f"{EXPERIMENT_ID}: runner.preparation_authority_gate must remain "
        f"{HARP_V3_EXECUTION_AMENDMENT_GATE!r}"
    )
    assert preparation_authority_registration_error(
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        experiment_id="wrong.consumer",
    ) == (
        "wrong.consumer: runner.preparation_authority_gate "
        f"{HARP_V3_EXECUTION_AMENDMENT_GATE!r} is bound only to {EXPERIMENT_ID}"
    )
    assert validate_preparation_authority_extra_args(
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        ("--dry-run",),
    ) == ("--dry-run",)
    assert validate_preparation_authority_extra_args(
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        ("--confirm", HARP_V3_RUN_CONFIRMATION_TOKEN),
    ) == ("--confirm", HARP_V3_RUN_CONFIRMATION_TOKEN)
    assert validate_preparation_authority_extra_args(
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        (),
        preparation_only=True,
    ) == ()


@pytest.mark.parametrize(
    "extra_args",
    (
        (),
        ("--confirm",),
        ("--confirm", "wrong"),
        ("--confirm=" + HARP_V3_RUN_CONFIRMATION_TOKEN,),
        ("--dry-run", "--confirm", HARP_V3_RUN_CONFIRMATION_TOKEN),
        ("--confirm", HARP_V3_RUN_CONFIRMATION_TOKEN, "--dry-run"),
        ("--DRY-RUN",),
    ),
)
def test_v3_workspace_launch_allowlist_rejects_missing_or_variant_arguments(
    extra_args: tuple[str, ...],
) -> None:
    with pytest.raises(PreparationAuthorityError, match="accepts only"):
        validate_preparation_authority_extra_args(
            HARP_V3_EXECUTION_AMENDMENT_GATE,
            extra_args,
        )


def test_v3_internal_preparation_rejects_runner_arguments_and_force() -> None:
    with pytest.raises(PreparationAuthorityError, match="internal preparation"):
        validate_preparation_authority_extra_args(
            HARP_V3_EXECUTION_AMENDMENT_GATE,
            ("--dry-run",),
            preparation_only=True,
        )


def test_direct_cli_requires_exact_launch_confirmation_before_config_access() -> None:
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        cli.main(
            [
                "fixed-bank-harp-router-v3",
                "--config",
                "/definitely/not/a/readable/harp-v3-config.yaml",
            ]
        )


def test_retired_v3_publisher_cli_rejects_before_config_or_input_access() -> None:
    with pytest.raises(ProtocolError, match="disabled before config or input access"):
        cli.main(
            [
                "publish-fixed-bank-harp-router-v3-amendment",
                "--config",
                "/unread/config.yaml",
                "--expert-bank-root",
                "/unread/bank",
                "--generation-lock-root",
                "/unread/generation",
                "--prepared-cache-root",
                "/unread/cache",
                "--development-manifest",
                "/unread/development.csv",
                "--evaluation-manifest",
                "/unread/evaluation.csv",
                "--parent-ledger",
                "/unread/parent.json",
                "--amendment-path",
                "/unwritten/amendment.json",
                "--authorization-basis",
                "unread",
                "--authorization-date",
                "unread",
                "--repository-root",
                "/unread/repository",
            ]
        )


def test_v3_cli_launch_confirmation_is_mutually_exclusive_with_safe_modes() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "fixed-bank-harp-router-v3",
            "--config",
            "config.yaml",
            "--confirm",
            HARP_V3_RUN_CONFIRMATION_TOKEN,
        ]
    )
    assert parsed.confirm == HARP_V3_RUN_CONFIRMATION_TOKEN
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fixed-bank-harp-router-v3",
                "--config",
                "config.yaml",
                "--dry-run",
                "--confirm",
                HARP_V3_RUN_CONFIRMATION_TOKEN,
            ]
        )


@pytest.mark.parametrize(
    "extra_args",
    ((), ("--confirm",), ("--confirm", "wrong"), ("--dry-run", "extra")),
)
def test_workspace_run_rejects_v3_launch_variants_before_authority_or_prepare(
    extra_args: tuple[str, ...],
) -> None:
    events: list[str] = []
    experiment = SimpleNamespace(
        runnable=True,
        status="diagnostic",
        preparation_authority_gate=HARP_V3_EXECUTION_AMENDMENT_GATE,
        output_artifact_id="output",
        run_recovery_strategy=None,
    )
    workspace = SimpleNamespace(
        validate=lambda: events.append("validate"),
        get_experiment=lambda _experiment_id: (
            events.append("experiment") or experiment
        ),
        _enforce_preparation_authority=lambda _experiment: events.append(
            "authority"
        ),
        resolve_artifact=lambda *_args, **_kwargs: events.append("path")
        or Path("output"),
        prepare=lambda *_args, **_kwargs: events.append("prepare"),
        _execute=lambda *_args, **_kwargs: events.append("execute") or 0,
    )

    with pytest.raises(WorkspaceError, match="accepts only"):
        MidogppWorkspace.run(
            workspace,
            EXPERIMENT_ID,
            extra_args=extra_args,
        )

    assert events == ["validate", "experiment"]


def test_workspace_run_accepts_exact_v3_confirmation_then_prepares() -> None:
    events: list[str] = []
    experiment = SimpleNamespace(
        runnable=True,
        status="diagnostic",
        preparation_authority_gate=HARP_V3_EXECUTION_AMENDMENT_GATE,
        output_artifact_id="output",
        run_recovery_strategy=None,
    )
    prepared = object()
    def prepare(
        _experiment_id: str,
        *,
        require_inputs: bool,
        force: bool,
        _run_admitted: bool,
    ) -> object:
        assert require_inputs is True
        assert force is False
        assert _run_admitted is True
        events.append("prepare")
        return prepared

    workspace = SimpleNamespace(
        validate=lambda: events.append("validate"),
        get_experiment=lambda _experiment_id: (
            events.append("experiment") or experiment
        ),
        _enforce_preparation_authority=lambda _experiment: events.append(
            "authority"
        ),
        resolve_artifact=lambda *_args, **_kwargs: events.append("path")
        or Path("output"),
        prepare=prepare,
        _execute=lambda observed, *, extra_args: (
            events.append(("execute", observed, extra_args)) or 0
        ),
    )

    result = MidogppWorkspace.run(
        workspace,
        EXPERIMENT_ID,
        extra_args=("--confirm", HARP_V3_RUN_CONFIRMATION_TOKEN),
    )

    assert result == 0
    assert events == [
        "validate",
        "experiment",
        "authority",
        "path",
        "prepare",
        (
            "execute",
            prepared,
            ("--confirm", HARP_V3_RUN_CONFIRMATION_TOKEN),
        ),
    ]


def test_public_workspace_prepare_rejects_v3_before_authority_or_output() -> None:
    events: list[str] = []
    experiment = SimpleNamespace(
        runnable=True,
        status="diagnostic",
        preparation_authority_gate=HARP_V3_EXECUTION_AMENDMENT_GATE,
    )
    workspace = SimpleNamespace(
        validate=lambda: events.append("validate"),
        get_experiment=lambda _experiment_id: (
            events.append("experiment") or experiment
        ),
        _enforce_preparation_authority=lambda _experiment: events.append(
            "authority"
        ),
    )

    with pytest.raises(WorkspaceError, match="accepts only"):
        MidogppWorkspace.prepare(workspace, EXPERIMENT_ID)

    assert events == ["validate", "experiment"]


@pytest.mark.parametrize(
    ("run_admitted", "expected_preparation_only"),
    ((False, False), (True, True)),
)
def test_workspace_prepare_marks_only_run_admitted_call_as_internal(
    monkeypatch: pytest.MonkeyPatch,
    run_admitted: bool,
    expected_preparation_only: bool,
) -> None:
    observed: list[tuple[tuple[str, ...], bool]] = []

    def validate_args(
        _gate_id: str | None,
        extra_args: tuple[str, ...],
        *,
        force: bool,
        preparation_only: bool,
    ) -> tuple[str, ...]:
        assert force is False
        observed.append((extra_args, preparation_only))
        return extra_args

    class StopAfterArgumentGate(RuntimeError):
        pass

    experiment = SimpleNamespace(
        runnable=True,
        status="diagnostic",
        preparation_authority_gate=HARP_V3_EXECUTION_AMENDMENT_GATE,
    )
    workspace = SimpleNamespace(
        validate=lambda: None,
        get_experiment=lambda _experiment_id: experiment,
        _enforce_preparation_authority=lambda _experiment: (_ for _ in ()).throw(
            StopAfterArgumentGate()
        ),
    )
    monkeypatch.setattr(
        workspace_runtime,
        "validate_preparation_authority_extra_args",
        validate_args,
    )

    with pytest.raises(StopAfterArgumentGate):
        MidogppWorkspace.prepare(
            workspace,
            EXPERIMENT_ID,
            _run_admitted=run_admitted,
        )

    assert observed == [((), expected_preparation_only)]
    with pytest.raises(PreparationAuthorityError, match="reject --force"):
        validate_preparation_authority_extra_args(
            HARP_V3_EXECUTION_AMENDMENT_GATE,
            (),
            preparation_only=True,
            force=True,
        )


@pytest.mark.parametrize(
    "surface",
    (
        "fixed-bank-harp-router-v3",
        "prepare-fixed-bank-harp-router-v3-inputs",
        "publish-fixed-bank-harp-router-v3-amendment",
    ),
)
def test_v3_cli_surfaces_are_registered(surface: str) -> None:
    parser = build_parser()
    action = next(
        item
        for item in parser._actions
        if getattr(item, "dest", None) == "surface"
    )
    assert surface in action.choices


def test_v3_registration_determinism_and_workstation_controls_are_exact() -> None:
    assert authorization.WORKSPACE_RUNNER_ENV == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }
    assert "fixed-bank-harp-router-v3" in authorization.WORKSPACE_RUNNER_ARGV
    assert "fixed-bank-harp-router-v1" not in authorization.WORKSPACE_RUNNER_ARGV
    assert "fixed-bank-harp-router-v2" not in authorization.WORKSPACE_RUNNER_ARGV


def test_v3_canonical_workspace_command_contains_exact_launch_confirmation() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    command = shlex.split(workspace.central_command(EXPERIMENT_ID))
    assert command[-3:] == [
        "--",
        "--confirm",
        HARP_V3_RUN_CONFIRMATION_TOKEN,
    ]


def test_v3_catalog_cache_members_match_preparation_contract() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    cache = workspace.artifacts["midogpp_stage90_harp_consumed_test_cache_v3"]
    required = set(cache.required_files)
    assert {
        CACHE_INDEX.as_posix(),
        CONTENT_INDEX.as_posix(),
        CASE_PARTITION.as_posix(),
        LABEL_FREE_BARRIER.as_posix(),
        LABEL_FREE_CONTENT_INDEX.as_posix(),
        PREPARATION_RECEIPT.as_posix(),
    }.issubset(required)
    assert "manifests/case_partition.json" not in required
    assert "manifests/label_free_partition_barrier.json" not in required
    assert "manifests/label_free_content_index.json" not in required
