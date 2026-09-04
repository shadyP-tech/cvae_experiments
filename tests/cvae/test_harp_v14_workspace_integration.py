from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.config import (
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V14_EXECUTION_AMENDMENT_GATE,
    HARP_V14_EXPERIMENT_ID,
    HARP_V14_RUN_CONFIRMATION_TOKEN,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
AMENDMENT = ROOT / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
OUTPUT = ROOT / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
LEASE = authorization.lease_path(ROOT)
EXPECTED_INPUT_ARTIFACT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_harp_source_train_full_test_cache_v14",
    "midogpp_stage90_harp_source_train_label_capability_v14",
    "midogpp_stage90_harp_full_test_evaluation_release_v14",
    "midogpp_uniform_b_test_consumption_ledger_harp_parent_v14",
    "midogpp_uniform_b_test_consumption_ledger_harp_execution_amendment_v14",
)


def _planned_mutation_witness() -> tuple[bool, bool, bool]:
    """Observe only the three paths that inspection and rejection must not create."""

    return tuple(
        os.path.lexists(path)
        for path in (AMENDMENT, OUTPUT, LEASE)
    )  # type: ignore[return-value]


def test_v14_workspace_load_validate_and_registration_are_exact() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)

    assert HARP_V14_EXPERIMENT_ID == EXPERIMENT_ID
    assert INPUT_ARTIFACT_IDS == EXPECTED_INPUT_ARTIFACT_IDS
    assert experiment.input_artifact_ids == EXPECTED_INPUT_ARTIFACT_IDS
    assert len(experiment.input_artifact_ids) == 7
    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.preparation_authority_gate == (
        HARP_V14_EXECUTION_AMENDMENT_GATE
    )
    assert tuple(experiment.runner_argv) == authorization.WORKSPACE_RUNNER_ARGV
    assert dict(experiment.runner_env) == dict(authorization.WORKSPACE_RUNNER_ENV)


def test_v14_canonical_workspace_command_is_confirmation_bound() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    command = shlex.split(workspace.central_command(EXPERIMENT_ID))

    assert command[1:6] == [
        "-m",
        "midogpp_thesis",
        "workspace",
        "run",
        EXPERIMENT_ID,
    ]
    assert command[-3:] == [
        "--",
        "--confirm",
        HARP_V14_RUN_CONFIRMATION_TOKEN,
    ]


def test_v14_planned_workspace_launch_refuses_before_any_mutation() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    before = _planned_mutation_witness()

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.run(
            EXPERIMENT_ID,
            extra_args=("--confirm", HARP_V14_RUN_CONFIRMATION_TOKEN),
        )

    assert _planned_mutation_witness() == before


def test_v14_inspect_plan_is_path_free_and_mutation_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = _planned_mutation_witness()

    result = cli.main(
        [
            "fixed-bank-harp-router-v14",
            "--config",
            str(CONFIG),
            "--inspect-plan",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["status"] == "PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["execution_authorized"] is False
    assert payload["authorization_probed"] is False
    assert payload["paths_resolved"] is False
    assert payload["filesystem_mutations"] == 0
    assert payload["development_labels_opened"] is False
    assert payload["evaluation_labels_opened"] is False
    assert payload["publication_status"] == PUBLICATION_STATUS
    assert payload["terminal_decision"] == TERMINAL_DECISION
    assert payload["fresh_evidence"] is False
    assert payload["may_feed_another_experiment"] is False
    assert _planned_mutation_witness() == before


@pytest.mark.parametrize(
    "confirmation_args",
    ((), ("--confirm", "RUN_HARP_V14_WRONG_TOKEN")),
)
def test_v14_direct_run_requires_exact_token_before_config_access(
    confirmation_args: tuple[str, ...],
) -> None:
    before = _planned_mutation_witness()

    with pytest.raises(ProtocolError, match="exact confirmation token"):
        cli.main(
            [
                "fixed-bank-harp-router-v14",
                "--config",
                "/definitely/not/a/readable/harp-v14-config.yaml",
                *confirmation_args,
            ]
        )

    assert _planned_mutation_witness() == before
