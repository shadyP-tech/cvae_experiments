from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.config import (
    CONFIG_TOP_LEVEL,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.experiment_contracts import (
    AUTHORIZED_INPUT_ROLES,
    CANONICAL_OUTPUT_ROOT,
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.identity import (
    CLI_SURFACE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v2.yaml"
)
V2_SCOPED_INPUTS = frozenset(INPUT_ARTIFACT_IDS[2:])
FORBIDDEN_OUTPUT_PURPOSES = {
    "real_feature_reference_evidence",
    "cvae_preservation_evidence",
    "expert_bank_evidence",
    "generation_evidence",
    "all_candidate_utility_diagnostic",
    "routing_evidence",
    "expert_selection_evidence",
    "nelbo_compatibility_evidence",
    "synthetic_downstream_utility_evidence",
    "oracle_and_diagnostic_evidence",
}


def test_registration_is_exact_eight_single_use_and_terminal_only() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 8
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert output.canonical_path == CANONICAL_OUTPUT_ROOT
    assert output.availability == "generated_on_run"
    assert output.evidence_label == (
        "AUTHORIZED_TERMINAL_CONSUMED_TEST_DIAGNOSTIC_PENDING_RUN"
    )
    assert output.semantic_identities["publication_status"] == PUBLICATION_STATUS
    assert output.semantic_identities["terminal_decision"] == TERMINAL_DECISION
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"
    assert output.semantic_identities["nelbo_compatibility_claimed"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert set(output.forbidden_reuse) == FORBIDDEN_OUTPUT_PURPOSES
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_config_scaffold_is_path_deferred_and_fails_closed_until_resealed() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert set(payload) == set(CONFIG_TOP_LEVEL)
    assert tuple(payload["inputs"]["direct_input_artifact_ids"]) == (
        INPUT_ARTIFACT_IDS
    )
    assert tuple(payload["inputs"]["authorized_input_roles"]) == (
        AUTHORIZED_INPUT_ROLES
    )
    assert payload["inputs"]["direct_input_count"] == 8
    assert payload["protocol"]["fresh_evidence"] is False
    assert payload["protocol"]["routing_success_claimed"] is False
    assert payload["protocol"]["nelbo_compatibility_claimed"] is False
    assert payload["claim_boundary"]["adaptive_architecture_comparison"] is True
    assert payload["claim_boundary"]["comparisons_are_descriptive_only"] is True
    assert payload["claim_boundary"]["new_center_generalization_claimed"] is False
    assert payload["claim_boundary"]["may_feed_another_experiment"] is False

    pending = "__PENDING_" in CONFIG.read_text(encoding="utf-8")
    if pending:
        with pytest.raises(ProtocolError, match="pending"):
            load_config(CONFIG)
    else:
        config = load_config(CONFIG)
        assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
        assert config.execution_authorized is True


def test_scoped_aliases_are_fenced_to_the_v2_consumer() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    for artifact_id in V2_SCOPED_INPUTS:
        artifact = workspace.artifacts[artifact_id]
        identities = artifact.semantic_identities
        assert identities["authorized_consumer_experiment_ids"] == EXPERIMENT_ID
        assert identities["fresh_evidence"] == "false"
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False


def test_cli_surface_exposes_mutation_free_dry_run() -> None:
    args = build_parser().parse_args(
        [
            CLI_SURFACE,
            "--config",
            "config.yaml",
            "--artifact-root",
            "output",
            "--dry-run",
        ]
    )
    assert args.surface == CLI_SURFACE
    assert args.dry_run is True


def test_v1_is_not_a_v2_input_or_output_predecessor() -> None:
    assert all("sceptre_router_v1" not in value for value in INPUT_ARTIFACT_IDS)
    assert OUTPUT_ARTIFACT_ID not in INPUT_ARTIFACT_IDS
    workspace = MidogppWorkspace.load(ROOT)
    consumers = [
        experiment.experiment_id
        for experiment in workspace.experiments.values()
        if OUTPUT_ARTIFACT_ID in experiment.input_artifact_ids
    ]
    assert consumers == []
