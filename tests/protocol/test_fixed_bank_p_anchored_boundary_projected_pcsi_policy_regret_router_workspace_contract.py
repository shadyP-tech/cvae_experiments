from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router import (
    experiment_contracts as contracts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.config import (
    load_p_anchored_boundary_projected_pcsi_policy_regret_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.protocol import (
    FrozenProtocol,
    build_frozen_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_boundary_projected_"
    "pcsi_policy_regret_router_v1.yaml"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_registry_catalog_and_whole_test_geometry_are_fenced() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(contracts.EXPERIMENT_ID)
    output = workspace.artifacts[contracts.OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert experiment.run_recovery_strategy is None
    assert experiment.runner_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert experiment.runner_env["OMP_NUM_THREADS"] == "1"

    assert output.canonical_path == contracts.CANONICAL_OUTPUT_ROOT
    assert output.semantic_identities["eligible_test_row_count"] == "9928"
    assert output.semantic_identities["held_case_route_count"] == "218"
    assert output.semantic_identities["physical_probability_cell_count"] == "810"
    assert output.semantic_identities["outer_endpoint_model_fit_count"] == "3488"
    assert output.semantic_identities["target_local_posterior_model_fit_count"] == "436"
    assert output.semantic_identities["utility_model_fit_count"] == "1395"
    assert output.semantic_identities["whole_policy_pseudo_target_replay_count"] == "144"
    assert output.semantic_identities["transport_semantics"] == (
        "support_conditioned_endpoint_reconstructed_P_B_I_R"
    )
    assert output.semantic_identities[
        "transport_source_prior_labels_used_upstream"
    ] == "true"
    assert output.semantic_identities[
        "transport_route_local_support_labels_used_upstream"
    ] == "true"
    assert output.semantic_identities[
        "transport_held_case_evaluation_capability_used_directly"
    ] == "false"
    assert output.semantic_identities[
        "transport_pseudo_evaluation_capability_used_directly"
    ] == "false"
    assert output.semantic_identities[
        "transport_terminal_evaluation_capability_used_directly"
    ] == "false"
    assert output.semantic_identities["transport_label_free_claim"] == "false"
    assert output.semantic_identities[
        "transport_identity_level_route_noninterference_required"
    ] == "true"
    assert output.semantic_identities[
        "transport_identity_level_route_noninterference_proven"
    ] == "false"
    assert output.semantic_identities["transport_authorization_valid"] == "false"
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["transport_protocol_status"] == (
        "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
    )
    assert output.availability == "planned_protocol_blocked"
    assert output.evidence_label == "NEEDS_EVIDENCE_BLOCKED_IDENTITY_FEEDBACK"
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.required_files == REQUIRED_FILES
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert not any(
        contracts.OUTPUT_ARTIFACT_ID in candidate.input_artifact_ids
        for candidate in workspace.experiments.values()
    )


def test_four_consumer_fenced_aliases_are_terminal_only() -> None:
    workspace = _workspace()
    for artifact_id in (
        contracts.TEST_CACHE_ARTIFACT_ID,
        contracts.TEST_MANIFEST_ARTIFACT_ID,
        contracts.TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        contracts.LEDGER_AMENDMENT_ARTIFACT_ID,
    ):
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == contracts.EXPERIMENT_ID
        assert artifact.semantic_identities["fresh_evidence"] == "false"
        assert "routing_evidence" in artifact.forbidden_reuse
        assert "oracle_and_diagnostic_evidence" in artifact.forbidden_reuse
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False


def test_exact_six_input_fence_rejects_extra_and_stage90_inputs() -> None:
    config = load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(
        CONFIG
    )
    assert_input_fence(config)
    with pytest.raises(ProtocolError, match="exactly six fenced inputs"):
        assert_input_fence(
            SimpleNamespace(
                experiment_id=config.experiment_id,
                output_artifact_id=config.output_artifact_id,
                input_artifact_ids=(*contracts.INPUT_ARTIFACT_IDS, "unexpected"),
                expert_bank_root=config.expert_bank_root,
                generation_lock_root=config.generation_lock_root,
                test_cache_root=config.test_cache_root,
                test_manifest_path=config.test_manifest_path,
                test_consumption_ledger_path=config.test_consumption_ledger_path,
                ledger_amendment_path=config.ledger_amendment_path,
            )
        )
    poisoned = replace(
        config,
        ledger_amendment_path=Path(
            "/tmp/fixed_bank_p_anchored_crossfit_sample_influence_router/"
            "amendment.json"
        ),
    )
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(poisoned)


def test_protocol_rejects_freshness_or_claim_boundary_widening() -> None:
    protocol = build_frozen_protocol()
    assert protocol.protocol_hash == (
        "222a4ec08039330a94b35a43c7c73039ad4a17b1a10f0646d032c18938e607ab"
    )
    assert protocol.payload["double_exclusion_pair_count"] == 72
    assert protocol.payload["may_feed_another_experiment"] is False
    assert protocol.payload["transport_label_free_claim"] is False
    assert protocol.payload[
        "transport_held_case_evaluation_capability_used_directly"
    ] is False
    assert protocol.payload[
        "transport_identity_level_route_noninterference_required"
    ] is True
    assert protocol.payload[
        "transport_identity_level_route_noninterference_proven"
    ] is False
    assert protocol.payload["transport_authorization_valid"] is False
    assert protocol.payload["transport_protocol_status"] == (
        "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
    )

    widened = dict(protocol.payload)
    widened["fresh_evidence"] = True
    with pytest.raises(ProtocolError, match="frozen protocol drifted"):
        FrozenProtocol(widened)
    with pytest.raises(ProtocolError, match="protocol hash drifted"):
        FrozenProtocol(dict(protocol.payload), "0" * 64)
