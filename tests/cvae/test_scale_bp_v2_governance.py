from __future__ import annotations

import copy

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.config import (
    fixed_experiment_payload,
    fixed_inputs_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPECTED_PHYSICAL_CELL_COUNT,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
    GovernanceError,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.inputs import (
    _validate_cache_protocol,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.physical.library import (
    build_action_library,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.protocol import (
    frozen_protocol_payload,
    terminal_claim_firewall_payload,
    validate_protocol_payload,
    validate_terminal_claim_firewall,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.source_fence import (
    validate_source_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.workstation import (
    canonical_workstation_payload,
    validate_workstation_plan,
)


def test_closed_world_source_fence_and_exact_six_inputs() -> None:
    receipt = validate_source_fence()
    inputs = fixed_inputs_payload()

    assert receipt.member_count >= 50
    assert receipt.dynamic_import_count == 0
    assert len(DIRECT_INPUT_ARTIFACT_IDS) == 6
    assert tuple(inputs["direct_input_artifact_ids"]) == DIRECT_INPUT_ARTIFACT_IDS
    assert inputs["previous_stage90_outputs_used"] is False
    assert inputs["previous_stage90_amendments_used"] is False
    assert inputs["previous_stage90_scratch_or_checkpoints_used"] is False


def test_protocol_and_claim_firewall_are_terminal_and_fail_closed() -> None:
    protocol = frozen_protocol_payload()
    claims = terminal_claim_firewall_payload()

    validate_protocol_payload(protocol)
    validate_terminal_claim_firewall(claims)
    assert fixed_experiment_payload()["consumed_test_reuse_authorized"] is True
    assert protocol["fresh_evidence"] is False
    assert protocol["target_terminal_labels_open_only_after_durable_decision_seal"] is True
    assert protocol["exact_p_fallback_required"] is True
    for key in (
        "routing_success_claimed",
        "downstream_utility_claimed",
        "deployment_claimed",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
    ):
        assert claims[key] is False

    drifted = copy.deepcopy(protocol)
    drifted["fresh_evidence"] = True
    with pytest.raises(GovernanceError):
        validate_protocol_payload(drifted)

    drifted_claims = copy.deepcopy(claims)
    drifted_claims["routing_success_claimed"] = True
    with pytest.raises(GovernanceError):
        validate_terminal_claim_firewall(drifted_claims)


def test_physical_library_and_workstation_topology_are_frozen() -> None:
    library = build_action_library()
    runtime = canonical_workstation_payload()

    assert len(library) == 90
    assert len({row.action_hash for row in library}) == 90
    assert all(row.target_center not in row.counts_by_class[0] for row in library)
    assert runtime["persistent_generation_workers"] == 2
    assert runtime["physical_cells_materialized_once"] == EXPECTED_PHYSICAL_CELL_COUNT
    assert runtime["cpu_outer_workers"] == 4
    assert runtime["support_folds_inside_outer_worker"] == "SEQUENTIAL"
    assert runtime["nested_process_pools_allowed"] is False
    assert runtime["storage_dtype"] == "float32"
    assert runtime["reduction_dtype"] == "float64"
    validate_workstation_plan(runtime)

    changed = dict(runtime)
    changed["cpu_outer_workers"] = 8
    with pytest.raises(GovernanceError):
        validate_workstation_plan(changed)


def test_cache_protocol_uses_the_immutable_builders_nested_schema() -> None:
    frozen = {
        "cache_name": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "scoring_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "cache_extractor_protocol": {
            "representation_id": EXPECTED_TEST_CACHE_REPRESENTATION_ID,
        },
    }
    alignment = {"row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH}
    report = {
        "row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "row_count": EXPECTED_TEST_ROW_COUNT,
        "fresh_evidence": False,
    }
    validation = {"status": "PASS"}

    _validate_cache_protocol(frozen, alignment, report, validation)

    legacy_flattened = copy.deepcopy(frozen)
    legacy_flattened["representation_id"] = EXPECTED_TEST_CACHE_REPRESENTATION_ID
    legacy_flattened["manifest_sha256"] = EXPECTED_TEST_MANIFEST_SHA256
    del legacy_flattened["cache_extractor_protocol"]
    with pytest.raises(GovernanceError, match="test-cache protocol drifted"):
        _validate_cache_protocol(
            legacy_flattened,
            alignment,
            report,
            validation,
        )
