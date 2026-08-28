"""Frozen terminal-only scientific protocol for OE-PPUR v3.

Authorization is a lifecycle state, not a scientific-method parameter.  The
protocol therefore records the gates that *must* hold before execution while
the current issuance state lives exclusively in the config's experiment,
exact-input-contract, and claim-boundary projections.  Keeping those concerns
separate gives planned and authorization-ready configs one immutable scientific
protocol hash without allowing the protocol section to contradict either
state.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol import ProtocolError
from .execution.inputs import (
    build_authorized_seven_input_contract,
    build_planned_seven_input_contract,
)
from .hashing import canonical_hash
from .identity import (
    ACTION_IDS,
    CENTERS,
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXCLUDED_CENTERS,
    EXPERIMENT_ID,
    EXPECTED_CASE_COUNT,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_TEST_ROW_COUNT,
    FRESH_EVIDENCE,
    P_ACTION_ID,
    PROBABILITY_COLUMN_IDS,
    PUBLICATION_STATUS,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
    TERMINAL_DECISION,
)


def claim_boundary_payload(
    *, execution_authorized: bool = False
) -> dict[str, object]:
    if type(execution_authorized) is not bool:
        raise ProtocolError("OE-PPUR v3 claim authority state is untyped.")
    return {
        "schema_version": "oe_ppur_v3_terminal_claim_boundary_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "fresh_evidence": FRESH_EVIDENCE,
        "execution_authorized": execution_authorized,
        "consumed_test_reuse_authorized": execution_authorized,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "cvae_compatibility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "deployment_claimed": False,
        "promotion_allowed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def _claim_restrictions_payload() -> dict[str, object]:
    """Return state-neutral restrictions shared by every lifecycle state."""

    return {
        "schema_version": "oe_ppur_v3_terminal_claim_restrictions_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "fresh_evidence": FRESH_EVIDENCE,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "cvae_compatibility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "deployment_claimed": False,
        "promotion_allowed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def _protocol_body() -> dict[str, object]:
    planned_contract = build_planned_seven_input_contract()
    authorized_contract = build_authorized_seven_input_contract()
    return {
        "schema_version": "oe_ppur_v3_terminal_scientific_protocol_v2",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "claim_dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "held_case_route_count": EXPECTED_CASE_COUNT,
        "eligible_center_ids": list(CENTERS),
        "excluded_center_ids": list(EXCLUDED_CENTERS),
        "candidate_action_ids": list(ACTION_IDS),
        "protected_probability_baseline_id": P_ACTION_ID,
        "probability_matrix_column_ids": list(PROBABILITY_COLUMN_IDS),
        "probability_matrix_shape": list(EXPECTED_PROBABILITY_MATRIX_SHAPE),
        "probability_storage_dtype": "<f4",
        "reduction_dtype": "<f8",
        "source_only_training_supervision_direct_input_ordinal": 3,
        "source_supervision_materialization_required_before_execution": True,
        "source_supervision_required_members": list(
            SOURCE_SUPERVISION_REQUIRED_MEMBERS
        ),
        "source_supervision_target_test_rows_present": False,
        "source_supervision_target_test_labels_present": False,
        "target_H_excluded_from_every_fit_normalizer_calibrator_and_candidate_pool": True,
        "source_inner_cross_fitting_required": True,
        "query_to_compatibility_to_decision_to_expert_to_true_utility_order": True,
        "proxy_is_not_true_utility": True,
        "terminal_labels_closed_preterminal": True,
        "terminal_labels_ephemeral_aggregate_only": True,
        "terminal_capability_one_shot_process_local": True,
        "terminal_evaluated_case_count": EXPECTED_CASE_COUNT,
        "exact_P_on_any_failed_gate": True,
        "direct_input_count": 7,
        "direct_input_roles": list(DIRECT_INPUT_ROLES),
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "direct_input_order_exact": True,
        "direct_input_duplicates_forbidden": True,
        "authorization_amendment_input_ordinal": 7,
        "authorization_amendment_required_before_execution": True,
        "authorized_seven_input_contract_required_before_execution": True,
        "current_authority_state_owned_by_config_not_protocol": True,
        "planned_seven_input_contract_hash": planned_contract.receipt_hash,
        "authorized_seven_input_contract_hash": authorized_contract.receipt_hash,
        "predecessor_runtime_or_artifact_reuse": False,
        "structural_scientific_service_injection_allowed": False,
        "canonical_source_sealed_service_required": True,
        "runner_writes_allowed_in_planned_state": False,
        "experiment_launch_allowed_in_planned_state": False,
        "workstation": {
            "persistent_gpu_workers": 2,
            "spawn_cpu_workers": 4,
            "cuda_visible_to_cpu_workers": False,
            "blas_threads_per_cpu_worker": 1,
            "worker_transport": "pickle_primitive_dto_only",
        },
        "publication_and_claim_restrictions": _claim_restrictions_payload(),
    }


def frozen_protocol_payload() -> dict[str, object]:
    body = _protocol_body()
    return {**body, "protocol_hash": canonical_hash(body)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or dict(payload) != frozen_protocol_payload():
        raise ProtocolError("OE-PPUR v3 protocol contract drifted.")


def validate_claim_boundary(
    payload: Mapping[str, object],
    *,
    execution_authorized: bool = False,
) -> None:
    if not isinstance(payload, Mapping) or dict(payload) != claim_boundary_payload(
        execution_authorized=execution_authorized
    ):
        raise ProtocolError("OE-PPUR v3 terminal claim firewall drifted.")


__all__ = (
    "ProtocolError",
    "claim_boundary_payload",
    "frozen_protocol_payload",
    "validate_claim_boundary",
    "validate_protocol_payload",
)
