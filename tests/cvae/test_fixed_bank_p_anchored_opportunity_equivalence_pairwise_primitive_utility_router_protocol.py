from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import pickle

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.config import (
    build_planned_config,
    frozen_config_contract_payload,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.contracts import (
    EphemeralLabelView,
    OuterSelectionLineage,
    PreterminalPhaseReceipt,
    SelectionDecisionLedger,
    SelectionDecisionLedgerEntry,
    claim_boundary_payload,
    direct_input_policy_payload,
    open_terminal_label_view,
    terminal_case_manifest_hash,
    _issue_preterminal_phase_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.fold_scope import (
    FinalOuterScope,
    FoldScope,
    composite_case_key,
    validate_complete_k_rotation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.identity import (
    ACTION_IDS,
    ANNOTATION_MANIFEST_ARTIFACT_ID,
    EXPERIMENT_ID,
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.manifest_contract import (
    ANNOTATION_MANIFEST_CONTENT_SHA256,
    ANNOTATION_MANIFEST_MEMBER,
    CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER,
    CANONICAL_TERMINAL_CASE_INVENTORY,
    CANONICAL_TERMINAL_CASE_INVENTORY_HASH,
    CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER,
    CANONICAL_TERMINAL_SPLIT,
    build_canonical_terminal_manifest_receipt,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.contracts import (
    ActionSurface,
    P_ACTION_ID,
    SelectionDecision,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.opportunity import (
    build_opportunity_case_receipt,
    build_opportunity_set,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.protocol import (
    frozen_protocol_payload,
    validate_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.source_fence import (
    validate_source_fence,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "a" * 64
RANKING_SHA = "4" * 64
_BASELINE = (0.25, 0.75)
_ZERO_OPPORTUNITY = build_opportunity_set(
    _BASELINE,
    tuple(
        ActionSurface(
            action_id=action_id,
            family=action_id.split("::", maxsplit=1)[0],
            direction=action_id.split("::", maxsplit=1)[1],
            probabilities=_BASELINE,
        )
        for action_id in ACTION_IDS
    ),
    candidate_action_ids=ACTION_IDS,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_manifest_receipt(**overrides: object):
    values: dict[str, object] = {
        "annotation_artifact_id": ANNOTATION_MANIFEST_ARTIFACT_ID,
        "manifest_member": ANNOTATION_MANIFEST_MEMBER,
        "manifest_content_sha256": ANNOTATION_MANIFEST_CONTENT_SHA256,
        "split": CANONICAL_TERMINAL_SPLIT,
        "eligible_center_ids": CENTERS,
        "row_count": EXPECTED_TEST_ROW_COUNT,
        "case_count": EXPECTED_CASE_COUNT,
        "row_counts_by_center": CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER,
        "case_counts_by_center": CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER,
        "case_inventory": CANONICAL_TERMINAL_CASE_INVENTORY,
    }
    values.update(overrides)
    return build_canonical_terminal_manifest_receipt(**values)


def _complete_decision_ledger() -> SelectionDecisionLedger:
    manifest_receipt = _canonical_manifest_receipt()
    inventory = manifest_receipt.case_inventory
    outer_lineages = tuple(
        OuterSelectionLineage(
            outer_target_center=center,
            source_surface_receipt_hash=_sha(f"source::{center}"),
            candidate_pool_receipt_hash=_sha(f"candidate::{center}"),
            pairwise_model_hash=_sha(f"model::{center}"),
            uncertainty_calibration_hash=_sha(f"calibration::{center}"),
            bacc_ranking_policy_hash=RANKING_SHA,
        )
        for center in CENTERS
    )
    lineage_by_h = {row.outer_target_center: row for row in outer_lineages}
    entries = []
    for center, case in inventory:
        opportunity_receipt = build_opportunity_case_receipt(
            center_id=center,
            case_id=case,
            opportunity=_ZERO_OPPORTUNITY,
        )
        entries.append(
            SelectionDecisionLedgerEntry(
                center,
                case,
                opportunity_receipt,
                SelectionDecision(
                    selected_action_id=P_ACTION_ID,
                    raw_winner_action_id=P_ACTION_ID,
                    fallback_to_p=True,
                    reason="no_active_unique_action_opportunity",
                    active_representative_count=0,
                    runner_up_action_id=None,
                    selected_equivalent_action_ids=(P_ACTION_ID,),
                    candidate_pool_receipt_hash=(
                        lineage_by_h[center].candidate_pool_receipt_hash
                    ),
                    pairwise_model_hash=lineage_by_h[center].pairwise_model_hash,
                    uncertainty_calibration_hash=(
                        lineage_by_h[center].uncertainty_calibration_hash
                    ),
                    opportunity_case_receipt_hash=opportunity_receipt.receipt_hash,
                    bacc_ranking_policy_hash=RANKING_SHA,
                    opportunity_active_representative_ids=(),
                ),
            )
        )
    return SelectionDecisionLedger(
        manifest_receipt=manifest_receipt,
        entries=tuple(reversed(entries)),
        outer_lineages=outer_lineages,
    )


def test_identity_is_new_planned_terminal_only_and_has_three_legal_inputs() -> None:
    assert EXPERIMENT_ID.endswith("pairwise_primitive_utility_router.v1")
    assert OUTPUT_ARTIFACT_ID.endswith("pairwise_primitive_utility_router_v1")
    assert INPUT_ARTIFACT_IDS == (
        "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
        "midogpp_output_uniform_b_v2_generation_lock_v1",
        "midogpp_dataset_contract_annotation_patch_v1",
    )
    protocol = frozen_protocol_payload()
    claims = claim_boundary_payload()
    inputs = direct_input_policy_payload()
    assert protocol["execution_authorized"] is False
    assert protocol["fresh_evidence"] is False
    assert protocol["route_policy_proxy_is_cvae_compatibility"] is False
    assert protocol["route_policy_proxy_is_nelbo_compatibility"] is False
    assert protocol["may_feed_another_experiment"] is False
    assert claims["terminal_decision"] == TERMINAL_DECISION
    assert claims["may_feed_stage50"] is False
    assert inputs["direct_input_count"] == 3
    assert inputs["test_cache_capability_registered"] is False
    assert inputs["test_label_capability_registered"] is False
    assert inputs["test_consumption_ledger_capability_registered"] is False
    assert inputs["test_cache_resolution_status"] == (
        "PENDING_SEPARATE_FUTURE_AUTHORIZATION"
    )
    assert inputs["parent_consumption_ledger_resolution_status"] == (
        "PENDING_SEPARATE_FUTURE_AUTHORIZATION"
    )
    assert inputs["authorization_amendment_status"] == "ABSENT_NOT_AUTHORIZED"


def test_protocol_payload_is_exact_and_hash_bound() -> None:
    payload = frozen_protocol_payload()
    validate_protocol_payload(payload)
    assert payload["schema_version"] == "oe_ppur_v1_terminal_protocol_v3"
    assert payload["nested_K_rotation_centers"] == "EXACT_C_MINUS_H"
    assert payload["nested_K_rotation_complete_once_per_source_center"] is True
    assert payload["nested_scopes_share_one_outer_H"] is True
    assert payload["d_identity"] == (
        "EXPLICIT_CENTER_AND_WHOLE_CASE_TUPLE_WITH_AUDIT_HASH"
    )
    assert payload["d_recovered_in_legal_final_source_refit"] is True
    assert payload["primitive_computation_precedes_metric_normalization"] is True
    assert payload["normalization_policy"] == (
        "ACTION_INVARIANT_EXPECTED_CLASS_TOTALS_PER_CASE_SCOPE"
    )
    assert payload["pairwise_fit_response_metric"] == "EXPECTED_BACC_GAIN_ONLY"
    assert payload["brier_or_log_may_enter_pairwise_ranking_response"] is False
    assert payload[
        "typed_opportunity_receipt_required_at_pairwise_fit_and_selection"
    ] is True
    assert payload["opportunity_candidate_action_inventory"] == (
        "EXACT_FROZEN_CANDIDATE_ACTION_IDS"
    )
    assert payload[
        "typed_row_posterior_prediction_required_for_primitive_and_denominator"
    ] is True
    assert payload["primitive_action_id_exact_match_to_opportunity_member"] is True
    assert payload[
        "primitive_protected_baseline_probability_hash_exact_match_to_opportunity"
    ] is True
    assert payload[
        "primitive_candidate_probability_hash_exact_match_to_opportunity_member"
    ] is True
    assert payload["primitive_denominator_posterior_model_hash_exact_match"] is True
    assert payload[
        "pairwise_fit_exact_matches_utility_action_and_probability_surface_to_opportunity"
    ] is True
    assert payload[
        "selection_exact_matches_utility_action_and_probability_surface_to_opportunity"
    ] is True
    assert payload["selection_ledger_entry_requires_typed_opportunity_receipt"] is True
    assert payload[
        "selection_ledger_entry_exact_matches_opportunity_receipt_center_and_case"
    ] is True
    assert payload[
        "selection_ledger_entry_exact_matches_decision_opportunity_receipt_hash"
    ] is True
    manifest = payload["canonical_terminal_manifest_contract"]
    assert manifest["annotation_artifact_id"] == ANNOTATION_MANIFEST_ARTIFACT_ID
    assert manifest["manifest_content_sha256"] == ANNOTATION_MANIFEST_CONTENT_SHA256
    assert manifest["terminal_row_count"] == EXPECTED_TEST_ROW_COUNT
    assert manifest["terminal_case_count"] == EXPECTED_CASE_COUNT
    assert manifest["terminal_case_inventory_hash"] == (
        CANONICAL_TERMINAL_CASE_INVENTORY_HASH
    )
    assert payload[
        "canonical_terminal_manifest_receipt_required_by_selection_ledger"
    ] is True
    assert payload[
        "canonical_terminal_manifest_receipt_hash_bound_in_preterminal_phase"
    ] is True
    assert payload["terminal_label_gate_requires_canonical_terminal_manifest_receipt"] is True
    assert payload["residual_calibration_one_sided_alpha"] == 0.2
    assert payload["residual_calibration_minimum_distinct_L_centers"] == 4
    assert payload["terminal_admission_aggregation"] == "CASE_THEN_EQUAL_CENTER"
    assert payload[
        "prelabel_selection_decision_hash_frozen_before_terminal_label_open"
    ] is True
    assert payload["outer_selection_lineage_inventory"] == (
        "EXACT_ONE_PER_ELIGIBLE_H"
    )
    assert payload[
        "preterminal_phase_uses_canonical_per_H_lineage_surface_hashes"
    ] is True
    poisoned = dict(payload)
    poisoned["fresh_evidence"] = True
    with pytest.raises(ProtocolError, match="protocol contract drifted"):
        validate_protocol_payload(poisoned)


def test_fold_scope_is_hashable_and_enforces_all_role_deletions() -> None:
    scope = FoldScope("0", "1", "2", "3", "case-1-007")
    assert hash(scope) == hash(scope)
    assert scope.excluded_fit_centers == ("0", "1", "2", "3")
    fit_cases = tuple(
        (center, f"case-{center}") for center in scope.nested_training_centers
    )
    scope.assert_fit_exclusions(
        centers=scope.nested_training_centers,
        case_keys=fit_cases,
    )
    scope.assert_hyperparameter_validation(
        center="2", case_keys=(("2", "other"),)
    )
    scope.assert_residual_calibration(
        center="3", case_keys=(("3", "other"),)
    )
    scope.assert_held_case_center("1")
    with pytest.raises(ProtocolError, match="H/J/K/L deletion"):
        scope.assert_fit_exclusions(
            centers=("0", *scope.nested_training_centers[1:]),
            case_keys=fit_cases,
        )
    with pytest.raises(ProtocolError, match="held whole case"):
        scope.assert_fit_exclusions(
            centers=scope.nested_training_centers,
            case_keys=(*fit_cases[:-1], (scope.J, scope.d)),
        )


def test_fold_scope_converts_to_neutral_receipt_with_typed_case_keys() -> None:
    scope = FoldScope("0", "1", "2", "3", "case-007")
    cases = tuple((center, f"case-{center}") for center in scope.nested_training_centers)
    receipt = scope.to_source_scope_receipt(training_cases=cases)
    assert receipt.outer_target_center == scope.H
    assert receipt.query_center == scope.J
    assert receipt.hyperparameter_center == scope.K
    assert receipt.calibration_center == scope.L
    assert receipt.heldout_case_center == scope.J
    assert receipt.heldout_case_id == scope.d
    assert receipt.training_center_ids == scope.nested_training_centers
    assert (scope.J, scope.d) not in receipt.training_case_keys
    assert receipt.training_case_keys == tuple(sorted(cases))
    assert composite_case_key("1", "same") != composite_case_key("2", "same")
    with pytest.raises(ProtocolError, match="not exact C-minus-H/J/K/L"):
        scope.to_source_scope_receipt(training_cases=cases[:-1])


def test_complete_K_rotation_requires_every_C_minus_H_center_once_and_shared_H() -> None:
    source_centers = ("1", "2", "3", "5", "6", "7", "8", "9")
    scopes = tuple(
        FoldScope(
            "0",
            source_centers[(index + 1) % len(source_centers)],
            center,
            source_centers[(index + 2) % len(source_centers)],
            f"case-{center}",
        )
        for index, center in enumerate(source_centers)
    )
    rotation = validate_complete_k_rotation(
        tuple(reversed(scopes)), outer_target_center="0"
    )
    assert tuple(scope.K for scope in rotation) == source_centers
    with pytest.raises(ProtocolError, match="complete shared-H K rotation"):
        validate_complete_k_rotation(scopes[:-1], outer_target_center="0")
    with pytest.raises(ProtocolError, match="complete shared-H K rotation"):
        validate_complete_k_rotation(
            (FoldScope("1", "0", "2", "3", "poison"), *scopes[1:]),
            outer_target_center="0",
        )


@pytest.mark.parametrize(
    "values",
    (
        ("0", "0", "2", "3", "case"),
        ("0", "1", "1", "3", "case"),
        ("0", "1", "2", "2", "case"),
        ("0", "1", "2", "4", "case"),
        ("0", "1", "2", "3", ""),
    ),
)
def test_fold_scope_role_poison_fails_closed(values: tuple[str, ...]) -> None:
    with pytest.raises(ProtocolError):
        FoldScope(*values)


def test_K_and_L_are_role_specific_and_d_is_excluded() -> None:
    scope = FoldScope("0", "1", "2", "3", "held-d")
    with pytest.raises(ProtocolError, match="K validation"):
        scope.assert_hyperparameter_validation(
            center="3", case_keys=(("3", "x"),)
        )
    with pytest.raises(ProtocolError, match="K validation"):
        scope.assert_hyperparameter_validation(
            center="2", case_keys=((scope.J, scope.d),)
        )
    scope.assert_hyperparameter_validation(
        center="2", case_keys=(("2", scope.d),)
    )
    with pytest.raises(ProtocolError, match="L calibration"):
        scope.assert_residual_calibration(
            center="2", case_keys=(("2", "x"),)
        )
    with pytest.raises(ProtocolError, match="pseudo-target J"):
        scope.assert_held_case_center("5")


def test_final_outer_scope_refits_exact_C_minus_H_after_choices_are_frozen() -> None:
    scope = FinalOuterScope("2", SHA, "b" * 64)
    assert scope.legal_source_centers == ("0", "1", "3", "5", "6", "7", "8", "9")
    assert hash(scope) == hash(scope)
    scope.assert_source_only_component(
        role="normalizer", centers=("0", "1", "3")
    )
    scope.assert_candidate_pool(scope.legal_source_centers)
    recovered = ("1", "nested-d")
    source_cases = tuple(
        (center, "nested-d" if center == "1" else f"case-{center}")
        for center in scope.legal_source_centers
    )
    scope.assert_final_estimator_fit(
        centers=scope.legal_source_centers,
        case_keys=source_cases,
        target_case_keys=((scope.H, "target-c"),),
        required_recovered_case_keys=(recovered,),
    )
    pool = scope.build_candidate_pool_receipt(
        expert_inventory=tuple(
            (f"expert-{center}", center) for center in scope.legal_source_centers
        ),
        bank_lock_hash=SHA,
        source_surface_receipt_hash="c" * 64,
    )
    assert pool.outer_target_center == scope.H
    assert pool.candidate_center_ids == scope.legal_source_centers
    assert pool.source_surface_receipt_hash == "c" * 64
    with pytest.raises(ProtocolError, match="included target H"):
        scope.assert_source_only_component(role="calibrator", centers=("0", "2"))
    with pytest.raises(ProtocolError, match="exact C-minus-H"):
        scope.assert_final_estimator_fit(
            centers=scope.legal_source_centers[:-1],
            case_keys=source_cases,
            target_case_keys=((scope.H, "target-c"),),
            required_recovered_case_keys=(recovered,),
        )
    with pytest.raises(ProtocolError, match="target-H case"):
        scope.assert_final_estimator_fit(
            centers=scope.legal_source_centers,
            case_keys=(*source_cases[:-1], (scope.H, "target-c")),
            target_case_keys=((scope.H, "target-c"),),
            required_recovered_case_keys=(recovered,),
        )
    with pytest.raises(ProtocolError, match="recover every legal nested d"):
        scope.assert_final_estimator_fit(
            centers=scope.legal_source_centers,
            case_keys=source_cases,
            target_case_keys=((scope.H, "target-c"),),
            required_recovered_case_keys=(("1", "missing-d"),),
        )


def test_preterminal_phase_binds_complete_decision_ledger_and_keeps_labels_closed() -> None:
    ledger = _complete_decision_ledger()
    assert len(ledger.entries) == 218
    assert tuple(row.outer_target_center for row in ledger.outer_lineages) == CENTERS
    assert len(
        {row.pairwise_model_hash for row in ledger.outer_lineages}
    ) == len(CENTERS)
    assert len(ledger.ledger_hash) == 64
    assert len(ledger.opportunity_surface_receipt_hash) == 64
    with pytest.raises(ProtocolError, match="guarded factory"):
        PreterminalPhaseReceipt(
            config_contract_hash="5" * 64,
            protocol_contract_hash=frozen_protocol_payload()["protocol_hash"],
            source_fence_receipt_hash="6" * 64,
            source_surface_lineage_hash=ledger.source_surface_lineage_hash,
            annotation_manifest_receipt_hash=(
                ledger.annotation_manifest_receipt_hash
            ),
            case_manifest_hash=ledger.case_manifest_hash,
            candidate_pool_lineage_hash=ledger.candidate_pool_lineage_hash,
            pairwise_model_lineage_hash=ledger.pairwise_model_lineage_hash,
            uncertainty_calibration_lineage_hash=(
                ledger.uncertainty_calibration_lineage_hash
            ),
            opportunity_surface_receipt_hash=(
                ledger.opportunity_surface_receipt_hash
            ),
            bacc_ranking_policy_hash=RANKING_SHA,
            decision_ledger=ledger,
        )
    phase = _issue_preterminal_phase_receipt(
        config_contract_hash="5" * 64,
        protocol_contract_hash=frozen_protocol_payload()["protocol_hash"],
        source_fence_receipt_hash="6" * 64,
        decision_ledger=ledger,
    )
    assert phase.execution_authorized is False
    assert phase.terminal_label_capability_openable is False
    assert pickle.loads(pickle.dumps(phase)) == phase

    class _PoisonLabels:
        def __iter__(self):
            raise AssertionError("closed label gate inspected terminal labels")

    with pytest.raises(ProtocolError, match="sealed preterminal phase"):
        EphemeralLabelView(
            _PoisonLabels(),
            scope_hash=phase.phase_hash,
            decision_ledger_hash=ledger.ledger_hash,
        )
    with pytest.raises(ProtocolError, match="terminal labels remain closed"):
        open_terminal_label_view(
            _PoisonLabels(),
            phase_receipt=phase,
            decision_ledger_hash=ledger.ledger_hash,
        )
    with pytest.raises(ProtocolError, match="decision ledger drifted"):
        open_terminal_label_view(
            _PoisonLabels(),
            phase_receipt=phase,
            decision_ledger_hash="8" * 64,
        )


def test_canonical_manifest_receipt_rejects_replacement_subset_and_row_drift() -> None:
    receipt = _canonical_manifest_receipt()
    assert receipt.case_inventory == CANONICAL_TERMINAL_CASE_INVENTORY
    assert receipt.case_inventory_hash == CANONICAL_TERMINAL_CASE_INVENTORY_HASH
    assert terminal_case_manifest_hash(receipt.case_inventory) == (
        CANONICAL_TERMINAL_CASE_INVENTORY_HASH
    )
    replacement = list(CANONICAL_TERMINAL_CASE_INVENTORY)
    replacement[0] = (replacement[0][0], "replacement-case")
    with pytest.raises(ProtocolError, match="annotation-manifest identity drifted"):
        _canonical_manifest_receipt(case_inventory=tuple(replacement))
    with pytest.raises(ProtocolError, match="case manifest is not exact"):
        _canonical_manifest_receipt(
            case_inventory=CANONICAL_TERMINAL_CASE_INVENTORY[:-1]
        )
    with pytest.raises(ProtocolError, match="annotation-manifest identity drifted"):
        _canonical_manifest_receipt(row_count=EXPECTED_TEST_ROW_COUNT - 1)
    with pytest.raises(ProtocolError, match="annotation-manifest identity drifted"):
        _canonical_manifest_receipt(manifest_content_sha256="f" * 64)


def test_selection_ledger_rejects_incomplete_or_mixed_lineage() -> None:
    ledger = _complete_decision_ledger()
    with pytest.raises(ProtocolError, match="complete canonical case inventory"):
        SelectionDecisionLedger(
            manifest_receipt=ledger.manifest_receipt,
            entries=ledger.entries,
            outer_lineages=ledger.outer_lineages[:-1],
        )
    with pytest.raises(ProtocolError, match="manifest receipt is untyped"):
        SelectionDecisionLedger(
            manifest_receipt=object(),
            entries=ledger.entries,
            outer_lineages=ledger.outer_lineages,
        )
    with pytest.raises(ProtocolError, match="complete canonical case inventory"):
        SelectionDecisionLedger(
            manifest_receipt=ledger.manifest_receipt,
            entries=ledger.entries[:-1],
            outer_lineages=ledger.outer_lineages,
        )
    poisoned_decision = replace(
        ledger.entries[0].decision,
        pairwise_model_hash="9" * 64,
    )
    poisoned_entry = SelectionDecisionLedgerEntry(
        ledger.entries[0].center_id,
        ledger.entries[0].case_id,
        ledger.entries[0].opportunity_receipt,
        poisoned_decision,
    )
    with pytest.raises(ProtocolError, match="mixed decision lineage"):
        SelectionDecisionLedger(
            manifest_receipt=ledger.manifest_receipt,
            entries=(poisoned_entry, *ledger.entries[1:]),
            outer_lineages=ledger.outer_lineages,
        )


def test_selection_ledger_entry_rejects_cross_case_opportunity_swap() -> None:
    ledger = _complete_decision_ledger()
    left, right = ledger.entries[:2]
    with pytest.raises(ProtocolError, match="opportunity receipt identity drifted"):
        SelectionDecisionLedgerEntry(
            left.center_id,
            left.case_id,
            right.opportunity_receipt,
            left.decision,
        )
    poisoned_decision = replace(
        left.decision,
        opportunity_case_receipt_hash=right.opportunity_receipt.receipt_hash,
    )
    with pytest.raises(ProtocolError, match="opportunity receipt identity drifted"):
        SelectionDecisionLedgerEntry(
            left.center_id,
            left.case_id,
            left.opportunity_receipt,
            poisoned_decision,
        )


def test_source_fence_rejects_predecessor_import_and_artifact_literal(
    tmp_path: Path,
) -> None:
    poison_import = tmp_path / "import_poison.py"
    poison_import.write_text(
        "from midogpp_thesis.cvae.diagnostics."
        "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
        "boundary_projected_router_v2 import runner\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="forbidden predecessor"):
        validate_source_fence(tmp_path)
    poison_import.write_text(
        "VALUE = 'artifacts/midogpp/90_oracles_and_diagnostics/"
        "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
        "boundary_projected_router/v1/reports/run_state.json'\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="forbidden predecessor"):
        validate_source_fence(tmp_path)


def test_strict_six_section_config_roundtrip_and_authorization_poison(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oe-ppur.yaml"
    payload = frozen_config_contract_payload()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loaded = load_config(path)
    assert loaded == replace(build_planned_config(), source_path=path.resolve())
    payload["experiment"]["execution_authorized"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="planned config contract drifted"):
        load_config(path)


def test_config_recomputes_and_rejects_combined_source_provenance_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oe-ppur-source-drift.yaml"
    payload = frozen_config_contract_payload()
    assert payload["source_provenance"]["source_scopes_are_disjoint"] is True
    assert payload["source_provenance"]["recompute_and_exact_match_on_load"] is True
    payload["source_provenance"]["core_tree_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="combined adapter/core source seal drifted"):
        load_config(path)
