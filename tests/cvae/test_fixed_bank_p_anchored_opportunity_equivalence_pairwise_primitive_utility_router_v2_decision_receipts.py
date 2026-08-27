from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution.decision_receipts import (
    CANONICAL_CASE_INVENTORY,
    TypedCaseDecisionReceipt,
    TypedPreterminalDecisionLedgerReceipt,
    derive_decision_source_hash,
    seal_typed_preterminal_decision_ledger,
    validate_typed_preterminal_decision_ledger,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution.probability_matrix_receipts import (
    EXPECTED_PROBABILITY_COLUMNS,
    PROBABILITY_STORAGE_BYTE_ORDER,
    PROBABILITY_STORAGE_DTYPE,
    PROBABILITY_STORAGE_MEMORY_ORDER,
    ROW_BYTE_WIDTH,
    _issue_parsed_probability_matrix_receipt,
    _issue_parsed_probability_shard_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution_admission import (
    _issue_six_input_admission_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.identity import (
    ACTION_IDS,
    CENTERS,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
    P_ACTION_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.phase_contracts import (
    OuterFoldExecutionReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.row_binding import (
    derive_admitted_row_binding,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.admission import (
    seal_admission_decision,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.contracts import (
    ActionQuery,
    ActionSurface,
    BaccRankingPolicy,
    CandidatePoolReceipt,
    NormalizedUtility,
    PairwiseRankerModel,
    UncertaintyCalibration,
    UncertaintyComponent,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.opportunity import (
    build_opportunity_case_receipt,
    build_opportunity_set,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.pairwise_features import (
    design_names,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.selection import (
    assemble_action_selection_evidence,
    select_fail_closed_action,
)


ACTIVE_ACTION = "B::zero_to_one"
ACTION_SCHEMA = tuple(
    sorted(
        (
            action_id,
            action_id.split("::", maxsplit=1)[0],
            action_id.split("::", maxsplit=1)[1],
        )
        for action_id in ACTION_IDS
    )
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _admission(seed: str = "canonical"):
    validated = SimpleNamespace(
        input_binding_hash=_sha(f"{seed}:input-binding"),
        input_location_binding_sha256=_sha(
            f"{seed}:input-location-binding"
        ),
        bank_content_index_sha256=EXPECTED_BANK_CONTENT_INDEX_SHA256,
        generation_content_index_sha256=(
            EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ),
        cache_content_sha256=EXPECTED_TEST_CACHE_CONTENT_HASH,
        cache_row_order_sha256=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        manifest_sha256=EXPECTED_TEST_MANIFEST_SHA256,
        parent_ledger_sha256=EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
        artifact_root=f"/safe/{seed}/output",
        scratch_root=f"/safe/{seed}/scratch",
    )
    return _issue_six_input_admission_receipt(
        config=SimpleNamespace(contract_hash=_sha(f"{seed}:config")),
        validated=validated,
        protocol_hash=_sha(f"{seed}:protocol"),
        source_hash=_sha(f"{seed}:source"),
        amendment_sha256=_sha(f"{seed}:amendment"),
    )


def _matrix(admission):
    binding = derive_admitted_row_binding(admission)
    column_hashes = tuple(_sha(f"matrix-column-{index}") for index in range(7))
    worker = _sha("gpu-worker")
    result_file = _sha("gpu-result-file")
    shard = _issue_parsed_probability_shard_receipt(
        shard_ordinal=0,
        file_sha256=result_file,
        gpu_worker_result_sha256=worker,
        row_start=0,
        row_stop=9_928,
        shape=(9_928, 7),
        column_ids=EXPECTED_PROBABILITY_COLUMNS,
        column_content_sha256s=column_hashes,
        dtype=PROBABILITY_STORAGE_DTYPE,
        byte_order=PROBABILITY_STORAGE_BYTE_ORDER,
        memory_order=PROBABILITY_STORAGE_MEMORY_ORDER,
        byte_length=9_928 * ROW_BYTE_WIDTH,
        value_count=9_928 * 7,
        minimum_probability=0.01,
        maximum_probability=0.99,
        descriptor_read_only=True,
        no_follow_used=True,
        stable_identity_revalidated=True,
    )
    return _issue_parsed_probability_matrix_receipt(
        six_input_admission_hash=admission.receipt_hash,
        input_binding_hash=admission.input_binding_hash,
        row_binding_hash=binding.receipt_hash,
        cache_content_sha256=binding.cache_content_sha256,
        cache_row_order_sha256=binding.cache_row_order_sha256,
        manifest_sha256=binding.manifest_sha256,
        case_inventory_sha256=binding.case_inventory_sha256,
        row_index_sha256=binding.row_index_sha256,
        row_alignment_receipt_hash=binding.row_alignment_receipt_hash,
        gpu_prediction_batch_hash=_sha("gpu-batch"),
        gpu_result_surface_sha256=_sha("gpu-surface"),
        gpu_worker_result_hashes=(worker,),
        gpu_result_file_hashes=(result_file,),
        shards=(shard,),
        matrix_content_sha256=_sha("matrix-content"),
        column_content_sha256s=column_hashes,
        scientific_values_validated=True,
    )


@lru_cache(maxsize=None)
def _center_assets(center: str, variant: str = "canonical"):
    sources = tuple(value for value in CENTERS if value != center)
    policy = BaccRankingPolicy()
    pool = CandidatePoolReceipt(
        outer_target_center=center,
        all_center_ids=CENTERS,
        candidate_center_ids=sources,
        expert_inventory=tuple(
            (f"expert-{source}", source) for source in sources
        ),
        bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        source_surface_receipt_hash=_sha(f"source-surface:{center}:{variant}"),
    )
    names = design_names(("x",), ACTION_SCHEMA)
    coefficients = tuple(
        0.30 if name == f"action_intercept::{ACTIVE_ACTION}" else 0.0
        for name in names
    )
    model = PairwiseRankerModel(
        feature_names=("x",),
        feature_mean=(0.0,),
        feature_scale=(1.0,),
        action_schema=ACTION_SCHEMA,
        candidate_action_ids=tuple(sorted(ACTION_IDS)),
        design_names=names,
        coefficients=coefficients,
        selected_alpha=1.0,
        alpha_grid=(1.0,),
        delete_center_losses=tuple((1.0, source, 0.1) for source in sources),
        alpha_selection_summary=((1.0, 0.1, 0.1),),
        training_center_ids=sources,
        training_case_count=len(sources),
        training_contrast_count=len(sources),
        source_scope_receipt_hash=_sha(f"model-scope:{center}:{variant}"),
        candidate_pool_receipt_hash=pool.receipt_hash,
        opportunity_surface_receipt_hash=_sha(
            f"model-opportunity:{center}:{variant}"
        ),
        bacc_ranking_policy_hash=policy.policy_hash,
    )
    scopes = tuple(_sha(f"calibration:{center}:{index}") for index in range(4))
    specifications = (
        ("bacc", "lower"),
        ("brier", "upper"),
        ("log", "upper"),
        ("pairwise", "lower"),
    )
    components = tuple(
        sorted(
            (
                UncertaintyComponent(
                    action_id=ACTIVE_ACTION,
                    comparator_id=P_ACTION_ID,
                    metric=metric,
                    side=side,
                    offset=0.01,
                    alpha=0.1,
                    center_count=4,
                    case_count=4,
                    scope_receipt_hashes=scopes,
                )
                for metric, side in specifications
            ),
            key=lambda row: (
                row.action_id,
                row.comparator_id,
                row.metric,
                row.side,
            ),
        )
    )
    calibration = UncertaintyCalibration(
        components=components,
        outer_target_center=center,
        calibration_scope_receipt_hashes=scopes,
    )
    return pool, model, calibration, policy


def _opportunity(center: str, case_id: str, *, active: bool):
    baseline = (0.4, 0.6)
    surfaces = tuple(
        ActionSurface(
            action_id,
            family,
            direction,
            (0.8, 0.6) if active and action_id == ACTIVE_ACTION else baseline,
        )
        for action_id, family, direction in ACTION_SCHEMA
    )
    return build_opportunity_case_receipt(
        center_id=center,
        case_id=case_id,
        opportunity=build_opportunity_set(
            baseline,
            surfaces,
            candidate_action_ids=tuple(sorted(ACTION_IDS)),
        ),
    )


def _decision(
    center: str,
    case_id: str,
    *,
    active: bool = False,
    variant: str = "canonical",
):
    pool, model, calibration, policy = _center_assets(center, variant)
    opportunity = _opportunity(center, case_id, active=active)
    if active:
        member = opportunity.opportunity.member(ACTIVE_ACTION)
        query = ActionQuery(
            ACTIVE_ACTION,
            member.family,
            member.direction,
            model.feature_names,
            (0.0,),
        )
        utility = NormalizedUtility(
            bacc_gain=0.20,
            brier_loss_delta=-0.10,
            log_loss_delta=-0.10,
            action_id=ACTIVE_ACTION,
            baseline_probability_hash=opportunity.opportunity.baseline_hash,
            candidate_probability_hash=member.probability_hash,
            denominator_scope_id=f"case::{center}::{case_id}",
            denominator_eta_hash=_sha(f"eta:{center}:{case_id}"),
            row_manifest_hash=_sha(f"rows:{center}:{case_id}"),
            primitive_response_hash=_sha(f"primitive:{center}:{case_id}"),
            posterior_model_hash=_sha(f"posterior-model:{center}"),
            posterior_scope_receipt_hash=_sha(f"posterior-scope:{center}"),
        )
        evidence = (
            assemble_action_selection_evidence(
                query=query,
                equivalent_action_ids=(ACTIVE_ACTION,),
                utility=utility,
                comparator_queries=(ActionQuery.p_anchor(model.feature_names),),
                candidate_pool=pool,
                pairwise_model=model,
                uncertainty_calibration=calibration,
                opportunity_receipt=opportunity,
                ranking_policy=policy,
            ),
        )
    else:
        evidence = ()
    selected = select_fail_closed_action(
        evidence,
        candidate_pool=pool,
        pairwise_model=model,
        uncertainty_calibration=calibration,
        opportunity_receipt=opportunity,
        ranking_policy=policy,
    )
    return seal_admission_decision(
        center_id=center,
        case_id=case_id,
        decision=selected,
        candidate_evidence=evidence,
        candidate_pool=pool,
        pairwise_model=model,
        uncertainty_calibration=calibration,
        opportunity_receipt=opportunity,
        ranking_policy=policy,
    )


def _decisions(*, one_active: bool = True):
    return tuple(
        _decision(center, case_id, active=one_active and ordinal == 0)
        for ordinal, (center, case_id) in enumerate(CANONICAL_CASE_INVENTORY)
    )


def _ledger_bundle(*, one_active: bool = True):
    admission = _admission()
    matrix = _matrix(admission)
    decisions = _decisions(one_active=one_active)
    source_hash = derive_decision_source_hash(
        admission_receipt=admission,
        matrix_receipt=matrix,
        decisions=decisions,
    )
    outer = OuterFoldExecutionReceipt(
        parsed_probability_matrix_receipt_hash=matrix.receipt_hash,
        outer_center_ids=CENTERS,
        ordered_outer_result_hashes=tuple(
            _sha(f"outer-result:{center}") for center in CENTERS
        ),
        decision_source_hash=source_hash,
    )
    ledger = seal_typed_preterminal_decision_ledger(
        admission_receipt=admission,
        matrix_receipt=matrix,
        outer_fold_receipt=outer,
        decisions=decisions,
    )
    return admission, matrix, decisions, outer, ledger


def _rehash_ledger_after_case_mutation(ledger) -> None:
    object.__setattr__(
        ledger,
        "ordered_case_decision_hashes",
        tuple(row.decision_hash for row in ledger.decisions),
    )
    object.__setattr__(ledger, "receipt_hash", canonical_hash(ledger._payload()))


def _rehash_ledger_after_outer_mutation(ledger) -> None:
    object.__setattr__(
        ledger,
        "outer_lineage_surface_hash",
        canonical_hash(
            {
                "schema_version": (
                    "oe_ppur_v2_outer_decision_lineage_surface_v1"
                ),
                "ordered_outer_lineage_hashes": [
                    (row.center_id, row.lineage_hash)
                    for row in ledger.outer_lineages
                ],
            }
        ),
    )
    _rehash_ledger_after_case_mutation(ledger)


def test_seals_canonical_typed_decisions_and_derives_every_summary() -> None:
    admission, matrix, _, outer, ledger = _ledger_bundle()

    assert ledger.case_count == 218
    assert ledger.exact_p_fallback_count == 217
    assert dict(ledger.selected_action_counts) == {
        P_ACTION_ID: 217,
        **{action_id: int(action_id == ACTIVE_ACTION) for action_id in ACTION_IDS},
    }
    assert tuple((row.center_id, row.case_id) for row in ledger.decisions) == (
        CANONICAL_CASE_INVENTORY
    )
    assert ledger.decisions[0].selected_action_id == ACTIVE_ACTION
    assert ledger.decisions[0].selected_probability_column_index == 1
    assert ledger.decisions[0].selected_probability_column_sha256 == (
        matrix.column_content_sha256s[1]
    )
    assert ledger.decisions[1].selected_probability_column_index == 0
    assert ledger.decisions[1].selected_probability_column_sha256 == (
        matrix.column_content_sha256s[0]
    )
    assert len(ledger.outer_lineages) == len(CENTERS)
    assert all(
        row.outer_lineage_hash
        == ledger.outer_lineages[CENTERS.index(row.center_id)].lineage_hash
        for row in ledger.decisions
    )
    assert ledger.to_payload()["terminal_labels_opened"] is False
    assert ledger.to_payload()["raw_labels_persisted"] is False
    assert validate_typed_preterminal_decision_ledger(
        ledger,
        admission_receipt=admission,
        matrix_receipt=matrix,
        outer_fold_receipt=outer,
    ) is ledger


def test_order_is_canonical_and_hash_is_deterministic() -> None:
    admission, matrix, decisions, outer, ledger = _ledger_bundle()
    reversed_ledger = seal_typed_preterminal_decision_ledger(
        admission_receipt=admission,
        matrix_receipt=matrix,
        outer_fold_receipt=outer,
        decisions=tuple(reversed(decisions)),
    )
    assert reversed_ledger == ledger
    assert reversed_ledger.receipt_hash == ledger.receipt_hash


def test_rejects_duplicate_missing_and_added_case_inventory() -> None:
    admission, matrix, decisions, outer, _ = _ledger_bundle()
    with pytest.raises(ProtocolError, match="duplicates"):
        seal_typed_preterminal_decision_ledger(
            admission_receipt=admission,
            matrix_receipt=matrix,
            outer_fold_receipt=outer,
            decisions=(*decisions[:-1], decisions[0]),
        )
    with pytest.raises(ProtocolError, match="missing or adds"):
        derive_decision_source_hash(
            admission_receipt=admission,
            matrix_receipt=matrix,
            decisions=decisions[:-1],
        )


def test_rejects_invalid_action_and_mixed_outer_model_lineage() -> None:
    admission, matrix, decisions, outer, _ = _ledger_bundle()
    poisoned_action = _decision(*CANONICAL_CASE_INVENTORY[0], active=True)
    object.__setattr__(
        poisoned_action.selection_decision,
        "selected_action_id",
        "NOT_A_FROZEN_ACTION",
    )
    with pytest.raises(ProtocolError):
        derive_decision_source_hash(
            admission_receipt=admission,
            matrix_receipt=matrix,
            decisions=(poisoned_action, *decisions[1:]),
        )

    mixed = _decision(*CANONICAL_CASE_INVENTORY[1], variant="mixed-model")
    mixed_decisions = (decisions[0], mixed, *decisions[2:])
    mixed_source_hash = derive_decision_source_hash(
        admission_receipt=admission,
        matrix_receipt=matrix,
        decisions=mixed_decisions,
    )
    mixed_outer = replace(outer, decision_source_hash=mixed_source_hash)
    with pytest.raises(ProtocolError, match="mixed model, pool, or policy"):
        seal_typed_preterminal_decision_ledger(
            admission_receipt=admission,
            matrix_receipt=matrix,
            outer_fold_receipt=mixed_outer,
            decisions=mixed_decisions,
        )


def test_rejects_admission_matrix_and_outer_lineage_mismatch() -> None:
    admission, matrix, decisions, outer, _ = _ledger_bundle()
    unrelated = _admission("unrelated")
    with pytest.raises(ProtocolError, match="unrelated row binding"):
        derive_decision_source_hash(
            admission_receipt=unrelated,
            matrix_receipt=matrix,
            decisions=decisions,
        )
    with pytest.raises(ProtocolError, match="outer decision lineage"):
        seal_typed_preterminal_decision_ledger(
            admission_receipt=admission,
            matrix_receipt=matrix,
            outer_fold_receipt=replace(
                outer,
                parsed_probability_matrix_receipt_hash=_sha("other-matrix"),
            ),
            decisions=decisions,
        )
    with pytest.raises(ProtocolError, match="outer decision lineage"):
        seal_typed_preterminal_decision_ledger(
            admission_receipt=admission,
            matrix_receipt=matrix,
            outer_fold_receipt=replace(
                outer,
                decision_source_hash=_sha("invented-decision-source"),
            ),
            decisions=decisions,
        )


def test_receipts_reject_free_form_construction_and_replacement() -> None:
    _, _, _, _, ledger = _ledger_bundle()
    case = ledger.decisions[0]
    with pytest.raises(ProtocolError, match="bypassed typed sealing"):
        replace(case, selected_action_id=P_ACTION_ID)
    with pytest.raises(ProtocolError, match="bypassed typed sealing"):
        replace(ledger, decisions=ledger.decisions)
    with pytest.raises(ProtocolError, match="bypassed typed sealing"):
        TypedCaseDecisionReceipt(
            center_id=case.center_id,
            case_id=case.case_id,
            selected_action_id=case.selected_action_id,
            admission_decision_receipt_hash=(
                case.admission_decision_receipt_hash
            ),
            selection_decision_hash=case.selection_decision_hash,
            opportunity_case_receipt_hash=case.opportunity_case_receipt_hash,
            opportunity_hash=case.opportunity_hash,
            posterior_model_hashes=case.posterior_model_hashes,
            posterior_scope_receipt_hashes=(
                case.posterior_scope_receipt_hashes
            ),
            outer_lineage_hash=case.outer_lineage_hash,
            outer_result_hash=case.outer_result_hash,
            six_input_admission_hash=case.six_input_admission_hash,
            parsed_probability_matrix_receipt_hash=(
                case.parsed_probability_matrix_receipt_hash
            ),
            matrix_content_sha256=case.matrix_content_sha256,
            row_binding_hash=case.row_binding_hash,
            selected_probability_column_sha256=(
                case.selected_probability_column_sha256
            ),
            outer_fold_receipt_hash=case.outer_fold_receipt_hash,
            decision_source_hash=case.decision_source_hash,
        )


def test_validator_rechecks_every_persisted_case_matrix_column_binding() -> None:
    admission, matrix, _, outer, ledger = _ledger_bundle()
    case = ledger.decisions[0]
    object.__setattr__(
        case,
        "selected_probability_column_sha256",
        _sha("coherently-forged-selected-column"),
    )
    object.__setattr__(case, "decision_hash", canonical_hash(case._payload()))
    _rehash_ledger_after_case_mutation(ledger)

    # The nested and top-level receipt hashes now agree with the forged
    # persistence payload.  Revalidation must still reopen the matrix lineage
    # and reject the selected action's wrong column digest.
    with pytest.raises(ProtocolError, match="case/matrix-column lineage"):
        validate_typed_preterminal_decision_ledger(
            ledger,
            admission_receipt=admission,
            matrix_receipt=matrix,
            outer_fold_receipt=outer,
        )


def test_validator_rechecks_every_persisted_center_outer_result_binding() -> None:
    admission, matrix, _, outer, ledger = _ledger_bundle()
    lineage = ledger.outer_lineages[0]
    forged_result_hash = _sha("coherently-forged-outer-result")
    object.__setattr__(lineage, "outer_result_hash", forged_result_hash)
    object.__setattr__(lineage, "lineage_hash", canonical_hash(lineage._payload()))
    for case in ledger.decisions:
        if case.center_id != lineage.center_id:
            continue
        object.__setattr__(case, "outer_result_hash", forged_result_hash)
        object.__setattr__(case, "outer_lineage_hash", lineage.lineage_hash)
        object.__setattr__(case, "decision_hash", canonical_hash(case._payload()))
    _rehash_ledger_after_outer_mutation(ledger)

    # All internal case/outer hashes are coherent, so only exact matching to
    # the supplied outer-fold result inventory can detect this substitution.
    with pytest.raises(ProtocolError, match="center/outer-result lineage"):
        validate_typed_preterminal_decision_ledger(
            ledger,
            admission_receipt=admission,
            matrix_receipt=matrix,
            outer_fold_receipt=outer,
        )


def test_exact_p_count_is_derived_not_caller_supplied() -> None:
    _, _, _, _, ledger = _ledger_bundle(one_active=False)
    assert ledger.exact_p_fallback_count == 218
    assert dict(ledger.selected_action_counts)[P_ACTION_ID] == 218
    assert "exact_p_fallback_count" not in (
        seal_typed_preterminal_decision_ledger.__annotations__
    )
    with pytest.raises(TypeError):
        TypedPreterminalDecisionLedgerReceipt(
            six_input_admission_hash=ledger.six_input_admission_hash,
            input_binding_hash=ledger.input_binding_hash,
            parsed_probability_matrix_receipt_hash=(
                ledger.parsed_probability_matrix_receipt_hash
            ),
            matrix_content_sha256=ledger.matrix_content_sha256,
            row_binding_hash=ledger.row_binding_hash,
            outer_fold_receipt_hash=ledger.outer_fold_receipt_hash,
            decision_source_hash=ledger.decision_source_hash,
            decisions=ledger.decisions,
            outer_lineages=ledger.outer_lineages,
            exact_p_fallback_count=0,
        )
