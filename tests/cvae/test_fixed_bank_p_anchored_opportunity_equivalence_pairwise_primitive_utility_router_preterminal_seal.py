from __future__ import annotations

import hashlib

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1 import identity
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.config import (
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.contracts import (
    OuterSelectionLineage,
    SelectionDecisionLedger,
    SelectionDecisionLedgerEntry,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution import memmap
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.input_lineage import (
    PreterminalInputLineage,
    _build_strict_test_upstream_receipts,
    build_planned_config_protocol_receipt,
    build_preterminal_input_lineage,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.dtos import (
    OuterFoldTaskDTO,
    PredictionTaskDTO,
    WorkerResultDTO,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.pools import (
    ExecutionBatchResult,
    run_cpu_outer_pool,
    run_persistent_gpu_prediction_pool,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.preterminal import (
    LabelFreePreterminalInputs,
    seal_label_free_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.surfaces import (
    CandidateProbabilitySurfaceReceipt,
    build_candidate_probability_surface_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.fold_scope import (
    FoldScope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.manifest_contract import (
    ANNOTATION_MANIFEST_CONTENT_SHA256,
    ANNOTATION_MANIFEST_MEMBER,
    CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER,
    CANONICAL_TERMINAL_CASE_INVENTORY,
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
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "c" * 64
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
        for action_id in identity.ACTION_IDS
    ),
    candidate_action_ids=identity.ACTION_IDS,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest_receipt():
    return build_canonical_terminal_manifest_receipt(
        annotation_artifact_id=identity.ANNOTATION_MANIFEST_ARTIFACT_ID,
        manifest_member=ANNOTATION_MANIFEST_MEMBER,
        manifest_content_sha256=ANNOTATION_MANIFEST_CONTENT_SHA256,
        split=CANONICAL_TERMINAL_SPLIT,
        eligible_center_ids=identity.CENTERS,
        row_count=identity.EXPECTED_TEST_ROW_COUNT,
        case_count=identity.EXPECTED_CASE_COUNT,
        row_counts_by_center=CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER,
        case_counts_by_center=CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER,
        case_inventory=CANONICAL_TERMINAL_CASE_INVENTORY,
    )


def _manifest_rows() -> tuple[memmap.CanonicalRowIdentity, ...]:
    rows: list[memmap.CanonicalRowIdentity] = []
    ordinal = 0
    for center, center_count in CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER:
        cases = tuple(
            case
            for case_center, case in CANONICAL_TERMINAL_CASE_INVENTORY
            if case_center == center
        )
        base, remainder = divmod(center_count, len(cases))
        for case_index, case in enumerate(cases):
            for _ in range(base + int(case_index < remainder)):
                rows.append(
                    memmap.CanonicalRowIdentity(
                        row_ordinal=ordinal,
                        manifest_row_index=ordinal,
                        evaluation_row_id=f"preterminal-row-{ordinal:05d}",
                        center_id=center,
                        case_id=case,
                    )
                )
                ordinal += 1
    return tuple(rows)


def _decision_ledger(manifest) -> SelectionDecisionLedger:
    lineages = tuple(
        OuterSelectionLineage(
            outer_target_center=center,
            source_surface_receipt_hash=_sha(f"source::{center}"),
            candidate_pool_receipt_hash=_sha(f"candidate::{center}"),
            pairwise_model_hash=_sha(f"model::{center}"),
            uncertainty_calibration_hash=_sha(f"calibration::{center}"),
            bacc_ranking_policy_hash=RANKING_SHA,
        )
        for center in identity.CENTERS
    )
    by_center = {row.outer_target_center: row for row in lineages}
    entries = []
    for center, case in manifest.case_inventory:
        opportunity = build_opportunity_case_receipt(
            center_id=center,
            case_id=case,
            opportunity=_ZERO_OPPORTUNITY,
        )
        lineage = by_center[center]
        decision = SelectionDecision(
            selected_action_id=P_ACTION_ID,
            raw_winner_action_id=P_ACTION_ID,
            fallback_to_p=True,
            reason="no_active_unique_action_opportunity",
            active_representative_count=0,
            runner_up_action_id=None,
            selected_equivalent_action_ids=(P_ACTION_ID,),
            candidate_pool_receipt_hash=lineage.candidate_pool_receipt_hash,
            pairwise_model_hash=lineage.pairwise_model_hash,
            uncertainty_calibration_hash=lineage.uncertainty_calibration_hash,
            opportunity_case_receipt_hash=opportunity.receipt_hash,
            bacc_ranking_policy_hash=RANKING_SHA,
            opportunity_active_representative_ids=(),
        )
        entries.append(
            SelectionDecisionLedgerEntry(center, case, opportunity, decision)
        )
    return SelectionDecisionLedger(manifest, tuple(entries), lineages)


def _synthetic_preterminal_prediction_callback(
    task: PredictionTaskDTO,
) -> WorkerResultDTO:
    return WorkerResultDTO(
        task.task_hash,
        (task.output_path,),
        (_sha(task.task_hash),),
        9928,
        task.row_index_sha256,
        task.source_surface_sha256,
    )


def _synthetic_preterminal_outer_callback(
    task: OuterFoldTaskDTO,
) -> WorkerResultDTO:
    return WorkerResultDTO(
        task.task_hash,
        (f"/tmp/oe-ppur-preterminal-{task.task_hash}.fixture",),
        (_sha(task.task_hash),),
        9928,
        task.row_index_sha256,
        task.candidate_probability_surface_sha256,
    )


@pytest.fixture()
def execution_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    memmap.CanonicalRowAlignmentReceipt,
    CandidateProbabilitySurfaceReceipt,
    PreterminalInputLineage,
    ExecutionBatchResult,
    ExecutionBatchResult,
]:
    manifest = _manifest_receipt()
    manifest_rows = _manifest_rows()
    row_order_hash = canonical_hash(
        [row.evaluation_row_id for row in manifest_rows]
    )
    monkeypatch.setattr(
        memmap,
        "EXPECTED_EXECUTABLE_TEST_CACHE_ROW_ORDER_SHA256",
        row_order_hash,
    )
    alignment = memmap.build_canonical_row_alignment_receipt(
        manifest_receipt=manifest,
        manifest_rows=manifest_rows,
        rows=manifest_rows,
        cache_content_sha256=(
            memmap.EXPECTED_EXECUTABLE_TEST_CACHE_CONTENT_SHA256
        ),
        cache_row_order_sha256=row_order_hash,
    )
    row_index_sha256 = memmap.ImmutableRowIndexReceipt(
        tuple(row.evaluation_row_id for row in manifest_rows),
        ANNOTATION_MANIFEST_CONTENT_SHA256,
    ).row_index_sha256
    predictions = tuple(
        PredictionTaskDTO(
            task_id=f"preterminal-{index}",
            device_index=index,
            input_paths=(f"/tmp/oe-ppur-preterminal-input-{index}.bin",),
            input_hashes=(SHA,),
            row_index_sha256=row_index_sha256,
            output_path=f"/tmp/oe-ppur-preterminal-output-{index}.bin",
            route_ids=(f"route-{index}",),
        )
        for index in (0, 1)
    )
    gpu = run_persistent_gpu_prediction_pool(
        predictions,
        callback=_synthetic_preterminal_prediction_callback,
    )
    candidate_surface = build_candidate_probability_surface_receipt(
        gpu,
        row_alignment_receipt=alignment,
    )
    bank, generation = _build_strict_test_upstream_receipts()
    lineage = build_preterminal_input_lineage(
        config_protocol=build_planned_config_protocol_receipt(
            build_planned_config()
        ),
        expert_bank=bank,
        generation_lock=generation,
        candidate_surface=candidate_surface,
        manifest=manifest,
        rows=alignment,
    )
    roles = (
        ("0", "1", "2", "3"),
        ("0", "2", "3", "5"),
        ("1", "0", "2", "3"),
        ("1", "2", "3", "5"),
    )
    outer = []
    for index, values in enumerate(roles):
        scope = FoldScope(*values, f"preterminal-case-{index}")
        outer.append(
            OuterFoldTaskDTO(
                scope.H,
                scope.J,
                scope.K,
                scope.L,
                scope.d,
                scope.scope_hash,
                "/tmp/oe-ppur-preterminal-probabilities.f32",
                candidate_surface.output_file_hashes[0],
                candidate_surface.candidate_probability_surface_sha256,
                row_index_sha256,
            )
        )
    cpu = run_cpu_outer_pool(
        outer,
        callback=_synthetic_preterminal_outer_callback,
        candidate_surface=candidate_surface,
    )
    return (
        alignment,
        candidate_surface,
        lineage,
        gpu,
        cpu,
    )


def test_preterminal_requires_real_typed_complete_ledger_and_canonical_rows(
    execution_batches: tuple[
        memmap.CanonicalRowAlignmentReceipt,
        CandidateProbabilitySurfaceReceipt,
        PreterminalInputLineage,
        ExecutionBatchResult,
        ExecutionBatchResult,
    ],
) -> None:
    manifest = _manifest_receipt()
    alignment, candidate_surface, lineage, gpu, cpu = execution_batches
    inputs = LabelFreePreterminalInputs(
        lineage=lineage,
    )
    ledger = _decision_ledger(manifest)

    sealed = seal_label_free_preterminal_result(
        inputs,
        decision_ledger=ledger,
        gpu_prediction_batch=gpu,
        cpu_outer_batch=cpu,
    )

    assert sealed.decision_ledger is ledger
    assert sealed.phase_receipt.decision_ledger is ledger
    assert sealed.phase_receipt.execution_authorized is False
    assert sealed.phase_receipt.terminal_label_capability_openable is False
    assert sealed.telemetry.gpu_prediction_batch_hash == gpu.batch_hash
    assert sealed.telemetry.cpu_outer_batch_hash == cpu.batch_hash
    assert sealed.labels_opened is False
    assert sealed.terminal_capability_opened is False
    assert inputs.lineage.test_fixture_only is True
    assert inputs.candidate_probability_surface is candidate_surface
    assert inputs.row_alignment_receipt is alignment
    assert len(sealed.sealed_result_hash) == 64

    with pytest.raises(ProtocolError, match="decision ledger is untyped"):
        seal_label_free_preterminal_result(
            inputs,
            decision_ledger=object(),
            gpu_prediction_batch=gpu,
            cpu_outer_batch=cpu,
        )
    with pytest.raises(ProtocolError, match="input lineage is untyped"):
        LabelFreePreterminalInputs(lineage=object())
    with pytest.raises(TypeError):
        LabelFreePreterminalInputs(  # type: ignore[call-arg]
            config_contract_hash=SHA,
            protocol_contract_hash=SHA,
            source_fence_receipt_hash=SHA,
            fixed_bank_lock_hash=SHA,
            generation_lock_hash=SHA,
            candidate_probability_surface=candidate_surface,
            manifest_receipt=manifest,
            row_alignment_receipt=alignment,
        )
