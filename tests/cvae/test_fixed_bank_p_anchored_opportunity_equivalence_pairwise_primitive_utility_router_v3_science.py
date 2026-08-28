from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.action_compiler import (
    BasePredictionSurface,
    canonical_compiler_receipt,
    compile_action_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.candidate_pools import (
    build_final_outer_candidate_pool,
    build_held_center_candidate_pool,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.folds import (
    build_outer_fold_plan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    CENTERS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.science.admission import (
    exact_p_fail_closed_reason,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.science.outer_orchestration import (
    OuterScienceResult,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.science.target_decision import (
    OuterTargetDecisionInput,
    TargetDecisionLedger,
    TargetRowBinding,
    assemble_exact_218_case_decisions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.science.target_inventory import (
    CANONICAL_TARGET_CASE_INVENTORY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_supervision import (
    SourceSupervisionContractReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.held_actions import (
    canonical_held_action_library,
)


_HASH = "1" * 64


def _inventory():
    return tuple((f"expert-{center}", center) for center in CENTERS)


def _contract(compiler):
    library = canonical_held_action_library()
    return SourceSupervisionContractReceipt(
        compiler_receipt_hash=compiler.receipt_hash,
        producer_source_seal_sha256="2" * 64,
        held_action_library_sha256=library.library_hash,
        held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
    )


def _pools(h: str, compiler, contract):
    held = tuple(
        build_held_center_candidate_pool(
            outer_target_center=h,
            held_center=q,
            all_center_ids=CENTERS,
            expert_inventory=_inventory(),
            bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
            source_supervision_contract_hash=contract.contract_hash,
            compiler=compiler,
        )
        for q in CENTERS
        if q != h
    )
    final = build_final_outer_candidate_pool(
        outer_target_center=h,
        all_center_ids=CENTERS,
        expert_inventory=_inventory(),
        bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        source_supervision_contract_hash=contract.contract_hash,
        compiler=compiler,
    )
    return held, final


def test_fold_plan_covers_every_actual_j_d_once() -> None:
    compiler = canonical_compiler_receipt()
    contract = _contract(compiler)
    held, final = _pools("0", compiler, contract)
    cases = {center: (f"{center}-a", f"{center}-b") for center in CENTERS}
    plan = build_outer_fold_plan(
        outer_target_center="0",
        cases_by_center=cases,
        held_pool_receipts=held,
        final_pool_receipt=final,
        compiler=compiler,
        source_supervision_contract_hash=contract.contract_hash,
    )
    expected = {(center, case) for center in CENTERS if center != "0" for case in cases[center]}
    assert {(scope.J, scope.d) for scope in plan.case_crossfit_scopes} == expected
    assert len(plan.scopes) == 8
    assert len(plan.case_crossfit_scopes) == 16


def _target_inputs():
    compiler = canonical_compiler_receipt()
    contract = _contract(compiler)
    cases_by_center = {
        center: tuple(case for value, case in CANONICAL_TARGET_CASE_INVENTORY if value == center)
        for center in CENTERS
    }
    result = []
    for center, count in EXPECTED_TEST_ROWS_BY_CENTER:
        _held, final = _pools(center, compiler, contract)
        row_ids = tuple(f"{center}-row-{index:05d}" for index in range(count))
        candidates = final.candidate_center_ids
        base = BasePredictionSurface(
            outer_target_center=center,
            evaluated_center=center,
            row_ids=row_ids,
            equal_union_probabilities=tuple(0.40 for _ in row_ids),
            union_probabilities=tuple(0.60 for _ in row_ids),
            expert_probabilities=tuple((candidate, tuple(0.55 for _ in row_ids)) for candidate in candidates),
            candidate_pool_receipt_hash=final.receipt_hash,
        )
        surface = compile_action_surface(base, candidate_pool=final, compiler=compiler)
        outer = OuterScienceResult(
            outer_target_center=center,
            plan_hash=_HASH,
            source_surface_lineage_hash="3" * 64,
            admitted=False,
            row_posterior_model=None,
            row_oof_predictions=(),
            pairwise_model=None,
            uncertainty_calibration=None,
            admission=None,
            fallback=exact_p_fail_closed_reason(
                outer_target_center=center,
                reason_code="source_model_unavailable",
                evidence_hash="4" * 64,
            ),
        )
        case_ids = cases_by_center[center]
        bindings = tuple(
            TargetRowBinding(index, row_id, center, case_ids[index % len(case_ids)])
            for index, row_id in enumerate(row_ids)
        )
        result.append(OuterTargetDecisionInput(outer, surface, bindings, final))
    return tuple(result)


def test_exact_218_assembler_seals_unadmitted_outer_science_to_p() -> None:
    ledger = assemble_exact_218_case_decisions(
        _target_inputs(),
        expected_case_inventory=CANONICAL_TARGET_CASE_INVENTORY,
    )
    assert len(ledger.decisions) == 218
    assert ledger.exact_p_count == 218
    assert ledger.rank_unavailable_count == 218
    assert all(row.selected_action_id == "P_PROTECTED" for row in ledger.decisions)
    assert all(row.admission_decision_receipt is None for row in ledger.decisions)
    with pytest.raises(ProtocolError):
        TargetDecisionLedger(ledger.decisions, ledger.expected_case_inventory)


def test_outer_target_input_type_checks_before_dereference() -> None:
    with pytest.raises(ProtocolError):
        OuterTargetDecisionInput(object(), object(), (), object())  # type: ignore[arg-type]
