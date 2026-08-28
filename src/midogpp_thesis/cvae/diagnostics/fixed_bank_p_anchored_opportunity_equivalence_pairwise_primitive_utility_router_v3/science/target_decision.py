"""Label-free exact-218 target decision assembly with per-case exact-P fallback."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility import (
    ActionQuery,
    AdmissionDecisionReceipt,
    ActionSurface,
    BaccRankingPolicy,
    RowPosteriorPrediction,
    SelectionDecision,
    assemble_action_selection_evidence,
    build_expected_denominators,
    build_opportunity_set,
    canonical_sha256,
    expected_additive_utility,
    normalize_expected_utility,
    seal_admission_decision,
    select_fail_closed_action,
)
from ....routing.pairwise_primitive_utility.opportunity import (
    build_opportunity_case_receipt,
)
from ....routing.pairwise_primitive_utility.row_posterior_crossfit import (
    predict_source_row_posterior,
)
from ..action_compiler import CompiledActionSurface
from ..candidate_pools import (
    ALL_ACTION_IDS,
    CANDIDATE_ACTION_IDS,
    P_ACTION_ID,
    FinalOuterCandidatePoolReceipt,
)
from ..feature_engineering import (
    ACTION_FEATURE_NAMES,
    ROW_FEATURE_NAMES,
    build_action_features,
    build_row_features,
)
from ..hashing import require_sha256
from ..identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    EXPECTED_TEST_ROW_COUNT,
)
from .outer_orchestration import OuterScienceResult
from .target_inventory import (
    CANONICAL_TARGET_CASE_INVENTORY,
    target_case_inventory_sha256,
)


_LEDGER_TOKEN = object()


@dataclass(frozen=True, slots=True)
class TargetRowBinding:
    row_index: int
    row_id: str
    center_id: str
    case_id: str

    def __post_init__(self) -> None:
        if int(self.row_index) < 0 or not self.row_id or self.center_id not in CENTERS or not self.case_id:
            raise ProtocolError("OE-PPUR v3 target row binding drifted.")
        object.__setattr__(self, "row_index", int(self.row_index))


@dataclass(frozen=True, slots=True)
class TargetCaseDecision:
    center_id: str
    case_id: str
    selected_action_id: str
    reason: str
    row_indices: tuple[int, ...]
    row_manifest_hash: str
    outer_result_hash: str
    predicted_action_scores: tuple[tuple[str, float | None], ...]
    rank_available: bool
    admission_decision_receipt: AdmissionDecisionReceipt | None
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        indices = tuple(int(value) for value in self.row_indices)
        scores = tuple((str(action), None if value is None else float(value)) for action, value in self.predicted_action_scores)
        score_values = tuple(value for _action, value in scores)
        admission = self.admission_decision_receipt
        if (
            self.center_id not in CENTERS
            or not self.case_id
            or self.selected_action_id not in ALL_ACTION_IDS
            or not self.reason
            or tuple(sorted(set(indices))) != indices
            or tuple(action for action, _value in scores) != ALL_ACTION_IDS
            or type(self.rank_available) is not bool
            or self.rank_available != all(value is not None and math.isfinite(value) for value in score_values)
            or (not self.rank_available and any(value is not None for value in score_values))
            or (admission is None and self.selected_action_id != P_ACTION_ID)
            or (
                admission is not None
                and (
                    not isinstance(admission, AdmissionDecisionReceipt)
                    or (admission.center_id, admission.case_id) != (self.center_id, self.case_id)
                    or admission.selection_decision.selected_action_id != self.selected_action_id
                )
            )
        ):
            raise ProtocolError("OE-PPUR v3 target case decision is not fail-closed.")
        object.__setattr__(self, "row_indices", indices)
        object.__setattr__(self, "row_manifest_hash", require_sha256(self.row_manifest_hash, "target row manifest hash"))
        object.__setattr__(self, "outer_result_hash", require_sha256(self.outer_result_hash, "target outer result hash"))
        object.__setattr__(self, "predicted_action_scores", scores)
        object.__setattr__(self, "decision_hash", canonical_sha256({
            "schema": "oe_ppur_v3_preterminal_target_case_decision_v1",
            "center_id": self.center_id,
            "case_id": self.case_id,
            "selected_action_id": self.selected_action_id,
            "reason": self.reason,
            "row_indices": indices,
            "row_manifest_hash": self.row_manifest_hash,
            "outer_result_hash": self.outer_result_hash,
            "predicted_action_scores": scores,
            "rank_available": self.rank_available,
            "admission_decision_receipt_hash": None if admission is None else admission.receipt_hash,
            "selection_decision_hash": None if admission is None else admission.selection_decision.decision_hash,
            "exact_P_fallback": self.selected_action_id == P_ACTION_ID,
            "target_labels_used": False,
        }))

    @property
    def selection_decision(self) -> SelectionDecision | None:
        return None if self.admission_decision_receipt is None else self.admission_decision_receipt.selection_decision


@dataclass(frozen=True, slots=True)
class OuterTargetDecisionInput:
    outer_science: OuterScienceResult
    final_surface: CompiledActionSurface
    row_bindings: tuple[TargetRowBinding, ...]
    final_pool_receipt: FinalOuterCandidatePoolReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.outer_science, OuterScienceResult):
            raise ProtocolError("OE-PPUR v3 outer target decision science is untyped.")
        h = self.outer_science.outer_target_center
        bindings = tuple(self.row_bindings)
        if (
            not isinstance(self.outer_science, OuterScienceResult)
            or not isinstance(self.final_surface, CompiledActionSurface)
            or not isinstance(self.final_pool_receipt, FinalOuterCandidatePoolReceipt)
            or self.final_surface.receipt.outer_target_center != h
            or self.final_surface.receipt.evaluated_center != h
            or self.final_pool_receipt.outer_target_center != h
            or not bindings
            or any(row.center_id != h for row in bindings)
        ):
            raise ProtocolError("OE-PPUR v3 outer target decision input drifted.")


@dataclass(frozen=True, slots=True)
class TargetDecisionLedger:
    decisions: tuple[TargetCaseDecision, ...]
    expected_case_inventory: tuple[tuple[str, str], ...]
    _factory_token: InitVar[object | None] = None
    exact_p_count: int = field(init=False)
    rank_unavailable_count: int = field(init=False)
    ledger_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        expected = tuple(self.expected_case_inventory)
        decisions = tuple(self.decisions)
        keys = tuple((row.center_id, row.case_id) for row in decisions)
        if (
            _factory_token is not _LEDGER_TOKEN
            or expected != CANONICAL_TARGET_CASE_INVENTORY
            or target_case_inventory_sha256(expected) != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
            or len(expected) != EXPECTED_CASE_COUNT
            or len(set(expected)) != EXPECTED_CASE_COUNT
            or keys != expected
            or len({row.decision_hash for row in decisions}) != EXPECTED_CASE_COUNT
        ):
            raise ProtocolError("OE-PPUR v3 target decision ledger is not exact 218 cases.")
        object.__setattr__(self, "exact_p_count", sum(row.selected_action_id == P_ACTION_ID for row in decisions))
        object.__setattr__(self, "rank_unavailable_count", sum(not row.rank_available for row in decisions))
        object.__setattr__(self, "ledger_hash", canonical_sha256({
            "schema": "oe_ppur_v3_exact_218_case_preterminal_ledger_v1",
            "case_inventory": expected,
            "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            "decision_hashes": tuple(row.decision_hash for row in decisions),
            "exact_P_count": self.exact_p_count,
            "rank_unavailable_count": self.rank_unavailable_count,
            "rank_diagnostic_policy": "AVAILABLE_CASES_ONLY_NO_IMPUTATION",
            "terminal_labels_opened": False,
        }))


def _fallback_case(
    center: str,
    case_id: str,
    *,
    reason: str,
    evidence: object,
    outer_hash: str,
    row_indices: Sequence[int] = (),
) -> TargetCaseDecision:
    return TargetCaseDecision(
        center_id=center,
        case_id=case_id,
        selected_action_id=P_ACTION_ID,
        reason=reason,
        row_indices=tuple(row_indices),
        row_manifest_hash=canonical_sha256(evidence),
        outer_result_hash=outer_hash,
        predicted_action_scores=tuple((action, None) for action in ALL_ACTION_IDS),
        rank_available=False,
        admission_decision_receipt=None,
    )


def _assemble_one_case(
    outer: OuterTargetDecisionInput,
    rows: Sequence[TargetRowBinding],
    *,
    ranking_policy: BaccRankingPolicy,
) -> TargetCaseDecision:
    science = outer.outer_science
    case_rows = tuple(sorted(rows, key=lambda row: row.row_index))
    if (
        not science.admitted
        or science.row_posterior_model is None
        or science.pairwise_model is None
        or science.uncertainty_calibration is None
        or science.admission is None
        or not science.admission.admitted
    ):
        return _fallback_case(case_rows[0].center_id, case_rows[0].case_id, reason="outer_source_science_not_admitted", evidence=(science.result_hash, tuple(row.row_id for row in case_rows)), outer_hash=science.result_hash, row_indices=tuple(row.row_index for row in case_rows))
    matrix = outer.final_surface.probability_matrix(dtype="<f8")
    indices = np.asarray([row.row_index for row in case_rows], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(matrix)) or tuple(outer.final_surface.row_ids[int(index)] for index in indices) != tuple(row.row_id for row in case_rows):
        raise ProtocolError("OE-PPUR v3 target case row binding drifted from compiled surface.")
    values = np.ascontiguousarray(matrix[indices], dtype=np.float64)
    protected = values[:, 0]
    eta = tuple(
        predict_source_row_posterior(
            science.row_posterior_model,
            feature_names=ROW_FEATURE_NAMES,
            feature_values=build_row_features(row).values,
        )
        for row in values
    )
    row_manifest_hash = canonical_sha256(tuple(row.row_id for row in case_rows))
    scope_id = canonical_sha256({"schema": "oe_ppur_v3_target_case_scope_v1", "H": case_rows[0].center_id, "case_id": case_rows[0].case_id, "row_manifest_hash": row_manifest_hash})
    denominators = build_expected_denominators(eta, scope_id=scope_id, row_manifest_hash=row_manifest_hash)
    actions = tuple(ActionSurface(action_id, action_id.split("::", 1)[0], action_id.split("::", 1)[1], tuple(float(value) for value in values[:, ALL_ACTION_IDS.index(action_id)])) for action_id in CANDIDATE_ACTION_IDS)
    opportunity = build_opportunity_set(protected, actions, candidate_action_ids=CANDIDATE_ACTION_IDS)
    receipt = build_opportunity_case_receipt(center_id=case_rows[0].center_id, case_id=case_rows[0].case_id, opportunity=opportunity)
    queries: dict[str, ActionQuery] = {}
    utilities = {}
    for action in actions:
        feature = build_action_features(protected, action.probabilities)
        queries[action.action_id] = ActionQuery(action.action_id, action.family, action.direction, ACTION_FEATURE_NAMES, feature.values)
        utilities[action.action_id] = normalize_expected_utility(
            expected_additive_utility(protected, action.probabilities, eta, action_id=action.action_id, scope_id=scope_id, row_manifest_hash=row_manifest_hash),
            denominators,
        )
    active = receipt.active_representative_ids
    evidence = []
    for action in active:
        comparators = (ActionQuery.p_anchor(ACTION_FEATURE_NAMES), *(queries[value] for value in active if value != action))
        evidence.append(assemble_action_selection_evidence(
            query=queries[action],
            equivalent_action_ids=opportunity.equivalent_action_ids(action),
            utility=utilities[action],
            comparator_queries=comparators,
            candidate_pool=outer.final_pool_receipt.to_neutral(),
            pairwise_model=science.pairwise_model,
            uncertainty_calibration=science.uncertainty_calibration,
            opportunity_receipt=receipt,
            ranking_policy=ranking_policy,
        ))
    decision = select_fail_closed_action(
        evidence,
        candidate_pool=outer.final_pool_receipt.to_neutral(),
        pairwise_model=science.pairwise_model,
        uncertainty_calibration=science.uncertainty_calibration,
        opportunity_receipt=receipt,
        ranking_policy=ranking_policy,
    )
    admission_receipt = seal_admission_decision(
        center_id=case_rows[0].center_id,
        case_id=case_rows[0].case_id,
        decision=decision,
        candidate_evidence=evidence,
        candidate_pool=outer.final_pool_receipt.to_neutral(),
        pairwise_model=science.pairwise_model,
        uncertainty_calibration=science.uncertainty_calibration,
        opportunity_receipt=receipt,
        ranking_policy=ranking_policy,
    )
    representative_scores = {row.action_id: row.ranking_score for row in evidence}
    full_scores = [(P_ACTION_ID, 0.0)]
    for action in CANDIDATE_ACTION_IDS:
        representative = opportunity.member(action).representative_action_id
        full_scores.append((action, 0.0 if representative is None else representative_scores[representative]))
    return TargetCaseDecision(
        center_id=case_rows[0].center_id,
        case_id=case_rows[0].case_id,
        selected_action_id=decision.selected_action_id,
        reason=decision.reason,
        row_indices=tuple(row.row_index for row in case_rows),
        row_manifest_hash=row_manifest_hash,
        outer_result_hash=science.result_hash,
        predicted_action_scores=tuple(full_scores),
        rank_available=True,
        admission_decision_receipt=admission_receipt,
    )


def assemble_exact_218_case_decisions(
    outer_inputs: Sequence[OuterTargetDecisionInput],
    *,
    expected_case_inventory: Sequence[tuple[object, object]],
    ranking_policy: BaccRankingPolicy | None = None,
) -> TargetDecisionLedger:
    """Assemble all target cases; any malformed/unsupported case becomes exact P."""

    expected = tuple((str(center), str(case)) for center, case in expected_case_inventory)
    if expected != CANONICAL_TARGET_CASE_INVENTORY or target_case_inventory_sha256(expected) != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256:
        raise ProtocolError("OE-PPUR v3 expected target inventory is not canonical 218 cases.")
    inputs = tuple(outer_inputs)
    by_center = {row.outer_science.outer_target_center: row for row in inputs}
    policy = BaccRankingPolicy() if ranking_policy is None else ranking_policy
    if len(by_center) != len(inputs) or tuple(by_center) != CENTERS or not isinstance(policy, BaccRankingPolicy):
        evidence = canonical_sha256({"schema": "oe_ppur_v3_bad_outer_input_inventory_v1", "expected": expected})
        decisions = tuple(_fallback_case(center, case, reason="malformed_outer_input_inventory", evidence=(evidence, center, case), outer_hash=evidence) for center, case in expected)
        return TargetDecisionLedger(decisions, expected, _factory_token=_LEDGER_TOKEN)
    binding_by_case: dict[tuple[str, str], list[TargetRowBinding]] = {}
    topology_ok = True
    row_count = 0
    for center in CENTERS:
        outer = by_center[center]
        bindings = tuple(outer.row_bindings)
        row_count += len(bindings)
        if tuple(row.row_index for row in bindings) != tuple(range(len(bindings))) or tuple(row.row_id for row in bindings) != outer.final_surface.row_ids or len({row.row_id for row in bindings}) != len(bindings):
            topology_ok = False
        for row in bindings:
            binding_by_case.setdefault((row.center_id, row.case_id), []).append(row)
    if row_count != EXPECTED_TEST_ROW_COUNT or set(binding_by_case) != set(expected):
        topology_ok = False
    decisions: list[TargetCaseDecision] = []
    for center, case_id in expected:
        outer = by_center[center]
        rows = tuple(binding_by_case.get((center, case_id), ()))
        if not topology_ok or not rows:
            decisions.append(_fallback_case(center, case_id, reason="malformed_target_case_topology", evidence=(outer.outer_science.result_hash, center, case_id, tuple(row.row_id for row in rows)), outer_hash=outer.outer_science.result_hash, row_indices=tuple(row.row_index for row in rows)))
            continue
        try:
            decisions.append(_assemble_one_case(outer, rows, ranking_policy=policy))
        except (ProtocolError, ValueError, FloatingPointError) as exc:
            decisions.append(_fallback_case(center, case_id, reason="case_science_fail_closed", evidence=(outer.outer_science.result_hash, center, case_id, type(exc).__name__, canonical_sha256(str(exc))), outer_hash=outer.outer_science.result_hash, row_indices=tuple(row.row_index for row in rows)))
    return TargetDecisionLedger(tuple(decisions), expected, _factory_token=_LEDGER_TOKEN)


__all__ = (
    "OuterTargetDecisionInput",
    "TargetCaseDecision",
    "TargetDecisionLedger",
    "TargetRowBinding",
    "assemble_exact_218_case_decisions",
)
