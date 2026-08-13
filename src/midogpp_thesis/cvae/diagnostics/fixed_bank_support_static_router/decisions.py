"""S4, source/target-excluding G_static, and blocked-null decisions."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CENTERS,
    NULL_DERANGEMENT_ALGORITHM,
    OOF_FOLD_COUNT,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    TIE_TOLERANCE,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
    decision_action_ids,
    source_from_action,
)
from .hashing import canonical_hash, require_sha256
from .partitions import CaseFold, CaseOOFPartition
from .products import (
    ActionGain,
    CaseActionCounts,
    DecisionSeal,
    GStaticSeal,
    NullRouteSelection,
    NullSelectionPlan,
    RouteDecision,
    StaticSelection,
)
from .scoring import counts_surface_hash, pooled_bacc


def _action_rows_hash(
    rows: Sequence[CaseActionCounts], *, label_scope: str, action_id: str
) -> str:
    return canonical_hash(
        {
            "schema_version": "fixed_bank_support_static_router_action_contributions_v1",
            "label_scope": label_scope,
            "action_id": action_id,
            "rows": [row.to_payload() for row in rows],
            "additive_sufficient_statistics": True,
            "per_case_bacc_used": False,
        }
    )


def _choose_action(
    action_order: Sequence[str], gains: Mapping[str, float], *, tie_tolerance: float
) -> str:
    if float(tie_tolerance) != TIE_TOLERANCE:
        raise ProtocolError("Tie tolerance drifted from the frozen 1e-12 value.")
    order = tuple(action_order)
    if tuple(gains) != order:
        raise ProtocolError("Action-gain order drifted from B then numeric source center.")
    maximum = max(gains.values())
    eligible = tuple(action for action in order if maximum - gains[action] <= tie_tolerance)
    return eligible[0]


def select_support_static_action(
    fold: CaseFold,
    support_counts: Sequence[CaseActionCounts],
    *,
    prerequisite_seal_hash: str,
    tie_tolerance: float = TIE_TOLERANCE,
) -> StaticSelection:
    """Select S4 from B plus all eight A1 actions on the other four folds."""

    require_sha256(prerequisite_seal_hash, "prerequisite_seal_hash")
    target = fold.target_center
    cases = tuple(fold.support_case_ids)
    action_order = decision_action_ids(target)
    rows = tuple(support_counts)
    if not rows or any(row.target_center != target for row in rows):
        raise ProtocolError("S4 counts must be non-empty and target-local.")
    if any(row.case_id not in set(cases) for row in rows):
        raise ProtocolError("S4 counts escaped the route support cases.")
    allowed = set(action_order) | {U_ACTION_ID}
    if any(row.action_id not in allowed for row in rows):
        raise ProtocolError("S4 counts contain an unknown action.")
    by_key = {(row.action_id, row.case_id): row for row in rows}
    if len(by_key) != len(rows):
        raise ProtocolError("S4 count rows are duplicated.")
    expected = {(action, case) for action in action_order for case in cases}
    if not expected.issubset(by_key):
        raise ProtocolError("S4 must score B and all eight A1 actions on every support case.")
    u_keys = {(U_ACTION_ID, case) for case in cases}
    observed_u = {key for key in by_key if key[0] == U_ACTION_ID}
    if observed_u and observed_u != u_keys:
        raise ProtocolError("U may be present only as a complete, unused control surface.")
    if set(by_key) not in (expected, expected | u_keys):
        raise ProtocolError("S4 count coverage is not a closed support/action surface.")
    for case in cases:
        class_counts = {by_key[(action, case)].class_counts for action in action_order}
        if len(class_counts) != 1:
            raise ProtocolError("S4 action class denominators drifted within a case.")

    label_scope = f"same_H_other_four_folds::{fold.fold_id}"
    case_keys = tuple((target, case) for case in cases)
    n_positive = sum(by_key[(B_ACTION_ID, case)].n_positive for case in cases)
    n_negative = sum(by_key[(B_ACTION_ID, case)].n_negative for case in cases)
    single_class = n_positive <= 0 or n_negative <= 0
    action_gains: list[ActionGain] = []
    if single_class:
        for action in action_order:
            action_rows = tuple(by_key[(action, case)] for case in cases)
            action_gains.append(
                ActionGain(
                    action,
                    None if action == B_ACTION_ID else source_from_action(action),
                    None,
                    None,
                    None,
                    "pooled_exact_bacc",
                    label_scope,
                    (target,),
                    case_keys,
                    _action_rows_hash(action_rows, label_scope=label_scope, action_id=action),
                )
            )
        selected_action = B_ACTION_ID
        selected_gain = 0.0
        baseline_value = None
        selected_value = None
        reason = "support_single_class_fallback_B"
    else:
        baseline_rows = tuple(by_key[(B_ACTION_ID, case)] for case in cases)
        baseline = pooled_bacc(baseline_rows)
        for action in action_order:
            action_rows = tuple(by_key[(action, case)] for case in cases)
            score = baseline if action == B_ACTION_ID else pooled_bacc(action_rows)
            action_gains.append(
                ActionGain(
                    action,
                    None if action == B_ACTION_ID else source_from_action(action),
                    score.exact_bacc,
                    baseline.exact_bacc,
                    score.exact_bacc - baseline.exact_bacc,
                    "pooled_exact_bacc",
                    label_scope,
                    (target,),
                    case_keys,
                    _action_rows_hash(action_rows, label_scope=label_scope, action_id=action),
                )
            )
        gains = {row.action_id: float(row.gain) for row in action_gains}
        selected_action = _choose_action(action_order, gains, tie_tolerance=tie_tolerance)
        # B is first in the tie order; a challenger must beat zero outside tolerance.
        if selected_action == B_ACTION_ID or gains[selected_action] <= 0.0:
            selected_action = B_ACTION_ID
            reason = "no_strictly_positive_support_gain"
        else:
            reason = None
        selected_row = next(row for row in action_gains if row.action_id == selected_action)
        selected_gain = float(selected_row.gain)
        baseline_value = float(selected_row.baseline_score)
        selected_value = float(selected_row.action_score)
    return StaticSelection(
        target_center=target,
        method_id="S4",
        action_id=selected_action,
        selected_source=None if selected_action == B_ACTION_ID else source_from_action(selected_action),
        selected_gain=selected_gain,
        baseline_score=baseline_value,
        selected_score=selected_value,
        score_type="pooled_exact_bacc",
        action_gains=tuple(action_gains),
        label_case_ids=tuple(cases),
        label_case_keys=case_keys,
        label_scope=label_scope,
        prerequisite_seal_hash=prerequisite_seal_hash,
        fallback_reason=reason,
    )


def select_global_static_action(
    heldout_target: object,
    donor_counts_by_action: Mapping[str, Sequence[CaseActionCounts]],
    *,
    prerequisite_seal_hash: str,
    tie_tolerance: float = TIE_TOLERANCE,
) -> StaticSelection:
    """Equal-center G_static over q not in {H,e}, never using H support."""

    require_sha256(prerequisite_seal_hash, "prerequisite_seal_hash")
    target = str(heldout_target)
    candidates = tuple(a1_action_id(source) for source in candidate_sources(target))
    if tuple(donor_counts_by_action) != candidates:
        raise ProtocolError("G_static donor mapping must follow numeric candidate-source order.")
    label_scope = f"loco_equal_center_q_notin_H_e::heldout_H={target}"
    candidate_rows: dict[str, tuple[CaseActionCounts, ...]] = {}
    baseline_union: dict[tuple[str, str], CaseActionCounts] = {}
    all_case_keys: set[tuple[str, str]] = set()
    action_gains_by_id: dict[str, ActionGain] = {}
    unavailable = False
    for action in candidates:
        source = source_from_action(action)
        rows = tuple(donor_counts_by_action[action])
        allowed_centers = tuple(center for center in CENTERS if center not in {target, source})
        if not rows or any(row.target_center not in allowed_centers for row in rows):
            raise ProtocolError("G_static donor rows violate q not in {H,e}.")
        if any(row.action_id not in {B_ACTION_ID, action} for row in rows):
            raise ProtocolError("G_static candidate grant may contain only B and its A1 action.")
        by_key = {(row.target_center, row.case_id, row.action_id): row for row in rows}
        if len(by_key) != len(rows):
            raise ProtocolError("G_static donor count rows are duplicated.")
        case_keys = tuple(sorted({(row.target_center, row.case_id) for row in rows}))
        if {center for center, _case in case_keys} != set(allowed_centers):
            raise ProtocolError("G_static candidate lacks one or more legal donor centers.")
        expected = {
            (center, case, candidate_action)
            for center, case in case_keys
            for candidate_action in (B_ACTION_ID, action)
        }
        if set(by_key) != expected:
            raise ProtocolError("G_static candidate lacks paired B/A1 donor cases.")
        for center, case in case_keys:
            baseline = by_key[(center, case, B_ACTION_ID)]
            challenger = by_key[(center, case, action)]
            if baseline.class_counts != challenger.class_counts:
                raise ProtocolError("G_static B/A1 class denominators drifted.")
            previous = baseline_union.setdefault((center, case), baseline)
            if previous != baseline:
                raise ProtocolError("G_static repeated B sufficient statistics disagree.")
        all_case_keys.update(case_keys)
        bacc_b: list[float] = []
        bacc_a: list[float] = []
        candidate_unavailable = False
        for query_center in allowed_centers:
            center_cases = tuple(case for center, case in case_keys if center == query_center)
            b_rows = tuple(by_key[(query_center, case, B_ACTION_ID)] for case in center_cases)
            a_rows = tuple(by_key[(query_center, case, action)] for case in center_cases)
            n_positive = sum(row.n_positive for row in b_rows)
            n_negative = sum(row.n_negative for row in b_rows)
            if n_positive <= 0 or n_negative <= 0:
                candidate_unavailable = True
                break
            bacc_b.append(pooled_bacc(b_rows).exact_bacc)
            bacc_a.append(pooled_bacc(a_rows).exact_bacc)
        unavailable |= candidate_unavailable
        ordered_rows = tuple(
            by_key[(center, case, action_id)]
            for center, case in case_keys
            for action_id in (B_ACTION_ID, action)
        )
        candidate_rows[action] = ordered_rows
        mean_b = None if candidate_unavailable else sum(bacc_b) / len(bacc_b)
        mean_a = None if candidate_unavailable else sum(bacc_a) / len(bacc_a)
        action_gains_by_id[action] = ActionGain(
            action,
            source,
            mean_a,
            mean_b,
            None if candidate_unavailable else float(mean_a) - float(mean_b),
            "equal_center_mean_of_per_q_pooled_exact_bacc",
            f"{label_scope}::candidate_e={source}",
            allowed_centers,
            case_keys,
            _action_rows_hash(ordered_rows, label_scope=label_scope, action_id=action),
        )

    union_keys = tuple(sorted(all_case_keys))
    union_centers = tuple(center for center in CENTERS if center != target)
    b_scores: list[float] = []
    baseline_unavailable = False
    for center in union_centers:
        rows = tuple(baseline_union[key] for key in union_keys if key[0] == center)
        if not rows or sum(row.n_positive for row in rows) <= 0 or sum(row.n_negative for row in rows) <= 0:
            baseline_unavailable = True
            break
        b_scores.append(pooled_bacc(rows).exact_bacc)
    unavailable |= baseline_unavailable
    baseline_score = None if baseline_unavailable else sum(b_scores) / len(b_scores)
    baseline_rows = tuple(baseline_union[key] for key in union_keys)
    baseline_gain = ActionGain(
        B_ACTION_ID,
        None,
        baseline_score,
        baseline_score,
        None if baseline_score is None else 0.0,
        "equal_center_mean_of_per_q_pooled_exact_bacc",
        f"{label_scope}::baseline_union",
        union_centers,
        union_keys,
        _action_rows_hash(baseline_rows, label_scope=label_scope, action_id=B_ACTION_ID),
    )
    action_gains = (baseline_gain, *(action_gains_by_id[action] for action in candidates))
    if unavailable:
        # Do not partially compare candidates over different valid donor panels.
        unavailable_rows = tuple(
            ActionGain(
                row.action_id,
                row.selected_source,
                None,
                None,
                None,
                row.score_type,
                row.label_scope,
                row.donor_centers,
                row.label_case_keys,
                row.contribution_hash,
            )
            for row in action_gains
        )
        action_gains = unavailable_rows
        selected_action = B_ACTION_ID
        selected_gain = 0.0
        baseline_value = None
        selected_value = None
        reason = "loco_cell_single_class_fallback_B"
    else:
        gains = {row.action_id: float(row.gain) for row in action_gains}
        selected_action = _choose_action(
            decision_action_ids(target), gains, tie_tolerance=tie_tolerance
        )
        if selected_action == B_ACTION_ID or gains[selected_action] <= 0.0:
            selected_action = B_ACTION_ID
            reason = "no_strictly_positive_equal_center_loco_gain"
        else:
            reason = None
        chosen = next(row for row in action_gains if row.action_id == selected_action)
        selected_gain = float(chosen.gain)
        baseline_value = float(chosen.baseline_score)
        selected_value = float(chosen.action_score)
    label_ids = tuple(f"{center}::{case}" for center, case in union_keys)
    return StaticSelection(
        target_center=target,
        method_id="G_static",
        action_id=selected_action,
        selected_source=None if selected_action == B_ACTION_ID else source_from_action(selected_action),
        selected_gain=selected_gain,
        baseline_score=baseline_value,
        selected_score=selected_value,
        score_type="equal_center_mean_of_per_q_pooled_exact_bacc",
        action_gains=tuple(action_gains),
        label_case_ids=label_ids,
        label_case_keys=union_keys,
        label_scope=label_scope,
        prerequisite_seal_hash=prerequisite_seal_hash,
        fallback_reason=reason,
    )


def seal_global_static_selections(
    selections: Sequence[StaticSelection], *, probability_seal_hash: str
) -> GStaticSeal:
    canonical = tuple(sorted(selections, key=lambda row: CENTERS.index(row.target_center)))
    payload = {
        "schema_version": "fixed_bank_support_static_router_g_static_seal_v1",
        "probability_seal_hash": probability_seal_hash,
        "selections": [row.to_payload() for row in canonical],
        "source_excluding_target_excluding_LOCO": True,
        "same_H_support_labels_used": False,
        "held_evaluation_labels_used": False,
    }
    return GStaticSeal(canonical, probability_seal_hash, canonical_hash(payload))


def make_route_decision(
    fold: CaseFold,
    *,
    g_static_seal: GStaticSeal,
    s4_selection: StaticSelection,
    probability_seal_hash: str,
) -> RouteDecision:
    if g_static_seal.probability_seal_hash != probability_seal_hash:
        raise ProtocolError("Route G_static and probability seals differ.")
    global_selection = g_static_seal.selection(fold.target_center)
    return RouteDecision(
        target_center=fold.target_center,
        fold_ordinal=fold.fold_ordinal,
        fold_hash=fold.fold_hash,
        support_case_ids=fold.support_case_ids,
        evaluation_case_ids=fold.evaluation_case_ids,
        g_static=global_selection,
        s4=s4_selection,
        g_static_seal_hash=g_static_seal.seal_hash,
        probability_seal_hash=probability_seal_hash,
    )


def seal_route_decisions(
    decisions: Sequence[RouteDecision],
    *,
    partition: CaseOOFPartition,
    probability_seal_hash: str,
) -> DecisionSeal:
    canonical = tuple(
        sorted(
            decisions,
            key=lambda row: (CENTERS.index(row.target_center), row.fold_ordinal),
        )
    )
    if len(canonical) != len(CENTERS) * OOF_FOLD_COUNT:
        raise ProtocolError("All 45 route decisions are required for the terminal seal.")
    for decision, fold in zip(canonical, partition.folds):
        if (
            decision.target_center != fold.target_center
            or decision.fold_ordinal != fold.fold_ordinal
            or decision.fold_hash != fold.fold_hash
            or decision.support_case_ids != fold.support_case_ids
            or decision.evaluation_case_ids != fold.evaluation_case_ids
        ):
            raise ProtocolError("Route decision drifted from the five-fold partition.")
    payload = {
        "schema_version": "fixed_bank_support_static_router_all_route_decision_seal_v1",
        "partition_hash": partition.partition_hash,
        "probability_seal_hash": probability_seal_hash,
        "decisions": [row.to_payload() for row in canonical],
        "all_route_decisions_sealed_before_terminal_aggregation": True,
        "each_route_decision_sealed_before_own_evaluation_labels": True,
        "evaluation_labels_used": False,
        "terminal_oracles_used": False,
    }
    return DecisionSeal(
        canonical,
        partition.partition_hash,
        probability_seal_hash,
        canonical_hash(payload),
    )


def permute_support_candidate_blocks(
    fold: CaseFold,
    support_counts: Sequence[CaseActionCounts],
    *,
    permutation_index: int,
    permutation_seed: int = PERMUTATION_SEED,
) -> tuple[CaseActionCounts, ...]:
    """Derange complete A1 count blocks within each support case; keep B/U fixed."""

    if isinstance(permutation_index, bool) or not isinstance(permutation_index, int) or permutation_index < 0:
        raise ProtocolError("permutation_index must be a non-negative integer.")
    if permutation_seed != PERMUTATION_SEED:
        raise ProtocolError("Permutation seed drifted from the predeclared value.")
    rows = tuple(support_counts)
    by_key = {(row.action_id, row.case_id): row for row in rows}
    if len(by_key) != len(rows):
        raise ProtocolError("Null input count rows are duplicated.")
    candidates = tuple(a1_action_id(source) for source in candidate_sources(fold.target_center))
    expected = {
        (action, case)
        for case in fold.support_case_ids
        for action in (B_ACTION_ID, *candidates)
    }
    u_keys = {(U_ACTION_ID, case) for case in fold.support_case_ids}
    if not expected.issubset(by_key) or set(by_key) not in (expected, expected | u_keys):
        raise ProtocolError("Null input must contain complete B/eight-A1 support blocks.")
    result: list[CaseActionCounts] = []
    for case in fold.support_case_ids:
        result.append(by_key[(B_ACTION_ID, case)])
        if (U_ACTION_ID, case) in by_key:
            result.append(by_key[(U_ACTION_ID, case)])
        ordered = _case_candidate_order(
            candidates,
            permutation_seed=permutation_seed,
            fold_id=fold.fold_id,
            case_id=case,
        )
        shift = _case_shift(
            permutation_seed=permutation_seed,
            fold_id=fold.fold_id,
            permutation_index=permutation_index,
            case_id=case,
        )
        for index, recipient in enumerate(ordered):
            donor = by_key[(ordered[(index + shift) % len(ordered)], case)]
            result.append(
                CaseActionCounts(
                    target_center=fold.target_center,
                    case_id=case,
                    action_id=recipient,
                    n_positive=donor.n_positive,
                    true_positive=donor.true_positive,
                    n_negative=donor.n_negative,
                    true_negative=donor.true_negative,
                )
            )
    order = decision_action_ids(fold.target_center)
    return tuple(
        sorted(
            result,
            key=lambda row: (
                fold.support_case_ids.index(row.case_id),
                (B_ACTION_ID, U_ACTION_ID, *order[1:]).index(row.action_id),
            ),
        )
    )


def build_null_selection_plan(
    fold: CaseFold,
    support_counts: Sequence[CaseActionCounts],
    *,
    prerequisite_seal_hash: str,
    permutation_seed: int = PERMUTATION_SEED,
    permutation_count: int = PERMUTATION_COUNT,
) -> NullSelectionPlan:
    """Return sealed null action selections only; no p-value or gate is computed."""

    require_sha256(prerequisite_seal_hash, "prerequisite_seal_hash")
    if permutation_seed != PERMUTATION_SEED or permutation_count != PERMUTATION_COUNT:
        raise ProtocolError("Null permutation count/seed drifted from the predeclared values.")
    support_hash = counts_surface_hash(support_counts)
    frozen = _vectorized_null_route_selections(
        fold,
        support_counts,
        prerequisite_seal_hash=prerequisite_seal_hash,
        permutation_seed=permutation_seed,
        permutation_count=permutation_count,
    )
    payload = {
        "schema_version": "fixed_bank_support_static_router_null_selection_plan_v1",
        "target_center": fold.target_center,
        "fold_ordinal": fold.fold_ordinal,
        "fold_hash": fold.fold_hash,
        "permutation_seed": permutation_seed,
        "permutation_count": permutation_count,
        "support_counts_hash": support_hash,
        "prerequisite_seal_hash": prerequisite_seal_hash,
        "selections": [row.to_payload() for row in frozen],
        "complete_candidate_blocks_permuted_within_support_case": True,
        "baseline_block_permuted": False,
        "nonzero_cyclic_shifts_only": True,
        "evaluation_labels_used": False,
        "descriptive_exceedance_only": True,
        "p_value_computed": False,
        "gate_computed": False,
    }
    return NullSelectionPlan(
        fold.target_center,
        fold.fold_ordinal,
        fold.fold_hash,
        permutation_seed,
        permutation_count,
        support_hash,
        prerequisite_seal_hash,
        frozen,
        canonical_hash(payload),
    )


def _vectorized_null_route_selections(
    fold: CaseFold,
    support_counts: Sequence[CaseActionCounts],
    *,
    prerequisite_seal_hash: str,
    permutation_seed: int,
    permutation_count: int,
) -> tuple[NullRouteSelection, ...]:
    """Vectorized exact replay of the scalar complete-block derangement."""

    if permutation_count <= 0:
        raise ProtocolError("permutation_count must be positive.")
    # Reuse the observed selector as the closed-surface and denominator audit.
    select_support_static_action(
        fold,
        support_counts,
        prerequisite_seal_hash=prerequisite_seal_hash,
    )
    rows = tuple(support_counts)
    by_key = {(row.action_id, row.case_id): row for row in rows}
    cases = tuple(fold.support_case_ids)
    candidates = tuple(a1_action_id(source) for source in candidate_sources(fold.target_center))
    candidate_index = {action: index for index, action in enumerate(candidates)}
    true_positive = np.asarray(
        [[by_key[(action, case)].true_positive for action in candidates] for case in cases],
        dtype=np.int64,
    )
    true_negative = np.asarray(
        [[by_key[(action, case)].true_negative for action in candidates] for case in cases],
        dtype=np.int64,
    )
    baseline_rows = tuple(by_key[(B_ACTION_ID, case)] for case in cases)
    n_positive = sum(row.n_positive for row in baseline_rows)
    n_negative = sum(row.n_negative for row in baseline_rows)
    # Single-class support is a pre-scoring B fallback for every null replicate.
    if n_positive <= 0 or n_negative <= 0:
        return tuple(
            NullRouteSelection(
                fold.target_center, fold.fold_ordinal, index, B_ACTION_ID, 0.0
            )
            for index in range(permutation_count)
        )
    baseline_score = 0.5 * (
        sum(row.true_positive for row in baseline_rows) / n_positive
        + sum(row.true_negative for row in baseline_rows) / n_negative
    )

    # mapping[case, nonzero_shift-1, canonical_recipient] -> canonical donor.
    mapping = np.empty((len(cases), 7, len(candidates)), dtype=np.int8)
    shifts = np.empty((permutation_count, len(cases)), dtype=np.int8)
    counters = np.arange(1, permutation_count + 1, dtype=np.uint64)
    for case_index, case_id in enumerate(cases):
        ordered = _case_candidate_order(
            candidates,
            permutation_seed=permutation_seed,
            fold_id=fold.fold_id,
            case_id=case_id,
        )
        order = np.asarray([candidate_index[action] for action in ordered], dtype=np.int8)
        for shift in range(1, 8):
            donor = np.roll(order, -shift)
            mapping[case_index, shift - 1, order] = donor
        base = np.uint64(
            int.from_bytes(
                hashlib.sha256(
                    f"{permutation_seed}::{fold.fold_id}::{case_id}::shift".encode()
                ).digest()[:8],
                "big",
            )
        )
        with np.errstate(over="ignore"):
            values = base + counters * np.uint64(0x9E3779B97F4A7C15)
            values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            values ^= values >> np.uint64(31)
        shifts[:, case_index] = (1 + values % np.uint64(7)).astype(np.int8)

    pooled_tp = np.zeros((permutation_count, len(candidates)), dtype=np.int64)
    pooled_tn = np.zeros_like(pooled_tp)
    for case_index in range(len(cases)):
        donor = mapping[case_index, shifts[:, case_index] - 1, :].astype(np.int64)
        pooled_tp += true_positive[case_index, donor]
        pooled_tn += true_negative[case_index, donor]
    scores = 0.5 * (
        pooled_tp.astype(np.float64) / float(n_positive)
        + pooled_tn.astype(np.float64) / float(n_negative)
    )
    gains = scores - baseline_score
    maximum = np.maximum(0.0, np.max(gains, axis=1))
    # B precedes numeric source centers and wins every tolerance tie with zero.
    choose_b = maximum <= TIE_TOLERANCE
    eligible = maximum[:, None] - gains <= TIE_TOLERANCE
    candidate_codes = np.argmax(eligible, axis=1)
    chosen_gains = gains[np.arange(permutation_count), candidate_codes]
    selections: list[NullRouteSelection] = []
    for index in range(permutation_count):
        action = B_ACTION_ID if choose_b[index] else candidates[int(candidate_codes[index])]
        gain = 0.0 if choose_b[index] else float(chosen_gains[index])
        selections.append(
            NullRouteSelection(
                fold.target_center,
                fold.fold_ordinal,
                index,
                action,
                gain,
            )
        )
    return tuple(selections)


def _case_candidate_order(
    candidates: tuple[str, ...], *, permutation_seed: int, fold_id: str, case_id: str
) -> tuple[str, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda action: (
                hashlib.sha256(
                    f"{permutation_seed}::{fold_id}::{case_id}::{action}".encode()
                ).hexdigest(),
                action,
            ),
        )
    )


def _case_shift(
    *, permutation_seed: int, fold_id: str, permutation_index: int, case_id: str
) -> int:
    base = int.from_bytes(
        hashlib.sha256(
            f"{permutation_seed}::{fold_id}::{case_id}::shift".encode()
        ).digest()[:8],
        "big",
    )
    return 1 + _splitmix64(
        base + (permutation_index + 1) * 0x9E3779B97F4A7C15
    ) % 7


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value &= mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return value & mask


__all__ = (
    "NULL_DERANGEMENT_ALGORITHM",
    "build_null_selection_plan",
    "make_route_decision",
    "permute_support_candidate_blocks",
    "seal_global_static_selections",
    "seal_route_decisions",
    "select_global_static_action",
    "select_support_static_action",
)
