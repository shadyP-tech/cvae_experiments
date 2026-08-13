from __future__ import annotations

from collections import OrderedDict

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.actions import (
    actions_for_target,
    build_action_library,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.constants import (
    B_ACTION_ID,
    CENTERS,
    PERMUTATION_SEED,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
    decision_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.decisions import (
    _vectorized_null_route_selections,
    permute_support_candidate_blocks,
    select_global_static_action,
    select_support_static_action,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.partitions import (
    CaseFold,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.products import (
    BinaryLabelRow,
    BinaryPredictionRow,
    CaseActionCounts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.scoring import (
    pooled_bacc,
    score_case_action_counts,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "1" * 64


def _fold() -> CaseFold:
    return CaseFold("0", 0, ("s0", "s1", "s2", "s3"), ("eval",))


def _support_counts(
    *,
    winner: str = "1",
    tie: str | None = None,
    single_class: bool = False,
) -> tuple[CaseActionCounts, ...]:
    rows: list[CaseActionCounts] = []
    actions = (B_ACTION_ID, U_ACTION_ID, *(a1_action_id(s) for s in candidate_sources("0")))
    for case in _fold().support_case_ids:
        for action in actions:
            n_positive = 0 if single_class else 10
            n_negative = 10
            if action == B_ACTION_ID:
                tp, tn = (0 if single_class else 6), 6
            elif action == U_ACTION_ID:
                tp, tn = (0 if single_class else 10), 10
            elif action in {a1_action_id(winner), *( () if tie is None else (a1_action_id(tie),) )}:
                tp, tn = (0 if single_class else 8), 8
            else:
                tp, tn = (0 if single_class else 5), 5
            rows.append(
                CaseActionCounts("0", case, action, n_positive, tp, n_negative, tn)
            )
    return tuple(rows)


def test_action_library_is_exact_target_keyed_B_U_eight_A1() -> None:
    library = build_action_library()
    assert tuple(library) == CENTERS
    assert all(len(library[target]) == 10 for target in CENTERS)
    target = actions_for_target("0")
    assert tuple(row.action_id for row in target) == (
        B_ACTION_ID,
        U_ACTION_ID,
        *(a1_action_id(source) for source in candidate_sources("0")),
    )
    assert all(row.to_payload()["target_expert_excluded"] is True for row in target)
    assert target[2].counts_by_class[0]["1"] == 256
    assert target[2].sample_weight_by_source["1"] == pytest.approx(23.0 / 16.0)


def test_exact_scoring_pools_additive_counts_and_never_defines_case_bacc() -> None:
    labels = (
        BinaryLabelRow("0", "case", "p", 1),
        BinaryLabelRow("0", "case", "n", 0),
    )
    rows = []
    for action in (B_ACTION_ID, U_ACTION_ID, *(a1_action_id(s) for s in candidate_sources("0"))):
        for sample, probability in (("p", 0.8), ("n", 0.2)):
            rows.append(BinaryPredictionRow("0", "case", sample, action, probability, SHA))
    counts = score_case_action_counts(rows, labels)
    assert len(counts) == 10
    assert not hasattr(counts[0], "bacc")
    score = pooled_bacc((counts[0],))
    assert score.exact_bacc == 1.0


def test_s4_scores_all_A1_vs_B_ignores_U_and_uses_numeric_tie_order() -> None:
    selected = select_support_static_action(
        _fold(), _support_counts(winner="1", tie="2"), prerequisite_seal_hash=SHA
    )
    assert selected.action_id == a1_action_id("1")
    assert selected.selected_gain == pytest.approx(0.2)
    assert tuple(row.action_id for row in selected.action_gains) == decision_action_ids("0")
    # U is deliberately perfect but never appears in the decision candidates.
    assert all(row.action_id != U_ACTION_ID for row in selected.action_gains)


def test_s4_falls_back_to_B_for_nonpositive_or_single_class_support() -> None:
    nonpositive = tuple(
        CaseActionCounts(
            row.target_center,
            row.case_id,
            row.action_id,
            row.n_positive,
            6 if row.n_positive else 0,
            row.n_negative,
            6,
        )
        for row in _support_counts()
    )
    assert select_support_static_action(
        _fold(), nonpositive, prerequisite_seal_hash=SHA
    ).action_id == B_ACTION_ID
    single = select_support_static_action(
        _fold(), _support_counts(single_class=True), prerequisite_seal_hash=SHA
    )
    assert single.action_id == B_ACTION_ID
    assert single.fallback_reason == "support_single_class_fallback_B"
    assert all(row.gain is None for row in single.action_gains)


def _g_donors() -> OrderedDict[str, tuple[CaseActionCounts, ...]]:
    result: OrderedDict[str, tuple[CaseActionCounts, ...]] = OrderedDict()
    for source in candidate_sources("0"):
        action = a1_action_id(source)
        rows: list[CaseActionCounts] = []
        for query in CENTERS:
            if query in {"0", source}:
                continue
            # Unequal query case counts make accidental row pooling observably different.
            for ordinal in range(1 if query == "1" else 2):
                case = f"{query}-{ordinal}"
                rows.append(CaseActionCounts(query, case, B_ACTION_ID, 10, 5, 10, 5))
                correct = 8 if source == "1" else 4
                rows.append(CaseActionCounts(query, case, action, 10, correct, 10, correct))
        result[action] = tuple(rows)
    return result


def test_g_static_enforces_H_e_exclusion_and_equal_center_mean() -> None:
    selected = select_global_static_action("0", _g_donors(), prerequisite_seal_hash=SHA)
    assert selected.action_id == a1_action_id("1")
    assert selected.score_type == "equal_center_mean_of_per_q_pooled_exact_bacc"
    gain = selected.action_gains[1]
    assert set(gain.donor_centers) == set(CENTERS) - {"0", "1"}
    poisoned = _g_donors()
    poisoned[a1_action_id("1")] += (
        CaseActionCounts("0", "held", B_ACTION_ID, 10, 10, 10, 10),
        CaseActionCounts("0", "held", a1_action_id("1"), 10, 10, 10, 10),
    )
    with pytest.raises(ProtocolError, match="q not in"):
        select_global_static_action("0", poisoned, prerequisite_seal_hash=SHA)


def test_vectorized_block_null_matches_scalar_selector_for_small_replay() -> None:
    counts = _support_counts(winner="1")
    fast = _vectorized_null_route_selections(
        _fold(),
        counts,
        prerequisite_seal_hash=SHA,
        permutation_seed=PERMUTATION_SEED,
        permutation_count=31,
    )
    slow = []
    for index in range(31):
        permuted = permute_support_candidate_blocks(
            _fold(), counts, permutation_index=index
        )
        selection = select_support_static_action(
            _fold(), permuted, prerequisite_seal_hash=SHA
        )
        slow.append((selection.action_id, selection.selected_gain))
    assert tuple((row.action_id, row.selected_gain) for row in fast) == tuple(slow)


def test_candidate_block_null_is_deterministic_nonzero_and_preserves_blocks() -> None:
    counts = _support_counts()
    first = permute_support_candidate_blocks(_fold(), counts, permutation_index=0)
    repeat = permute_support_candidate_blocks(_fold(), counts, permutation_index=0)
    assert first == repeat
    original = {(row.case_id, row.action_id): row for row in counts}
    permuted = {(row.case_id, row.action_id): row for row in first}
    for case in _fold().support_case_ids:
        original_blocks = {
            (
                original[(case, action)].n_positive,
                original[(case, action)].true_positive,
                original[(case, action)].n_negative,
                original[(case, action)].true_negative,
            )
            for action in decision_action_ids("0")[1:]
        }
        permuted_blocks = {
            (
                permuted[(case, action)].n_positive,
                permuted[(case, action)].true_positive,
                permuted[(case, action)].n_negative,
                permuted[(case, action)].true_negative,
            )
            for action in decision_action_ids("0")[1:]
        }
        assert original_blocks == permuted_blocks
        assert permuted[(case, B_ACTION_ID)] == original[(case, B_ACTION_ID)]
