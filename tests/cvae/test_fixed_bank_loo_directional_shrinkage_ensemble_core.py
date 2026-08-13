from __future__ import annotations

from fractions import Fraction
import json
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.actions import (
    actions_for_target,
    build_action_library,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.constants import (
    ARM_IDS,
    CENTERS,
    DIRECTION_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    HARD_THRESHOLD,
    a1_action_id,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.config_payloads import (
    canonical_nulls_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.decisions import (
    rank_sources_by_prior,
    select_arm_decisions,
    select_direction_decision,
    select_matched_g_decisions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.donor_priors import (
    compute_donor_prior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.endpoint_library import (
    build_endpoint_arms,
    build_matched_endpoint_libraries,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.ensemble import (
    compose_control_predictions,
    compose_direction_decomposition_predictions,
    compose_frequency_weighted_probability,
    compose_hard_vote_comparator,
    compose_method_predictions,
    compose_unique_source_mean_comparator,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.label_capabilities import (
    LabelCapabilityFirewall,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.loo_plans import (
    WholeCaseLooPlan,
    build_whole_case_loo_plans,
    seal_loo_plans,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.nulls import (
    build_candidate_identity_null_plan,
    select_scrambled_endpoints_scalar,
    select_scrambled_endpoints_vectorized,
    validate_candidate_identity_null_plan_contract,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
    exact_nine_mean,
    exact_nine_means,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.products import (
    BinaryLabel,
    CaseActionConfusion,
    CaseControlDecision,
    DirectionalGain,
    DirectionalControlDecision,
    DonorPrior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.scoring import (
    directional_flip_counts,
    directional_flip_counts_scalar,
    directional_hard_flip_gain,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _gain(
    query: str,
    source: str,
    direction: str,
    *,
    favorable: int = 0,
    adverse: int = 0,
    n_positive: int = 10,
    n_negative: int = 10,
    excluded: str | None = None,
) -> DirectionalGain:
    return DirectionalGain(
        query_center=query,
        excluded_case_id=excluded,
        source=source,
        direction=direction,
        n_positive=n_positive,
        n_negative=n_negative,
        favorable_count=favorable,
        adverse_count=adverse,
        contributing_case_ids=("support",),
        label_scope="test_scope",
    )


def _prior(
    target: str,
    source: str,
    direction: str,
    *,
    favorable: int = 0,
    adverse: int = 0,
    n_positive: int = 10,
    n_negative: int = 10,
) -> DonorPrior:
    queries = tuple(center for center in CENTERS if center not in {target, source})
    return DonorPrior(
        target,
        source,
        direction,
        tuple(
            _gain(
                query,
                source,
                direction,
                favorable=favorable,
                adverse=adverse,
                n_positive=n_positive,
                n_negative=n_negative,
            )
            for query in queries
        ),
    )


def _gain_surfaces(
    target: str = "0",
    *,
    winning_source: str = "1",
    tiny_positive: bool = False,
) -> tuple[tuple[DirectionalGain, ...], tuple[DonorPrior, ...]]:
    support: list[DirectionalGain] = []
    priors: list[DonorPrior] = []
    for source in candidate_sources(target):
        for direction in DIRECTION_IDS:
            favorable = 1 if source == winning_source else 0
            denominator = 10**15 if tiny_positive and source == winning_source else 10
            support.append(
                _gain(
                    target,
                    source,
                    direction,
                    favorable=favorable,
                    n_positive=denominator,
                    n_negative=denominator,
                    excluded="held",
                )
            )
            priors.append(
                _prior(
                    target,
                    source,
                    direction,
                    favorable=favorable,
                    n_positive=denominator,
                    n_negative=denominator,
                )
            )
    return tuple(support), tuple(priors)


def _probability_surface(
    *,
    target: str = "0",
    case_id: str = "held",
    samples: tuple[tuple[str, float], ...] = (("s0", 0.4),),
    overrides: dict[tuple[str, str], float] | None = None,
) -> ExactNineProbabilitySurface:
    overrides = overrides or {}
    rows = []
    for sample_id, baseline in samples:
        for action in physical_action_ids(target):
            value = overrides.get((sample_id, action), baseline)
            rows.append(
                ExactNineProbabilityRow(
                    target,
                    case_id,
                    sample_id,
                    action,
                    (value,) * 9,
                )
            )
    return ExactNineProbabilitySurface(tuple(rows), "store-v1")


def test_physical_action_library_and_exact_nine_are_closed_and_json_native() -> None:
    assert len(build_action_library()) == len(CENTERS) * 10
    assert tuple(row.action_id for row in actions_for_target("0")) == physical_action_ids("0")
    surface = _probability_surface()
    assert len(surface.rows) == 10
    assert surface.rows[0].probability_mean == pytest.approx(0.4)
    assert isinstance(json.loads(json.dumps(surface.to_payload()))["rows"], list)


def test_exact_nine_scalar_vectorized_parity_and_half_threshold() -> None:
    matrix = np.asarray(
        [np.linspace(0.1, 0.9, 9), np.full(9, 0.5)], dtype=np.float64
    )
    vector = exact_nine_means(matrix)
    scalar = np.asarray([exact_nine_mean(row) for row in matrix])
    assert np.array_equal(vector, scalar)
    assert scalar[1] == HARD_THRESHOLD
    assert ExactNineProbabilityRow("0", "c", "s", "B", (0.5,) * 9).hard_prediction == 1


def test_frequency_committee_is_float64_weighted_not_modal_endpoint() -> None:
    # OFF is modal (3/5), but two source-1 votes are sufficiently high that the
    # repeated-vote mean crosses the sole final threshold.  Modal-only would
    # incorrectly retain B=0.4/class 0.
    rows = (
        (None, 3, 5),
        ("1", 2, 5),
        *((source, 0, 1) for source in candidate_sources("0") if source != "1"),
    )
    direction_01 = DirectionalControlDecision(
        "LOO_frequency_committee", "0", "held", "zero_to_one", None, rows, 5
    )
    direction_10 = DirectionalControlDecision(
        "LOO_frequency_committee", "0", "held", "one_to_zero", None, rows, 5
    )
    decision = CaseControlDecision(
        "LOO_frequency_committee", "0", "held", direction_01, direction_10
    )
    surface = _probability_surface(
        overrides={("s0", a1_action_id("1")): 0.9}
    )
    prediction = compose_control_predictions(
        surface, (decision,), method_id="LOO_frequency_committee"
    )[0]
    repeated = np.asarray([0.4, 0.4, 0.4, 0.9, 0.9], dtype=np.float64)
    assert prediction.probability == np.mean(repeated, dtype=np.float64)
    assert prediction.hard_prediction == 1
    assert direction_01.selected_source is None
    assert compose_frequency_weighted_probability(
        (0.4, 0.9), (3, 2), nested_count=5
    ) == np.mean(repeated, dtype=np.float64)


def test_frequency_committee_float64_threshold_near_half_matches_repeated_votes() -> None:
    values = (np.nextafter(0.5, 0.0), np.nextafter(0.5, 1.0))
    result = compose_frequency_weighted_probability(values, (1, 1), nested_count=2)
    repeated = np.asarray(values, dtype=np.float64)
    assert result == np.mean(repeated, dtype=np.float64)
    assert int(result >= HARD_THRESHOLD) == int(
        np.mean(repeated, dtype=np.float64) >= HARD_THRESHOLD
    )


def _assert_null_selector_parity(
    support: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    prior: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    permutations: tuple[tuple[int, ...], ...],
) -> None:
    rankings = tuple(
        tuple(sorted(range(8), key=lambda ordinal: (-prior[d][ordinal], ordinal)))
        for d in range(2)
    )
    vector = select_scrambled_endpoints_vectorized(
        np.asarray([support], dtype=np.float64),
        np.asarray([prior], dtype=np.float64),
        np.asarray([rankings], dtype=np.int8),
        np.asarray([[permutation] for permutation in permutations], dtype=np.uint8),
    )
    scalar = np.asarray(
        [
            select_scrambled_endpoints_scalar(
                support, prior, rankings, permutation
            )
            for permutation in permutations
        ],
        dtype=np.int8,
    )[:, None, :, :]
    assert np.array_equal(vector, scalar)


def test_null_selector_scalar_vectorized_exact_tie_and_tolerance_parity() -> None:
    zero = Fraction(0)
    support = (
        (
            Fraction(1, 5 * 10**12),  # final score 1e-13 at w=1/2 -> OFF
            Fraction(3, 10**12),  # final score 1.5e-12 -> active
            Fraction(1, 7),
            Fraction(1, 7),  # algebraic tie, numeric ordinal wins
            zero,
            zero,
            zero,
            zero,
        ),
        (
            Fraction(-1, 13),
            Fraction(1, 11),
            Fraction(2, 11),
            Fraction(3, 11),
            zero,
            zero,
            zero,
            zero,
        ),
    )
    # Exact prior ties exercise numeric ranking, while the small exact gaps are
    # applied only at final selection.
    prior = (
        (zero, zero, Fraction(2, 9), Fraction(2, 9), zero, zero, zero, zero),
        (zero, zero, zero, zero, Fraction(1, 8), Fraction(1, 8), zero, zero),
    )
    _assert_null_selector_parity(
        support,
        prior,
        (
            tuple(range(8)),
            tuple(reversed(range(8))),
            (1, 0, 3, 2, 5, 4, 7, 6),
        ),
    )


def test_null_selector_scalar_vectorized_integer_count_fraction_parity() -> None:
    rng = np.random.default_rng(20260813)
    permutations = tuple(tuple(int(value) for value in rng.permutation(8)) for _ in range(12))
    for _ in range(8):
        support = tuple(
            tuple(
                Fraction(int(rng.integers(-40, 41)), int(rng.integers(81, 400)))
                for _source in range(8)
            )
            for _direction in range(2)
        )
        prior = tuple(
            tuple(
                Fraction(int(rng.integers(-40, 41)), int(rng.integers(81, 400)))
                for _source in range(8)
            )
            for _direction in range(2)
        )
        _assert_null_selector_parity(support, prior, permutations)  # type: ignore[arg-type]


def test_null_plan_scalar_permutation_matches_materialized_matrix() -> None:
    route_keys = tuple(
        (center, f"c{ordinal:03d}")
        for center in CENTERS
        for ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )
    plan = build_candidate_identity_null_plan(route_keys)
    check = validate_candidate_identity_null_plan_contract(
        plan, canonical_nulls_payload()
    )
    assert check["candidate_identity_null_contract_exact"] is True
    tampered = {
        **canonical_nulls_payload(),
        "scrambled_surface": "whole_endpoint_score_surface",
    }
    with pytest.raises(ProtocolError, match="plan/config contract drifted"):
        validate_candidate_identity_null_plan_contract(plan, tampered)
    matrix = plan.materialize()
    for replicate, route_ordinal in ((0, 0), (17, 61), (9_999, 217)):
        assert plan.permutation(replicate, route_ordinal) == tuple(
            int(value) for value in matrix[replicate, route_ordinal]
        )


def test_directional_counts_scalar_vectorized_parity_and_exact_formula() -> None:
    baseline = (0.4, 0.4, 0.6, 0.6, 0.5)
    action = (0.6, 0.6, 0.4, 0.4, 0.5)
    labels = (1, 0, 1, 0, 1)
    assert directional_flip_counts(baseline, action, labels) == (1, 1, 1, 1)
    assert directional_flip_counts_scalar(baseline, action, labels) == (1, 1, 1, 1)
    row = CaseActionConfusion("0", "a", a1_action_id("1"), 3, 2, 2, 1, 1, 1, 1, 1)
    gain_01 = directional_hard_flip_gain(
        (row,),
        query_center="0",
        source="1",
        direction="zero_to_one",
        contributing_case_ids=("a",),
        label_scope="route",
    )
    gain_10 = directional_hard_flip_gain(
        (row,),
        query_center="0",
        source="1",
        direction="one_to_zero",
        contributing_case_ids=("a",),
        label_scope="route",
    )
    assert gain_01.exact == Fraction(1, 6) - Fraction(1, 4)
    assert gain_10.exact == Fraction(1, 4) - Fraction(1, 6)


def test_whole_case_loo_builds_all_218_and_excludes_held_case() -> None:
    identities = tuple(
        SimpleNamespace(target_center=center, case_id=f"c{ordinal:03d}", sample_id=f"s{ordinal:03d}")
        for center in CENTERS
        for ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )
    plans = build_whole_case_loo_plans(identities, probability_surface_hash="surface")
    assert len(plans) == 218
    assert all(plan.case_id not in plan.support_case_ids for plan in plans)
    assert all(
        len(plan.support_case_ids) == EXPECTED_CASE_COUNTS_BY_CENTER[plan.target_center] - 1
        for plan in plans
    )
    seal = seal_loo_plans(plans, probability_surface_hash="surface")
    assert len(seal.plan_seal_hash) == 64


def test_donor_prior_uses_equal_centers_and_strict_q_notin_H_e() -> None:
    rows = []
    for query in CENTERS:
        if query in {"0", "1"}:
            continue
        rows.append(
            CaseActionConfusion(
                query,
                f"case-{query}",
                a1_action_id("1"),
                10,
                6,
                10,
                6,
                int(query) % 3,
                0,
                0,
                0,
            )
        )
    prior = compute_donor_prior(
        rows,
        heldout_center="0",
        source="1",
        direction="zero_to_one",
    )
    assert tuple(row.query_center for row in prior.query_gains) == tuple(
        center for center in CENTERS if center not in {"0", "1"}
    )
    assert prior.exact == sum((row.exact for row in prior.query_gains), Fraction()) / 7
    assert all(row.query_center not in {prior.heldout_center, prior.source} for row in prior.query_gains)


def test_endpoint_library_is_exact_nine_and_matched_G_has_identical_arms() -> None:
    dcse, matched = build_matched_endpoint_libraries()
    assert tuple(arm.arm_id for arm in dcse.arms) == ARM_IDS
    assert tuple(arm.to_payload() for arm in dcse.arms) == tuple(
        arm.to_payload() for arm in matched.arms
    )
    assert {arm.k for arm in build_endpoint_arms()} == {4, 5, 6}
    assert {arm.weight for arm in build_endpoint_arms()} == {
        Fraction(1, 2),
        Fraction(3, 5),
        Fraction(7, 10),
    }


def test_exact_rational_score_above_tolerance_beats_OFF() -> None:
    support: list[DirectionalGain] = []
    priors: list[DonorPrior] = []
    for source in candidate_sources("0"):
        for direction in DIRECTION_IDS:
            favorable = 1 if source == "1" else 0
            support.append(
                _gain(
                    "0", source, direction, favorable=favorable,
                    n_positive=10**11, n_negative=10**11, excluded="held",
                )
            )
            priors.append(
                _prior(
                    "0", source, direction, favorable=favorable,
                    n_positive=10**11, n_negative=10**11,
                )
            )
    decision = select_direction_decision(
        method_id="DCSE_LOO",
        target_center="0",
        case_id="held",
        arm=build_endpoint_arms()[0],
        direction="zero_to_one",
        support_gains=tuple(support),
        donor_priors=tuple(priors),
    )
    assert float(decision.selected_score) > 1.0e-12
    assert decision.selected_source == "1"


def test_exact_rational_1e_minus_13_gap_ties_with_OFF() -> None:
    support: list[DirectionalGain] = []
    priors: list[DonorPrior] = []
    for source in candidate_sources("0"):
        for direction in DIRECTION_IDS:
            favorable = 1 if source == "1" else 0
            support.append(
                _gain(
                    "0", source, direction, favorable=favorable,
                    n_positive=5 * 10**12, n_negative=5 * 10**12, excluded="held",
                )
            )
            priors.append(
                _prior(
                    "0", source, direction, favorable=favorable,
                    n_positive=5 * 10**12, n_negative=5 * 10**12,
                )
            )
    decision = select_direction_decision(
        method_id="DCSE_LOO",
        target_center="0",
        case_id="held",
        arm=build_endpoint_arms()[0],
        direction="zero_to_one",
        support_gains=tuple(support),
        donor_priors=tuple(priors),
    )
    assert max(row.score for row in decision.scores) == Fraction(1, 10**13)
    assert decision.selected_source is None


def test_near_equal_sources_within_exact_tolerance_use_numeric_tie_order() -> None:
    support: list[DirectionalGain] = []
    priors: list[DonorPrior] = []
    for source in candidate_sources("0"):
        for direction in DIRECTION_IDS:
            favorable = (
                2 * 10**12
                if source == "1"
                else 2 * 10**12 + 1
                if source == "2"
                else 0
            )
            support.append(
                _gain(
                    "0", source, direction, favorable=favorable,
                    n_positive=10**13, n_negative=10**13, excluded="held",
                )
            )
            priors.append(
                _prior(
                    "0", source, direction, favorable=favorable,
                    n_positive=10**13, n_negative=10**13,
                )
            )
    decision = select_direction_decision(
        method_id="DCSE_LOO",
        target_center="0",
        case_id="held",
        arm=build_endpoint_arms()[0],
        direction="zero_to_one",
        support_gains=tuple(support),
        donor_priors=tuple(priors),
    )
    assert decision.scores[1].source == "2"
    assert decision.scores[2].source == "1"
    assert decision.scores[1].score > decision.scores[2].score
    assert decision.scores[1].score - decision.scores[2].score < Fraction(1, 10**12)
    assert decision.selected_source == "1"


def test_top_K_prior_ranking_has_no_tolerance_at_boundary() -> None:
    priors = {
        source: Fraction(10 - ordinal, 100)
        for ordinal, source in enumerate(candidate_sources("0"))
    }
    # Source 6 is numerically earlier than 7 but source 7 is exactly 1e-13
    # larger; exact G ranking must place 7 ahead even at the K=4 boundary.
    priors["6"] = Fraction(1, 10)
    priors["7"] = Fraction(1, 10) + Fraction(1, 10**13)
    ranked = rank_sources_by_prior(priors, target_center="0")
    assert ranked.index("7") < ranked.index("6")


def test_exact_zero_tie_selects_OFF_then_numeric_source_ties_are_stable() -> None:
    support, priors = _gain_surfaces(winning_source="9")
    zero_support = tuple(
        _gain("0", source, direction, excluded="held")
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    zero_priors = tuple(
        _prior("0", source, direction)
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    decision = select_direction_decision(
        method_id="DCSE_LOO",
        target_center="0",
        case_id="held",
        arm=build_endpoint_arms()[0],
        direction="zero_to_one",
        support_gains=zero_support,
        donor_priors=zero_priors,
    )
    assert decision.selected_source is None
    assert decision.selected_action_id == "OFF"
    del support, priors


def test_matched_G_collapse_OFF_and_duplicate_nine_arm_composition() -> None:
    support, priors = _gain_surfaces()
    decisions = select_arm_decisions(
        method_id="DCSE_LOO",
        target_center="0",
        case_id="held",
        support_gains=support,
        donor_priors=priors,
    )
    matched = select_matched_g_decisions(
        target_center="0", case_id="held", donor_priors=priors
    )
    assert len(decisions) == len(matched) == 9
    assert all(row.zero_to_one.selected_source == "1" for row in decisions)
    assert all(row.zero_to_one.selected_source == "1" for row in matched)
    surface = _probability_surface(
        overrides={("s0", a1_action_id("1")): 0.8}
    )
    prediction = compose_method_predictions(surface, decisions, method_id="DCSE_LOO")[0]
    assert prediction.probability == pytest.approx(0.8)
    assert prediction.selected_sources_by_arm == ("1",) * 9


def test_B_hard_class_mask_switches_direction_and_comparators_are_distinct() -> None:
    support, priors = _gain_surfaces(winning_source="1")
    # Make source 2 win the 1->0 branch while source 1 still wins 0->1.
    support = tuple(
        _gain("0", row.source, row.direction, favorable=(1 if (row.direction == "zero_to_one" and row.source == "1") or (row.direction == "one_to_zero" and row.source == "2") else 0), excluded="held")
        for row in support
    )
    priors = tuple(
        _prior("0", row.source, row.direction, favorable=(1 if (row.direction == "zero_to_one" and row.source == "1") or (row.direction == "one_to_zero" and row.source == "2") else 0))
        for row in priors
    )
    decisions = select_arm_decisions(
        method_id="DCSE_LOO",
        target_center="0",
        case_id="held",
        support_gains=support,
        donor_priors=priors,
    )
    surface = _probability_surface(
        samples=(("low", 0.4), ("high", 0.6)),
        overrides={
            ("low", a1_action_id("1")): 0.9,
            ("high", a1_action_id("2")): 0.1,
        },
    )
    composed = compose_method_predictions(surface, decisions, method_id="DCSE_LOO")
    assert tuple(row.probability for row in composed) == pytest.approx((0.1, 0.9)) or tuple(row.probability for row in composed) == pytest.approx((0.9, 0.1))
    by_sample = {row.sample_id: row for row in composed}
    assert by_sample["low"].selected_sources_by_arm == ("1",) * 9
    assert by_sample["high"].selected_sources_by_arm == ("2",) * 9
    hard = compose_hard_vote_comparator(surface, decisions)
    unique = compose_unique_source_mean_comparator(surface, decisions)
    assert {row.sample_id: row.hard_prediction for row in hard} == {"low": 1, "high": 0}
    assert {row.sample_id: row.probability for row in unique} == pytest.approx({"low": 0.9, "high": 0.1})


def test_label_firewall_blocks_terminal_until_all_218_route_decisions() -> None:
    identities = tuple(
        SimpleNamespace(target_center=center, case_id=f"c{ordinal:03d}", sample_id=f"s{ordinal:03d}")
        for center in CENTERS
        for ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )
    plans = build_whole_case_loo_plans(identities, probability_surface_hash="surface")
    seal = seal_loo_plans(plans, probability_surface_hash="surface")
    raw_labels = tuple(
        SimpleNamespace(
            target_center=row.target_center,
            case_id=row.case_id,
            sample_id=row.sample_id,
            value=0,
        )
        for row in identities
    )
    firewall = LabelCapabilityFirewall(
        seal,
        lambda allowed: tuple(
            row
            for row in raw_labels
            if (row.target_center, row.case_id, row.sample_id) in allowed
        ),
    )
    with pytest.raises(ProtocolError, match="all 218"):
        firewall.open_terminal_labels()
    for plan in plans:
        labels = firewall.open_route_support_labels(
            plan.target_center, plan.case_id, plan_hash=plan.plan_hash
        )
        assert plan.case_id not in {row.case_id for row in labels}
        firewall.record_route_decision_seal(
            plan.target_center,
            plan.case_id,
            canonical_hash({"plan_hash": plan.plan_hash}),
        )
    with pytest.raises(ProtocolError, match="aggregate"):
        firewall.open_terminal_labels()
    barrier = firewall.decision_barrier_payload()
    aggregate_hash = canonical_hash(
        {
            "plan_seal_hash": seal.plan_seal_hash,
            "decision_barrier_hash": barrier["decision_barrier_hash"],
            "persisted_and_read_back": True,
        }
    )
    firewall.record_aggregate_plan_decision_seal(
        aggregate_hash,
        plan_seal_hash=seal.plan_seal_hash,
        decision_barrier_hash=str(barrier["decision_barrier_hash"]),
    )
    terminal = firewall.open_terminal_labels()
    assert len(terminal) == 218
    assert firewall.report_payload()["all_218_plan_and_decision_seals_before_terminal_labels"] is True
    assert len(json.loads(json.dumps(firewall.decision_barrier_payload()))["decision_seals"]) == 218


def test_label_firewall_does_not_decode_forbidden_label_values() -> None:
    identities = tuple(
        SimpleNamespace(target_center=center, case_id=f"c{ordinal:03d}", sample_id=f"s{ordinal:03d}")
        for center in CENTERS
        for ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )
    plans = build_whole_case_loo_plans(identities, probability_surface_hash="surface")
    seal = seal_loo_plans(plans, probability_surface_hash="surface")

    decoded: list[tuple[str, str, str]] = []

    class PoisonRow:
        def __init__(self, identity: object, allowed: frozenset[tuple[str, str, str]]) -> None:
            self.target_center = identity.target_center
            self.case_id = identity.case_id
            self.sample_id = identity.sample_id
            self._allowed = allowed

        @property
        def value(self) -> int:
            key = (self.target_center, self.case_id, self.sample_id)
            if key not in self._allowed:
                raise AssertionError("forbidden label value was decoded")
            decoded.append(key)
            return 0

    requested: list[frozenset[tuple[str, str, str]]] = []

    def bad_loader(allowed: frozenset[tuple[str, str, str]]) -> tuple[PoisonRow, ...]:
        requested.append(allowed)
        # An unscoped provider is rejected during identity inspection, before
        # any forbidden label value is decoded.
        return tuple(PoisonRow(identity, allowed) for identity in identities)

    firewall = LabelCapabilityFirewall(seal, bad_loader)
    with pytest.raises(ProtocolError, match="unauthorized identity"):
        firewall.open_donor_labels("0", "1")
    assert all(key in requested[0] for key in decoded)

    def loader(allowed: frozenset[tuple[str, str, str]]) -> tuple[PoisonRow, ...]:
        requested.append(allowed)
        return tuple(
            PoisonRow(identity, allowed)
            for identity in identities
            if (identity.target_center, identity.case_id, identity.sample_id) in allowed
        )

    requested.clear()
    decoded.clear()
    firewall = LabelCapabilityFirewall(seal, loader)
    donor = firewall.open_donor_labels("0", "1")
    assert {row.target_center for row in donor} == set(CENTERS).difference({"0", "1"})
    assert all(key[0] not in {"0", "1"} for key in requested[0])

    # A fresh manager demonstrates H-minus-c value isolation independently of
    # the donor-before-support ordering constraint.
    requested.clear()
    firewall = LabelCapabilityFirewall(seal, loader)
    held = plans[0]
    support = firewall.open_route_support_labels(
        held.target_center, held.case_id, plan_hash=held.plan_hash
    )
    assert held.case_id not in {row.case_id for row in support}
    assert all(key[:2] != held.key for key in requested[0])
