from __future__ import annotations

from types import SimpleNamespace
from fractions import Fraction

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.actions import (
    actions_for_target,
    build_action_library,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.constants import (
    ACTION_COUNT_PER_TARGET,
    CANDIDATE_FEATURE_PERMUTATION_ALGORITHM,
    CANDIDATE_FEATURE_PERMUTATION_SEED,
    CENTERS,
    DIRECTION_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TOTAL_CASE_COUNT,
    FEATURE_NAMES,
    OFF_ACTION_ID,
    SEED_PAIR_COUNT,
    a1_action_id,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.held_case_plans import (
    HeldCasePlan,
    build_held_case_plans,
    seal_held_case_plans,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.label_capabilities import (
    DirectionalCorrectnessLabelFirewall,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.model import (
    fit_directional_correctness_model,
    predict_directional_correctness,
    support_denominator_case_proxy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.features import (
    build_label_free_case_candidate_features,
    candidate_feature_permutation,
    case_directional_features,
    permute_route_candidate_feature_blocks,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.donor_priors import (
    compute_donor_priors,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.ensemble import (
    compose_case_predictions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.decisions import (
    select_case_directional_abstention_decision,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
    SeedProbabilityRow,
    aggregate_exact_nine,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.products import (
    BinaryLabel,
    CandidateDirectionalScore,
    CaseAbstentionDecision,
    DirectionalAbstentionDecision,
    DirectionalCorrectnessObservation,
    DirectionalCorrectnessModel,
    DirectionalGain,
    DonorDirectionalPrior,
    LabelFreeDirectionalFeatures,
    MethodPrediction,
    SupportClassDenominators,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.scoring import (
    score_directional_correctness_observations,
    score_permuted_directional_correctness_observations,
    support_class_denominators,
)
from midogpp_thesis.cvae.protocol import ProtocolError


STABLE_HASH = "probability-store-v1"


def test_method_prediction_rejects_unknown_target_even_when_off() -> None:
    with pytest.raises(ProtocolError, match="method prediction drifted"):
        MethodPrediction("4", "case", "sample", "B", 0.4, 0, 0, None)


def _all_case_identities() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            center=center,
            case_id=f"case-{center}-{case_ordinal:02d}",
            sample_id=f"sample-{center}-{case_ordinal:02d}",
            group_id=f"group-{center}-{case_ordinal:02d}",
        )
        for center in CENTERS
        for case_ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )


def _seed_rows(target: str = "0") -> tuple[SeedProbabilityRow, ...]:
    return tuple(
        SeedProbabilityRow(
            target,
            "case-0",
            "sample-0",
            action_id,
            seed_pair_ordinal,
            0.1 + 0.01 * seed_pair_ordinal,
            STABLE_HASH,
        )
        for action_id in physical_action_ids(target)
        for seed_pair_ordinal in range(SEED_PAIR_COUNT)
    )


def _feature_probability_surface() -> ExactNineProbabilitySurface:
    rows: list[ExactNineProbabilityRow] = []
    for case_id in ("support", "held"):
        for sample_id, baseline in (("low", 0.4), ("high", 0.6)):
            for action_id in physical_action_ids("0"):
                probabilities = (baseline,) * SEED_PAIR_COUNT
                if action_id == a1_action_id("1") and sample_id == "low":
                    probabilities = (0.6,) * 6 + (0.4,) * 3
                rows.append(
                    ExactNineProbabilityRow(
                        "0", case_id, f"{case_id}-{sample_id}", action_id, probabilities
                    )
                )
    return ExactNineProbabilitySurface(tuple(rows), STABLE_HASH)


def _composition_probability_surface() -> ExactNineProbabilitySurface:
    rows = []
    for sample_id, baseline in (("low", 0.4), ("high", 0.6)):
        for action_id in physical_action_ids("0"):
            probability = baseline
            if sample_id == "low" and action_id == a1_action_id("1"):
                probability = 0.9
            if sample_id == "high" and action_id == a1_action_id("2"):
                probability = 0.1
            rows.append(
                ExactNineProbabilityRow(
                    "0", "held", sample_id, action_id, (probability,) * 9
                )
            )
    return ExactNineProbabilitySurface(tuple(rows), STABLE_HASH)


def _directional_decision(
    direction: str, selected_source: str | None
) -> DirectionalAbstentionDecision:
    scores = tuple(
        CandidateDirectionalScore(
            "0",
            "held",
            direction,
            source,
            0.0,
            0,
            0.0,
            float(source is not None and source == selected_source),
            float(source is not None and source == selected_source),
            None,
        )
        for source in (None, *candidate_sources("0"))
    )
    return DirectionalAbstentionDecision(
        "CDCA_LOO", "0", "held", direction, scores, selected_source
    )


def test_constants_actions_and_exact_nine_surface_are_closed() -> None:
    assert CENTERS == ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    assert DIRECTION_IDS == ("zero_to_one", "one_to_zero")
    assert len(FEATURE_NAMES) == 6
    assert OFF_ACTION_ID not in physical_action_ids("0")

    sources = candidate_sources("0")
    expected_actions = ("B", "U", *(a1_action_id(source) for source in sources))
    assert len(sources) == 8
    assert physical_action_ids("0") == expected_actions
    assert tuple(action.action_id for action in actions_for_target("0")) == expected_actions
    assert len(build_action_library()) == len(CENTERS) * ACTION_COUNT_PER_TARGET

    surface = aggregate_exact_nine(_seed_rows())
    assert len(surface.rows) == ACTION_COUNT_PER_TARGET
    assert all(len(row.seed_probabilities) == SEED_PAIR_COUNT for row in surface.rows)
    assert all(row.probability_mean == pytest.approx(0.14) for row in surface.rows)
    assert all(row.to_payload()["mean_before_threshold"] is True for row in surface.rows)

    with pytest.raises(ProtocolError, match="exact-nine"):
        aggregate_exact_nine(_seed_rows()[:-1])


def test_all_218_whole_case_plans_are_exact_H_minus_c() -> None:
    identities = _all_case_identities()
    plans = build_held_case_plans(
        identities,
        probability_surface_hash=STABLE_HASH,
    )
    seal = seal_held_case_plans(plans, probability_surface_hash=STABLE_HASH)

    assert len(plans) == EXPECTED_TOTAL_CASE_COUNT == 218
    assert len(seal.plans) == EXPECTED_TOTAL_CASE_COUNT
    for plan in plans:
        expected_support = {
            row.case_id
            for row in identities
            if row.center == plan.target_center and row.case_id != plan.case_id
        }
        assert set(plan.support_case_ids) == expected_support
        assert plan.case_id not in plan.support_case_ids
        assert plan.evaluation_sample_ids == (
            f"sample-{plan.target_center}-{int(plan.case_id.rsplit('-', 1)[1]):02d}",
        )
        assert plan.to_payload()["held_case_and_group_excluded"] is True
        assert plan.to_payload()["labels_used"] is False

    assert seal.to_payload()["sealed_before_route_labels"] is True
    assert seal.to_payload()["terminal_labels_used"] is False


def test_six_directional_features_are_label_free_and_use_exact_nine_means() -> None:
    surface = _feature_probability_surface()
    feature = case_directional_features(
        surface, "0", "held", "1", "zero_to_one"
    )

    assert feature.feature_names == FEATURE_NAMES
    assert feature.directional_flip_count == 1
    assert feature.case_size == 2
    assert feature.values == pytest.approx(
        (
            0.5,
            0.1,
            1.0 / 30.0,
            2.0 / 15.0,
            2.0 / 3.0,
            4.0 / 9.0,
        )
    )
    payload = feature.to_payload()
    assert payload["labels_used"] is False
    assert not ({"label", "outcome", "successes", "trials"} & set(payload))

    one_to_zero = case_directional_features(
        _composition_probability_surface(), "0", "held", "2", "one_to_zero"
    )
    assert one_to_zero.directional_flip_count == 1
    assert one_to_zero.values[3] == pytest.approx(-0.5)

    all_features = build_label_free_case_candidate_features(surface)
    assert len(all_features) == 2 * 8 * 2
    assert len({row.key for row in all_features}) == len(all_features)


def test_splitmix_permutation_moves_whole_candidate_blocks_deterministically() -> None:
    assert CANDIDATE_FEATURE_PERMUTATION_ALGORITHM == (
        "splitmix64_route_direction_candidate_block_permutation_v1"
    )
    plan = HeldCasePlan(
        "0", "held", "held-group", ("support",), ("held-sample",), STABLE_HASH
    )
    features = tuple(
        LabelFreeDirectionalFeatures(
            "0",
            case_id,
            source,
            direction,
            FEATURE_NAMES,
            (
                float(int(source)),
                float(int(source) + 10),
                float(int(source) + 20),
                float(int(source) + 30),
                float(int(source) + 40),
                float(int(source) + 50),
            ),
            int(source) % 2,
            2,
        )
        for case_id in ("support", "held")
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )

    first = permute_route_candidate_feature_blocks(features, plan)
    second = permute_route_candidate_feature_blocks(tuple(reversed(features)), plan)
    assert first == second
    assert len(first) == len(features)

    original = {row.key: row for row in features}
    permuted = {row.key: row for row in first}
    for direction in DIRECTION_IDS:
        mapping = candidate_feature_permutation(
            "0", "held", direction, seed=CANDIDATE_FEATURE_PERMUTATION_SEED
        )
        assert tuple(mapping) == candidate_sources("0")
        assert set(mapping.values()) == set(candidate_sources("0"))
        for case_id in ("support", "held"):
            for destination, donor in mapping.items():
                actual = permuted[("0", case_id, destination, direction)]
                expected = original[("0", case_id, donor, direction)]
                assert actual.values == expected.values
                assert actual.directional_flip_count == expected.directional_flip_count

    assert candidate_feature_permutation("0", "held", "zero_to_one") == (
        candidate_feature_permutation("0", "held", "zero_to_one")
    )


def test_route_observations_and_denominators_use_exact_H_minus_c_labels() -> None:
    surface = _feature_probability_surface()
    plan = HeldCasePlan(
        "0",
        "held",
        "held-group",
        ("support",),
        ("held-low", "held-high"),
        surface.surface_hash,
    )
    labels = (
        BinaryLabel("0", "support", "support-low", 1, "route_support"),
        BinaryLabel("0", "support", "support-high", 0, "route_support"),
    )

    denominators = support_class_denominators(
        labels, plan, probability_surface_or_rows=surface
    )
    observations = score_directional_correctness_observations(
        surface, labels, plan
    )
    source_1_zero_to_one = next(
        row
        for row in observations
        if row.source == "1" and row.direction == "zero_to_one"
    )

    assert denominators.n_positive == denominators.n_negative == 1
    assert denominators.support_case_ids == ("support",)
    assert denominators.to_payload()["held_case_labels_used"] is False
    assert source_1_zero_to_one.route_case_id == "held"
    assert source_1_zero_to_one.support_case_id == "support"
    assert (source_1_zero_to_one.successes, source_1_zero_to_one.trials) == (1, 1)
    assert all(row.route_case_id != row.support_case_id for row in observations)

    leaked = (*labels, BinaryLabel("0", "held", "held-low", 0, "route_support"))
    with pytest.raises(ProtocolError, match="exactly H-minus-held-case"):
        score_directional_correctness_observations(surface, leaked, plan)


def test_descriptive_permutation_moves_features_but_not_response_outcomes() -> None:
    surface = _feature_probability_surface()
    plan = HeldCasePlan(
        "0",
        "held",
        "held-group",
        ("support",),
        ("held-low", "held-high"),
        surface.surface_hash,
    )
    labels = (
        BinaryLabel("0", "support", "support-low", 1, "route_support"),
        BinaryLabel("0", "support", "support-high", 0, "route_support"),
    )
    original_features = build_label_free_case_candidate_features(surface)
    canonical = score_directional_correctness_observations(
        surface, labels, plan, features=original_features
    )
    moved_features = permute_route_candidate_feature_blocks(
        original_features, plan
    )

    with pytest.raises(ProtocolError, match="flip count changed"):
        score_directional_correctness_observations(
            surface, labels, plan, features=moved_features
        )
    descriptive = score_permuted_directional_correctness_observations(
        surface, labels, plan, permuted_features=moved_features
    )

    canonical_by_key = {row.key: row for row in canonical}
    descriptive_by_key = {row.key: row for row in descriptive}
    moved_by_key = {row.key: row for row in moved_features}
    assert set(descriptive_by_key) == set(canonical_by_key)
    assert any(
        descriptive_by_key[key].feature_values
        != canonical_by_key[key].feature_values
        for key in canonical_by_key
    )
    for key, observation in descriptive_by_key.items():
        reference = canonical_by_key[key]
        assert (observation.successes, observation.trials) == (
            reference.successes,
            reference.trials,
        )
        feature_key = (
            observation.target_center,
            observation.support_case_id,
            observation.source,
            observation.direction,
        )
        assert observation.feature_values == moved_by_key[feature_key].values


def test_donor_G_is_equal_center_mean_over_q_not_in_H_or_e() -> None:
    by_source: dict[str, tuple[DirectionalGain, ...]] = {}
    expected_by_key: dict[tuple[str, str], Fraction] = {}
    for source in candidate_sources("0"):
        rows: list[DirectionalGain] = []
        eligible = tuple(center for center in CENTERS if center not in {"0", source})
        for direction in DIRECTION_IDS:
            exact_values: list[Fraction] = []
            for ordinal, query in enumerate(eligible, start=1):
                exact = Fraction(ordinal, 100)
                exact_values.append(exact)
                rows.append(
                    DirectionalGain(
                        query,
                        source,
                        direction,
                        ordinal,
                        0,
                        10,
                        10,
                        exact.numerator,
                        exact.denominator,
                        float(exact),
                    )
                )
            expected_by_key[(source, direction)] = sum(
                exact_values, Fraction()
            ) / len(eligible)
        by_source[source] = tuple(rows)

    priors = compute_donor_priors(by_source, heldout_center="0")
    assert len(priors) == 8 * 2
    for prior in priors:
        assert prior.query_centers == tuple(
            center for center in CENTERS if center not in {"0", prior.source}
        )
        assert prior.fraction == expected_by_key[(prior.source, prior.direction)]
        assert prior.to_payload()["query_excludes_heldout_and_source"] is True

    contaminated = dict(by_source)
    contaminated["1"] = (
        DirectionalGain("0", "1", "zero_to_one", 0, 0, 10, 10, 0, 1, 0.0),
        *contaminated["1"][1:],
    )
    with pytest.raises(ProtocolError, match="contains H or e"):
        compute_donor_priors(contaminated, heldout_center="0")


def test_pure_numpy_ridge_fit_is_deterministic_and_route_local() -> None:
    successes = (1, 2, 4, 6, 8, 9)
    observations = tuple(
        DirectionalCorrectnessObservation(
            "0",
            "held",
            f"support-{ordinal}",
            "1",
            "zero_to_one",
            (
                float(ordinal),
                float(ordinal % 2),
                float(ordinal * ordinal),
                float(5 - ordinal),
                float(ordinal) / 5.0,
                float((ordinal + 1) % 3),
            ),
            successes[ordinal],
            10,
        )
        for ordinal in range(6)
    )

    first = fit_directional_correctness_model(
        observations,
        target_center="0",
        case_id="held",
        source="1",
        direction="zero_to_one",
    )
    second = fit_directional_correctness_model(
        tuple(reversed(observations)),
        target_center="0",
        case_id="held",
        source="1",
        direction="zero_to_one",
    )

    assert first == second
    assert first.model_hash == second.model_hash
    assert first.converged is True
    assert first.training_case_ids == tuple(f"support-{i}" for i in range(6))
    assert "held" not in first.training_case_ids
    assert first.training_trial_count == 60
    assert first.valid_observation_count == 6
    assert first.to_payload()["intercept_penalized"] is False
    held_features = LabelFreeDirectionalFeatures(
        "0", "held", "1", "zero_to_one", FEATURE_NAMES, (2.5,) * 6, 2, 4
    )
    assert predict_directional_correctness(first, held_features) == (
        predict_directional_correctness(second, held_features)
    )

    wrong_route = DirectionalCorrectnessObservation(
        "0", "another-held", "support-extra", "1", "zero_to_one",
        (0.0,) * 6, 1, 2,
    )
    with pytest.raises(ProtocolError, match="fit scope"):
        fit_directional_correctness_model(
            (*observations, wrong_route),
            target_center="0",
            case_id="held",
            source="1",
            direction="zero_to_one",
        )


def test_support_denominator_case_proxy_matches_frozen_formula() -> None:
    assert support_denominator_case_proxy(
        0.75, 4, "zero_to_one", 20, 10
    ) == pytest.approx(0.025)
    assert support_denominator_case_proxy(
        0.75, 4, "one_to_zero", 20, 10
    ) == pytest.approx(0.125)
    assert support_denominator_case_proxy(
        0.75, 4, "zero_to_one", 20, 10, valid_model=False
    ) == 0.0
    with pytest.raises(ProtocolError, match="case-proxy inputs"):
        support_denominator_case_proxy(0.75, 4, "zero_to_one", 0, 10)


def test_all_eight_plus_OFF_exact_tie_selects_OFF_first() -> None:
    models = tuple(
        DirectionalCorrectnessModel(
            "0",
            "held",
            source,
            direction,
            FEATURE_NAMES,
            (0.0,) * 6,
            (1.0,) * 6,
            (0.0,) * 7,
            ("support",),
            0,
            0,
            False,
            0,
        )
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    features = tuple(
        LabelFreeDirectionalFeatures(
            "0", "held", source, direction, FEATURE_NAMES, (0.0,) * 6, 0, 2
        )
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    priors = tuple(
        DonorDirectionalPrior(
            "0",
            source,
            direction,
            tuple(center for center in CENTERS if center not in {"0", source}),
            tuple("a" * 64 for _ in range(7)),
            0,
            1,
            0.0,
        )
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    denominators = SupportClassDenominators(
        "0", "held", 1, 1, ("support",)
    )

    decision = select_case_directional_abstention_decision(
        method_id="CDCA_LOO",
        target_center="0",
        case_id="held",
        models=models,
        held_features=features,
        donor_priors=priors,
        denominators=denominators,
    )
    expected_sources = (None, *candidate_sources("0"))
    for directional in (decision.zero_to_one, decision.one_to_zero):
        assert tuple(score.source for score in directional.candidate_scores) == (
            expected_sources
        )
        assert all(score.final_score == 0.0 for score in directional.candidate_scores)
        off_score = directional.candidate_scores[0]
        assert (
            off_score.predicted_correctness,
            off_score.directional_flip_count,
            off_score.case_proxy,
            off_score.donor_prior,
            off_score.final_score,
            off_score.model_hash,
        ) == (0.0, 0, 0.0, 0.0, 0.0, None)
        assert directional.selected_source is None
        assert directional.to_payload()["selection_order"] == "OFF_then_numeric_source"

    descriptive = select_case_directional_abstention_decision(
        method_id="CDCA_feature_block_permutation_descriptive",
        target_center="0",
        case_id="held",
        models=models,
        held_features=features,
        donor_priors=priors,
        denominators=denominators,
    )
    assert descriptive.method_id == "CDCA_feature_block_permutation_descriptive"
    assert descriptive.zero_to_one.selected_source is None
    assert descriptive.one_to_zero.selected_source is None


def test_B_A1_composition_switches_only_on_the_matching_baseline_branch() -> None:
    surface = _composition_probability_surface()
    decision = CaseAbstentionDecision(
        "CDCA_LOO",
        "0",
        "held",
        _directional_decision("zero_to_one", "1"),
        _directional_decision("one_to_zero", "2"),
    )

    composed = {row.sample_id: row for row in compose_case_predictions(surface, decision)}
    assert composed["low"].baseline_hard_prediction == 0
    assert composed["low"].selected_source == "1"
    assert composed["low"].probability == pytest.approx(0.9)
    assert composed["low"].hard_prediction == 1
    assert composed["high"].baseline_hard_prediction == 1
    assert composed["high"].selected_source == "2"
    assert composed["high"].probability == pytest.approx(0.1)
    assert composed["high"].hard_prediction == 0

    abstained = CaseAbstentionDecision(
        "CDCA_LOO",
        "0",
        "held",
        _directional_decision("zero_to_one", None),
        _directional_decision("one_to_zero", None),
    )
    off = {row.sample_id: row for row in compose_case_predictions(surface, abstained)}
    assert {sample: row.probability for sample, row in off.items()} == pytest.approx(
        {"low": 0.4, "high": 0.6}
    )
    assert all(row.selected_source is None for row in off.values())


def test_label_firewall_requires_all_218_route_seals_and_aggregate_before_terminal() -> None:
    identities = _all_case_identities()
    plans = build_held_case_plans(
        identities, probability_surface_hash=STABLE_HASH
    )
    plan_seal = seal_held_case_plans(
        plans, probability_surface_hash=STABLE_HASH
    )
    raw_labels = tuple(
        SimpleNamespace(
            target_center=row.center,
            case_id=row.case_id,
            sample_id=row.sample_id,
            value=0,
        )
        for row in identities
    )
    firewall = DirectionalCorrectnessLabelFirewall(
        plan_seal,
        lambda allowed: tuple(
            row
            for row in raw_labels
            if (row.target_center, row.case_id, row.sample_id) in allowed
        ),
    )

    with pytest.raises(ProtocolError, match="all 218 routes and aggregate"):
        firewall.open_terminal_labels()
    first_plan = plans[0]
    with pytest.raises(ProtocolError, match="all 72 donor grants"):
        firewall.open_route_support_labels(
            first_plan.target_center,
            first_plan.case_id,
            plan_hash=first_plan.plan_hash,
        )
    for heldout in CENTERS:
        for source in candidate_sources(heldout):
            donor = firewall.open_donor_labels(heldout, source)
            assert {row.target_center for row in donor} == set(CENTERS).difference(
                {heldout, source}
            )
    for plan in plans:
        support = firewall.open_route_support_labels(
            plan.target_center, plan.case_id, plan_hash=plan.plan_hash
        )
        assert plan.case_id not in {row.case_id for row in support}
        firewall.record_route_decision_seal(
            plan.target_center,
            plan.case_id,
            canonical_hash({"route_plan_hash": plan.plan_hash}),
        )

    assert firewall.decision_seal_count == EXPECTED_TOTAL_CASE_COUNT
    with pytest.raises(ProtocolError, match="aggregate"):
        firewall.open_terminal_labels()
    barrier = firewall.decision_barrier_payload()
    assert barrier["route_count"] == EXPECTED_TOTAL_CASE_COUNT
    assert barrier["terminal_labels_used"] is False
    aggregate_hash = canonical_hash(
        {
            "plan_seal_hash": plan_seal.plan_seal_hash,
            "decision_barrier_hash": barrier["decision_barrier_hash"],
            "persisted_and_read_back": True,
        }
    )
    firewall.record_aggregate_plan_decision_seal(
        aggregate_hash,
        plan_seal_hash=plan_seal.plan_seal_hash,
        decision_barrier_hash=str(barrier["decision_barrier_hash"]),
    )
    terminal = firewall.open_terminal_labels()
    report = firewall.report_payload()
    assert len(terminal) == EXPECTED_TOTAL_CASE_COUNT
    assert report["status"] == "PASS"
    assert report["donor_grant_count"] == 72
    assert report["all_72_donor_grants_before_route_support"] is True
    assert report["all_218_route_seals_before_terminal"] is True
    assert report["aggregate_seal_before_terminal"] is True
    assert report["raw_labels_persisted"] is False
