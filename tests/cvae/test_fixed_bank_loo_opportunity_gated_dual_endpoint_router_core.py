from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.actions import (
    action_library_by_target,
    build_action_library,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.constants import (
    ARM_IDS,
    CENTERS,
    DIRECTION_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TOTAL_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    FEATURE_NAMES,
    a1_action_id,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.correctness_products import (
    DirectionalCorrectnessModel,
    LabelFreeDirectionalFeatures,
    SupportClassDenominators,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.candidate_feature_permutation import permute_route_candidate_feature_blocks
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.donor_prior import DonorPrior
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.held_case_features import build_label_free_features, case_directional_features
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.identification import select_case_identification_decision
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.identification_products import IdentificationCandidateScore
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.label_capabilities import DualEndpointLabelFirewall
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.prediction_products import MethodPrediction
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.predictions import compose_identification_case_predictions, compose_robust_case_predictions
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.probability_surfaces import ExactNineProbabilityRow, ExactNineProbabilitySurface
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.response_products import (
    BinaryLabel,
    CaseActionConfusion,
    CaseActionSufficientStat,
    DirectionalGain,
    deduplicate_sufficient_stats,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.robust import select_robust_arm_decisions
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.runner_runtime import compute_route_job
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.split_plans import WholeCaseLooPlan, build_whole_case_loo_plans, seal_whole_case_loo_plans
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.terminal_products import IdentificationMetrics, ProbabilityMetrics
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import validate_action_library


STORE = "test-store"
SHA = canonical_hash({"test": "stable"})


def _surface(*, cases: tuple[str, ...] = ("held",), two_samples: bool = True) -> ExactNineProbabilitySurface:
    rows = []
    samples = (("low", 0.4), ("high", 0.6)) if two_samples else (("low", 0.4),)
    for case in cases:
        for sample, baseline in samples:
            for action in physical_action_ids("0"):
                probability = baseline
                if sample == "low" and action == a1_action_id("1"):
                    probability = 0.9
                elif sample == "low" and action == a1_action_id("2"):
                    probability = 0.7
                elif sample == "high" and action == a1_action_id("2"):
                    probability = 0.1
                rows.append(ExactNineProbabilityRow("0", case, f"{case}-{sample}", action, (probability,) * 9))
    return ExactNineProbabilitySurface(tuple(rows), STORE)


def _model(source: str, direction: str, predicted: float) -> DirectionalCorrectnessModel:
    intercept = math.log(predicted / (1.0 - predicted))
    return DirectionalCorrectnessModel(
        "0", "held", source, direction,
        (0.0,) * 6, (1.0,) * 6, (intercept, *(0.0,) * 6),
        ("support",), (SHA,), 1, 1, True, 1,
    )


def _prior(source: str, direction: str, value_numerator: int = 0) -> DonorPrior:
    queries = tuple(center for center in CENTERS if center not in {"0", source})
    return DonorPrior(
        "0", source, direction, queries, (SHA,) * len(queries),
        value_numerator, 100, value_numerator / 100,
    )


def _identification_inputs(*, zero_opportunity: bool = False):
    plan = WholeCaseLooPlan("0", "held", "held", ("support",), ("held-low",), STORE)
    denominators = SupportClassDenominators("0", "held", 10, 10, ("support",))
    features = []
    models = []
    priors = []
    for source in candidate_sources("0"):
        for direction in DIRECTION_IDS:
            winning = (direction == "zero_to_one" and source in {"1", "2"}) or (
                direction == "one_to_zero" and source == "2"
            )
            flips = 0 if zero_opportunity else int(winning)
            features.append(
                LabelFreeDirectionalFeatures(
                    "0", "held", source, direction, FEATURE_NAMES,
                    (float(flips), 0.1, 0.1, 0.2, 1.0, 0.0) if flips else (0.0,) * 6,
                    flips, 2,
                )
            )
            models.append(_model(source, direction, 0.9 if winning else 0.5))
            priors.append(_prior(source, direction, 1 if source == "1" else 0))
    return plan, denominators, tuple(features), tuple(models), tuple(priors)


def test_neutral_action_contract_exact_nine_and_json_roundtrip() -> None:
    payload, library_hash = validate_action_library(action_library_by_target())
    assert len(payload) == len(CENTERS)
    assert len(build_action_library()) == 90
    assert len(library_hash) > 0
    surface = _surface()
    reloaded = ExactNineProbabilitySurface.from_payload(json.loads(json.dumps(surface.to_payload())))
    assert reloaded == surface
    assert all(row.to_payload()["mean_before_threshold"] is True for row in surface.rows)


def test_all_218_plans_exclude_c_and_group_before_labels() -> None:
    identities = tuple(
        SimpleNamespace(
            center=center,
            case_id=f"case-{center}-{ordinal:02d}",
            sample_id=f"sample-{center}-{ordinal:02d}",
            group_id=f"group-{center}-{ordinal:02d}",
        )
        for center in CENTERS
        for ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )
    plans = build_whole_case_loo_plans(identities, probability_surface_hash=STORE)
    seal = seal_whole_case_loo_plans(plans, probability_surface_hash=STORE)
    assert len(plans) == EXPECTED_TOTAL_CASE_COUNT
    assert all(plan.case_id not in plan.support_case_ids for plan in plans)
    assert seal.to_payload()["terminal_labels_used"] is False


def test_six_label_free_features_and_zero_opportunity_are_exact() -> None:
    feature = case_directional_features(_surface(), "0", "held", "1", "zero_to_one")
    assert feature.feature_names == FEATURE_NAMES
    assert feature.directional_flip_count == 1
    assert feature.case_size == 2
    assert feature.values[0] == pytest.approx(0.5)
    assert feature.to_payload()["labels_used"] is False
    assert case_directional_features(_surface(), "0", "held", "1", "one_to_zero").directional_flip_count == 0


def test_identification_normalizes_eight_candidates_gates_off_and_exposes_controls() -> None:
    plan, denominators, features, models, priors = _identification_inputs()
    decision = select_case_identification_decision(plan, features, models, denominators, priors)
    assert decision.zero_to_one.selected_source == "1"
    assert decision.zero_to_one.eligible_sources == ("1", "2")
    assert decision.one_to_zero.selected_source == "2"
    assert all(row.case_scale == pytest.approx(0.01) for row in decision.zero_to_one.candidate_scores)
    surface = _surface(two_samples=False)
    primary = compose_identification_case_predictions(surface, decision)[0]
    gate = compose_identification_case_predictions(surface, decision, control="gate_only")[0]
    source = compose_identification_case_predictions(surface, decision, control="source_only")[0]
    assert primary.probability == pytest.approx(0.9)
    assert gate.probability == pytest.approx(0.8)
    assert source.probability == pytest.approx(0.9)

    zero = _identification_inputs(zero_opportunity=True)
    off = select_case_identification_decision(zero[0], zero[2], zero[3], zero[1], zero[4])
    assert off.zero_to_one.selected_source is None
    assert off.zero_to_one.source_only_selected_source is None
    assert off.zero_to_one.fail_closed is True

    with pytest.raises(ProtocolError, match="candidate score drifted"):
        IdentificationCandidateScore(
            "0", "held", "zero_to_one", "1", 0.5, 1, True,
            float("nan"), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            True, False, "bad", SHA,
        )


def test_robust_endpoint_preserves_nine_duplicate_arms_and_exact_off_tolerance() -> None:
    plan = WholeCaseLooPlan("0", "held", "held", ("support",), ("held-low",), STORE)
    gains = []
    priors = []
    for source in candidate_sources("0"):
        for direction in DIRECTION_IDS:
            favorable = 2 if source == "1" else 0
            gains.append(
                DirectionalGain("0", "held", source, direction, 10, 10, favorable, 0, ("support",), "support")
            )
            priors.append(_prior(source, direction, 10 if source == "1" else 0))
    decisions = select_robust_arm_decisions(plan, gains, priors)
    assert tuple(row.arm_id for row in decisions) == ARM_IDS
    assert len(decisions) == 9
    assert [row.zero_to_one.selected_source for row in decisions] == ["1"] * 9
    robust = compose_robust_case_predictions(_surface(two_samples=False), decisions)[0]
    assert robust.probability == pytest.approx(0.9)
    assert len(robust.selected_sources_by_arm) == 9
    assert all(plan.case_id not in gain.contributing_case_ids for gain in gains)
    assert all("0" not in prior.query_centers and prior.source not in prior.query_centers for prior in priors)

    tiny_gains = tuple(
        DirectionalGain(
            "0", "held", source, direction, 10**13, 10**13,
            int(source == "1"), 0, ("support",), "support",
        )
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    zeros = tuple(_prior(source, direction, 0) for source in candidate_sources("0") for direction in DIRECTION_IDS)
    tiny = select_robust_arm_decisions(plan, tiny_gains, zeros)
    assert all(row.zero_to_one.selected_source is None for row in tiny)


def test_prediction_payload_roundtrip_and_label_barrier() -> None:
    prediction = MethodPrediction(
        "0", "held", "sample", "OGDE_PORTFOLIO", 0.7, 1, 0,
        "portfolio", "1", ("1", None), "test",
    )
    assert MethodPrediction.from_payload(json.loads(json.dumps(prediction.to_payload()))) == prediction

    identities = tuple(
        SimpleNamespace(
            center=center,
            case_id=f"c-{center}-{ordinal:02d}",
            sample_id=f"s-{center}-{ordinal:02d}",
            group_id=f"g-{center}-{ordinal:02d}",
        )
        for center in CENTERS
        for ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )
    plans = build_whole_case_loo_plans(identities, probability_surface_hash=STORE)
    seal = seal_whole_case_loo_plans(plans, probability_surface_hash=STORE)
    values = {(row.center, row.case_id, row.sample_id): ordinal % 2 for ordinal, row in enumerate(identities)}

    def loader(keys):
        return tuple(SimpleNamespace(target_center=k[0], case_id=k[1], sample_id=k[2], value=values[k]) for k in keys)

    firewall = DualEndpointLabelFirewall(seal, loader)
    with pytest.raises(ProtocolError, match="terminal labels require"):
        firewall.open_terminal_labels()
    for target in CENTERS:
        for source_center in candidate_sources(target):
            labels = firewall.open_donor_labels(target, source_center)
            assert all(row.target_center not in {target, source_center} for row in labels)
    for plan in plans:
        labels = firewall.open_route_support_labels(plan.target_center, plan.case_id, plan_hash=plan.plan_hash)
        assert plan.case_id not in {row.case_id for row in labels}
        firewall.record_route_decision_seal(plan.target_center, plan.case_id, SHA)
    barrier = firewall.decision_barrier_payload()
    firewall.record_aggregate_plan_decision_seal(
        SHA,
        plan_seal_hash=seal.plan_seal_hash,
        decision_barrier_hash=barrier["decision_barrier_hash"],
    )
    assert len(firewall.open_terminal_labels()) == EXPECTED_TOTAL_CASE_COUNT
    assert firewall.report_payload()["status"] == "PASS"


def test_scope_independent_sufficient_stats_dedupe_and_roundtrip() -> None:
    left = CaseActionConfusion(
        "0", "held", "B", 4, 3, 5, 4, 1, 2, 2, 1, "donor_capability",
    )
    right = CaseActionConfusion(
        "0", "held", "B", 4, 3, 5, 4, 1, 2, 2, 1, "route_support_capability",
    )
    assert left.confusion_hash != right.confusion_hash
    (stat,) = deduplicate_sufficient_stats((left, right))
    reloaded = CaseActionSufficientStat.from_payload(
        json.loads(json.dumps(stat.to_payload()))
    )
    assert reloaded == stat
    assert "label_scope" not in stat.to_payload()

    conflicting = CaseActionConfusion(
        "0", "held", "B", 4, 2, 5, 4, 1, 2, 2, 1, "third_capability",
    )
    with pytest.raises(ProtocolError, match="counts disagree"):
        deduplicate_sufficient_stats((left, conflicting))


def test_terminal_metric_dtos_pin_methods_counts_domains_and_center_order() -> None:
    identification = IdentificationMetrics(
        "I_OPPORTUNITY_GATED", 436, 100, 120,
        0.75, 0.80, 0.70, 0.60, 0.55, 336, 0.10, 0.02, 0.25,
    )
    probability = ProbabilityMetrics(
        "OGDE_PORTFOLIO", EXPECTED_TEST_ROW_COUNT,
        0.18, 0.52, 0.0, 1.0, 0.81,
        tuple((center, 0.81) for center in CENTERS),
    )
    json.dumps(identification.to_payload(), allow_nan=False)
    json.dumps(probability.to_payload(), allow_nan=False)

    with pytest.raises(ProtocolError, match="identification metrics drifted"):
        IdentificationMetrics(
            "R_NINE_ARM_ROBUST", 436, 0, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0,
        )
    with pytest.raises(ProtocolError, match="identification metrics drifted"):
        IdentificationMetrics(
            "I_OPPORTUNITY_GATED", 435, 0, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0,
        )
    with pytest.raises(ProtocolError, match="probability metrics drifted"):
        ProbabilityMetrics(
            "OGDE_PORTFOLIO", EXPECTED_TEST_ROW_COUNT,
            1.1, 0.52, 0.0, 1.0, 0.81,
            tuple((center, 0.81) for center in reversed(CENTERS)),
        )


def test_permuted_features_preserve_canonical_order_through_full_route_worker() -> None:
    surface = _surface(cases=("support", "held"))
    plan = WholeCaseLooPlan(
        "0", "held", "held", ("support",),
        ("held-low", "held-high"), STORE,
    )
    features = build_label_free_features(surface)
    permuted = permute_route_candidate_feature_blocks(features, plan)
    assert tuple(row.key for row in permuted) == tuple(
        ("0", case, source, direction)
        for case in ("support", "held")
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    labels = (
        BinaryLabel("0", "support", "support-low", 0, "route_support"),
        BinaryLabel("0", "support", "support-high", 1, "route_support"),
    )
    priors = tuple(
        _prior(source, direction)
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    result = compute_route_job(
        surface,
        {
            "plan": plan,
            "support_labels": labels,
            "donor_priors": priors,
            "route_features": features,
        },
    )
    assert tuple(row.method_id for row in result.identification_decisions) == (
        "I_OPPORTUNITY_GATED",
        "I_FEATURE_BLOCK_PERMUTED",
    )
    assert len(result.robust_arm_decisions) == 18
