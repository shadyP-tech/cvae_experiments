from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import math

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling import permutation_controls
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.core_contracts import (
    canonical_statistic_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.pooled_prior import (
    _select_best_candidate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.scientific_core import (
    BASELINE_ACTION_ID,
    MIDOGPP_CENTERS,
    NULL_DERANGEMENT_ALGORITHM,
    AggregatedProbabilityRow,
    BinaryLabelRow,
    CandidateGlobalEstimate,
    CaseActionSufficientStatistics,
    CaseIdentityRow,
    DecisionConfig,
    PairwisePriorEstimate,
    PosteriorConfig,
    PriorConfig,
    SeedProbabilityRow,
    action_ids,
    aggregate_exact_nine_probabilities,
    binary_balanced_accuracy,
    build_blocked_support_permutation,
    build_case_oof_partition,
    build_permutation_decision_plan,
    candidate_actions,
    decision_seal_from_payload,
    evaluate_decision_seal,
    evaluate_statistics_seal,
    evaluation_result_from_payload,
    fit_pooled_fold_posterior,
    fit_pooled_loco_prior,
    legal_donor_centers,
    make_fold_decision,
    make_statistics_surface,
    paired_whole_case_cluster_contrast,
    partition_from_payload,
    permutation_plan_from_payload,
    permute_fold_support_statistics,
    pooled_exact_bacc,
    pooled_fold_posterior_from_payload,
    pooled_loco_prior_from_payload,
    probability_surface_from_payload,
    routing_challengers,
    score_evaluation_statistics_after_preevaluation_seals,
    score_fold_support_statistics,
    score_loco_prior_statistics,
    seal_fold_decisions,
    statistics_surface_from_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _synthetic_inputs():
    identities: list[CaseIdentityRow] = []
    labels: list[BinaryLabelRow] = []
    seed_rows: list[SeedProbabilityRow] = []
    store_hash = _hash("v2-global-prelabel-store")
    for center in MIDOGPP_CENTERS:
        for case_index in range(10):
            # Retain an explicit negative-only and positive-only case per center;
            # the remaining cases are mixed, so every two-case held fold has both classes.
            case_labels = (0,) if case_index == 0 else (1,) if case_index == 1 else (0, 1)
            case_id = f"case-{center}-{case_index}"
            for row_index, label in enumerate(case_labels):
                sample_id = f"sample-{center}-{case_index}-{row_index}"
                identities.append(CaseIdentityRow(center, case_id, sample_id))
                labels.append(BinaryLabelRow(center, case_id, sample_id, label))
                for action in action_ids(center):
                    correct = action != BASELINE_ACTION_ID
                    base = (0.9 if label else 0.1) if correct else (0.1 if label else 0.9)
                    for seed_pair in range(9):
                        seed_rows.append(
                            SeedProbabilityRow(
                                target_center=center,
                                case_id=case_id,
                                sample_id=sample_id,
                                action_id=action,
                                seed_pair_ordinal=seed_pair,
                                probability=base + (seed_pair - 4) * 0.001,
                                probability_store_hash=store_hash,
                            )
                        )
    return tuple(identities), tuple(labels), tuple(seed_rows)


@pytest.fixture(scope="module")
def pooled_run():
    identities, labels, seed_rows = _synthetic_inputs()
    probabilities = aggregate_exact_nine_probabilities(seed_rows)
    partition = build_case_oof_partition(
        identities, partition_seed=90_902_026, expected_total_case_count=90
    )
    priors = {}
    support_surfaces = {}
    posteriors = {}
    decisions = []
    for target in MIDOGPP_CENTERS:
        prior = fit_pooled_loco_prior(
            target,
            score_loco_prior_statistics(probabilities, labels, target_center=target),
        )
        priors[target] = prior
        for fold_ordinal in range(5):
            fold = partition.fold(target, fold_ordinal)
            support = score_fold_support_statistics(
                probabilities, labels, fold=fold, global_prior=prior
            )
            support_surfaces[(target, fold_ordinal)] = support
            posterior = fit_pooled_fold_posterior(fold, support, prior)
            posteriors[(target, fold_ordinal)] = posterior
            decisions.append(make_fold_decision(fold, posterior, prior))
    decision_seal = seal_fold_decisions(decisions, partition, probabilities)
    permutation_plan = build_permutation_decision_plan(
        partition,
        probabilities,
        tuple(priors[center] for center in MIDOGPP_CENTERS),
        support_surfaces,
        permutation_seed=90_912_026,
        permutation_count=19,
        chunk_size=7,
    )
    evaluation_statistics = score_evaluation_statistics_after_preevaluation_seals(
        probabilities,
        labels,
        decision_seal=decision_seal,
        permutation_plan=permutation_plan,
    )
    result = evaluate_statistics_seal(
        decision_seal,
        partition,
        evaluation_statistics,
        permutation_plan=permutation_plan,
    )
    return {
        "identities": identities,
        "labels": labels,
        "probabilities": probabilities,
        "partition": partition,
        "priors": priors,
        "support_surfaces": support_surfaces,
        "posteriors": posteriors,
        "decisions": tuple(decisions),
        "decision_seal": decision_seal,
        "permutation_plan": permutation_plan,
        "evaluation_statistics": evaluation_statistics,
        "result": result,
    }


def test_single_class_cases_retain_counts_and_only_pooled_scope_gets_bacc(pooled_run) -> None:
    surface = pooled_run["evaluation_statistics"]
    negative = surface.by_key()[("0", "case-0-0", BASELINE_ACTION_ID)]
    positive = surface.by_key()[("0", "case-0-1", BASELINE_ACTION_ID)]
    assert (negative.n_positive, negative.n_negative) == (0, 1)
    assert (positive.n_positive, positive.n_negative) == (1, 0)
    assert "exact_bacc" not in negative.to_payload()
    assert "smooth" not in " ".join(negative.to_payload())
    with pytest.raises(ProtocolError, match="both classes"):
        pooled_exact_bacc((negative,))
    pooled = pooled_exact_bacc((negative, positive))
    assert pooled.exact_bacc == binary_balanced_accuracy((0, 1), (1, 0)) == 0.0


def test_pooled_exact_bacc_matches_raw_rows_at_donor_support_and_evaluation_scopes(
    pooled_run,
) -> None:
    probabilities = pooled_run["probabilities"]
    labels = pooled_run["labels"]
    probability_map = probabilities.probabilities()
    label_map = {(row.target_center, row.case_id, row.sample_id): row.label for row in labels}
    scopes = (
        (
            score_loco_prior_statistics(probabilities, labels, target_center="0"),
            "2",
            tuple(f"case-2-{index}" for index in range(10)),
            "1",
        ),
        (
            pooled_run["support_surfaces"][("0", 0)],
            "0",
            pooled_run["partition"].fold("0", 0).support_case_ids,
            "1",
        ),
        (
            pooled_run["evaluation_statistics"],
            "0",
            tuple(f"case-0-{index}" for index in range(10)),
            BASELINE_ACTION_ID,
        ),
    )
    for surface, center, cases, action in scopes:
        statistic_rows = tuple(
            surface.by_key()[(center, case, action)] for case in cases
        )
        expected_samples = tuple(
            identity
            for identity in probabilities.identities
            if identity.target_center == center and identity.case_id in set(cases)
        )
        truth = tuple(label_map[(center, row.case_id, row.sample_id)] for row in expected_samples)
        guessed = tuple(
            int(probability_map[(center, row.case_id, row.sample_id, action)] >= 0.5)
            for row in expected_samples
        )
        assert pooled_exact_bacc(statistic_rows).exact_bacc == pytest.approx(
            binary_balanced_accuracy(truth, guessed)
        )


def test_paired_cluster_influence_variance_and_tamper_are_exact() -> None:
    challenger = (
        CaseActionSufficientStatistics("0", "a", "1", 0, 0, 2, 2),
        CaseActionSufficientStatistics("0", "b", "1", 2, 2, 0, 0),
        CaseActionSufficientStatistics("0", "c", "1", 2, 1, 2, 1),
    )
    reference = (
        CaseActionSufficientStatistics("0", "a", "B", 0, 0, 2, 1),
        CaseActionSufficientStatistics("0", "b", "B", 2, 1, 0, 0),
        CaseActionSufficientStatistics("0", "c", "B", 2, 0, 2, 0),
    )
    contrast = paired_whole_case_cluster_contrast(
        challenger, reference, variance_floor=1.0e-9
    )
    positive_differences = (0, 1, 1)
    negative_differences = (1, 0, 1)
    positive_mean = sum(positive_differences) / 4
    negative_mean = sum(negative_differences) / 4
    expected_psi = tuple(
        0.5
        * (
            (dp - left.n_positive * positive_mean) / 4
            + (dn - left.n_negative * negative_mean) / 4
        )
        for left, dp, dn in zip(challenger, positive_differences, negative_differences)
    )
    expected_variance = 3 / 2 * sum(value**2 for value in expected_psi)
    assert contrast.pooled_bacc_difference == pytest.approx(0.5)
    assert tuple(value for _case, value in contrast.case_influences) == pytest.approx(
        expected_psi
    )
    assert contrast.cluster_variance == pytest.approx(max(expected_variance, 1.0e-9))
    assert sum(value for _case, value in contrast.case_influences) == pytest.approx(0.0)
    with pytest.raises(ProtocolError, match="mathematical identity"):
        replace(contrast, pooled_bacc_difference=contrast.pooled_bacc_difference + 0.1)


def test_loco_prior_has_exact_target_candidate_and_reference_exclusions(pooled_run) -> None:
    prior = pooled_run["priors"]["0"]
    assert prior.global_action_id == "1"
    assert tuple(value.action_id for value in prior.candidate_estimates) == candidate_actions("0")
    assert tuple(value.challenger_action_id for value in prior.pairwise_estimates) == (
        routing_challengers("0", "1")
    )
    for estimate in prior.candidate_estimates:
        assert tuple(center for center, _ in estimate.donor_center_effects) == (
            legal_donor_centers("0", estimate.action_id, BASELINE_ACTION_ID)
        )
        assert estimate.other_center_count == 7
    for estimate in prior.pairwise_estimates:
        assert tuple(center for center, _ in estimate.donor_center_effects) == (
            legal_donor_centers("0", estimate.challenger_action_id, "1")
        )
        assert estimate.donor_center_count == 6
        assert estimate.challenger_action_id != prior.global_action_id

    baseline_surface = _all_candidates_equal_baseline(
        score_loco_prior_statistics(
            pooled_run["probabilities"], pooled_run["labels"], target_center="0"
        )
    )
    baseline_prior = fit_pooled_loco_prior("0", baseline_surface)
    assert baseline_prior.global_action_id == BASELINE_ACTION_ID
    assert len(baseline_prior.pairwise_estimates) == 8
    assert all(value.donor_center_count == 7 for value in baseline_prior.pairwise_estimates)


def test_prior_donor_math_gate_and_near_tie_tampering_are_rejected(pooled_run) -> None:
    prior = pooled_run["priors"]["0"]
    estimate = prior.candidate_estimates[0]
    target_donor = (("0", estimate.donor_center_effects[0][1]), *estimate.donor_center_effects[1:])
    target_tamper = replace(estimate, donor_center_effects=target_donor)
    with pytest.raises(ProtocolError, match="donor-center exclusion"):
        replace(prior, candidate_estimates=(target_tamper, *prior.candidate_estimates[1:]))
    with pytest.raises(ProtocolError, match="donor exclusions"):
        replace(
            estimate,
            donor_center_effects=(
                (estimate.action_id, estimate.donor_center_effects[0][1]),
                *estimate.donor_center_effects[1:],
            ),
        )
    with pytest.raises(ProtocolError, match="seven legal donors"):
        replace(estimate, donor_center_effects=estimate.donor_center_effects[:-1])
    with pytest.raises(ProtocolError, match="donor exclusions"):
        replace(
            estimate,
            donor_center_effects=(
                estimate.donor_center_effects[0],
                estimate.donor_center_effects[0],
                *estimate.donor_center_effects[2:],
            ),
        )
    with pytest.raises(ProtocolError, match="seven legal donors"):
        replace(estimate, donor_center_effects=(*estimate.donor_center_effects, ("0", 0.0)))
    pairwise = prior.pairwise_estimates[0]
    with pytest.raises(ProtocolError, match="source reference as a donor"):
        replace(
            pairwise,
            donor_center_effects=(
                (pairwise.reference_action_id, pairwise.donor_center_effects[0][1]),
                *pairwise.donor_center_effects[1:],
            ),
        )
    math_tamper = replace(estimate, mean_gain_vs_b=estimate.mean_gain_vs_b + 0.01)
    with pytest.raises(ProtocolError, match="mathematical identity"):
        replace(prior, candidate_estimates=(math_tamper, *prior.candidate_estimates[1:]))
    pairwise_math_tamper = replace(
        pairwise, prior_mean=pairwise.prior_mean + 0.01
    )
    with pytest.raises(ProtocolError, match="mathematical identity"):
        replace(
            prior,
            pairwise_estimates=(
                pairwise_math_tamper,
                *prior.pairwise_estimates[1:],
            ),
        )
    with pytest.raises(ProtocolError, match="best-candidate/gate"):
        replace(prior, best_candidate_action_id="2")

    estimates = []
    for index, action in enumerate(candidate_actions("0")):
        donors = tuple(center for center in MIDOGPP_CENTERS if center != action)[:7]
        mean = 1.0 + (0.5e-12 if index == 1 else 0.0 if index == 0 else -1.0)
        estimates.append(
            CandidateGlobalEstimate(
                action_id=action,
                donor_center_effects=tuple((center, mean) for center in donors),
                donor_center_case_count=7,
                mean_gain_vs_b=mean,
                variance_of_mean=1.0,
                standard_error=1.0,
                lower_confidence_bound=mean - 1.96,
            )
        )
    assert _select_best_candidate(tuple(estimates), PriorConfig().tie_tolerance).action_id == "1"


def test_label_mutations_respect_loco_support_and_evaluation_capabilities(pooled_run) -> None:
    probabilities = pooled_run["probabilities"]
    labels = pooled_run["labels"]
    original = pooled_run["priors"]["0"]
    h_mutated = tuple(
        replace(row, label=1 - row.label) if row.target_center == "0" else row
        for row in labels
    )
    repeated = fit_pooled_loco_prior(
        "0", score_loco_prior_statistics(probabilities, h_mutated, target_center="0")
    )
    assert repeated.prior_hash == original.prior_hash

    fold = pooled_run["partition"].fold("0", 0)
    support = pooled_run["support_surfaces"][("0", 0)]
    eval_cases = set(fold.evaluation_case_ids)
    eval_mutated = tuple(
        replace(row, label=1 - row.label)
        if row.target_center == "0" and row.case_id in eval_cases
        else row
        for row in labels
    )
    unchanged = score_fold_support_statistics(
        probabilities, eval_mutated, fold=fold, global_prior=original
    )
    assert unchanged.statistics_surface_hash == support.statistics_surface_hash
    support_case = fold.support_case_ids[0]
    support_mutated = tuple(
        replace(row, label=1 - row.label)
        if row.target_center == "0" and row.case_id == support_case
        else row
        for row in labels
    )
    changed = score_fold_support_statistics(
        probabilities, support_mutated, fold=fold, global_prior=original
    )
    assert changed.statistics_surface_hash != support.statistics_surface_hash


def test_normal_normal_posterior_matches_hand_calculation(pooled_run) -> None:
    posterior = pooled_run["posteriors"][("0", 0)]
    estimate = posterior.estimates[0]
    expected_variance = 1.0 / (
        1.0 / estimate.prior_variance + 1.0 / estimate.support_cluster_variance
    )
    expected_mean = expected_variance * (
        estimate.prior_mean / estimate.prior_variance
        + estimate.support_pooled_difference / estimate.support_cluster_variance
    )
    assert estimate.posterior_variance == pytest.approx(expected_variance)
    assert estimate.posterior_mean == pytest.approx(expected_mean)
    assert estimate.lower_confidence_bound == pytest.approx(
        expected_mean - 1.96 * math.sqrt(expected_variance)
    )
    assert estimate.action_id != posterior.global_action_id
    with pytest.raises(ProtocolError, match="mathematical identity"):
        replace(posterior, estimates=(replace(estimate, posterior_mean=0.123), *posterior.estimates[1:]))


def test_high_index_null_is_linear_deterministic_bijective_and_vector_identical(
    pooled_run, monkeypatch
) -> None:
    fold = pooled_run["partition"].fold("0", 0)
    original_shift = permutation_controls._case_shift
    calls = []

    def counted_shift(**kwargs):
        calls.append(kwargs["case_id"])
        return original_shift(**kwargs)

    monkeypatch.setattr(permutation_controls, "_case_shift", counted_shift)
    first = build_blocked_support_permutation(
        fold, permutation_index=9_999, permutation_seed=90_912_026
    )
    assert len(calls) == len(fold.support_case_ids)
    calls.clear()
    second = build_blocked_support_permutation(
        fold, permutation_index=9_999, permutation_seed=90_912_026
    )
    assert first == second and len(calls) == len(fold.support_case_ids)
    for case in fold.support_case_ids:
        pairs = tuple(
            (recipient, donor)
            for row_case, recipient, donor in first.case_recipient_donor_actions
            if row_case == case
        )
        assert {recipient for recipient, _ in pairs} == {donor for _, donor in pairs}
        assert all(recipient != donor for recipient, donor in pairs)

    support = _nonconstant_support(pooled_run["support_surfaces"][("0", 0)])
    perm17 = build_blocked_support_permutation(
        fold, permutation_index=17, permutation_seed=90_912_026
    )
    permuted = permute_fold_support_statistics(support, fold, perm17)
    for case in fold.support_case_ids:
        before = sorted(
            (
                row.n_positive,
                row.true_positive,
                row.n_negative,
                row.true_negative,
            )
            for row in support.rows
            if row.case_id == case and row.action_id != BASELINE_ACTION_ID
        )
        after = sorted(
            (
                row.n_positive,
                row.true_positive,
                row.n_negative,
                row.true_negative,
            )
            for row in permuted.rows
            if row.case_id == case and row.action_id != BASELINE_ACTION_ID
        )
        assert after == before
        assert permuted.by_key()[("0", case, "B")] == support.by_key()[("0", case, "B")]
    support_by_fold = dict(pooled_run["support_surfaces"])
    support_by_fold[("0", 0)] = support
    plan = build_permutation_decision_plan(
        pooled_run["partition"],
        pooled_run["probabilities"],
        tuple(pooled_run["priors"][center] for center in MIDOGPP_CENTERS),
        support_by_fold,
        permutation_seed=90_912_026,
        permutation_count=19,
        chunk_size=5,
    )
    direct = make_fold_decision(
        fold,
        fit_pooled_fold_posterior(fold, permuted, pooled_run["priors"]["0"]),
        pooled_run["priors"]["0"],
    )
    assert plan.action_codes[17, 0] == action_ids("0").index(direct.routed_action_id)
    payload = plan.to_payload()
    assert payload["null_derangement_algorithm"] == NULL_DERANGEMENT_ALGORITHM
    assert payload["uniform_over_all_derangements"] is False


def test_terminal_fold_center_inference_and_null_contracts_are_complete(pooled_run) -> None:
    result = pooled_run["result"]
    assert len(result.fold_metric_rows) == 45
    assert len(result.center_metric_rows) == 9
    assert len(result.equal_center_inference_rows) == 9
    assert len(result.action_selection_rows) == 20
    assert len(result.permutation_null_summary_rows) == 1
    assert tuple(row.endpoint for row in result.equal_center_inference_rows)[3:6] == (
        "G_H-B",
        "R-G_H",
        "R-B",
    )
    assert all(row.n_positive > 0 and row.n_negative > 0 for row in result.fold_metric_rows)
    assert all(row.top1_accuracy in (0.0, 1.0) for row in result.fold_metric_rows)
    assert all(row.tie_aware_top1_accuracy in (0.0, 1.0) for row in result.fold_metric_rows)
    null = result.permutation_null_summary_rows[0]
    assert null.two_sided_p_value == pytest.approx(
        min(1.0, 2.0 * min(null.one_sided_p_value, null.lower_tail_p_value))
    )
    direct = evaluate_decision_seal(
        pooled_run["decision_seal"],
        pooled_run["partition"],
        pooled_run["probabilities"],
        pooled_run["labels"],
        permutation_plan=pooled_run["permutation_plan"],
    )
    assert direct.scientific_result_hash == result.scientific_result_hash
    assert "per_case_bacc_used" in result.to_payload()
    assert result.to_payload()["per_case_bacc_used"] is False


def test_semantic_payload_roundtrips_and_coherent_result_tamper_fails(pooled_run) -> None:
    assert probability_surface_from_payload(
        pooled_run["probabilities"].to_payload()
    ).surface_hash == pooled_run["probabilities"].surface_hash
    assert partition_from_payload(
        pooled_run["partition"].to_payload()
    ).partition_hash == pooled_run["partition"].partition_hash
    prior = pooled_run["priors"]["0"]
    assert pooled_loco_prior_from_payload(prior.to_payload()).prior_hash == prior.prior_hash
    posterior = pooled_run["posteriors"][("0", 0)]
    assert (
        pooled_fold_posterior_from_payload(posterior.to_payload()).posterior_hash
        == posterior.posterior_hash
    )
    assert decision_seal_from_payload(
        pooled_run["decision_seal"].to_payload()
    ).decision_seal_hash == pooled_run["decision_seal"].decision_seal_hash
    plan = pooled_run["permutation_plan"]
    assert permutation_plan_from_payload(plan.to_payload(), plan.action_codes).plan_hash == plan.plan_hash
    result = pooled_run["result"]
    assert evaluation_result_from_payload(
        result.to_payload()
    ).scientific_result_hash == result.scientific_result_hash
    assert statistics_surface_from_payload(
        pooled_run["evaluation_statistics"].to_payload()
    ).statistics_surface_hash == pooled_run["evaluation_statistics"].statistics_surface_hash
    with pytest.raises(ProtocolError, match="aggregate scientific arithmetic"):
        replace(result, mean_center_routed_bacc=result.mean_center_routed_bacc - 0.1)


def test_scientific_surface_contains_no_smooth_or_per_case_bacc_api() -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.scientific_core as core

    public = set(core.__all__)
    assert not any("smooth" in name.lower() for name in public)
    assert not any(name in public for name in ("CaseActionUtility", "CaseEvaluationMetric"))
    source = inspect.getsource(CaseActionSufficientStatistics)
    assert "exact_bacc:" not in source and "smooth_bacc" not in source


def _all_candidates_equal_baseline(surface):
    lookup = surface.by_key()
    rows = []
    for row in surface.rows:
        baseline = lookup[(row.target_center, row.case_id, BASELINE_ACTION_ID)]
        rows.append(
            CaseActionSufficientStatistics(
                target_center=row.target_center,
                case_id=row.case_id,
                action_id=row.action_id,
                n_positive=baseline.n_positive,
                true_positive=baseline.true_positive,
                n_negative=baseline.n_negative,
                true_negative=baseline.true_negative,
            )
        )
    return make_statistics_surface(
        rows,
        allowed_case_keys=surface.allowed_case_keys,
        label_scope=surface.label_scope,
        prerequisite_seal_hash=surface.prerequisite_seal_hash,
    )


def _nonconstant_support(surface):
    rows = []
    for row in surface.rows:
        if row.action_id == BASELINE_ACTION_ID:
            rows.append(row)
            continue
        action_index = candidate_actions(row.target_center).index(row.action_id)
        case_index = int(row.case_id.rsplit("-", 1)[1])
        rows.append(
            CaseActionSufficientStatistics(
                target_center=row.target_center,
                case_id=row.case_id,
                action_id=row.action_id,
                n_positive=row.n_positive,
                true_positive=max(0, row.n_positive - ((action_index + case_index) % 2)),
                n_negative=row.n_negative,
                true_negative=max(0, row.n_negative - ((action_index + 2 * case_index) % 2)),
            )
        )
    return make_statistics_surface(
        canonical_statistic_rows(rows),
        allowed_case_keys=surface.allowed_case_keys,
        label_scope=surface.label_scope,
        prerequisite_seal_hash=surface.prerequisite_seal_hash,
    )
