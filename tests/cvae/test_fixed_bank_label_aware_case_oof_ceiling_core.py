"""Protocol and mathematics regressions for the label-aware OOF ceiling."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.config_payloads import (
    canonical_decision_payload,
    canonical_evaluation_payload,
    canonical_global_prior_payload,
    canonical_posterior_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.scientific_core import (
    BASELINE_ACTION_ID,
    MIDOGPP_CENTERS,
    PERMUTATION_DECISION_TIE_BREAK,
    BinaryLabelRow,
    CaseActionUtility,
    CaseIdentityRow,
    CaseUtilitySurface,
    DecisionConfig,
    PosteriorConfig,
    PriorConfig,
    SeedProbabilityRow,
    action_ids,
    aggregate_exact_nine_probabilities,
    build_blocked_support_permutation,
    build_case_oof_partition,
    build_permutation_decision_plan,
    evaluate_decision_seal,
    fit_fold_local_posterior,
    fit_label_derived_loco_global_prior,
    make_fold_decision,
    permute_fold_support_utilities,
    replace_smooth_descriptive,
    score_evaluation_utilities_after_decision_seal,
    score_fold_support_utilities,
    score_loco_prior_utilities,
    seal_fold_decisions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.core_hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.core_contracts import (
    canonical_utility_rows,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _synthetic_inputs():
    identities: list[CaseIdentityRow] = []
    labels: list[BinaryLabelRow] = []
    seed_rows: list[SeedProbabilityRow] = []
    store_hash = _hash("globally-sealed-before-labels")
    for center in MIDOGPP_CENTERS:
        for case_index in range(5):
            case_id = f"case-{center}-{case_index}"
            # One large center-0 case makes row-level BACC visibly different
            # from an unapproved equal-case mean at the terminal endpoint.
            per_class = 10 if center == "0" and case_index == 0 else 1
            for label in (0, 1):
                for replicate in range(per_class):
                    sample_id = f"sample-{center}-{case_index}-{label}-{replicate}"
                    identities.append(CaseIdentityRow(center, case_id, sample_id))
                    labels.append(BinaryLabelRow(center, case_id, sample_id, label))
                    for action in action_ids(center):
                        if action == BASELINE_ACTION_ID:
                            baseline_correct = center == "0" and case_index == 0
                            correct = baseline_correct
                        else:
                            correct = True
                        base_probability = (
                            0.9 if label == 1 else 0.1
                        ) if correct else (0.4 if label == 1 else 0.6)
                        for seed_pair in range(9):
                            # Symmetric perturbations preserve the exact-nine mean.
                            probability = base_probability + (seed_pair - 4) * 0.001
                            seed_rows.append(
                                SeedProbabilityRow(
                                    target_center=center,
                                    case_id=case_id,
                                    sample_id=sample_id,
                                    action_id=action,
                                    seed_pair_ordinal=seed_pair,
                                    probability=probability,
                                    probability_store_hash=store_hash,
                                )
                            )
    return tuple(identities), tuple(labels), tuple(seed_rows)


@pytest.fixture(scope="module")
def synthetic_run():
    identities, labels, seed_rows = _synthetic_inputs()
    probabilities = aggregate_exact_nine_probabilities(seed_rows)
    partition = build_case_oof_partition(
        identities,
        partition_seed=90_902_026,
        expected_total_case_count=45,
    )
    priors = {}
    posteriors = {}
    support_surfaces = {}
    decisions = []
    for target in MIDOGPP_CENTERS:
        prior_utilities = score_loco_prior_utilities(
            probabilities, labels, target_center=target
        )
        prior = fit_label_derived_loco_global_prior(target, prior_utilities)
        priors[target] = prior
        for fold_ordinal in range(5):
            fold = partition.fold(target, fold_ordinal)
            support = score_fold_support_utilities(
                probabilities, labels, fold=fold, global_prior=prior
            )
            support_surfaces[(target, fold_ordinal)] = support
            posterior = fit_fold_local_posterior(fold, support, prior)
            posteriors[(target, fold_ordinal)] = posterior
            decisions.append(make_fold_decision(fold, posterior, prior))
    seal = seal_fold_decisions(decisions, partition, probabilities)
    permutation_plan = build_permutation_decision_plan(
        partition,
        probabilities,
        tuple(priors[center] for center in MIDOGPP_CENTERS),
        support_surfaces,
        permutation_seed=90_912_026,
        permutation_count=31,
    )
    result = evaluate_decision_seal(
        seal,
        partition,
        probabilities,
        labels,
        permutation_plan=permutation_plan,
    )
    return {
        "identities": identities,
        "labels": labels,
        "probabilities": probabilities,
        "partition": partition,
        "priors": priors,
        "posteriors": posteriors,
        "support_surfaces": support_surfaces,
        "decisions": tuple(decisions),
        "seal": seal,
        "permutation_plan": permutation_plan,
        "result": result,
    }


def test_config_hyperparameters_match_core_defaults() -> None:
    global_payload = canonical_global_prior_payload()
    posterior_payload = canonical_posterior_payload()
    decision_payload = canonical_decision_payload()
    prior = PriorConfig()
    posterior = PosteriorConfig()
    decision = DecisionConfig()
    assert prior.prior_strength == global_payload["prior_strength"] == 8.0
    assert prior.variance_floor == global_payload["variance_floor"]
    assert prior.confidence_multiplier == global_payload["confidence_multiplier"]
    assert posterior.prior_strength == posterior_payload["prior_strength"] == 8.0
    assert posterior.variance_floor == posterior_payload["variance_floor"]
    assert posterior.confidence_multiplier == posterior_payload["confidence_multiplier"]
    assert decision.minimum_gain == decision_payload["minimum_gain"]
    assert decision.tie_tolerance == decision_payload["tie_tolerance"]
    assert (
        canonical_evaluation_payload()["permutation_decision_tie_break"]
        == PERMUTATION_DECISION_TIE_BREAK
        == "lexicographic_action_id_no_evaluation_utility_access"
    )


def test_exact_nine_global_seal_and_target_exclusion() -> None:
    identities, _labels, seed_rows = _synthetic_inputs()
    surface = aggregate_exact_nine_probabilities(tuple(reversed(seed_rows)))
    assert surface.predictions_globally_sealed_before_labels is True
    assert surface.labels_readable_during_materialization is False
    assert len(surface.rows) == len(identities) * 9
    first = seed_rows[0]
    with pytest.raises(ProtocolError, match="exact seed ordinals"):
        aggregate_exact_nine_probabilities(seed_rows[1:])
    with pytest.raises(ProtocolError, match="target expert"):
        replace(first, action_id=first.target_center)


def test_five_fold_whole_case_partition_is_deterministic_and_exact(synthetic_run) -> None:
    partition = synthetic_run["partition"]
    rebuilt = build_case_oof_partition(
        tuple(reversed(synthetic_run["identities"])),
        partition_seed=90_902_026,
        expected_total_case_count=45,
    )
    assert rebuilt.partition_hash == partition.partition_hash
    assert len(partition.folds) == 45
    for center in MIDOGPP_CENTERS:
        eval_cases = [
            case
            for fold in partition.folds
            if fold.target_center == center
            for case in fold.evaluation_case_ids
        ]
        assert len(eval_cases) == len(set(eval_cases)) == 5
        for fold in (value for value in partition.folds if value.target_center == center):
            assert set(fold.support_case_ids).isdisjoint(fold.evaluation_case_ids)


def test_loco_prior_label_mutation_scope_and_equal_center_weighting(synthetic_run) -> None:
    probabilities = synthetic_run["probabilities"]
    labels = synthetic_run["labels"]
    original_h0 = fit_label_derived_loco_global_prior(
        "0", score_loco_prior_utilities(probabilities, labels, target_center="0")
    )
    # H labels are outside G_H by construction.
    mutated_h0 = tuple(
        replace(row, label=1 - row.label) if row.target_center == "0" else row
        for row in labels
    )
    repeated_h0 = fit_label_derived_loco_global_prior(
        "0", score_loco_prior_utilities(probabilities, mutated_h0, target_center="0")
    )
    assert repeated_h0.prior_hash == original_h0.prior_hash
    # Mutating H'=1 may affect G_0, but never G_1 because center 1 is excluded there.
    original_h1 = fit_label_derived_loco_global_prior(
        "1", score_loco_prior_utilities(probabilities, labels, target_center="1")
    )
    mutated_h1_labels = tuple(
        replace(row, label=1 - row.label) if row.target_center == "1" else row
        for row in labels
    )
    changed_h0 = fit_label_derived_loco_global_prior(
        "0", score_loco_prior_utilities(probabilities, mutated_h1_labels, target_center="0")
    )
    unchanged_h1 = fit_label_derived_loco_global_prior(
        "1", score_loco_prior_utilities(probabilities, mutated_h1_labels, target_center="1")
    )
    assert changed_h0.prior_hash != original_h0.prior_hash
    assert unchanged_h1.prior_hash == original_h1.prior_hash
    assert all(estimate.other_center_count == 7 for estimate in original_h0.estimates)

    unequal = _unequal_center_utility_surface(target="0", action="9")
    weighted = fit_label_derived_loco_global_prior("0", unequal)
    estimate = weighted.estimate("9")
    assert estimate is not None
    # One positive center with 40 cases and six zero centers with one case each:
    # fixed center-equal mean is 1/7, never the pooled 40/46.
    assert estimate.shrunk_mean_gain_vs_b == pytest.approx(1.0 / 15.0)
    assert estimate.other_center_case_count == 46


def test_support_and_evaluation_label_mutations_have_fold_local_scope(synthetic_run) -> None:
    probabilities = synthetic_run["probabilities"]
    partition = synthetic_run["partition"]
    labels = synthetic_run["labels"]
    prior = synthetic_run["priors"]["0"]
    fold = partition.fold("0", 0)
    original_support = score_fold_support_utilities(
        probabilities, labels, fold=fold, global_prior=prior
    )
    original_posterior = fit_fold_local_posterior(fold, original_support, prior)
    held_case = fold.evaluation_case_ids[0]
    eval_mutated = tuple(
        replace(row, label=1 - row.label)
        if row.target_center == "0" and row.case_id == held_case
        else row
        for row in labels
    )
    heldout_unchanged = fit_fold_local_posterior(
        fold,
        score_fold_support_utilities(
            probabilities, eval_mutated, fold=fold, global_prior=prior
        ),
        prior,
    )
    assert heldout_unchanged.posterior_hash == original_posterior.posterior_hash
    support_case = fold.support_case_ids[0]
    support_mutated = tuple(
        replace(row, label=1 - row.label)
        if row.target_center == "0" and row.case_id == support_case
        else row
        for row in labels
    )
    support_changed = fit_fold_local_posterior(
        fold,
        score_fold_support_utilities(
            probabilities, support_mutated, fold=fold, global_prior=prior
        ),
        prior,
    )
    assert support_changed.posterior_hash != original_posterior.posterior_hash


def test_smooth_poison_cannot_change_posterior_or_decision(synthetic_run) -> None:
    probabilities = synthetic_run["probabilities"]
    partition = synthetic_run["partition"]
    labels = synthetic_run["labels"]
    prior = synthetic_run["priors"]["0"]
    fold = partition.fold("0", 0)
    support = score_fold_support_utilities(
        probabilities, labels, fold=fold, global_prior=prior
    )
    poisoned = replace_smooth_descriptive(support, lambda row: 1.0 - row.smooth_bacc)
    assert poisoned.exact_surface_hash == support.exact_surface_hash
    assert poisoned.descriptive_surface_hash != support.descriptive_surface_hash
    clean_posterior = fit_fold_local_posterior(fold, support, prior)
    poison_posterior = fit_fold_local_posterior(fold, poisoned, prior)
    assert poison_posterior.posterior_hash == clean_posterior.posterior_hash
    assert make_fold_decision(fold, poison_posterior, prior) == make_fold_decision(
        fold, clean_posterior, prior
    )


def test_blocked_permutation_is_nondegenerate_bijective_and_case_blocked(synthetic_run) -> None:
    partition = synthetic_run["partition"]
    probabilities = synthetic_run["probabilities"]
    labels = synthetic_run["labels"]
    prior = synthetic_run["priors"]["0"]
    fold = partition.fold("0", 0)
    first = build_blocked_support_permutation(
        fold, permutation_index=17, permutation_seed=90_912_026
    )
    second = build_blocked_support_permutation(
        fold, permutation_index=17, permutation_seed=90_912_026
    )
    assert first == second
    for case_id in fold.support_case_ids:
        case_rows = [
            (recipient, donor)
            for case, recipient, donor in first.case_recipient_donor_actions
            if case == case_id
        ]
        assert {recipient for recipient, _ in case_rows} == {
            donor for _, donor in case_rows
        }
        assert all(recipient != donor for recipient, donor in case_rows)
    assert not {
        case for case, _, _ in first.case_recipient_donor_actions
    }.intersection(fold.evaluation_case_ids)
    support = score_fold_support_utilities(
        probabilities, labels, fold=fold, global_prior=prior
    )
    nonconstant = _nonconstant_support_surface(support)
    permuted = permute_fold_support_utilities(nonconstant, fold, first)
    for case_id in fold.support_case_ids:
        before = sorted(
            row.exact_bacc
            for row in nonconstant.rows
            if row.case_id == case_id and row.action_id != BASELINE_ACTION_ID
        )
        after = sorted(
            row.exact_bacc
            for row in permuted.rows
            if row.case_id == case_id and row.action_id != BASELINE_ACTION_ID
        )
        assert after == before
        before_b = nonconstant.by_key()[("0", case_id, BASELINE_ACTION_ID)]
        after_b = permuted.by_key()[("0", case_id, BASELINE_ACTION_ID)]
        assert after_b.exact_bacc == before_b.exact_bacc
    clean_posterior = fit_fold_local_posterior(fold, nonconstant, prior)
    null_posterior = fit_fold_local_posterior(fold, permuted, prior)
    assert null_posterior.posterior_hash != clean_posterior.posterior_hash

    nonconstant_by_fold = {
        key: _nonconstant_support_surface(value)
        for key, value in synthetic_run["support_surfaces"].items()
    }
    plan = build_permutation_decision_plan(
        partition,
        probabilities,
        tuple(synthetic_run["priors"][center] for center in MIDOGPP_CENTERS),
        nonconstant_by_fold,
        permutation_seed=90_912_026,
        permutation_count=64,
    )
    assert np.unique(plan.action_codes, axis=0).shape[0] > 1
    direct_null_decision = make_fold_decision(
        fold,
        fit_fold_local_posterior(fold, permuted, prior),
        prior,
    )
    assert plan.action_codes[17, 0] == action_ids("0").index(
        direct_null_decision.routed_action_id
    )


def test_terminal_endpoint_is_row_level_and_requires_all_decisions(synthetic_run) -> None:
    result = synthetic_run["result"]
    center_zero = result.center_metrics[0]
    assert center_zero.target_center == "0"
    assert center_zero.baseline_bacc == pytest.approx(20.0 / 28.0)
    evaluation_utility = score_evaluation_utilities_after_decision_seal(
        synthetic_run["probabilities"],
        synthetic_run["labels"],
        decision_seal=synthetic_run["seal"],
    )
    per_case_baseline = [
        row.exact_bacc
        for row in evaluation_utility.rows
        if row.target_center == "0" and row.action_id == BASELINE_ACTION_ID
    ]
    assert sum(per_case_baseline) / len(per_case_baseline) == pytest.approx(0.2)
    assert center_zero.baseline_bacc != pytest.approx(sum(per_case_baseline) / len(per_case_baseline))
    assert result.mean_center_routed_bacc == pytest.approx(1.0)
    assert result.total_case_count == 45
    assert len(result.case_metric_rows) == 45
    assert len(result.center_metric_rows) == 9
    assert len(result.permutation_null_summary_rows) == 1
    assert result.permutation_null_summary_rows[0].permutation_count == 31
    assert synthetic_run["permutation_plan"].action_codes.shape == (31, 45)
    assert synthetic_run["permutation_plan"].action_codes.flags.writeable is False
    assert (
        synthetic_run["permutation_plan"].to_payload()[
            "permutation_decision_tie_break"
        ]
        == PERMUTATION_DECISION_TIE_BREAK
    )
    assert (
        synthetic_run["permutation_plan"].to_payload()[
            "evaluation_utility_used_for_permutation_tie_break"
        ]
        is False
    )
    assert result.mean_center_global_minus_baseline == pytest.approx(
        sum(row.global_minus_baseline for row in result.center_metric_rows) / 9
    )
    assert result.global_minus_baseline_ci95_lower <= (
        result.mean_center_global_minus_baseline
    ) <= result.global_minus_baseline_ci95_upper
    assert tuple(
        (row.method_id, row.action_id) for row in result.action_selection_rows
    ) == tuple(
        (method, action)
        for method in ("G_H", "R")
        for action in (BASELINE_ACTION_ID, *MIDOGPP_CENTERS)
    )
    for method in ("G_H", "R"):
        method_rows = [
            row for row in result.action_selection_rows if row.method_id == method
        ]
        assert sum(row.selection_count for row in method_rows) == 45
        assert sum(row.selection_share for row in method_rows) == pytest.approx(1.0)
    evaluation_contract = canonical_evaluation_payload()
    assert evaluation_contract["primary_contrasts"] == ["R-G_H", "R-B", "G_H-B"]
    assert set(evaluation_contract["metrics"]) == {
        "exact_bacc",
        "paired_R_minus_G_H",
        "paired_R_minus_B",
        "normalized_regret",
        "top1_accuracy",
        "tie_aware_top1_accuracy",
        "coverage",
        "source_selection_share",
    }
    # Every configured metric and contrast has a typed exact-only output.
    assert result.center_metric_rows[0].global_minus_baseline == pytest.approx(
        result.center_metric_rows[0].global_bacc
        - result.center_metric_rows[0].baseline_bacc
    )
    assert result.case_metric_rows[0].oracle_regret >= 0.0
    assert result.case_metric_rows[0].exact_top1 in (0.0, 1.0)
    assert result.case_metric_rows[0].tie_aware_top1 in (0.0, 1.0)
    assert 0.0 <= result.local_route_coverage <= 1.0
    with pytest.raises(ProtocolError, match="All 45"):
        seal_fold_decisions(
            synthetic_run["decisions"][:-1],
            synthetic_run["partition"],
            synthetic_run["probabilities"],
        )


def _unequal_center_utility_surface(*, target: str, action: str) -> CaseUtilitySurface:
    allowed = []
    rows = []
    for center in MIDOGPP_CENTERS:
        if center == target:
            continue
        count = 40 if center == "1" else 1
        for index in range(count):
            case_id = f"unequal-{center}-{index}"
            allowed.append((center, case_id))
            for candidate in action_ids(center):
                gain = 1.0 if candidate == action and center == "1" else 0.0
                exact = gain
                rows.append(
                    CaseActionUtility(
                        target_center=center,
                        case_id=case_id,
                        action_id=candidate,
                        sample_count=2,
                        exact_bacc=exact,
                        smooth_bacc=exact,
                        exact_gain_vs_b=gain,
                    )
                )
    rows = canonical_utility_rows(rows)
    allowed = tuple(sorted(allowed))
    probability_hash = _hash("unequal-probability-seal")
    exact_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_label_aware_case_utility_exact_v1",
            "label_scope": f"label_derived_LOCO_global_prior::heldout_H={target}",
            "prerequisite_seal_hash": probability_hash,
            "allowed_case_keys": [list(key) for key in allowed],
            "rows": [row.exact_payload() for row in rows],
        }
    )
    return CaseUtilitySurface(
        rows=rows,
        allowed_case_keys=allowed,
        label_scope=f"label_derived_LOCO_global_prior::heldout_H={target}",
        prerequisite_seal_hash=probability_hash,
        exact_surface_hash=exact_hash,
        descriptive_surface_hash=canonical_hash(
            {
                "exact_surface_hash": exact_hash,
                "smooth_bacc": [row.smooth_bacc for row in rows],
            }
        ),
    )


def _nonconstant_support_surface(surface: CaseUtilitySurface) -> CaseUtilitySurface:
    candidates_by_target = {
        center: tuple(action for action in action_ids(center) if action != BASELINE_ACTION_ID)
        for center in MIDOGPP_CENTERS
    }
    rows = []
    for row in surface.rows:
        if row.action_id == BASELINE_ACTION_ID:
            exact = 0.0
        else:
            exact = 0.1 * (1 + candidates_by_target[row.target_center].index(row.action_id))
        rows.append(
            CaseActionUtility(
                target_center=row.target_center,
                case_id=row.case_id,
                action_id=row.action_id,
                sample_count=row.sample_count,
                exact_bacc=exact,
                smooth_bacc=exact,
                exact_gain_vs_b=exact,
            )
        )
    rows = canonical_utility_rows(rows)
    exact_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_label_aware_case_utility_exact_v1",
            "label_scope": surface.label_scope,
            "prerequisite_seal_hash": surface.prerequisite_seal_hash,
            "allowed_case_keys": [list(key) for key in surface.allowed_case_keys],
            "rows": [row.exact_payload() for row in rows],
        }
    )
    return CaseUtilitySurface(
        rows=rows,
        allowed_case_keys=surface.allowed_case_keys,
        label_scope=surface.label_scope,
        prerequisite_seal_hash=surface.prerequisite_seal_hash,
        exact_surface_hash=exact_hash,
        descriptive_surface_hash=canonical_hash(
            {
                "exact_surface_hash": exact_hash,
                "smooth_bacc": [row.smooth_bacc for row in rows],
            }
        ),
    )
