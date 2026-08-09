"""Focused scientific-core tests for the fixed-bank decision audit."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.constants import (
    CASE_AWARE_BOUNDARY_EXACT,
    CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL,
    CASE_BALANCED_RICH_EXACT,
    CASE_BALANCED_SHIFT_EXACT,
    CENTERS,
    EXACT_FAMILY_IDS,
    EXPECTED_EXACT_FOLD_COUNT,
    EXPECTED_EXACT_PREDICTION_COUNT,
    GLOBAL_SOURCE_EXACT_CONTROL,
    NULL_TIED_EXACT_CONTROL,
    POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL,
    PRIMARY_R_FAMILY_ID,
    SMOOTH_SHIFT_DESCRIPTIVE,
    candidate_sources,
    expected_training_row_count,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.crossfit import (
    crossfit_exact_families,
    crossfit_smooth_descriptive,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.decision import (
    summarize_abstention_diagnostic,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.features import (
    blocked_permutation_donor_keys,
    build_exact_family_designs,
    build_fixed_bank_dataset,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.metric_contracts import (
    FixedBankDecisionAuditResult,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.metrics import (
    summarize_exact_crossfit,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.row_contracts import (
    FixedBankFeatureRow,
    FixedBankResponseRow,
    feature_row_from_payload,
    response_row_from_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dataset(*, smooth_poison: bool = False, reverse: bool = False):
    positions = {center: index for index, center in enumerate(CENTERS)}
    features: list[FixedBankFeatureRow] = []
    responses: list[FixedBankResponseRow] = []
    for outer in CENTERS:
        for query in (value for value in CENTERS if value != outer):
            outer_index = positions[outer]
            query_index = positions[query]
            for source in candidate_sources(outer, query):
                source_index = positions[source]
                distance = abs(source_index - query_index) / 8.0
                signed = (source_index - query_index) / 8.0
                feature = FixedBankFeatureRow(
                    outer_target_id=outer,
                    query_id=query,
                    candidate_source=source,
                    source_feature_row_hash=_hash(
                        f"source-feature::{outer}::{query}::{source}"
                    ),
                    metadata_similarity=1.0 - distance,
                    pooled_row_weighted_abs_shift=0.01 + 0.01 * distance,
                    equal_case_abs_shift=0.02 + 0.02 * distance,
                    case_abs_shift_sd=0.01 + 0.005 * ((source_index + query_index) % 4),
                    equal_case_signed_margin=0.08 * signed,
                    case_balanced_flip_rate=0.03 + 0.02 * distance,
                    case_balanced_entropy_change=-0.04 * signed,
                    case_balanced_reconstruction=0.5 + distance,
                    case_balanced_kl=0.3 + 0.5 * distance,
                    case_balanced_log_mmd=0.2 + 0.75 * distance,
                )
                # A source prior plus target-local compatibility creates a
                # unique oracle in every seven-candidate list.
                exact = float(
                    0.0004 * source_index
                    - 0.0030 * distance
                    + 0.00005 * outer_index
                )
                smooth = float(0.4 * exact + 0.002 * signed)
                if smooth_poison:
                    smooth = float(np.clip(-smooth + 0.05, -1.0, 1.0))
                response = FixedBankResponseRow(
                    outer_target_id=outer,
                    query_id=query,
                    candidate_source=source,
                    feature_row_hash=feature.feature_row_hash,
                    source_response_row_hash=_hash(
                        f"source-response::{outer}::{query}::{source}"
                    ),
                    exact_bacc_delta=exact,
                    smooth_bacc_delta=smooth,
                )
                features.append(feature)
                responses.append(response)
    if reverse:
        features.reverse()
        responses.reverse()
    return build_fixed_bank_dataset(features, responses)


@pytest.fixture(scope="module")
def dataset():
    return _dataset()


@pytest.fixture(scope="module")
def exact_crossfit(dataset):
    return crossfit_exact_families(dataset)


@pytest.fixture(scope="module")
def exact_summaries(exact_crossfit):
    return summarize_exact_crossfit(exact_crossfit)


def test_owned_rows_are_hash_bound_and_dataset_order_is_canonical(dataset) -> None:
    feature = dataset.feature_rows[0]
    response = dataset.response_rows[0]
    assert feature_row_from_payload(feature.to_payload()) == feature
    assert response_row_from_payload(response.to_payload()) == response
    assert feature.to_payload()["known_fixed_bank_reuse"] is True
    assert feature.to_payload()["unseen_expert_transfer"] is False
    assert feature.to_payload()["metadata_similarity"] == feature.metadata_similarity
    assert (
        feature.to_payload()["pooled_row_weighted_abs_shift"]
        == feature.pooled_row_weighted_abs_shift
    )
    with pytest.raises(ProtocolError, match="hash drifted"):
        feature_row_from_payload(
            {**feature.to_payload(), "metadata_similarity": 0.0}
        )
    with pytest.raises(ProtocolError, match="hash drifted"):
        response_row_from_payload(
            {**response.to_payload(), "exact_bacc_delta": 0.0}
        )
    reversed_dataset = _dataset(reverse=True)
    assert reversed_dataset.to_payload() == dataset.to_payload()


def test_family_designs_have_faithful_g_pooled_and_blocked_controls(dataset) -> None:
    designs = build_exact_family_designs(dataset)
    assert tuple(designs) == EXACT_FAMILY_IDS
    assert len(EXACT_FAMILY_IDS) == 9
    assert designs[POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL].spec.predictor_names == (
        "pooled_row_weighted_abs_shift",
    )
    assert designs[CASE_BALANCED_SHIFT_EXACT].spec.predictor_names != (
        "pooled_row_weighted_abs_shift",
    )
    first = dataset.feature_rows[0]
    assert designs[POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL].values[0, 0] == pytest.approx(
        first.pooled_row_weighted_abs_shift
    )
    donor_keys = blocked_permutation_donor_keys(dataset)
    donor_index = dataset.row_keys.index(donor_keys[0])
    assert designs[
        CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL
    ].values[0].tolist() == pytest.approx(
        designs[CASE_BALANCED_RICH_EXACT].values[donor_index].tolist()
    )
    assert (
        designs[CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL].design_hash
        == build_exact_family_designs(dataset)[
            CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL
        ].design_hash
    )


def test_crossfit_is_one_shared_model_per_hq_and_retains_legal_e_history(
    exact_crossfit,
) -> None:
    assert expected_training_row_count() == 210
    assert len(exact_crossfit.predictions) == EXPECTED_EXACT_PREDICTION_COUNT == 4_536
    assert len(exact_crossfit.fold_audits) == EXPECTED_EXACT_FOLD_COUNT == 648
    fold = next(
        row
        for row in exact_crossfit.fold_audits
        if row.family_id == CASE_BALANCED_RICH_EXACT
        and row.outer_target_id == "0"
        and row.query_id == "1"
    )
    assert len(fold.training_row_keys) == 210
    assert all({"0", "1"}.isdisjoint(key) for key in fold.training_row_keys)
    assert dict(fold.legal_candidate_source_history_counts)["2"] == 30
    assert sum(key[2] == "2" for key in fold.training_row_keys) == 30
    query_predictions = tuple(
        row
        for row in exact_crossfit.predictions
        if row.family_id == CASE_BALANCED_RICH_EXACT
        and row.outer_target_id == "0"
        and row.query_id == "1"
    )
    assert len(query_predictions) == 7
    assert {row.fold_hash for row in query_predictions} == {fold.fold_hash}
    payload = fold.to_payload()
    assert payload["strict_H_q_all_role_exclusion"] is True
    assert payload["candidate_e_history_retained_for_known_bank"] is True
    assert payload["same_model_scores_all_legal_e"] is True
    assert payload["unseen_expert_transfer"] is False


def test_tied_null_is_really_tied_and_tie_aware_hit_is_one_seventh(
    exact_crossfit, exact_summaries
) -> None:
    query_rows, _outer_rows, _family_rows = exact_summaries
    null_predictions = tuple(
        row
        for row in exact_crossfit.predictions
        if row.family_id == NULL_TIED_EXACT_CONTROL
        and row.outer_target_id == "0"
        and row.query_id == "1"
    )
    assert len({row.predicted_delta for row in null_predictions}) == 1
    metric = next(
        row
        for row in query_rows
        if row.family_id == NULL_TIED_EXACT_CONTROL
        and row.outer_target_id == "0"
        and row.query_id == "1"
    )
    assert metric.exact_top1 == 0.0
    assert metric.tie_aware_top1 == pytest.approx(1.0 / 7.0)


def test_metrics_predeclare_only_rich_as_primary_and_report_paired_r_minus_g(
    exact_summaries,
) -> None:
    query_rows, _outer_rows, family_rows = exact_summaries
    eligible = tuple(row for row in family_rows if row.publication_gate_eligible)
    assert tuple(row.family_id for row in eligible) == (PRIMARY_R_FAMILY_ID,)
    assert next(
        row for row in family_rows if row.family_id == CASE_BALANCED_SHIFT_EXACT
    ).scientific_role == "secondary_challenger_descriptive"
    assert next(
        row for row in family_rows if row.family_id == CASE_AWARE_BOUNDARY_EXACT
    ).exact_gate_passed is False
    assert next(
        row
        for row in family_rows
        if row.family_id == POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL
    ).scientific_role == "control"
    query = next(
        row
        for row in query_rows
        if row.family_id == CASE_BALANCED_RICH_EXACT
    )
    assert query.r_minus_g_exact_gain == pytest.approx(
        query.selected_exact_gain - query.global_selected_exact_gain
    )


def test_failed_primary_gate_forces_every_abstention_row_to_exact_b(
    exact_crossfit, exact_summaries
) -> None:
    _queries, _outers, summaries = exact_summaries
    forced_fail = tuple(
        replace(row, exact_gate_passed=False)
        if row.family_id == PRIMARY_R_FAMILY_ID
        else row
        for row in summaries
    )
    decisions, abstention = summarize_abstention_diagnostic(
        exact_crossfit, family_summaries=forced_fail
    )
    assert not any(row.routed for row in decisions)
    assert all(row.deployed_exact_gain == 0.0 for row in decisions)
    assert all(row.route_coverage == 0.0 for row in abstention)


def test_smooth_poison_cannot_change_exact_models_decisions_or_hashes(
    dataset, exact_crossfit, exact_summaries
) -> None:
    poisoned = _dataset(smooth_poison=True)
    assert poisoned.feature_surface_hash == dataset.feature_surface_hash
    assert poisoned.exact_response_surface_hash == dataset.exact_response_surface_hash
    assert poisoned.smooth_response_surface_hash != dataset.smooth_response_surface_hash
    repeated_exact = crossfit_exact_families(poisoned)
    assert repeated_exact.to_payload() == exact_crossfit.to_payload()

    query_rows, outer_rows, family_rows = exact_summaries
    abstention_rows, abstention_summaries = summarize_abstention_diagnostic(
        exact_crossfit, family_summaries=family_rows
    )
    smooth = crossfit_smooth_descriptive(dataset)
    poisoned_smooth = crossfit_smooth_descriptive(poisoned)
    assert smooth.result_hash != poisoned_smooth.result_hash
    gate = next(
        row.exact_gate_passed
        for row in family_rows
        if row.family_id == PRIMARY_R_FAMILY_ID
    )
    first = FixedBankDecisionAuditResult(
        exact_crossfit=exact_crossfit,
        smooth_descriptive_crossfit=smooth,
        query_metrics=query_rows,
        outer_metrics=outer_rows,
        family_summaries=family_rows,
        abstention_decisions=abstention_rows,
        abstention_summaries=abstention_summaries,
        primary_exact_gate_passed=gate,
    )
    second = replace(first, smooth_descriptive_crossfit=poisoned_smooth)
    assert first.exact_decision_hash == second.exact_decision_hash
    assert first.result_hash != second.result_hash
    assert first.to_payload()["smooth_influences_exact_model_or_decision"] is False


def test_smooth_surface_is_separate_descriptive_only(dataset) -> None:
    smooth = crossfit_smooth_descriptive(
        dataset, family_ids=(SMOOTH_SHIFT_DESCRIPTIVE,)
    )
    payload = smooth.to_payload()
    assert payload["smooth_response_is_wholly_separate_descriptive_result"] is True
    assert payload["exact_model_or_decision_influence"] is False
    assert payload["terminal_decision_authorized"] is False
    assert all(row.response_name == "smooth_bacc_delta" for row in smooth.predictions)


def test_global_source_control_ignores_local_feature_values(dataset) -> None:
    changed_features = [
        replace(
            row,
            equal_case_abs_shift=min(1.0, row.equal_case_abs_shift + 0.5),
            case_balanced_reconstruction=row.case_balanced_reconstruction + 10.0,
        )
        for row in dataset.feature_rows
    ]
    changed_responses = [
        FixedBankResponseRow(
            outer_target_id=response.outer_target_id,
            query_id=response.query_id,
            candidate_source=response.candidate_source,
            feature_row_hash=feature.feature_row_hash,
            source_response_row_hash=response.source_response_row_hash,
            exact_bacc_delta=response.exact_bacc_delta,
            smooth_bacc_delta=response.smooth_bacc_delta,
        )
        for feature, response in zip(
            changed_features, dataset.response_rows, strict=True
        )
    ]
    changed = build_fixed_bank_dataset(changed_features, changed_responses)
    original_g = crossfit_exact_families(
        dataset, family_ids=(GLOBAL_SOURCE_EXACT_CONTROL,)
    )
    changed_g = crossfit_exact_families(
        changed, family_ids=(GLOBAL_SOURCE_EXACT_CONTROL,)
    )
    assert [row.predicted_delta for row in original_g.predictions] == pytest.approx(
        [row.predicted_delta for row in changed_g.predictions], abs=0.0, rel=0.0
    )
