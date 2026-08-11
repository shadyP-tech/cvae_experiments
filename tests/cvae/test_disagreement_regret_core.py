from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import replace
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing import disagreement_regret_core as core
from midogpp_thesis.cvae.routing.disagreement_regret_core import (
    CandidateContrastRow,
    DevelopmentContext,
    DevelopmentScope,
    DisagreementFeatureSurface,
    ExactRegretSurface,
    InferenceActionSchema,
    LabelFreeInferenceContext,
    ProbabilityRow,
    SourceOOFLabelRow,
    build_disagreement_feature_surface,
    build_label_free_inference_feature_surface,
    build_source_oof_training_feature_surface,
    build_exact_regret_surface,
    build_safe_selection_diagnostics,
    feature_surface_for_family,
    fit_known_bank_pairwise_models,
    freeze_pairwise_model_bank,
    deserialize_pairwise_model_bank,
    serialize_pairwise_model_bank,
    score_label_free_inference_candidate_contrasts,
    score_target_candidate_contrasts,
)
from midogpp_thesis.cvae.routing.disagreement_regret_core.hashing import canonical_sha256


DONOR_QUERY_IDS = ("0", "1", "2", "3", "4")
OUTER_TARGET_ID = "H"
ALL_QUERY_IDS = (*DONOR_QUERY_IDS, OUTER_TARGET_ID)
CASE_IDS = ("case-0", "case-1")
SAMPLE_IDS = ("sample-0", "sample-1", "sample-2", "sample-3")
BASELINE_ACTION_ID = "B"
CONTROL_ACTION_ID = "U"
CANDIDATE_ACTIONS = tuple(f"A::{source}" for source in DONOR_QUERY_IDS)
PREDICTION_SEAL_HASH = "a" * 64
MODEL_HASH = "b" * 64


def _context() -> DevelopmentContext:
    return DevelopmentContext(
        scope=DevelopmentScope.SYNTHETIC_TEST,
        dataset_family="SYNTHETIC",
        outer_target_id=OUTER_TARGET_ID,
    )


def _synthetic_probability_rows() -> tuple[ProbabilityRow, ...]:
    rows: list[ProbabilityRow] = []
    for query_id in ALL_QUERY_IDS:
        for case_id in CASE_IDS:
            for sample_index, sample_id in enumerate(SAMPLE_IDS):
                latent_truth = sample_index % 2
                wrong_probability = 0.49 if latent_truth else 0.51
                correct_probability = 0.51 if latent_truth else 0.49
                for action_id in (BASELINE_ACTION_ID, CONTROL_ACTION_ID):
                    rows.append(
                        ProbabilityRow(
                            query_id=query_id,
                            case_id=case_id,
                            sample_id=sample_id,
                            action_id=action_id,
                            source_id=None,
                            probability=wrong_probability,
                            probability_sd=0.01,
                            hard_vote_fraction=0.8,
                            prediction_seal_hash=PREDICTION_SEAL_HASH,
                        )
                    )
                for source_id, action_id in zip(
                    DONOR_QUERY_IDS, CANDIDATE_ACTIONS, strict=True
                ):
                    # A source expert is never an action for its own pseudoquery.
                    if source_id == query_id:
                        continue
                    if source_id == "0":
                        probability = correct_probability
                    elif source_id == "1" and sample_index < 2:
                        probability = correct_probability
                    else:
                        probability = wrong_probability
                    rows.append(
                        ProbabilityRow(
                            query_id=query_id,
                            case_id=case_id,
                            sample_id=sample_id,
                            action_id=action_id,
                            source_id=source_id,
                            probability=probability,
                            probability_sd=0.01,
                            hard_vote_fraction=0.9,
                            prediction_seal_hash=PREDICTION_SEAL_HASH,
                        )
                    )
    return tuple(rows)


def _synthetic_source_oof_labels() -> tuple[SourceOOFLabelRow, ...]:
    return tuple(
        SourceOOFLabelRow(
            query_id=query_id,
            case_id=case_id,
            sample_id=sample_id,
            label=sample_index % 2,
        )
        for query_id in DONOR_QUERY_IDS
        for case_id in CASE_IDS
        for sample_index, sample_id in enumerate(SAMPLE_IDS)
    )


@pytest.fixture(scope="module")
def probability_rows() -> tuple[ProbabilityRow, ...]:
    return _synthetic_probability_rows()


@pytest.fixture(scope="module")
def source_oof_labels() -> tuple[SourceOOFLabelRow, ...]:
    return _synthetic_source_oof_labels()


@pytest.fixture(scope="module")
def feature_surface(
    probability_rows: tuple[ProbabilityRow, ...],
) -> DisagreementFeatureSurface:
    return build_disagreement_feature_surface(
        probability_rows,
        baseline_action_id=BASELINE_ACTION_ID,
        control_action_id=CONTROL_ACTION_ID,
        context=_context(),
    )


@pytest.fixture(scope="module")
def response_surface(
    feature_surface: DisagreementFeatureSurface,
    source_oof_labels: tuple[SourceOOFLabelRow, ...],
) -> ExactRegretSurface:
    return build_exact_regret_surface(
        feature_surface,
        source_oof_labels,
        context=_context(),
    )


@pytest.fixture(scope="module")
def family_surfaces(
    feature_surface: DisagreementFeatureSurface,
) -> dict[str, DisagreementFeatureSurface]:
    return {
        family: feature_surface_for_family(feature_surface, family=family)
        for family in ("G", "R", "P")
    }


@pytest.fixture(scope="module")
def family_models(
    family_surfaces: dict[str, DisagreementFeatureSurface],
    response_surface: ExactRegretSurface,
) -> dict[str, tuple[core.PairwiseRegretModel, ...]]:
    return {
        family: fit_known_bank_pairwise_models(
            surface,
            response_surface,
            context=_context(),
            family=family,
            aligned_parent_features=(
                family_surfaces["R"] if family in ("G", "P") else None
            ),
        )
        for family, surface in family_surfaces.items()
    }


def _row_by_key(
    rows: tuple[ProbabilityRow, ...],
) -> dict[tuple[str, str, str, str], ProbabilityRow]:
    return {row.row_key: row for row in rows}


def _balanced_accuracy(
    *, labels: list[int], predictions: list[int]
) -> float:
    positive = [prediction for label, prediction in zip(labels, predictions) if label]
    negative = [prediction for label, prediction in zip(labels, predictions) if not label]
    return 0.5 * (
        sum(positive) / len(positive)
        + sum(1 - prediction for prediction in negative) / len(negative)
    )


def test_probability_and_label_order_do_not_change_canonical_surfaces(
    probability_rows: tuple[ProbabilityRow, ...],
    source_oof_labels: tuple[SourceOOFLabelRow, ...],
    feature_surface: DisagreementFeatureSurface,
    response_surface: ExactRegretSurface,
) -> None:
    assert set(feature_surface.query_ids) == set(ALL_QUERY_IDS)
    assert OUTER_TARGET_ID not in {row.query_id for row in source_oof_labels}
    assert all(
        row.source_id != row.query_id
        for row in probability_rows
        if row.source_id is not None
    )

    reversed_features = build_disagreement_feature_surface(
        tuple(reversed(probability_rows)),
        baseline_action_id=BASELINE_ACTION_ID,
        control_action_id=CONTROL_ACTION_ID,
        context=_context(),
    )
    reversed_responses = build_exact_regret_surface(
        reversed_features,
        tuple(reversed(source_oof_labels)),
        context=_context(),
    )

    assert reversed_features.rows == feature_surface.rows
    assert reversed_features.disagreements == feature_surface.disagreements
    assert reversed_features.surface_hash == feature_surface.surface_hash
    assert reversed_responses.rows == response_surface.rows
    assert reversed_responses.surface_hash == response_surface.surface_hash


def test_probability_surface_rejects_duplicates_missing_controls_and_target_expert(
    probability_rows: tuple[ProbabilityRow, ...],
) -> None:
    kwargs = {
        "baseline_action_id": BASELINE_ACTION_ID,
        "control_action_id": CONTROL_ACTION_ID,
        "context": _context(),
    }
    with pytest.raises(ProtocolError, match="duplicate"):
        build_disagreement_feature_surface(
            (*probability_rows, probability_rows[0]), **kwargs
        )

    first_sample = ("0", "case-0", "sample-0")
    for missing_action in (BASELINE_ACTION_ID, CONTROL_ACTION_ID):
        incomplete = tuple(
            row
            for row in probability_rows
            if not (row.sample_key == first_sample and row.action_id == missing_action)
        )
        with pytest.raises(ProtocolError, match="both B and U"):
            build_disagreement_feature_surface(incomplete, **kwargs)

    candidate_index = next(
        index
        for index, row in enumerate(probability_rows)
        if row.query_id == "0" and row.action_id == "A::1"
    )
    target_expert_rows = list(probability_rows)
    target_expert_rows[candidate_index] = replace(
        target_expert_rows[candidate_index], source_id="0"
    )
    with pytest.raises(ProtocolError, match="query/target expert"):
        build_disagreement_feature_surface(tuple(target_expert_rows), **kwargs)


def test_sparse_flip_rows_preserve_direction_and_structural_zero(
    feature_surface: DisagreementFeatureSurface,
    response_surface: ExactRegretSurface,
) -> None:
    feature_by_key = {row.row_key: row for row in feature_surface.rows}
    strong = feature_by_key[("3", "case-0", "A::0")]
    partial = feature_by_key[("3", "case-0", "A::1")]
    baseline = feature_by_key[("3", "case-0", BASELINE_ACTION_ID)]

    assert strong.sample_count == 4
    assert strong.disagreement_count == 4
    assert strong.values[:3] == pytest.approx((1.0, 0.5, 0.5))
    assert partial.disagreement_count == 2
    assert partial.values[:3] == pytest.approx((0.5, 0.25, 0.25))
    assert baseline.disagreement_count == 0
    assert baseline.values[:3] == pytest.approx((0.0, 0.0, 0.0))

    sparse = tuple(
        row
        for row in feature_surface.disagreements
        if (row.query_id, row.case_id, row.action_id) == ("3", "case-0", "A::0")
    )
    assert tuple(row.sample_id for row in sparse) == SAMPLE_IDS
    assert tuple(row.flip_direction for row in sparse) == (-1, 1, -1, 1)
    assert not any(
        row.action_id == BASELINE_ACTION_ID for row in feature_surface.disagreements
    )

    baseline_responses = tuple(
        row for row in response_surface.rows if row.action_id == BASELINE_ACTION_ID
    )
    assert baseline_responses
    assert all(row.disagreement_count == 0 for row in baseline_responses)
    assert all(row.exact_bacc_gain_vs_control == 0.0 for row in baseline_responses)


def test_feature_hashes_are_label_free_but_response_hashes_bind_labels(
    feature_surface: DisagreementFeatureSurface,
    source_oof_labels: tuple[SourceOOFLabelRow, ...],
    response_surface: ExactRegretSurface,
) -> None:
    feature_hashes_before = tuple(row.feature_hash for row in feature_surface.rows)
    inverted_labels = tuple(
        replace(row, label=1 - row.label) for row in source_oof_labels
    )
    inverted = build_exact_regret_surface(
        feature_surface,
        inverted_labels,
        context=_context(),
    )

    assert inverted.feature_surface_hash == feature_surface.surface_hash
    assert tuple(row.feature_hash for row in feature_surface.rows) == feature_hashes_before
    assert inverted.label_surface_hash != response_surface.label_surface_hash
    assert inverted.surface_hash != response_surface.surface_hash


def test_sparse_gain_sum_is_exactly_the_direct_balanced_accuracy_delta(
    probability_rows: tuple[ProbabilityRow, ...],
    source_oof_labels: tuple[SourceOOFLabelRow, ...],
    response_surface: ExactRegretSurface,
) -> None:
    predictions = _row_by_key(probability_rows)
    labels_by_query: dict[str, list[SourceOOFLabelRow]] = defaultdict(list)
    for label in source_oof_labels:
        labels_by_query[label.query_id].append(label)

    gains_by_query_action: dict[tuple[str, str], float] = defaultdict(float)
    for row in response_surface.rows:
        gains_by_query_action[(row.query_id, row.action_id)] += (
            row.exact_bacc_gain_vs_control
        )

    for (query_id, action_id), sparse_gain in gains_by_query_action.items():
        ordered_labels = sorted(labels_by_query[query_id], key=lambda row: row.row_key)
        truth = [row.label for row in ordered_labels]
        action_predictions = [
            int(
                predictions[
                    (query_id, row.case_id, row.sample_id, action_id)
                ].probability
                >= 0.5
            )
            for row in ordered_labels
        ]
        control_predictions = [
            int(
                predictions[
                    (query_id, row.case_id, row.sample_id, CONTROL_ACTION_ID)
                ].probability
                >= 0.5
            )
            for row in ordered_labels
        ]
        direct_gain = _balanced_accuracy(
            labels=truth, predictions=action_predictions
        ) - _balanced_accuracy(labels=truth, predictions=control_predictions)
        assert sparse_gain == pytest.approx(direct_gain, abs=1.0e-15)


@pytest.mark.parametrize("role", ("target_support", "target_evaluation", "eval"))
def test_response_surface_rejects_forbidden_label_roles(role: str) -> None:
    with pytest.raises(ProtocolError, match="forbidden"):
        SourceOOFLabelRow(
            query_id="0",
            case_id="case-0",
            sample_id="sample-0",
            label=0,
            role=role,
        )


def test_response_surface_rejects_outer_target_and_incomplete_labels(
    feature_surface: DisagreementFeatureSurface,
    source_oof_labels: tuple[SourceOOFLabelRow, ...],
) -> None:
    outer_label = SourceOOFLabelRow(
        query_id=OUTER_TARGET_ID,
        case_id="case-0",
        sample_id="sample-0",
        label=0,
    )
    with pytest.raises(ProtocolError, match="Outer-target labels"):
        build_exact_regret_surface(
            feature_surface,
            (*source_oof_labels, outer_label),
            context=_context(),
        )

    wrong_identity = (
        replace(source_oof_labels[0], sample_id="same-count-wrong-sample"),
        *source_oof_labels[1:],
    )
    with pytest.raises(ProtocolError, match="sample identities"):
        build_exact_regret_surface(
            feature_surface,
            wrong_identity,
            context=_context(),
        )

    with pytest.raises(ProtocolError, match="complete case"):
        build_exact_regret_surface(
            feature_surface,
            source_oof_labels[1:],
            context=_context(),
        )


def test_sparse_surface_rejects_forged_sample_and_source_lineage(
    feature_surface: DisagreementFeatureSurface,
) -> None:
    first = feature_surface.disagreements[0]
    for forged in (
        replace(first, sample_id="unknown-sample"),
        replace(first, source_id="wrong-source"),
    ):
        with pytest.raises(ProtocolError, match="ordering|lineage"):
            DisagreementFeatureSurface(
                rows=feature_surface.rows,
                disagreements=(forged, *feature_surface.disagreements[1:]),
                baseline_action_id=feature_surface.baseline_action_id,
                control_action_id=feature_surface.control_action_id,
                candidate_source_by_action=feature_surface.candidate_source_by_action,
                prediction_seal_hash=feature_surface.prediction_seal_hash,
                sample_keys=feature_surface.sample_keys,
                development_context_hash=feature_surface.development_context_hash,
                dataset_family=feature_surface.dataset_family,
                outer_target_id=feature_surface.outer_target_id,
            )


def test_feature_surface_rejects_incomplete_per_case_candidate_blocks(
    feature_surface: DisagreementFeatureSurface,
) -> None:
    removed = next(
        row
        for row in feature_surface.rows
        if row.query_id == "1"
        and row.case_id == CASE_IDS[0]
        and row.action_id != BASELINE_ACTION_ID
    )
    rows = tuple(row for row in feature_surface.rows if row.row_key != removed.row_key)
    disagreements = tuple(
        row
        for row in feature_surface.disagreements
        if (row.query_id, row.case_id, row.action_id) != removed.row_key
    )
    with pytest.raises(ProtocolError, match="every legal non-query candidate"):
        DisagreementFeatureSurface(
            rows=rows,
            disagreements=disagreements,
            baseline_action_id=feature_surface.baseline_action_id,
            control_action_id=feature_surface.control_action_id,
            candidate_source_by_action=feature_surface.candidate_source_by_action,
            prediction_seal_hash=feature_surface.prediction_seal_hash,
            sample_keys=feature_surface.sample_keys,
            development_context_hash=feature_surface.development_context_hash,
            dataset_family=feature_surface.dataset_family,
            outer_target_id=feature_surface.outer_target_id,
        )

def test_g_r_p_controls_are_key_matched_and_p_is_a_block_derangement(
    feature_surface: DisagreementFeatureSurface,
    response_surface: ExactRegretSurface,
    family_surfaces: dict[str, DisagreementFeatureSurface],
) -> None:
    assert family_surfaces["R"] is feature_surface
    reference_keys = tuple(row.row_key for row in feature_surface.rows)
    response_keys = tuple(row.row_key for row in response_surface.rows)
    labeled_queries = {row.query_id for row in response_surface.rows}
    reference_by_key = {row.row_key: row for row in feature_surface.rows}

    for surface in family_surfaces.values():
        assert tuple(row.row_key for row in surface.rows) == reference_keys
        assert tuple(
            row.row_key for row in surface.rows if row.query_id in labeled_queries
        ) == response_keys
        assert surface.prediction_seal_hash == feature_surface.prediction_seal_hash

    assert all(
        row.values == (0.0,) * len(row.values)
        and row.disagreement_count == 0
        for row in family_surfaces["G"].rows
    )

    permuted = family_surfaces["P"]
    for row in permuted.rows:
        if row.action_id == BASELINE_ACTION_ID:
            assert row == reference_by_key[row.row_key]
            continue
        assert row.feature_origin_action_id != row.action_id
        donor = reference_by_key[
            (row.query_id, row.case_id, row.feature_origin_action_id)
        ]
        assert row.values == donor.values
        assert row.disagreement_count == donor.disagreement_count

    for query_id in ALL_QUERY_IDS:
        for case_id in CASE_IDS:
            original_multiset = sorted(
                row.values
                for row in feature_surface.rows
                if row.query_id == query_id
                and row.case_id == case_id
                and row.action_id != BASELINE_ACTION_ID
            )
            permuted_multiset = sorted(
                row.values
                for row in permuted.rows
                if row.query_id == query_id
                and row.case_id == case_id
                and row.action_id != BASELINE_ACTION_ID
            )
            assert permuted_multiset == original_multiset


def test_control_contracts_reject_nonzero_g_and_non_deranged_p(
    feature_surface: DisagreementFeatureSurface,
    family_surfaces: dict[str, DisagreementFeatureSurface],
) -> None:
    g_surface = family_surfaces["G"]
    bad_g_rows = (
        replace(g_surface.rows[0], values=(1.0,) * len(g_surface.rows[0].values)),
        *g_surface.rows[1:],
    )
    with pytest.raises(ProtocolError, match="exact zero"):
        DisagreementFeatureSurface(
            rows=bad_g_rows,
            disagreements=(),
            baseline_action_id=g_surface.baseline_action_id,
            control_action_id=g_surface.control_action_id,
            candidate_source_by_action=g_surface.candidate_source_by_action,
            prediction_seal_hash=g_surface.prediction_seal_hash,
            sample_keys=g_surface.sample_keys,
            development_context_hash=g_surface.development_context_hash,
            dataset_family=g_surface.dataset_family,
            outer_target_id=g_surface.outer_target_id,
            family="G",
            parent_surface_hash=feature_surface.surface_hash,
        )

    p_surface = family_surfaces["P"]
    candidate_index = next(
        index
        for index, row in enumerate(p_surface.rows)
        if row.action_id != BASELINE_ACTION_ID
    )
    bad_p_rows = list(p_surface.rows)
    bad_p_rows[candidate_index] = replace(
        bad_p_rows[candidate_index],
        feature_origin_action_id=bad_p_rows[candidate_index].action_id,
    )
    with pytest.raises(ProtocolError, match="candidate derangement"):
        DisagreementFeatureSurface(
            rows=tuple(bad_p_rows),
            disagreements=(),
            baseline_action_id=p_surface.baseline_action_id,
            control_action_id=p_surface.control_action_id,
            candidate_source_by_action=p_surface.candidate_source_by_action,
            prediction_seal_hash=p_surface.prediction_seal_hash,
            sample_keys=p_surface.sample_keys,
            development_context_hash=p_surface.development_context_hash,
            dataset_family=p_surface.dataset_family,
            outer_target_id=p_surface.outer_target_id,
            family="P",
            parent_surface_hash=feature_surface.surface_hash,
        )


def test_all_control_models_converge_score_canonically_and_select_safely(
    family_surfaces: dict[str, DisagreementFeatureSurface],
    family_models: dict[str, tuple[core.PairwiseRegretModel, ...]],
) -> None:
    diagnostics_by_family = {}
    for family in ("G", "R", "P"):
        models = family_models[family]
        assert tuple(model.candidate_action_id for model in models) == CANDIDATE_ACTIONS
        assert all(model.family == family and model.converged for model in models)
        assert all(model.observation_count > 0 for model in models)
        assert all(OUTER_TARGET_ID in model.excluded_query_ids for model in models)
        assert all(
            model.candidate_source_id in model.excluded_query_ids for model in models
        )

        contrasts = score_target_candidate_contrasts(
            models,
            family_surfaces[family],
            context=_context(),
        )
        assert tuple(sorted(contrasts, key=lambda row: row.row_key)) == contrasts
        assert len(contrasts) == len(CASE_IDS) * len(CANDIDATE_ACTIONS)
        assert all(
            row.target_query_id == OUTER_TARGET_ID
            and math.isfinite(row.predicted_preference_margin_vs_control)
            and math.isfinite(row.predicted_preference_margin_vs_baseline)
            and row.standard_error_vs_control >= 0.0
            and row.standard_error_vs_baseline >= 0.0
            for row in contrasts
        )
        diagnostics_by_family[family] = build_safe_selection_diagnostics(
            contrasts,
            baseline_action_id=BASELINE_ACTION_ID,
            control_action_id=CONTROL_ACTION_ID,
            context=_context(),
        )

    assert {
        diagnostic.safe_action_id for diagnostic in diagnostics_by_family["R"]
    } == {"A::0"}
    for family in ("G", "P"):
        assert {
            diagnostic.safe_action_id
            for diagnostic in diagnostics_by_family[family]
        } == {BASELINE_ACTION_ID}
        assert all(
            diagnostic.fallback_reason
            == "simultaneous_lcb_nonpositive_vs_b_or_u"
            for diagnostic in diagnostics_by_family[family]
        )

    for diagnostics in diagnostics_by_family.values():
        assert all(
            diagnostic.may_authorize_routing is False
            and diagnostic.may_authorize_promotion is False
            and diagnostic.claim_role == core.DEVELOPMENT_CLAIM_ROLE
            for diagnostic in diagnostics
        )


def test_control_parent_and_scoring_surface_lineage_fail_closed(
    feature_surface: DisagreementFeatureSurface,
    response_surface: ExactRegretSurface,
    family_surfaces: dict[str, DisagreementFeatureSurface],
    family_models: dict[str, tuple[core.PairwiseRegretModel, ...]],
) -> None:
    forged_control = replace(
        family_surfaces["P"],
        parent_surface_hash="f" * 64,
    )
    with pytest.raises(ProtocolError, match="aligned R parent"):
        fit_known_bank_pairwise_models(
            forged_control,
            response_surface,
            context=_context(),
            family="P",
            aligned_parent_features=feature_surface,
        )

    forged_row = next(
        row
        for row in family_surfaces["P"].rows
        if row.action_id != BASELINE_ACTION_ID
    )
    forged_rows = tuple(
        replace(row, values=tuple(value + 1.0 for value in row.values))
        if row.row_key == forged_row.row_key
        else row
        for row in family_surfaces["P"].rows
    )
    forged_values = DisagreementFeatureSurface(
        rows=forged_rows,
        disagreements=(),
        baseline_action_id=feature_surface.baseline_action_id,
        control_action_id=feature_surface.control_action_id,
        candidate_source_by_action=feature_surface.candidate_source_by_action,
        prediction_seal_hash=feature_surface.prediction_seal_hash,
        sample_keys=feature_surface.sample_keys,
        development_context_hash=feature_surface.development_context_hash,
        dataset_family=feature_surface.dataset_family,
        outer_target_id=feature_surface.outer_target_id,
        family="P",
        parent_surface_hash=feature_surface.surface_hash,
    )
    with pytest.raises(ProtocolError, match="parent-row attestation"):
        fit_known_bank_pairwise_models(
            forged_values,
            response_surface,
            context=_context(),
            family="P",
            aligned_parent_features=feature_surface,
        )

    poisoned_response = next(
        row for row in response_surface.rows if row.source_id is not None
    )
    wrong_source_rows = tuple(
        replace(row, source_id="wrong-source")
        if row.row_key == poisoned_response.row_key
        else row
        for row in response_surface.rows
    )
    wrong_source_surface = ExactRegretSurface(
        rows=wrong_source_rows,
        feature_surface_hash=response_surface.feature_surface_hash,
        label_surface_hash=response_surface.label_surface_hash,
        prediction_seal_hash=response_surface.prediction_seal_hash,
        development_context_hash=response_surface.development_context_hash,
    )
    with pytest.raises(ProtocolError, match="source identity"):
        fit_known_bank_pairwise_models(
            feature_surface,
            wrong_source_surface,
            context=_context(),
            family="R",
        )

    substituted_target = replace(feature_surface, dataset_family="OTHER")
    with pytest.raises(ProtocolError, match="scoring feature lineage"):
        score_target_candidate_contrasts(
            family_models["R"],
            substituted_target,
            context=_context(),
        )

    with pytest.raises(ProtocolError, match="complete candidate model bank"):
        score_target_candidate_contrasts(
            family_models["R"][:-1],
            feature_surface,
            context=_context(),
        )

    model = family_models["R"][0]
    bad_covariance = np.array(model.coefficient_covariance, copy=True)
    bad_covariance[0, 0] = -1.0
    with pytest.raises(ProtocolError, match="symmetric PSD"):
        replace(model, coefficient_covariance=bad_covariance)


def test_safe_lcb_is_strict_and_fallback_is_exactly_immutable_b() -> None:
    contrasts = tuple(
        CandidateContrastRow(
            family="R",
            target_query_id=OUTER_TARGET_ID,
            case_id=case_id,
            candidate_action_id=action_id,
            candidate_source_id=source_id,
            predicted_preference_margin_vs_control=margin,
            standard_error_vs_control=standard_error,
            predicted_preference_margin_vs_baseline=margin,
            standard_error_vs_baseline=standard_error,
            model_hash=MODEL_HASH,
        )
        for case_id, specifications in (
            ("safe", (("A::0", "0", 1.0, 0.01), ("A::1", "1", 0.2, 0.2))),
            ("fallback", (("A::0", "0", 0.0, 0.0), ("A::1", "1", 0.2, 0.2))),
        )
        for action_id, source_id, margin, standard_error in specifications
    )
    diagnostics = build_safe_selection_diagnostics(
        contrasts,
        baseline_action_id=BASELINE_ACTION_ID,
        control_action_id=CONTROL_ACTION_ID,
        context=_context(),
    )
    by_case = {row.case_id: row for row in diagnostics}

    assert by_case["safe"].safe_action_id == "A::0"
    assert by_case["safe"].safe_margin > 0.0
    assert by_case["fallback"].safe_action_id == BASELINE_ACTION_ID
    assert by_case["fallback"].safe_margin == 0.0
    assert by_case["fallback"].fallback_reason == (
        "simultaneous_lcb_nonpositive_vs_b_or_u"
    )
    assert all(
        row.may_authorize_routing is False and row.may_authorize_promotion is False
        for row in diagnostics
    )
    with pytest.raises(ProtocolError, match="cannot authorize"):
        replace(by_case["safe"], may_authorize_routing=True)


def test_nested_h_q_e_exclusion_is_invariant_to_excluded_row_poison(
    feature_surface: DisagreementFeatureSurface,
    response_surface: ExactRegretSurface,
) -> None:
    heldout_query_id = "4"
    clean_models = fit_known_bank_pairwise_models(
        feature_surface,
        response_surface,
        context=_context(),
        family="R",
        heldout_query_id=heldout_query_id,
    )
    clean = next(model for model in clean_models if model.candidate_source_id == "0")
    assert set(clean.excluded_query_ids) == {OUTER_TARGET_ID, "0", heldout_query_id}
    assert set(clean.training_query_ids) == {"1", "2", "3"}

    poisoned_queries = {OUTER_TARGET_ID, "0", heldout_query_id}
    poisoned_feature_rows = tuple(
        replace(row, values=tuple(value + 1000.0 for value in row.values))
        if row.query_id in poisoned_queries
        else row
        for row in feature_surface.rows
    )
    poisoned_features = DisagreementFeatureSurface(
        rows=poisoned_feature_rows,
        disagreements=feature_surface.disagreements,
        baseline_action_id=feature_surface.baseline_action_id,
        control_action_id=feature_surface.control_action_id,
        candidate_source_by_action=feature_surface.candidate_source_by_action,
        prediction_seal_hash=feature_surface.prediction_seal_hash,
        sample_keys=feature_surface.sample_keys,
        development_context_hash=feature_surface.development_context_hash,
        dataset_family=feature_surface.dataset_family,
        outer_target_id=feature_surface.outer_target_id,
    )
    poisoned_gain_by_key = {
        row.row_key: (
            -row.exact_bacc_gain_vs_control
            if row.query_id in poisoned_queries
            else row.exact_bacc_gain_vs_control
        )
        for row in response_surface.rows
    }
    best_by_case: dict[tuple[str, str], float] = {}
    for row in response_surface.rows:
        case_key = (row.query_id, row.case_id)
        best_by_case[case_key] = max(
            best_by_case.get(case_key, 0.0),
            poisoned_gain_by_key[row.row_key],
        )
    poisoned_response_rows = tuple(
        replace(
            row,
            exact_bacc_gain_vs_control=poisoned_gain_by_key[row.row_key],
            exact_regret_from_case_best=(
                best_by_case[(row.query_id, row.case_id)]
                - poisoned_gain_by_key[row.row_key]
            ),
        )
        for row in response_surface.rows
    )
    poisoned_responses = ExactRegretSurface(
        rows=poisoned_response_rows,
        feature_surface_hash=poisoned_features.surface_hash,
        label_surface_hash=response_surface.label_surface_hash,
        prediction_seal_hash=response_surface.prediction_seal_hash,
        development_context_hash=response_surface.development_context_hash,
    )
    poisoned_models = fit_known_bank_pairwise_models(
        poisoned_features,
        poisoned_responses,
        context=_context(),
        family="R",
        heldout_query_id=heldout_query_id,
    )
    poisoned = next(
        model for model in poisoned_models if model.candidate_source_id == "0"
    )

    assert poisoned.training_query_ids == clean.training_query_ids
    assert poisoned.excluded_query_ids == clean.excluded_query_ids
    assert poisoned.observation_count == clean.observation_count
    assert poisoned.iteration_count == clean.iteration_count
    np.testing.assert_array_equal(poisoned.feature_mean, clean.feature_mean)
    np.testing.assert_array_equal(poisoned.feature_scale, clean.feature_scale)
    np.testing.assert_array_equal(poisoned.coefficients, clean.coefficients)
    np.testing.assert_array_equal(
        poisoned.coefficient_covariance, clean.coefficient_covariance
    )


def _inference_context(
    bank: core.PairwiseRegretModelBank,
    *,
    prediction_seal_hash: str = "c" * 64,
    cache_content_hash: str = "d" * 64,
    cache_order_hash: str = "e" * 64,
) -> LabelFreeInferenceContext:
    return LabelFreeInferenceContext(
        dataset_family="MIDOG++_CONSUMED_TEST_LABEL_FREE",
        outer_target_id=OUTER_TARGET_ID,
        target_cache_content_hash=cache_content_hash,
        target_cache_order_hash=cache_order_hash,
        prediction_seal_hash=prediction_seal_hash,
        action_schema=bank.action_schema,
        model_bank_hash=bank.model_bank_hash,
    )


def _target_inference_rows(
    probability_rows: tuple[ProbabilityRow, ...],
    *,
    prediction_seal_hash: str = "c" * 64,
) -> tuple[ProbabilityRow, ...]:
    return tuple(
        replace(row, prediction_seal_hash=prediction_seal_hash)
        for row in probability_rows
        if row.query_id == OUTER_TARGET_ID
    )


def test_frozen_model_bank_has_byte_exact_canonical_roundtrip(
    family_models: dict[str, tuple[core.PairwiseRegretModel, ...]],
) -> None:
    bank = freeze_pairwise_model_bank(family_models["R"])
    serialized = serialize_pairwise_model_bank(bank)
    replayed = deserialize_pairwise_model_bank(serialized)

    assert replayed.model_bank_hash == bank.model_bank_hash
    assert replayed.action_schema == bank.action_schema
    assert tuple(model.model_hash for model in replayed.models) == tuple(
        model.model_hash for model in bank.models
    )
    assert serialize_pairwise_model_bank(replayed) == serialized

    with pytest.raises(ProtocolError, match="not canonical JSON"):
        deserialize_pairwise_model_bank(" " + serialized)
    poisoned = serialized.replace(bank.models[0].model_hash, "f" * 64, 1)
    with pytest.raises(ProtocolError, match="hash"):
        deserialize_pairwise_model_bank(poisoned)


def test_frozen_bank_scores_a_separately_sealed_label_free_surface(
    probability_rows: tuple[ProbabilityRow, ...],
    family_surfaces: dict[str, DisagreementFeatureSurface],
    family_models: dict[str, tuple[core.PairwiseRegretModel, ...]],
) -> None:
    bank = freeze_pairwise_model_bank(family_models["R"])
    context = _inference_context(bank)
    inference_surface = build_label_free_inference_feature_surface(
        _target_inference_rows(probability_rows),
        context=context,
    )
    contrasts = score_label_free_inference_candidate_contrasts(
        bank,
        inference_surface,
        context=context,
    )

    assert inference_surface.query_ids == (OUTER_TARGET_ID,)
    assert inference_surface.prediction_seal_hash != bank.models[0].prediction_seal_hash
    assert inference_surface.surface_hash != family_surfaces["R"].surface_hash
    assert inference_surface.development_context_hash == context.context_hash
    assert len(contrasts) == len(CASE_IDS) * len(CANDIDATE_ACTIONS)
    assert {row.model_hash for row in contrasts} == {
        model.model_hash for model in bank.models
    }


def test_authorized_source_training_freezes_before_target_surface_admission(
    probability_rows: tuple[ProbabilityRow, ...],
    source_oof_labels: tuple[SourceOOFLabelRow, ...],
) -> None:
    train_rows = tuple(
        row for row in probability_rows if row.query_id in DONOR_QUERY_IDS
    )
    authorized_keys = tuple(sorted(row.row_key for row in source_oof_labels))
    development = DevelopmentContext(
        scope=DevelopmentScope.AUTHORIZED_SOURCE_OOF,
        dataset_family="MIDOG++_SOURCE_DISCOVERY",
        outer_target_id=OUTER_TARGET_ID,
        authorization_hash="6" * 64,
        authorization_unused=True,
        authorized_query_ids=DONOR_QUERY_IDS,
        authorized_sample_keys_hash=canonical_sha256(
            {"sample_keys": [list(key) for key in authorized_keys]}
        ),
    )
    training_features = build_source_oof_training_feature_surface(
        train_rows,
        baseline_action_id=BASELINE_ACTION_ID,
        control_action_id=CONTROL_ACTION_ID,
        context=development,
    )
    training_responses = build_exact_regret_surface(
        training_features,
        source_oof_labels,
        context=development,
    )
    models = fit_known_bank_pairwise_models(
        training_features,
        training_responses,
        context=development,
        family="R",
    )
    bank = freeze_pairwise_model_bank(models)

    assert OUTER_TARGET_ID not in training_features.query_ids
    assert all(
        model.feature_surface_hash == training_features.surface_hash for model in models
    )
    inference = _inference_context(bank)
    inference_features = build_label_free_inference_feature_surface(
        _target_inference_rows(probability_rows),
        context=inference,
    )
    contrasts = score_label_free_inference_candidate_contrasts(
        bank,
        inference_features,
        context=inference,
    )
    assert contrasts
    assert inference_features.surface_hash != training_features.surface_hash


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("target_cache_content_hash", "1" * 64, "feature/cache/action lineage"),
        ("target_cache_order_hash", "2" * 64, "feature/cache/action lineage"),
        ("prediction_seal_hash", "3" * 64, "feature/cache/action lineage"),
        ("model_bank_hash", "4" * 64, "model-bank seal"),
    ),
)
def test_inference_scoring_rejects_cache_order_prediction_and_model_seal_drift(
    field: str,
    value: str,
    message: str,
    probability_rows: tuple[ProbabilityRow, ...],
    family_models: dict[str, tuple[core.PairwiseRegretModel, ...]],
) -> None:
    bank = freeze_pairwise_model_bank(family_models["R"])
    context = _inference_context(bank)
    surface = build_label_free_inference_feature_surface(
        _target_inference_rows(probability_rows),
        context=context,
    )
    drifted = replace(context, **{field: value})
    with pytest.raises(ProtocolError, match=message):
        score_label_free_inference_candidate_contrasts(
            bank,
            surface,
            context=drifted,
        )


def test_inference_action_schema_is_bound_at_build_and_score(
    probability_rows: tuple[ProbabilityRow, ...],
    family_models: dict[str, tuple[core.PairwiseRegretModel, ...]],
) -> None:
    bank = freeze_pairwise_model_bank(family_models["R"])
    context = _inference_context(bank)
    rows = _target_inference_rows(probability_rows)
    surface = build_label_free_inference_feature_surface(rows, context=context)
    drifted_mapping = tuple(
        ("A::replacement", source_id) if action_id == CANDIDATE_ACTIONS[0] else (
            action_id,
            source_id,
        )
        for action_id, source_id in bank.action_schema.candidate_source_by_action
    )
    drifted_schema = InferenceActionSchema(
        family="R",
        baseline_action_id=BASELINE_ACTION_ID,
        control_action_id=CONTROL_ACTION_ID,
        candidate_source_by_action=drifted_mapping,
    )
    drifted = replace(context, action_schema=drifted_schema)

    with pytest.raises(ProtocolError, match="frozen action schema"):
        build_label_free_inference_feature_surface(rows, context=drifted)
    with pytest.raises(ProtocolError, match="frozen action schema"):
        score_label_free_inference_candidate_contrasts(bank, surface, context=drifted)


@pytest.mark.parametrize(
    "overrides",
    (
        {"target_labels_accessed": True},
        {"target_labels_accessed": 0},
        {"fresh_evidence": True},
        {"terminal_diagnostic_only": False},
        {"may_feed_another_experiment": True},
        {"may_authorize_routing": True},
        {"may_authorize_promotion": True},
        {"consumed_target_data": False},
        {"claim_role": "routing_success"},
    ),
)
def test_label_free_inference_context_rejects_labels_and_consumed_claim_drift(
    overrides: dict[str, object],
    family_models: dict[str, tuple[core.PairwiseRegretModel, ...]],
) -> None:
    bank = freeze_pairwise_model_bank(family_models["R"])
    values: dict[str, object] = {
        "dataset_family": "MIDOG++_CONSUMED_TEST_LABEL_FREE",
        "outer_target_id": OUTER_TARGET_ID,
        "target_cache_content_hash": "d" * 64,
        "target_cache_order_hash": "e" * 64,
        "prediction_seal_hash": "c" * 64,
        "action_schema": bank.action_schema,
        "model_bank_hash": bank.model_bank_hash,
    }
    values.update(overrides)
    with pytest.raises(ProtocolError):
        LabelFreeInferenceContext(**values)


def test_package_has_a_static_non_runnable_import_and_io_firewall() -> None:
    package_root = Path(inspect.getfile(core)).resolve().parent
    forbidden_import_parts = {
        "artifact",
        "artifacts",
        "config",
        "configs",
        "diagnostic",
        "diagnostics",
        "os",
        "pathlib",
        "runner",
        "workspace",
    }

    for source_path in sorted(package_root.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_names = [node.module or "", *(alias.name for alias in node.names)]
            else:
                imported_names = []
            for imported_name in imported_names:
                assert forbidden_import_parts.isdisjoint(imported_name.split(".")), (
                    source_path,
                    imported_name,
                )
            if isinstance(node, ast.Name):
                assert node.id not in {"Path", "open"}, (source_path, node.id)
            if isinstance(node, ast.Attribute):
                assert node.attr != "open", (source_path, node.attr)
        assert not hasattr(core, "main")
