"""Focused pure-core tests for the case-aware proxy-information audit."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib

import numpy as np
import pytest

import midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit as core
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit import (
    contracts as contract_facade,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.audit import (
    run_case_aware_proxy_information_audit,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.case_features import (
    build_case_aware_feature_row,
    build_case_aware_feature_surface,
    feature_row_from_payload,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.contracts import (
    CASE_AWARE_HYBRID_COMPACT,
    CENTERS,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
    EXACT_BACC_DELTA,
    FAMILY_IDS,
    MIN_SUPPORT_CASE_COUNT_PER_CENTER,
    POOLED_ROW_WEIGHTED_SHIFT_CONTROL,
    RESPONSE_NAMES,
    RIDGE_ALPHA,
    SMOOTH_BACC_DELTA,
    CaseAwareProxyFeatureRow,
    CaseAwareResponseRow,
    SupportCaseVectors,
    candidate_sources,
    expected_strict_training_row_count,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.crossfit import (
    crossfit_fold_audit_from_payload,
    crossfit_proxy_families,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.family_designs import (
    PROXY_FAMILY_SPECS,
    build_family_designs,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.feature_contracts import (
    CaseAwareProxyFeatureRow as LeafCaseAwareProxyFeatureRow,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.response_contracts import (
    CaseAwareResponseRow as LeafCaseAwareResponseRow,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit import (
    experiment_contracts,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.response_surfaces import (
    ExactNineEvaluationVectors,
    build_response_row,
    build_response_surface,
    exact_nine_response_values,
    mean_exact_nine_probabilities,
    response_row_from_payload,
    soft_balanced_accuracy,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import (
    array_sha256,
    canonical_sha256,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _support_case(
    index: int,
    *,
    shift: float,
    repeat: int = 1,
) -> SupportCaseVectors:
    base_row = np.asarray([0.2, 0.8], dtype=np.float64)
    tail_row = np.asarray([0.2 + shift, 0.8 - shift], dtype=np.float64)
    base = np.tile(np.tile(base_row, repeat), (9, 1))
    tail = np.tile(np.tile(tail_row, repeat), (9, 1))
    return SupportCaseVectors(
        case_id=f"case-{index}",
        case_hash=_hash(f"case::{index}"),
        row_hash=_hash(f"rows::{index}::repeat::{repeat}"),
        provenance_hash=_hash(f"provenance::{index}"),
        base_probabilities=base,
        tail_probabilities=tail,
        reconstruction_summary=0.1 + index,
        kl_summary=0.2 + index,
        log_mmd_summary=0.3 + index,
    )


def _single_feature(cases: tuple[SupportCaseVectors, ...]) -> CaseAwareProxyFeatureRow:
    return build_case_aware_feature_row(
        outer_target_id="0",
        query_id="1",
        candidate_source="2",
        support_partition_hash=_hash("support-partition"),
        prediction_seal_hash=_hash("support-prediction-seal"),
        metadata_similarity=0.5,
        cases=cases,
    )


def _complete_surfaces():
    feature_rows: list[CaseAwareProxyFeatureRow] = []
    center_position = {center: index for index, center in enumerate(CENTERS)}
    seal = _hash("global-label-free-prediction-seal")
    for outer in CENTERS:
        for query in (value for value in CENTERS if value != outer):
            sources = candidate_sources(outer, query)
            raw = np.asarray(
                [float(center_position[source]) for source in sources],
                dtype=np.float64,
            )
            centered = raw - float(np.mean(raw, dtype=np.float64))
            z = centered / float(np.sqrt(np.mean(centered * centered)))
            case_hashes = tuple(
                _hash(f"support-case::{query}::{index}")
                for index in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
            )
            row_hashes = tuple(
                _hash(f"support-rows::{query}::{index}")
                for index in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
            )
            provenance = tuple(
                _hash(f"support-provenance::{outer}::{query}::{index}")
                for index in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
            )
            base_hashes = tuple(
                tuple(
                    _hash(f"base::{outer}::{query}::{case}::{seed}")
                    for seed in range(9)
                )
                for case in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
            )
            for source_index, source in enumerate(sources):
                feature_rows.append(
                    CaseAwareProxyFeatureRow(
                        outer_target_id=outer,
                        query_id=query,
                        candidate_source=source,
                        candidate_source_count=len(sources),
                        support_partition_hash=_hash(f"support::{query}"),
                        prediction_seal_hash=seal,
                        support_case_count=MIN_SUPPORT_CASE_COUNT_PER_CENTER,
                        support_row_count=32,
                        support_case_hashes=case_hashes,
                        support_row_hashes=row_hashes,
                        support_provenance_hashes=provenance,
                        base_vector_hashes_by_case=base_hashes,
                        tail_vector_hashes_by_case=tuple(
                            tuple(
                                _hash(
                                    f"tail::{outer}::{query}::{source}::{case}::{seed}"
                                )
                                for seed in range(9)
                            )
                            for case in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
                        ),
                        metadata_similarity=float(center_position[source] / 8.0),
                        pooled_row_weighted_abs_shift=float(
                            0.02 + 0.005 * source_index
                        ),
                        equal_case_abs_shift=float(
                            0.03 + 0.006 * source_index
                        ),
                        case_abs_shift_sd=float(0.01 + 0.001 * source_index),
                        equal_case_signed_margin=float(0.04 * z[source_index]),
                        case_balanced_flip_rate=float(
                            0.03 + 0.01 * source_index
                        ),
                        case_balanced_entropy_change=float(
                            -0.02 * z[source_index]
                        ),
                        case_balanced_reconstruction=float(
                            0.5 + z[source_index]
                        ),
                        case_balanced_kl=float(0.7 - z[source_index]),
                        case_balanced_log_mmd=float(
                            0.4 + 0.5 * z[source_index]
                        ),
                    )
                )
    feature_surface = build_case_aware_feature_surface(feature_rows)
    persisted_feature_lock_hash = _hash("persisted-prelabel-feature-lock")
    response_rows: list[CaseAwareResponseRow] = []
    for feature in feature_surface.rows:
        source_index = center_position[feature.candidate_source]
        query_index = center_position[feature.query_id]
        exact_delta = float(
            np.clip(
                0.6 * feature.equal_case_signed_margin
                + 0.01 * feature.case_balanced_log_mmd
                + 0.001 * (query_index - 4),
                -0.2,
                0.2,
            )
        )
        smooth_delta = float(
            np.clip(
                0.5 * exact_delta + 0.001 * (source_index - 4),
                -0.2,
                0.2,
            )
        )
        response_rows.append(
            CaseAwareResponseRow(
                outer_target_id=feature.outer_target_id,
                query_id=feature.query_id,
                candidate_source=feature.candidate_source,
                support_partition_hash=feature.support_partition_hash,
                feature_row_hash=feature.feature_row_hash,
                feature_surface_seal_hash=persisted_feature_lock_hash,
                evaluation_partition_hash=_hash(
                    f"evaluation::{feature.query_id}"
                ),
                evaluation_case_hashes=tuple(
                    _hash(f"eval-case::{feature.query_id}::{index}")
                    for index in range(3)
                ),
                evaluation_row_hash=_hash(f"eval-rows::{feature.query_id}"),
                evaluation_label_sha256=_hash(
                    f"eval-labels::{feature.query_id}"
                ),
                response_prediction_hash=_hash(
                    f"response::{feature.outer_target_id}::{feature.query_id}::"
                    f"{feature.candidate_source}"
                ),
                exact_base_bacc=0.5,
                exact_tail_bacc=0.5 + exact_delta,
                exact_bacc_delta=exact_delta,
                smooth_base_bacc=0.5,
                smooth_tail_bacc=0.5 + smooth_delta,
                smooth_bacc_delta=smooth_delta,
            )
        )
    response_surface = build_response_surface(feature_surface, response_rows)
    return feature_surface, response_surface


@pytest.fixture(scope="module")
def audited_surfaces():
    features, responses = _complete_surfaces()
    result = run_case_aware_proxy_information_audit(features, responses)
    return features, responses, result


def test_equal_case_features_ignore_case_size_but_pooled_control_does_not() -> None:
    original_cases = tuple(
        _support_case(index, shift=0.1 if index == 0 else 0.2)
        for index in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
    )
    duplicated_cases = (
        _support_case(0, shift=0.1, repeat=5),
        *original_cases[1:],
    )
    original = _single_feature(original_cases)
    duplicated = _single_feature(duplicated_cases)

    invariant_names = (
        "equal_case_abs_shift",
        "case_abs_shift_sd",
        "equal_case_signed_margin",
        "case_balanced_flip_rate",
        "case_balanced_entropy_change",
        "case_balanced_reconstruction",
        "case_balanced_kl",
        "case_balanced_log_mmd",
    )
    for name in invariant_names:
        assert getattr(duplicated, name) == pytest.approx(getattr(original, name))
    assert duplicated.pooled_row_weighted_abs_shift != pytest.approx(
        original.pooled_row_weighted_abs_shift
    )
    assert duplicated.support_row_count != original.support_row_count
    assert duplicated.support_row_hashes != original.support_row_hashes


def test_exact_nine_mean_precedes_threshold_and_soft_bacc_is_exact() -> None:
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    base = np.empty((9, 4), dtype=np.float64)
    base[:5, :2] = 0.51
    base[5:, :2] = 0.0
    base[:5, 2:] = 0.49
    base[5:, 2:] = 1.0
    tail = np.tile(np.asarray([0.6, 0.6, 0.4, 0.4]), (9, 1))

    mean = mean_exact_nine_probabilities(base)
    assert np.all(mean[:2] < 0.5)
    assert np.all(mean[2:] >= 0.5)
    exact_base, exact_tail, exact_delta, smooth_base, smooth_tail, smooth_delta = (
        exact_nine_response_values(base, tail, labels)
    )
    assert (exact_base, exact_tail, exact_delta) == pytest.approx((0.0, 1.0, 1.0))
    assert smooth_base == pytest.approx(float(mean[0]))
    assert smooth_tail == pytest.approx(0.6)
    assert smooth_delta == pytest.approx(smooth_tail - smooth_base)
    assert soft_balanced_accuracy(
        [1, 1, 0, 0], [0.8, 0.6, 0.3, 0.1]
    ) == pytest.approx(0.75)


def test_response_builder_enforces_disjoint_post_seal_scoring() -> None:
    feature = _single_feature(
        tuple(
            _support_case(index, shift=0.1)
            for index in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
        )
    )
    evaluation = ExactNineEvaluationVectors(
        evaluation_partition_hash=_hash("evaluation-partition"),
        evaluation_case_hashes=(_hash("remaining-case-a"), _hash("remaining-case-b")),
        evaluation_row_hash=_hash("evaluation-rows"),
        prediction_provenance_hash=_hash("evaluation-prediction-seal"),
        base_probabilities=np.tile(np.asarray([0.2, 0.8]), (9, 1)),
        tail_probabilities=np.tile(np.asarray([0.8, 0.2]), (9, 1)),
        labels=np.asarray([1, 0]),
    )
    row = build_response_row(
        feature_row=feature,
        feature_surface_seal_hash=_hash("feature-surface-seal"),
        evaluation=evaluation,
    )
    assert row.exact_bacc_delta == pytest.approx(1.0)
    assert row.smooth_bacc_delta == pytest.approx(0.6)
    assert row.evaluation_label_sha256 == array_sha256(evaluation.labels)
    assert response_row_from_payload(row.to_payload()) == row
    with pytest.raises(ProtocolError, match="hash drifted"):
        response_row_from_payload(
            {**row.to_payload(), "evaluation_label_sha256": _hash("wrong-labels")}
        )
    with pytest.raises(ProtocolError, match="overlap"):
        build_response_row(
            feature_row=feature,
            feature_surface_seal_hash=_hash("feature-surface-seal"),
            evaluation=ExactNineEvaluationVectors(
                evaluation_partition_hash=_hash("bad-partition"),
                evaluation_case_hashes=(feature.support_case_hashes[0],),
                evaluation_row_hash=_hash("bad-rows"),
                prediction_provenance_hash=_hash("bad-provenance"),
                base_probabilities=np.tile(np.asarray([0.2, 0.8]), (9, 1)),
                tail_probabilities=np.tile(np.asarray([0.8, 0.2]), (9, 1)),
                labels=np.asarray([1, 0]),
            ),
        )


def test_feature_schema_is_label_free_hash_bound_and_deterministic() -> None:
    cases = tuple(
        _support_case(index, shift=0.1 + 0.01 * index)
        for index in range(MIN_SUPPORT_CASE_COUNT_PER_CENTER)
    )
    first = _single_feature(cases)
    second = _single_feature(tuple(reversed(cases)))
    assert first.feature_row_hash == second.feature_row_hash
    assert first.to_payload() == second.to_payload()
    payload = first.to_payload()
    assert "labels" not in payload
    assert payload["labels_used"] is False
    assert feature_row_from_payload(payload) == first
    with pytest.raises(ProtocolError, match="exact schema"):
        feature_row_from_payload({**payload, "labels": [0, 1]})
    with pytest.raises(ProtocolError, match="hash drifted"):
        feature_row_from_payload({**payload, "equal_case_abs_shift": 0.0})


def test_family_designs_include_fixed_capacity_and_cyclic_donor(audited_surfaces) -> None:
    features, _responses, _result = audited_surfaces
    designs = build_family_designs(features)
    assert tuple(designs) == FAMILY_IDS
    assert RIDGE_ALPHA == 1.0
    assert all(spec.predictor_count <= 3 for spec in PROXY_FAMILY_SPECS.values())
    assert PROXY_FAMILY_SPECS[CASE_AWARE_HYBRID_COMPACT].predictor_names == (
        "metadata_similarity",
        "case_balanced_log_mmd_z",
        "equal_case_abs_shift",
    )
    assert PROXY_FAMILY_SPECS[POOLED_ROW_WEIGHTED_SHIFT_CONTROL].predictor_names == (
        "pooled_row_weighted_abs_shift",
    )
    first = features.rows[0]
    sources = candidate_sources(first.outer_target_id, first.query_id)
    donor = next(
        row
        for row in features.rows
        if row.row_key == (first.outer_target_id, first.query_id, sources[1])
    )
    assert designs[CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL].values[0].tolist() == pytest.approx(
        [
            donor.equal_case_signed_margin,
            donor.case_balanced_flip_rate,
            donor.case_balanced_entropy_change,
        ]
    )


def test_crossfit_count_is_geometry_derived_and_excludes_every_role(audited_surfaces) -> None:
    _features, _responses, result = audited_surfaces
    assert expected_strict_training_row_count(9) == 120
    assert expected_strict_training_row_count(tuple(range(10))) == 210
    assert len(result.crossfit.predictions) == len(FAMILY_IDS) * len(RESPONSE_NAMES) * 504
    for audit in result.crossfit.fold_audits:
        excluded = set(audit.predicted_row_key)
        assert audit.training_row_count == expected_strict_training_row_count(CENTERS)
        assert audit.ridge_alpha == 1.0
        assert all(excluded.isdisjoint(key) for key in audit.training_row_keys)
        assert audit.to_payload()["hyperparameter_selection"] == "none_fixed_predeclared"

    first = result.crossfit.fold_audits[0]
    assert crossfit_fold_audit_from_payload(first.to_payload()) == first
    with pytest.raises(ProtocolError, match="fitted provenance hash drifted"):
        replace(first, intercept=first.intercept + 1.0)
    tampered = {**first.to_payload(), "intercept": first.intercept + 1.0}
    with pytest.raises(ProtocolError, match="fitted provenance hash drifted"):
        crossfit_fold_audit_from_payload(tampered)
    fold_lock = result.fold_lock_payload()
    supplied_lock_hash = fold_lock.pop("crossfit_fold_lock_hash")
    assert canonical_sha256(fold_lock) == supplied_lock_hash


def test_nine_outer_units_exact_primary_and_smooth_diagnostic_only(audited_surfaces) -> None:
    features, responses, result = audited_surfaces
    assert len(result.query_metrics) == len(FAMILY_IDS) * len(RESPONSE_NAMES) * 72
    assert len(result.outer_metrics) == len(FAMILY_IDS) * len(RESPONSE_NAMES) * 9
    assert len(result.family_summaries) == len(FAMILY_IDS) * len(RESPONSE_NAMES)
    assert all(row.outer_count == 9 for row in result.family_summaries)
    assert all(
        not row.screening_eligible and not row.screening_passed
        for row in result.family_summaries
        if row.response_name == SMOOTH_BACC_DELTA
    )
    assert all(
        row.response_name == EXACT_BACC_DELTA
        for row in result.family_summaries
        if row.screening_eligible
    )
    payload = result.to_payload()
    assert payload["publication_status"] == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert payload["policy_update_authorized"] is False
    assert payload["stage60_feed_authorized"] is False
    assert payload["stage70_feed_authorized"] is False
    assert payload["target_actions_authorized"] is False
    assert responses.feature_surface_hash == features.surface_hash
    assert responses.feature_surface_seal_hash == _hash(
        "persisted-prelabel-feature-lock"
    )
    assert responses.feature_surface_seal_hash != responses.feature_surface_hash
    assert build_case_aware_feature_surface(tuple(reversed(features.rows))).surface_hash == (
        features.surface_hash
    )
    assert build_response_surface(
        features, tuple(reversed(responses.rows))
    ).surface_hash == responses.surface_hash
    with pytest.raises(ProtocolError, match="one persisted pre-label feature seal"):
        build_response_surface(
            features,
            (
                replace(
                    responses.rows[0],
                    feature_surface_seal_hash=_hash("different-feature-lock"),
                ),
                *responses.rows[1:],
            ),
        )


def test_public_core_has_no_policy_or_action_api() -> None:
    assert all(
        "policy" not in name.lower() and "action" not in name.lower()
        for name in core.__all__
    )
    feature_field_names = {field.name for field in fields(CaseAwareProxyFeatureRow)}
    assert "labels" not in feature_field_names
    assert "evaluation_labels" not in feature_field_names


def test_contract_facade_reexports_leaf_objects_and_canonical_identity() -> None:
    assert contract_facade.CaseAwareProxyFeatureRow is LeafCaseAwareProxyFeatureRow
    assert contract_facade.CaseAwareResponseRow is LeafCaseAwareResponseRow
    assert contract_facade.EXPERIMENT_ID is experiment_contracts.EXPERIMENT_ID
    assert contract_facade.STAGE_ID is experiment_contracts.STAGE_ID
