"""Focused pure-core tests for the ensemble proxy-information audit."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.contracts import (
    CENTERS,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
    EQUAL_UNION_NULL,
    EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT,
    EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
    FAMILY_IDS,
    HYBRID_COMPACT,
    INPUT_ARTIFACT_IDS,
    RIDGE_ALPHA,
    ProxyFeatureRow,
    ProxyUtilityRow,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.crossfit import (
    crossfit_proxy_families,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.metrics import (
    run_proxy_information_audit,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.proxy_features import (
    PROXY_FAMILY_SPECS,
    build_proxy_family_designs,
    build_proxy_feature_surface,
    proxy_feature_row_from_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _surfaces() -> tuple[tuple[ProxyFeatureRow, ...], tuple[ProxyUtilityRow, ...]]:
    features: list[ProxyFeatureRow] = []
    utility: list[ProxyUtilityRow] = []
    center_position = {center: index for index, center in enumerate(CENTERS)}
    seal = _hash("development-seal")
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            sources = candidate_sources(outer, query)
            source_raw = np.asarray(
                [float(center_position[source]) for source in sources], dtype=np.float64
            )
            source_z = (source_raw - float(np.mean(source_raw))) / float(
                np.sqrt(np.mean((source_raw - float(np.mean(source_raw))) ** 2))
            )
            query_effect = 0.01 * float(center_position[query] - 4)
            for candidate_index, source in enumerate(sources):
                support_hash = _hash(f"support::{outer}::{query}")
                signed = float(np.clip(0.08 * source_z[candidate_index], -0.9, 0.9))
                flip = float(0.05 + 0.02 * candidate_index)
                entropy = float(-0.03 * source_z[candidate_index])
                feature = ProxyFeatureRow(
                    outer_target_id=outer,
                    query_id=query,
                    candidate_source=source,
                    candidate_source_count=7,
                    support_partition_hash=support_hash,
                    support_case_count=2,
                    support_row_count=8 + center_position[query],
                    seed_pair_count=9,
                    seed_feature_row_hashes=tuple(
                        _hash(f"feature::{outer}::{query}::{source}::{index}")
                        for index in range(9)
                    ),
                    base_support_vector_hashes=tuple(
                        _hash(f"base::{outer}::{query}::{index}")
                        for index in range(9)
                    ),
                    tail_support_vector_hashes=tuple(
                        _hash(f"tail::{outer}::{query}::{source}::{index}")
                        for index in range(9)
                    ),
                    metadata_similarity=float(center_position[source] / 8.0),
                    absolute_ensemble_shift=float(0.02 + 0.01 * candidate_index),
                    reconstruction_mean_within_query_z=float(source_z[candidate_index]),
                    kl_mean_within_query_z=float(-source_z[candidate_index]),
                    log_distribution_mmd_within_query_z=float(
                        0.5 * source_z[candidate_index]
                    ),
                    signed_margin_projection=signed,
                    threshold_flip_rate=flip,
                    mean_entropy_change=entropy,
                    development_prediction_seal_hash=seal,
                )
                features.append(feature)
                response = float(
                    np.clip(
                        0.7 * signed
                        + 0.08 * feature.log_distribution_mmd_within_query_z
                        + query_effect,
                        -0.95,
                        0.95,
                    )
                )
                utility.append(
                    ProxyUtilityRow(
                        outer_target_id=outer,
                        query_id=query,
                        candidate_source=source,
                        candidate_source_count=7,
                        support_partition_hash=support_hash,
                        utility_delta=response,
                        response_hash=_hash(
                            f"response::{outer}::{query}::{source}::{response}"
                        ),
                    )
                )
    return tuple(features), tuple(utility)


@pytest.fixture(scope="module")
def audited():
    features, utility = _surfaces()
    return features, utility, run_proxy_information_audit(features, utility)


def test_frozen_contract_uses_candidate_responses_not_seed_observations() -> None:
    assert EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT == 504
    assert EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT == 4536
    assert len(INPUT_ARTIFACT_IDS) == 5
    assert RIDGE_ALPHA == 1.0
    assert tuple(PROXY_FAMILY_SPECS) == FAMILY_IDS
    assert all(spec.predictor_count <= 3 for spec in PROXY_FAMILY_SPECS.values())
    assert PROXY_FAMILY_SPECS[EQUAL_UNION_NULL].predictor_count == 0
    assert PROXY_FAMILY_SPECS[HYBRID_COMPACT].predictor_names == (
        "metadata_similarity",
        "log_distribution_mmd_within_query_z",
        "signed_margin_projection",
    )


def test_strict_label_free_payload_parser_is_hash_bound() -> None:
    features, _utility = _surfaces()
    payload = features[0].to_payload()
    assert proxy_feature_row_from_payload(payload) == features[0]
    with pytest.raises(ProtocolError, match="exact schema"):
        proxy_feature_row_from_payload({**payload, "labels": [0, 1]})
    with pytest.raises(ProtocolError, match="hash drifted"):
        proxy_feature_row_from_payload({**payload, "metadata_similarity": 0.0})
    with pytest.raises(ProtocolError, match="label/probability boundary"):
        replace(features[0], labels_used=True)


def test_surface_and_cyclic_control_are_deterministic_with_fixed_capacity() -> None:
    features, _utility = _surfaces()
    first = build_proxy_feature_surface(features)
    second = build_proxy_feature_surface(tuple(reversed(features)))
    assert first.surface_hash == second.surface_hash
    assert first.row_keys == second.row_keys
    designs = build_proxy_family_designs(first)
    assert designs[EQUAL_UNION_NULL].values.shape == (504, 0)
    assert all(design.values.shape[1] <= 3 for design in designs.values())
    row = first.rows[0]
    sources = candidate_sources(row.outer_target_id, row.query_id)
    donor = next(
        item
        for item in first.rows
        if item.row_key == (row.outer_target_id, row.query_id, sources[1])
    )
    assert designs[CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL].values[0].tolist() == pytest.approx(
        [
            donor.signed_margin_projection,
            donor.threshold_flip_rate,
            donor.mean_entropy_change,
        ]
    )


def test_crossfit_excludes_H_q_e_from_all_outer_query_source_roles(audited) -> None:
    _features, _utility, result = audited
    crossfit = result.crossfit
    assert len(crossfit.predictions) == len(FAMILY_IDS) * 504
    assert crossfit.fold_lock.fold_count == len(FAMILY_IDS) * 504
    assert crossfit.fold_lock.ridge_alpha == 1.0
    for audit in crossfit.fold_audits:
        outer, query, source = audit.predicted_row_key
        excluded = {outer, query, source}
        assert audit.training_row_count == 120
        assert audit.to_payload()["ridge_cluster_unit"] == "outer_target_query"
        assert not excluded.intersection(audit.training_outer_target_ids)
        assert not excluded.intersection(audit.training_query_ids)
        assert not excluded.intersection(audit.training_source_ids)
        assert all(not excluded.intersection(key) for key in audit.training_row_keys)
        assert audit.learned_scaling_fit_on_training_fold_only is True
        assert audit.precomputed_candidate_list_transforms_are_label_free is True
    assert crossfit.fold_lock.to_payload()[
        "descriptive_seed_rows_may_feed_model"
    ] is False
    assert crossfit.fold_lock.to_payload()["ridge_cluster_unit"] == "outer_target_query"


def test_metrics_use_queries_then_nine_outer_H_units_and_remain_diagnostic(audited) -> None:
    features, utility, result = audited
    assert len(result.query_metrics) == len(FAMILY_IDS) * 72
    assert len(result.outer_metrics) == len(FAMILY_IDS) * 9
    assert len(result.family_summaries) == len(FAMILY_IDS)
    assert all(row.candidate_count == 7 for row in result.query_metrics)
    assert all(row.query_count == 8 for row in result.outer_metrics)
    assert all(row.outer_count == 9 for row in result.family_summaries)
    assert all(0.0 <= row.pairwise_accuracy <= 1.0 for row in result.query_metrics)
    assert all(0.0 <= row.normalized_oracle_regret <= 1.0 for row in result.query_metrics)
    assert result.to_payload()["screening_gate_may_authorize_policy"] is False
    assert result.to_payload()["policy_update_authorized"] is False
    assert result.to_payload()["technical_seed_row_count"] == 4536
    assert result.result_hash == run_proxy_information_audit(
        tuple(reversed(features)), tuple(reversed(utility))
    ).result_hash
