from __future__ import annotations

from dataclasses import replace
import inspect
import math

import numpy as np
import pytest

import midogpp_thesis.cvae.routing.utility_aligned as utility_aligned_public
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_contracts as ensemble_contracts
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_endpoint_contracts as endpoint_contracts
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_feature_contracts as feature_contracts
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_model_contracts as model_contracts
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_policy as ensemble_policy_facade
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_policy_contracts as policy_contracts
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_target_policy as target_policy
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_transfer as ensemble_transfer
import midogpp_thesis.cvae.routing.utility_aligned.ensemble_utility_contracts as utility_contracts
from midogpp_thesis.cvae.metrics import balanced_accuracy
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned import (
    ENSEMBLE_SEED_KEYS,
    ENSEMBLE_UTILITY_SEMANTICS,
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    CandidateFeatureRow,
    ScoredEnsembleUtilityResponse,
    SeedProbabilityVector,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
    SupportActionProbabilityShift,
    TargetSupportActionShiftCase,
    aggregate_candidate_seed_features,
    build_ensemble_feature_surface,
    build_ensemble_utility_response,
    build_ensemble_utility_policy,
    build_case_bootstrap_plan,
    build_target_ensemble_feature_surfaces,
    build_target_support_action_shift_case,
    cyclically_permute_target_scalar,
    derive_label_free_global_source_control,
    evaluate_ensemble_cardinality_transfer,
    fit_ensemble_utility_model,
    mean_exact_nine_positive_class_probabilities,
    score_nine_seed_probability_ensemble,
    scored_ensemble_utility_response_from_payload,
    support_action_probability_shift,
    validate_ensemble_utility_responses,
)


OUTER = "H"
SOURCES = tuple(f"d{index}" for index in range(8))


def test_split_contract_and_policy_facades_preserve_object_identity() -> None:
    endpoint_names = (
        "ProbabilityEnsembleEndpoint",
        "SeedProbabilityVector",
        "SupportActionProbabilityShift",
    )
    utility_names = (
        "EnsembleUtilityResponse",
        "EnsembleUtilitySurface",
        "ScoredEnsembleUtilityResponse",
    )
    feature_names = (
        "EnsembleCandidateFeatureRow",
        "EnsembleFeatureSurface",
        "GlobalSourceControl",
        "TargetEnsembleFeatureProduction",
        "TargetSupportActionShiftCase",
    )
    model_names = (
        "EnsembleCapacityReport",
        "EnsembleCardinalityTransferResult",
        "EnsembleFoldAudit",
        "EnsembleUtilityModel",
    )
    for module, names in (
        (endpoint_contracts, endpoint_names),
        (utility_contracts, utility_names),
        (feature_contracts, feature_names),
        (model_contracts, model_names),
        (policy_contracts, ("EnsembleUtilityPolicy",)),
    ):
        for name in names:
            assert getattr(ensemble_contracts, name) is getattr(module, name)
            assert getattr(utility_aligned_public, name) is getattr(module, name)
    assert (
        ensemble_policy_facade.build_ensemble_utility_policy
        is target_policy.build_ensemble_utility_policy
        is utility_aligned_public.build_ensemble_utility_policy
    )
    assert (
        ensemble_policy_facade.evaluate_ensemble_cardinality_transfer
        is ensemble_transfer.evaluate_ensemble_cardinality_transfer
        is utility_aligned_public.evaluate_ensemble_cardinality_transfer
    )


def _probability_vectors(
    values_by_seed: tuple[tuple[float, ...], ...], *, action: str
) -> tuple[SeedProbabilityVector, ...]:
    return tuple(
        SeedProbabilityVector(
            training_seed=training_seed,
            generation_seed=generation_seed,
            row_identity_hash="evaluation-rows-q",
            prediction_provenance_hash=f"{action}-{training_seed}-{generation_seed}",
            positive_class_probabilities=np.asarray(values, dtype=np.float32),
        )
        for (training_seed, generation_seed), values in zip(
            ENSEMBLE_SEED_KEYS, values_by_seed
        )
    )


def _candidate_seed_rows() -> tuple[CandidateFeatureRow, ...]:
    rows: list[CandidateFeatureRow] = []
    for query_index, query in enumerate(SOURCES):
        for source in SOURCES:
            if source == query:
                continue
            source_index = SOURCES.index(source)
            for seed_index, (training_seed, generation_seed) in enumerate(
                ENSEMBLE_SEED_KEYS
            ):
                scalar = (
                    0.05
                    + 0.07 * float((3 * query_index + 2 * source_index) % 11)
                    + 1.0e-4 * float(seed_index)
                )
                reconstruction = 0.8 + 0.01 * source_index + 1.0e-4 * seed_index
                kl = 0.2 + 0.01 * query_index + 2.0e-4 * seed_index
                rows.append(
                    CandidateFeatureRow(
                        role=INNER_ROLE,
                        outer_target_id=OUTER,
                        query_id=query,
                        candidate_source=source,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        candidate_source_count=INNER_CANDIDATE_COUNT,
                        support_partition_hash=f"support-{OUTER}-{query}",
                        support_case_count=12,
                        reconstruction_mean=reconstruction,
                        reconstruction_std=0.04 + 1.0e-4 * seed_index,
                        reconstruction_q25=reconstruction - 0.05,
                        reconstruction_q50=reconstruction,
                        reconstruction_q75=reconstruction + 0.05,
                        kl_mean=kl,
                        kl_std=0.02 + 1.0e-4 * seed_index,
                        kl_q25=kl - 0.025,
                        kl_q50=kl,
                        kl_q75=kl + 0.025,
                        replica_disagreement=0.01 + 1.0e-4 * seed_index,
                        distribution_mmd=0.1 + 0.02 * source_index,
                        metadata_similarity=scalar,
                    )
                )
    return tuple(rows)


def _scored_responses(
    aggregated,
) -> tuple[ScoredEnsembleUtilityResponse, ...]:
    output: list[ScoredEnsembleUtilityResponse] = []
    for row in aggregated:
        query_index = SOURCES.index(row.query_id)
        source_index = SOURCES.index(row.candidate_source)
        global_control = float(source_index) / 7.0
        scalar = float(row.target_local_scalar)
        delta = (
            0.015
            + 0.025 * global_control
            + 0.035 * scalar
            + 0.003 * math.sin(float(2 * query_index + source_index))
        )
        output.append(
            ScoredEnsembleUtilityResponse(
                outer_target_id=OUTER,
                query_id=row.query_id,
                candidate_source=row.candidate_source,
                candidate_source_count=INNER_CANDIDATE_COUNT,
                support_partition_hash=row.support_partition_hash,
                evaluation_partition_hash=f"evaluation-{row.query_id}",
                prediction_seal_hash=f"seal-{OUTER}",
                evaluation_row_identity_hash=f"evaluation-{row.query_id}",
                evaluation_label_hash=f"labels-{row.query_id}",
                base_endpoint_hash=f"base-endpoint-{row.query_id}",
                tail_endpoint_hash=f"tail-endpoint-{row.query_id}-{row.candidate_source}",
                base_probability_cell_hashes_hash=f"base-cells-{row.query_id}",
                tail_probability_cell_hashes_hash=f"tail-cells-{row.query_id}-{row.candidate_source}",
                base_ensemble_probability_hash=f"base-prob-{row.query_id}",
                tail_ensemble_probability_hash=f"tail-prob-{row.query_id}-{row.candidate_source}",
                base_ensemble_prediction_hash=f"base-pred-{row.query_id}",
                tail_ensemble_prediction_hash=f"tail-pred-{row.query_id}-{row.candidate_source}",
                source_response_hash=None,
                source_endpoint_row_hash=None,
                base_component_vector_hashes=tuple(
                    f"base-{row.query_id}-{index}" for index in range(9)
                ),
                tail_component_vector_hashes=tuple(
                    f"tail-{row.query_id}-{row.candidate_source}-{index}"
                    for index in range(9)
                ),
                base_bacc=0.60,
                tail_bacc=0.60 + delta,
                support_eval_disjoint=True,
                predictions_sealed_before_labels=True,
                source_expert_frozen=True,
            )
        )
    return tuple(output)


def _target_seed_rows(plan) -> tuple[CandidateFeatureRow, ...]:
    rows: list[CandidateFeatureRow] = []
    for source_index, source in enumerate(SOURCES):
        for seed_index, (training_seed, generation_seed) in enumerate(
            ENSEMBLE_SEED_KEYS
        ):
            reconstruction = 0.75 + 0.01 * source_index + 1.0e-4 * seed_index
            kl = 0.18 + 0.005 * source_index + 1.0e-4 * seed_index
            rows.append(
                CandidateFeatureRow(
                    role="fresh_target_support",
                    outer_target_id=OUTER,
                    query_id=OUTER,
                    candidate_source=source,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    candidate_source_count=8,
                    support_partition_hash=plan.support_partition_hash,
                    support_case_count=len(plan.support_case_ids),
                    reconstruction_mean=reconstruction,
                    reconstruction_std=0.04,
                    reconstruction_q25=reconstruction - 0.05,
                    reconstruction_q50=reconstruction,
                    reconstruction_q75=reconstruction + 0.05,
                    kl_mean=kl,
                    kl_std=0.02,
                    kl_q25=kl - 0.025,
                    kl_q50=kl,
                    kl_q75=kl + 0.025,
                    replica_disagreement=0.01,
                    distribution_mmd=0.1 + 0.01 * source_index,
                    metadata_similarity=0.1 + 0.1 * source_index,
                )
            )
    return tuple(rows)


def _target_case_shifts(plan) -> tuple[TargetSupportActionShiftCase, ...]:
    rows: list[TargetSupportActionShiftCase] = []
    for source_index, source in enumerate(SOURCES):
        for case_index, case_id in enumerate(plan.support_case_ids):
            shifts = tuple(
                0.005
                + 0.002 * source_index
                + 0.003 * case_index
                + 0.0001 * seed_index
                for seed_index in range(9)
            )
            rows.append(
                TargetSupportActionShiftCase(
                    target_id=OUTER,
                    candidate_source=source,
                    case_id=case_id,
                    support_row_identity_hash=f"support-row-{case_id}",
                    support_row_count=case_index + 1,
                    seed_keys=ENSEMBLE_SEED_KEYS,
                    per_seed_mean_absolute_shifts=shifts,
                    base_component_vector_hashes=tuple(
                        f"base-{source}-{case_id}-{seed_index}"
                        for seed_index in range(9)
                    ),
                    tail_component_vector_hashes=tuple(
                        f"tail-{source}-{case_id}-{seed_index}"
                        for seed_index in range(9)
                    ),
                    # Deliberately smaller than the descriptive seed mean so
                    # the bootstrap test detects a forbidden re-aggregation
                    # of per-seed absolutes into the model scalar.
                    ensemble_mean_absolute_shift=0.5 * float(np.mean(shifts)),
                    base_ensemble_probability_hash=(
                        f"base-ensemble-{source}-{case_id}"
                    ),
                    tail_ensemble_probability_hash=(
                        f"tail-ensemble-{source}-{case_id}"
                    ),
                    ensemble_absolute_difference_hash=(
                        f"ensemble-difference-{source}-{case_id}"
                    ),
                )
            )
    return tuple(rows)


def _source_inner_action_shift_contracts(seed_rows):
    legacy = aggregate_candidate_seed_features(
        seed_rows, legacy_target_local_scalar_name="metadata_similarity"
    )
    contracts = {}
    for row in legacy:
        base_hashes = tuple(
            f"base-{row.query_id}-{row.candidate_source}-{index}"
            for index in range(9)
        )
        tail_hashes = tuple(
            f"tail-{row.query_id}-{row.candidate_source}-{index}"
            for index in range(9)
        )
        shifts = tuple(
            row.target_local_scalar + (index - 4) * 1.0e-4
            for index in range(9)
        )
        values = np.asarray(shifts, dtype=np.float64)
        value = float(np.mean(values, dtype=np.float64))
        standard_deviation = float(np.std(values, ddof=0, dtype=np.float64))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        base_ensemble_hash = f"base-ensemble-{row.query_id}-{row.candidate_source}"
        tail_ensemble_hash = f"tail-ensemble-{row.query_id}-{row.candidate_source}"
        difference_hash = f"difference-{row.query_id}-{row.candidate_source}"
        unhashed = {
            "schema_version": SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
            "row_identity_hash": f"support-{row.query_id}",
            "seed_pair_count": len(ENSEMBLE_SEED_KEYS),
            "seed_keys": [list(key) for key in ENSEMBLE_SEED_KEYS],
            "base_component_vector_hashes": list(base_hashes),
            "tail_component_vector_hashes": list(tail_hashes),
            "per_seed_mean_absolute_shifts": list(shifts),
            "technical_seed_spread_semantics": (
                SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
            ),
            "technical_seed_values_may_feed_model": False,
            "base_ensemble_probability_sha256": base_ensemble_hash,
            "tail_ensemble_probability_sha256": tail_ensemble_hash,
            "ensemble_absolute_difference_sha256": difference_hash,
            "value": value,
            "seed_standard_deviation": standard_deviation,
            "seed_minimum": minimum,
            "seed_maximum": maximum,
            "seed_range": maximum - minimum,
            "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
            "scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
            "labels_used": False,
        }
        contracts[row.row_key] = SupportActionProbabilityShift(
            row_identity_hash=f"support-{row.query_id}",
            seed_keys=ENSEMBLE_SEED_KEYS,
            base_component_vector_hashes=base_hashes,
            tail_component_vector_hashes=tail_hashes,
            per_seed_mean_absolute_shifts=shifts,
            base_ensemble_probability_hash=base_ensemble_hash,
            tail_ensemble_probability_hash=tail_ensemble_hash,
            ensemble_absolute_difference_hash=difference_hash,
            value=value,
            seed_standard_deviation=standard_deviation,
            seed_minimum=minimum,
            seed_maximum=maximum,
            seed_range=maximum - minimum,
            shift_hash=canonical_sha256(unhashed),
        )
    return contracts


def test_primary_endpoint_means_probabilities_not_seed_baccs() -> None:
    truth = np.asarray([0, 1], dtype=np.uint8)
    base = _probability_vectors(((0.4, 0.6),) * 9, action="base")
    tail = _probability_vectors(
        ((0.4, 0.6),) * 5 + ((0.9, 0.1),) * 4,
        action="tail",
    )

    endpoint = score_nine_seed_probability_ensemble(tail, truth)
    per_seed_bacc = [
        balanced_accuracy(
            truth.tolist(),
            (cell.positive_class_probabilities >= 0.5).astype(np.uint8).tolist(),
        )
        for cell in tail
    ]
    response = build_ensemble_utility_response(
        outer_target_id="H",
        query_id="q",
        candidate_source="e",
        base_vectors=base,
        tail_vectors=tail,
        labels=truth,
        support_partition_hash="support-q",
        evaluation_partition_hash="evaluation-q",
        prediction_seal_hash="seal-q",
        support_eval_disjoint=True,
        predictions_sealed_before_labels=True,
        source_expert_frozen=True,
    )

    assert np.mean(per_seed_bacc) == pytest.approx(5.0 / 9.0)
    assert endpoint.balanced_accuracy == 0.0
    assert response.utility_delta == -1.0
    assert response.utility_delta != pytest.approx(np.mean(per_seed_bacc) - 1.0)


def test_representative_endpoint_hashes_match_versioned_reference() -> None:
    base = tuple(
        SeedProbabilityVector(
            training_seed=training_seed,
            generation_seed=generation_seed,
            row_identity_hash="rows-v1",
            prediction_provenance_hash=(
                f"base-{training_seed}-{generation_seed}"
            ),
            positive_class_probabilities=np.asarray(
                [0.2 + 0.01 * index, 0.8 - 0.01 * index],
                dtype=np.float64,
            ),
        )
        for index, (training_seed, generation_seed) in enumerate(
            ENSEMBLE_SEED_KEYS
        )
    )
    tail = tuple(
        SeedProbabilityVector(
            training_seed=training_seed,
            generation_seed=generation_seed,
            row_identity_hash="rows-v1",
            prediction_provenance_hash=(
                f"tail-{training_seed}-{generation_seed}"
            ),
            positive_class_probabilities=np.asarray(
                [0.4 + 0.01 * index, 0.6 - 0.01 * index],
                dtype=np.float64,
            ),
        )
        for index, (training_seed, generation_seed) in enumerate(
            ENSEMBLE_SEED_KEYS
        )
    )
    response = build_ensemble_utility_response(
        outer_target_id="H",
        query_id="q",
        candidate_source="e",
        base_vectors=base,
        tail_vectors=tail,
        labels=np.asarray([0, 1], dtype=np.uint8),
        support_partition_hash="support-v1",
        evaluation_partition_hash="evaluation-v1",
        prediction_seal_hash="seal-v1",
        support_eval_disjoint=True,
        predictions_sealed_before_labels=True,
        source_expert_frozen=True,
    )
    shift = support_action_probability_shift(base, tail)

    assert {
        "first_vector": base[0].vector_hash,
        "base_endpoint": response.base_endpoint.endpoint_hash,
        "tail_endpoint": response.tail_endpoint.endpoint_hash,
        "response": response.row_hash,
        "support_shift": shift.shift_hash,
    } == {
        "first_vector": (
            "e3cdf9fb7be3997fa881863225d0a2d04893fa0a923dbc29b61d6b9d7a18815d"
        ),
        "base_endpoint": (
            "d813946d66d2dd6cd3811921941407fcfda758751cba2afa293ffb8f18037305"
        ),
        "tail_endpoint": (
            "f895b91fafb1383a40f3cdf002f0202033919e5694bb65cd997c4e289818338b"
        ),
        "response": (
            "08a192ac05d9e350de0873154adfc492e4e72225f8694f6a9f52e1a92e2965eb"
        ),
        "support_shift": (
            "1b88e927e2333866f6efb791c6f84a2c5fda4ba8b79a1844931ef8b2a5ea3030"
        ),
    }


def test_exact_nine_order_duplicate_geometry_and_range_fail_closed() -> None:
    cells = _probability_vectors(((0.25, 0.75),) * 9, action="action")
    observed = mean_exact_nine_positive_class_probabilities(cells)
    assert observed.dtype == np.float64
    assert not observed.flags.writeable
    assert observed.tolist() == [0.25, 0.75]

    with pytest.raises(ProtocolError, match="exactly nine"):
        mean_exact_nine_positive_class_probabilities(cells[:-1])
    with pytest.raises(ProtocolError, match="duplicate seed"):
        mean_exact_nine_positive_class_probabilities((cells[0], cells[0], *cells[2:]))
    with pytest.raises(ProtocolError, match="canonical"):
        mean_exact_nine_positive_class_probabilities(tuple(reversed(cells)))
    with pytest.raises(ProtocolError, match=r"\[0, 1\]"):
        SeedProbabilityVector(
            training_seed=17,
            generation_seed=17,
            row_identity_hash="rows",
            prediction_provenance_hash="bad",
            positive_class_probabilities=np.asarray([1.01]),
        )
    drifted = list(cells)
    drifted[-1] = replace(drifted[-1], row_identity_hash="different-rows")
    with pytest.raises(ProtocolError, match="row geometry"):
        mean_exact_nine_positive_class_probabilities(tuple(drifted))


def test_support_action_shift_is_label_free_and_keeps_seed_spread() -> None:
    base = _probability_vectors(((0.2, 0.8),) * 9, action="base-support")
    tail = _probability_vectors(
        tuple((0.2 + 0.01 * index, 0.8 - 0.01 * index) for index in range(9)),
        action="tail-support",
    )
    shift = support_action_probability_shift(base, tail)
    assert shift.value == pytest.approx(0.04)
    assert shift.seed_range == pytest.approx(0.08)
    assert shift.seed_standard_deviation > 0.0
    assert shift.to_payload()["labels_used"] is False

    with pytest.raises(ProtocolError, match="aggregate hash"):
        replace(shift, shift_hash="caller-supplied-untrusted-hash")
    with pytest.raises(ProtocolError, match="technical-seed bound"):
        replace(shift, value=shift.value + 0.1)
    with pytest.raises(ProtocolError, match="canonical exact-nine"):
        replace(shift, seed_keys=tuple(reversed(shift.seed_keys)))
    with pytest.raises(ProtocolError, match="exact-nine unique"):
        replace(
            shift,
            base_component_vector_hashes=(
                shift.base_component_vector_hashes[0],
            )
            * len(shift.base_component_vector_hashes),
        )


def test_support_action_shift_ensembles_before_absolute_value_and_matches_target_case() -> None:
    base = _probability_vectors(((0.5, 0.5),) * 9, action="base-cancel")
    tail_values = (
        ((0.9, 0.1),) * 4
        + ((0.1, 0.9),) * 4
        + ((0.5, 0.5),)
    )
    tail = _probability_vectors(tail_values, action="tail-cancel")

    source_inner_shift = support_action_probability_shift(base, tail)
    target_case = build_target_support_action_shift_case(
        target_id="H",
        candidate_source="d0",
        case_id="case-cancel",
        base_vectors=base,
        tail_vectors=tail,
    )

    assert np.mean(source_inner_shift.per_seed_mean_absolute_shifts) > 0.35
    assert source_inner_shift.value == pytest.approx(0.0, abs=2.0e-8)
    assert target_case.ensemble_mean_absolute_shift == pytest.approx(
        source_inner_shift.value
    )
    assert (
        target_case.ensemble_absolute_difference_hash
        == source_inner_shift.ensemble_absolute_difference_hash
    )
    assert target_case.per_seed_mean_absolute_shifts == (
        source_inner_shift.per_seed_mean_absolute_shifts
    )


def test_seed_aggregation_is_order_invariant_and_not_pseudo_replication() -> None:
    seed_rows = _candidate_seed_rows()
    forward = aggregate_candidate_seed_features(
        seed_rows, legacy_target_local_scalar_name="metadata_similarity"
    )
    reversed_rows = aggregate_candidate_seed_features(
        tuple(reversed(seed_rows)),
        legacy_target_local_scalar_name="metadata_similarity",
    )
    assert len(seed_rows) == 8 * 7 * 9
    assert len(forward) == 8 * 7
    assert [row.row_hash for row in forward] == [row.row_hash for row in reversed_rows]
    assert all(row.seed_pair_count == 9 for row in forward)
    assert all(
        row.feature_seed_standard_deviation_by_name["metadata_similarity"] > 0.0
        for row in forward
    )

    surface = build_ensemble_feature_surface(
        forward,
        global_source_control_by_source={
            source: float(index) / 7.0 for index, source in enumerate(SOURCES)
        },
        global_source_control_semantics="source_inner_global_reliability_control_v1",
        global_source_control_provenance_hash="global-control-lock",
    )
    assert surface.values.shape == (56, 2)
    assert surface.independent_query_count == 8
    assert len(surface.rows) != len(seed_rows)

    permuted_a = cyclically_permute_target_scalar(surface, permutation_seed=41)
    permuted_b = cyclically_permute_target_scalar(surface, permutation_seed=41)
    assert permuted_a.surface_hash == permuted_b.surface_hash
    assert np.array_equal(permuted_a.values[:, 0], surface.values[:, 0])
    assert not np.array_equal(permuted_a.values[:, 1], surface.values[:, 1])


def test_model_uses_strict_h_q_e_exclusion_group_tuning_and_capacity_gate() -> None:
    aggregated = aggregate_candidate_seed_features(
        _candidate_seed_rows(), legacy_target_local_scalar_name="metadata_similarity"
    )
    surface = build_ensemble_feature_surface(
        aggregated,
        global_source_control_by_source={
            source: float(index) / 7.0 for index, source in enumerate(SOURCES)
        },
        global_source_control_semantics="source_inner_global_reliability_control_v1",
        global_source_control_provenance_hash="global-control-lock",
    )
    utility = validate_ensemble_utility_responses(_scored_responses(aggregated))
    model = fit_ensemble_utility_model(surface, utility, alphas=(0.1,))

    assert model.routing_tuning_endpoint == "mean_normalized_oracle_regret"
    assert len(model.crossfit_predictions) == 56
    assert len(model.fold_audits) == 56
    for audit in model.fold_audits:
        outer, query, source = audit.predicted_row_key
        assert {outer, query, source} <= set(audit.excluded_domain_ids)
        assert not set(audit.excluded_domain_ids) & set(audit.training_query_ids)
        assert not set(audit.excluded_domain_ids) & set(audit.training_source_ids)
    assert all(report.gate_passed for report in model.candidate_capacity_reports.values())
    assert all(
        report.independent_query_count == 7
        and report.observation_count == 42
        and report.predictor_column_count == 2
        and report.design_rank == 3
        and report.sandwich_rank_ceiling == 3
        for report in model.candidate_capacity_reports.values()
    )

    over_capacity = replace(
        surface,
        feature_names=(*surface.feature_names, "unapproved_extra_scalar"),
        values=np.column_stack((surface.values, np.arange(len(surface.rows)))),
    )
    with pytest.raises(ProtocolError, match="M0/M1 capacity"):
        fit_ensemble_utility_model(over_capacity, utility, alphas=(0.1,))


def test_persisted_stage60_endpoint_row_coerces_without_raw_vectors() -> None:
    payload = {
        "schema_version": "midogpp_exact_tail_ensemble_endpoint_row_v1",
        "outer_target": "H",
        "pseudo_query": "q",
        "candidate_source": "e",
        "base_bacc": 0.61,
        "tail_bacc": 0.66,
        "delta_bacc": 0.05,
        "evaluation_row_count": 20,
        "evaluation_case_count": 10,
        "evaluation_row_hash": "evaluation-rows-q",
        "evaluation_label_sha256": "evaluation-labels-q",
        "support_partition_hash": "support-q",
        "prediction_seal_hash": "prediction-seal",
        "base_probability_cell_hashes_hash": "base-cell-hashes",
        "tail_probability_cell_hashes_hash": "tail-cell-hashes",
        "base_ensemble_probability_sha256": "base-ensemble-prob",
        "tail_ensemble_probability_sha256": "tail-ensemble-prob",
        "base_ensemble_prediction_sha256": "base-ensemble-pred",
        "tail_ensemble_prediction_sha256": "tail-ensemble-pred",
        "base_endpoint_hash": "base-endpoint",
        "tail_endpoint_hash": "tail-endpoint",
        "ensemble_utility_response_hash": "raw-response-hash",
        "seed_pair_count": 9,
        "seed_pairs_hash": "seed-pairs-hash",
        "threshold": 0.5,
        "primary_metric": "balanced_accuracy",
        "primary_utility_endpoint": "tail_minus_base_bacc",
        "aggregation_semantics": "mean_probabilities_then_threshold",
        "response_semantics": "ensemble_tail_minus_base",
        "endpoint_role": "source_inner_candidate_utility",
        "development_labels_used_for_scoring_only": True,
        "technical_seed_repeats_are_not_independent_units": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "seed_selection_performed": False,
        "endpoint_row_hash": "persisted-endpoint-row-hash",
    }
    response = scored_ensemble_utility_response_from_payload(payload)
    assert isinstance(response, ScoredEnsembleUtilityResponse)
    assert response.row_key == ("H", "q", "e")
    assert response.utility_delta == pytest.approx(0.05)
    assert response.source_response_hash == "raw-response-hash"
    assert response.source_endpoint_row_hash == "persisted-endpoint-row-hash"
    assert not hasattr(response, "base_endpoint")
    assert response.utility_semantics == ENSEMBLE_UTILITY_SEMANTICS


def test_label_free_global_control_and_whole_case_target_bootstraps() -> None:
    source_inner_rows = _candidate_seed_rows()
    control = derive_label_free_global_source_control(source_inner_rows)
    control_payload = control.to_payload()
    assert control_payload["labels_used"] is False
    assert control_payload["utility_responses_used"] is False
    assert control.source_inner_seed_row_count == 8 * 7 * 9
    for source in SOURCES:
        expected = np.mean(
            [
                row.metadata_similarity
                for row in source_inner_rows
                if row.candidate_source == source
            ]
        )
        assert control.value_by_source[source] == pytest.approx(expected)

    plan = build_case_bootstrap_plan(
        target_id=OUTER,
        support_case_ids=tuple(f"case-{index}" for index in range(8)),
        replicate_count=32,
    )
    case_shifts = _target_case_shifts(plan)
    production = build_target_ensemble_feature_surfaces(
        _target_seed_rows(plan),
        case_shifts,
        plan,
        global_source_control=control,
    )
    assert len(production.bootstrap_surfaces) == 32
    assert production.to_payload()["resampling_unit"] == "independent_target_support_case"
    assert production.to_payload()["labels_used"] is False
    assert production.to_payload()["utility_responses_used"] is False
    assert "utility" not in inspect.signature(
        build_target_ensemble_feature_surfaces
    ).parameters

    first_replicate = plan.replicates[0]
    first_surface = production.bootstrap_surfaces[0]
    first_row = next(row for row in first_surface.rows if row.candidate_source == "d0")
    shift_by_case = {
        row.case_id: row
        for row in case_shifts
        if row.candidate_source == "d0"
    }
    selected_cases = tuple(
        shift_by_case[case_id] for case_id in first_replicate.sampled_case_ids
    )
    weights = np.asarray(
        [row.support_row_count for row in selected_cases], dtype=np.float64
    )
    expected_seed_means = np.average(
        np.asarray(
            [row.per_seed_mean_absolute_shifts for row in selected_cases],
            dtype=np.float64,
        ),
        axis=0,
        weights=weights,
    )
    assert first_row.target_local_scalar == pytest.approx(
        float(
            np.average(
                np.asarray(
                    [row.ensemble_mean_absolute_shift for row in selected_cases],
                    dtype=np.float64,
                ),
                weights=weights,
            )
        )
    )
    assert first_row.target_local_scalar_seed_standard_deviation == pytest.approx(
        float(np.std(expected_seed_means, ddof=0))
    )
    assert (
        first_row.support_partition_hash == first_replicate.support_partition_hash
    )


def test_transfer_bootstrap_and_policy_require_positive_combined_gain_lcb() -> None:
    seed_rows = _candidate_seed_rows()
    control = derive_label_free_global_source_control(seed_rows)
    global_rows = aggregate_candidate_seed_features(seed_rows)
    routed_rows = aggregate_candidate_seed_features(
        seed_rows,
        support_action_shift_by_candidate=_source_inner_action_shift_contracts(
            seed_rows
        ),
    )

    def build_surface(rows):
        return build_ensemble_feature_surface(
            rows,
            global_source_control_by_source=control.value_by_source,
            global_source_control_semantics=control.semantics,
            global_source_control_provenance_hash=control.provenance_hash,
        )

    global_surface = build_surface(global_rows)
    routed_surface = build_surface(routed_rows)
    permutation_surface = cyclically_permute_target_scalar(
        routed_surface, permutation_seed=41
    )
    utility = validate_ensemble_utility_responses(_scored_responses(routed_rows))
    global_model = fit_ensemble_utility_model(
        global_surface, utility, alphas=(0.1,)
    )
    routed_model = fit_ensemble_utility_model(
        routed_surface, utility, alphas=(0.1,)
    )
    permutation_model = fit_ensemble_utility_model(
        permutation_surface, utility, alphas=(0.1,)
    )
    transfer = evaluate_ensemble_cardinality_transfer(
        global_model, routed_model, permutation_model, utility
    )
    repeated = evaluate_ensemble_cardinality_transfer(
        global_model, routed_model, permutation_model, utility
    )
    assert transfer.transfer_hash == repeated.transfer_hash
    assert transfer.query_bootstrap_draw_count == 10_000
    assert transfer.independent_query_count == 8
    assert transfer.query_bootstrap_indices_hash == repeated.query_bootstrap_indices_hash
    assert "top1_lower_bound" in transfer.bootstrap_bounds_by_role["R"]
    assert "normalized_gap_reduction_lower_bound" in (
        transfer.paired_improvement_bounds["G"]
    )

    plan = build_case_bootstrap_plan(
        target_id=OUTER,
        support_case_ids=tuple(f"case-{index}" for index in range(8)),
        replicate_count=32,
    )
    production = build_target_ensemble_feature_surfaces(
        _target_seed_rows(plan),
        _target_case_shifts(plan),
        plan,
        global_source_control=control,
    )
    policy = build_ensemble_utility_policy(
        global_model,
        routed_model,
        permutation_model,
        production.point_surface,
        production.bootstrap_surfaces,
        transfer,
    )
    payload = policy.to_payload()
    assert payload["bootstrap_dispersion_divided_by_seed_repeat_sqrt"] is False
    assert payload["authorization_uncertainty_components"] == [
        "model_covariance_and_residual",
        "independent_whole_case_bootstrap",
    ]
    assert payload["target_scalar_seed_spread_role"] == (
        "descriptive_only_non_decision"
    )
    assert payload["target_scalar_seed_spread_enters_combined_standard_error"] is False
    for role in ("G", "R", "P"):
        candidate = min(
            policy.role_prediction_by_source[role],
            key=lambda source: (
                -policy.role_prediction_by_source[role][source],
                source,
            ),
        )
        model_se = policy.role_model_standard_error_by_source[role][candidate]
        case_sd = policy.role_bootstrap_standard_deviation_by_source[role][candidate]
        combined = policy.role_combined_standard_error_by_source[role][candidate]
        assert combined**2 == pytest.approx(model_se**2 + case_sd**2)
        lower_bound = policy.role_lower_confidence_bound_by_source[role][candidate]
        if lower_bound <= 0.0:
            assert policy.role_selected_action[role] == "B"
            assert policy.role_selected_source[role] is None
    if not transfer.authorized_for_target_policy:
        assert policy.exact_b_fallback is True
        assert policy.selected_action_role == "B"
        assert policy.fallback_reason == (
            "source_inner_cardinality_or_capacity_gate_failed"
        )


def test_per_seed_support_shift_diagnostics_cannot_change_target_policy() -> None:
    seed_rows = _candidate_seed_rows()
    control = derive_label_free_global_source_control(seed_rows)
    global_rows = aggregate_candidate_seed_features(seed_rows)
    routed_rows = aggregate_candidate_seed_features(
        seed_rows,
        support_action_shift_by_candidate=_source_inner_action_shift_contracts(
            seed_rows
        ),
    )

    def build_surface(rows):
        return build_ensemble_feature_surface(
            rows,
            global_source_control_by_source=control.value_by_source,
            global_source_control_semantics=control.semantics,
            global_source_control_provenance_hash=control.provenance_hash,
        )

    global_surface = build_surface(global_rows)
    routed_surface = build_surface(routed_rows)
    permutation_surface = cyclically_permute_target_scalar(
        routed_surface, permutation_seed=41
    )
    utility = validate_ensemble_utility_responses(_scored_responses(routed_rows))
    global_model = fit_ensemble_utility_model(global_surface, utility, alphas=(0.1,))
    routed_model = fit_ensemble_utility_model(routed_surface, utility, alphas=(0.1,))
    permutation_model = fit_ensemble_utility_model(
        permutation_surface, utility, alphas=(0.1,)
    )
    transfer = evaluate_ensemble_cardinality_transfer(
        global_model, routed_model, permutation_model, utility
    )
    plan = build_case_bootstrap_plan(
        target_id=OUTER,
        support_case_ids=tuple(f"case-{index}" for index in range(8)),
        replicate_count=32,
    )
    original_shifts = _target_case_shifts(plan)
    perturbed_shifts = tuple(
        replace(
            row,
            per_seed_mean_absolute_shifts=tuple(
                0.8 + 0.02 * index for index in range(9)
            ),
        )
        for row in original_shifts
    )
    original = build_target_ensemble_feature_surfaces(
        _target_seed_rows(plan), original_shifts, plan, global_source_control=control
    )
    perturbed = build_target_ensemble_feature_surfaces(
        _target_seed_rows(plan), perturbed_shifts, plan, global_source_control=control
    )
    np.testing.assert_array_equal(original.point_surface.values, perturbed.point_surface.values)
    for left, right in zip(
        original.bootstrap_surfaces, perturbed.bootstrap_surfaces, strict=True
    ):
        np.testing.assert_array_equal(left.values, right.values)
    assert (
        original.point_surface.rows[0].target_local_scalar_seed_standard_deviation
        != perturbed.point_surface.rows[0].target_local_scalar_seed_standard_deviation
    )

    original_policy = build_ensemble_utility_policy(
        global_model,
        routed_model,
        permutation_model,
        original.point_surface,
        original.bootstrap_surfaces,
        transfer,
    )
    perturbed_policy = build_ensemble_utility_policy(
        global_model,
        routed_model,
        permutation_model,
        perturbed.point_surface,
        perturbed.bootstrap_surfaces,
        transfer,
    )
    assert (
        original_policy.role_target_scalar_seed_standard_deviation_by_source
        != perturbed_policy.role_target_scalar_seed_standard_deviation_by_source
    )
    for field_name in (
        "role_prediction_by_source",
        "role_model_standard_error_by_source",
        "role_bootstrap_standard_deviation_by_source",
        "role_combined_standard_error_by_source",
        "role_lower_confidence_bound_by_source",
        "role_selected_source",
        "role_selected_action",
        "selected_action_role",
        "selected_source",
        "exact_b_fallback",
        "fallback_reason",
    ):
        assert getattr(original_policy, field_name) == getattr(
            perturbed_policy, field_name
        )
