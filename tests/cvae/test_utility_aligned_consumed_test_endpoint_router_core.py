from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.contracts import (
    CENTERS,
    DEVELOPMENT_RESPONSE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER,
    EXPECTED_TARGET_ACTION_COUNT,
    EXPECTED_TERMINAL_SCORE_COUNT,
    PRIMARY_CONTRASTS,
    SEED_PAIRS,
    SUPPORT_BOOTSTRAP_REPLICATES,
    SUPPORT_BOOTSTRAP_SEED,
    candidate_sources,
    inner_candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.endpoint_adapter import (
    build_label_free_support_case_shift,
    score_sealed_probability_ensemble,
    validate_development_endpoint_responses,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.features import (
    build_source_inner_feature_surfaces,
    build_target_case_bootstrap_plan,
    build_target_feature_production,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.inference import (
    CenterBaccContrast,
    summarize_center_contrasts,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.models import (
    fit_endpoint_router_models,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.partitions import (
    LabelFreeCaseRow,
    build_consumed_test_partitions,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.policy import (
    build_target_policy,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned import (
    INNER_ROLE,
    TARGET_ROLE,
    CandidateFeatureRow,
    ScoredEnsembleUtilityResponse,
    SeedProbabilityVector,
    support_action_probability_shift,
)


OUTER = "0"
DEVELOPMENT_SEAL = "development-prediction-seal"


def _allocate(total: int, groups: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, groups)
    return tuple(quotient + (index < remainder) for index in range(groups))


def _label_free_rows() -> tuple[LabelFreeCaseRow, ...]:
    support_totals = {center: 320 for center in CENTERS}
    support_totals[CENTERS[-1]] = 342
    rows: list[LabelFreeCaseRow] = []
    ordinal = 0
    for center in CENTERS:
        case_count = EXPECTED_CASE_COUNTS_BY_CENTER[center]
        counts = (
            *_allocate(support_totals[center], 8),
            *_allocate(
                EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER[center],
                case_count - 8,
            ),
        )
        for case_index, row_count in enumerate(counts):
            case_id = f"center-{center}-case-{case_index:02d}"
            for within_case in range(row_count):
                rows.append(
                    LabelFreeCaseRow(
                        row_ordinal=ordinal,
                        manifest_row_index=ordinal,
                        evaluation_row_id=f"row-{ordinal:05d}-{within_case}",
                        case_id=case_id,
                        center=center,
                    )
                )
                ordinal += 1
    return tuple(rows)


def _vectors(
    values: tuple[tuple[float, ...], ...], *, row_hash: str, name: str
) -> tuple[SeedProbabilityVector, ...]:
    return tuple(
        SeedProbabilityVector(
            training_seed=training_seed,
            generation_seed=generation_seed,
            row_identity_hash=row_hash,
            prediction_provenance_hash=(
                f"{name}::{training_seed}::{generation_seed}"
            ),
            positive_class_probabilities=np.asarray(value, dtype=np.float32),
        )
        for (training_seed, generation_seed), value in zip(
            SEED_PAIRS, values, strict=True
        )
    )


def _shift(outer: str, query: str, source: str, amount: float):
    base = tuple((0.25 + 0.001 * index, 0.75) for index in range(9))
    tail = tuple(
        (left + amount, right - amount / 2.0)
        for left, right in base
    )
    row_hash = f"support::{outer}::{query}::{source}"
    return support_action_probability_shift(
        _vectors(base, row_hash=row_hash, name=f"base::{row_hash}"),
        _vectors(tail, row_hash=row_hash, name=f"tail::{row_hash}"),
    )


def _feature_row(
    *, role: str, outer: str, query: str, source: str, support_hash: str
) -> list[CandidateFeatureRow]:
    sources = candidate_sources(outer)
    source_index = sources.index(source)
    query_index = 0 if role == TARGET_ROLE else sources.index(query)
    output: list[CandidateFeatureRow] = []
    for seed_index, (training_seed, generation_seed) in enumerate(SEED_PAIRS):
        reconstruction = 0.5 + 0.02 * source_index + 0.004 * query_index
        kl = 0.2 + 0.01 * query_index + 0.003 * source_index
        output.append(
            CandidateFeatureRow(
                role=role,
                outer_target_id=outer,
                query_id=query,
                candidate_source=source,
                training_seed=training_seed,
                generation_seed=generation_seed,
                candidate_source_count=7 if role == INNER_ROLE else 8,
                support_partition_hash=support_hash,
                support_case_count=8,
                reconstruction_mean=reconstruction + seed_index * 1.0e-4,
                reconstruction_std=0.03,
                reconstruction_q25=reconstruction - 0.04,
                reconstruction_q50=reconstruction,
                reconstruction_q75=reconstruction + 0.04,
                kl_mean=kl,
                kl_std=0.02,
                kl_q25=kl - 0.02,
                kl_q50=kl,
                kl_q75=kl + 0.02,
                replica_disagreement=0.01 + seed_index * 1.0e-4,
                distribution_mmd=0.08 + 0.01 * abs(source_index - query_index),
                metadata_similarity=0.1 + 0.1 * source_index,
            )
        )
    return output


def _source_features():
    rows: list[CandidateFeatureRow] = []
    shifts = {}
    sources = candidate_sources(OUTER)
    for query_index, query in enumerate(sources):
        for source_index, source in enumerate(inner_candidate_sources(OUTER, query)):
            support_hash = f"support::{OUTER}::{query}"
            rows.extend(
                _feature_row(
                    role=INNER_ROLE,
                    outer=OUTER,
                    query=query,
                    source=source,
                    support_hash=support_hash,
                )
            )
            shifts[(OUTER, query, source)] = _shift(
                OUTER,
                query,
                source,
                0.003 + 0.001 * ((2 * query_index + source_index) % 6),
            )
    return build_source_inner_feature_surfaces(
        rows,
        support_action_shift_by_candidate=shifts,
        outer_target_id=OUTER,
        feature_input_seal_hash="feature-input-seal",
    )


def _development_responses(*, poison_center_zero_labels: bool):
    rows: list[ScoredEnsembleUtilityResponse] = []
    for outer in CENTERS:
        queries = candidate_sources(outer)
        for query_index, query in enumerate(queries):
            for source_index, source in enumerate(
                inner_candidate_sources(outer, query)
            ):
                delta = 0.005 + 0.002 * query_index + 0.001 * source_index
                tail_bacc = 0.55 + delta
                label_hash = f"labels::{query}"
                if poison_center_zero_labels and query == OUTER and outer != OUTER:
                    tail_bacc = 0.15 + 0.01 * source_index
                    label_hash = f"poisoned-labels::{query}"
                rows.append(
                    ScoredEnsembleUtilityResponse(
                        outer_target_id=outer,
                        query_id=query,
                        candidate_source=source,
                        candidate_source_count=7,
                        support_partition_hash=f"support::{outer}::{query}",
                        evaluation_partition_hash=f"evaluation::{query}",
                        prediction_seal_hash=DEVELOPMENT_SEAL,
                        evaluation_row_identity_hash=f"rows::{query}",
                        evaluation_label_hash=label_hash,
                        base_endpoint_hash=f"base-endpoint::{outer}::{query}",
                        tail_endpoint_hash=f"tail-endpoint::{outer}::{query}::{source}",
                        base_probability_cell_hashes_hash=f"base-cells::{outer}::{query}",
                        tail_probability_cell_hashes_hash=f"tail-cells::{outer}::{query}::{source}",
                        base_ensemble_probability_hash=f"base-prob::{outer}::{query}",
                        tail_ensemble_probability_hash=f"tail-prob::{outer}::{query}::{source}",
                        base_ensemble_prediction_hash=f"base-pred::{outer}::{query}",
                        tail_ensemble_prediction_hash=f"tail-pred::{outer}::{query}::{source}",
                        source_response_hash=None,
                        source_endpoint_row_hash=None,
                        base_component_vector_hashes=tuple(
                            f"base::{outer}::{query}::{index}" for index in range(9)
                        ),
                        tail_component_vector_hashes=tuple(
                            f"tail::{outer}::{query}::{source}::{index}"
                            for index in range(9)
                        ),
                        base_bacc=0.55,
                        tail_bacc=tail_bacc,
                        support_eval_disjoint=True,
                        predictions_sealed_before_labels=True,
                        source_expert_frozen=True,
                    )
                )
    return validate_development_endpoint_responses(
        rows, development_prediction_seal_hash=DEVELOPMENT_SEAL
    )


def _target_features(source_features):
    case_ids = tuple(f"support-case-{index}" for index in range(8))
    plan = build_target_case_bootstrap_plan(
        target_id=OUTER, support_case_ids=case_ids
    )
    seed_rows: list[CandidateFeatureRow] = []
    case_shifts = []
    for source_index, source in enumerate(candidate_sources(OUTER)):
        seed_rows.extend(
            _feature_row(
                role=TARGET_ROLE,
                outer=OUTER,
                query=OUTER,
                source=source,
                support_hash=plan.support_partition_hash,
            )
        )
        for case_index, case_id in enumerate(case_ids):
            amount = 0.003 + 0.001 * source_index + 0.0002 * case_index
            base = tuple((0.25, 0.75) for _ in range(9))
            tail = tuple((0.25 + amount, 0.75 - amount) for _ in range(9))
            row_hash = f"case::{case_id}::{source}"
            case_shifts.append(
                build_label_free_support_case_shift(
                    target_id=OUTER,
                    candidate_source=source,
                    case_id=case_id,
                    base_vectors=_vectors(
                        base, row_hash=row_hash, name=f"base::{row_hash}"
                    ),
                    tail_vectors=_vectors(
                        tail, row_hash=row_hash, name=f"tail::{row_hash}"
                    ),
                )
            )
    return build_target_feature_production(
        seed_rows,
        case_shifts,
        source_features=source_features,
        case_bootstrap_plan=plan,
        support_partition_lock_hash="support-partition-lock",
        target_feature_seal_hash="target-feature-seal",
    )


def _contrast(
    target: str, contrast_id: str, left: str, right: str, delta: float
) -> CenterBaccContrast:
    payload = {
        "schema_version": "midogpp_consumed_test_center_bacc_contrast_v1",
        "target_center": target,
        "contrast_id": contrast_id,
        "left_action_id": left,
        "right_action_id": right,
        "left_bacc": 0.6 + delta,
        "right_bacc": 0.6,
        "paired_bacc_delta": delta,
        "score_set_hash": "score-set",
        "inference_unit": "target_center",
        "terminal_scores_may_update_plan": False,
        "consumed_test_diagnostic_only": True,
    }
    return CenterBaccContrast(
        target_id=target,
        contrast_id=contrast_id,
        left_action_id=left,
        right_action_id=right,
        left_bacc=0.6 + delta,
        right_bacc=0.6,
        paired_bacc_delta=delta,
        score_set_hash="score-set",
        contrast_hash=canonical_sha256(payload),
    )


def test_frozen_counts_and_label_free_lexical_partition() -> None:
    assert DEVELOPMENT_RESPONSE_COUNT == 504
    assert SUPPORT_BOOTSTRAP_SEED == 90703
    assert SUPPORT_BOOTSTRAP_REPLICATES == 32
    assert EXPECTED_TARGET_ACTION_COUNT == 13
    assert EXPECTED_TERMINAL_SCORE_COUNT == 117
    surface = build_consumed_test_partitions(tuple(reversed(_label_free_rows())))
    assert sum(len(rows) for rows in surface.support_rows_by_center.values()) == 2902
    assert sum(len(rows) for rows in surface.evaluation_rows_by_center.values()) == 7026
    assert sum(len(part.support_case_ids) for part in surface.by_center.values()) == 72
    assert sum(len(part.evaluation_case_ids) for part in surface.by_center.values()) == 146
    for center in CENTERS:
        part = surface.by_center[center]
        assert part.support_case_ids == tuple(
            f"center-{center}-case-{index:02d}" for index in range(8)
        )
        assert len(part.evaluation_case_ids) == EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER[center]
        assert set(part.support_case_ids).isdisjoint(part.evaluation_case_ids)
    assert surface.lock_payload["membership_seed"] is None
    assert surface.lock_payload["labels_used"] is False


def test_exact_nine_endpoint_means_probabilities_before_one_threshold() -> None:
    values = tuple((0.51, 0.51) if index < 5 else (0.0, 1.0) for index in range(9))
    vectors = _vectors(values, row_hash="evaluation-rows", name="B")
    endpoint = score_sealed_probability_ensemble(vectors, [0, 1])
    majority_vote = (
        np.mean(
            np.stack(
                [vector.positive_class_probabilities >= 0.5 for vector in vectors]
            ),
            axis=0,
        )
        >= 0.5
    ).astype(np.uint8)
    assert endpoint.predictions.tolist() == [0, 1]
    assert endpoint.balanced_accuracy == pytest.approx(1.0)
    assert majority_vote.tolist() == [1, 1]
    with pytest.raises(ProtocolError, match="exactly nine"):
        score_sealed_probability_ensemble(vectors[:-1], [0, 1])


def test_same_H_evaluation_label_poison_leaves_H_model_and_plan_unchanged() -> None:
    source_features = _source_features()
    base = _development_responses(poison_center_zero_labels=False)
    poisoned = _development_responses(poison_center_zero_labels=True)
    assert base.response_set_hash != poisoned.response_set_hash
    assert base.binding_hash_for_outer_target(OUTER) == poisoned.binding_hash_for_outer_target(OUTER)
    models = fit_endpoint_router_models(source_features, base, alphas=(1.0,))
    poisoned_models = fit_endpoint_router_models(
        source_features, poisoned, alphas=(1.0,)
    )
    assert models.model_hash == poisoned_models.model_hash
    assert models.global_model.model_hash == poisoned_models.global_model.model_hash
    assert models.routed_model.model_hash == poisoned_models.routed_model.model_hash
    target_features = _target_features(source_features)
    plan = build_target_policy(
        models, target_features, target_policy_seal_hash="policy-seal"
    )
    poisoned_plan = build_target_policy(
        poisoned_models, target_features, target_policy_seal_hash="policy-seal"
    )
    assert plan.policy_hash == poisoned_plan.policy_hash
    assert plan.to_payload()["same_outer_H_evaluation_labels_used"] is False


def test_aggregate_center_t_summaries_are_deterministic_and_hash_bound() -> None:
    delta_by_target = dict(zip(CENTERS, np.linspace(-0.04, 0.04, len(CENTERS))))
    rows = tuple(
        _contrast(target, contrast_id, left, right, float(delta_by_target[target]))
        for target in CENTERS
        for contrast_id, left, right in PRIMARY_CONTRASTS
    )
    first = summarize_center_contrasts(rows, score_set_hash="score-set")
    second = summarize_center_contrasts(rows, score_set_hash="score-set")
    assert tuple(row.to_payload() for row in first) == tuple(
        row.to_payload() for row in second
    )
    assert len(first) == 4
    for row in first:
        assert row.center_count == 9
        assert row.degrees_of_freedom == 8
        assert row.equal_center_mean_delta == pytest.approx(0.0, abs=1.0e-15)
        assert row.standard_error > 0.0
        assert row.two_sided_ci95_lower < 0.0 < row.two_sided_ci95_upper
        assert row.one_sided_lcb95 < 0.0
        assert row.two_sided_p_value == pytest.approx(1.0)
        with pytest.raises(ProtocolError, match="boundary drifted"):
            replace(row, one_sided_lcb95=0.0)


def test_all_fitting_and_policy_apis_exclude_target_label_parameters() -> None:
    prohibited = {"label", "labels", "bacc", "oracle", "target_utility"}
    for callable_ in (
        build_source_inner_feature_surfaces,
        build_target_feature_production,
        fit_endpoint_router_models,
        build_target_policy,
    ):
        assert not prohibited.intersection(inspect.signature(callable_).parameters)
