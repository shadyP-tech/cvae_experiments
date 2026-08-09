from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.actions import (
    build_target_ensemble_endpoint_actions,
    inner_action_library_for,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.config import (
    canonical_claim_boundary_payload,
    canonical_model_payload,
    canonical_protocol_payload,
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT,
    EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
    INPUT_ARTIFACT_IDS,
    ROUTED_ENSEMBLE_ACTION_ID,
    ROUTING_STATUS,
    SEED_PAIRS,
    candidate_sources,
    inner_candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.diagnostic_plan import (
    build_ensemble_endpoint_diagnostic_plan,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.endpoint_scoring import (
    build_support_action_shift,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.features import (
    build_heldout_ensemble_feature_surfaces,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.modeling import (
    fit_stage90_heldout_ensemble_models,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.scoring import (
    score_target_action_ensemble_endpoint,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.utility_aligned import (
    INNER_ROLE,
    TARGET_ROLE,
    CandidateFeatureRow,
    ScoredEnsembleUtilityResponse,
    SeedProbabilityVector,
    score_nine_seed_probability_ensemble,
)


OUTER = "0"
LOCK = "a" * 64
TARGET_SHIFT_LOCK = "b" * 64
PROBE_SEAL = "c" * 64


def _vectors(
    values: tuple[tuple[float, ...], ...],
    *,
    row_hash: str,
    action: str,
) -> tuple[SeedProbabilityVector, ...]:
    return tuple(
        SeedProbabilityVector(
            training_seed=training_seed,
            generation_seed=generation_seed,
            row_identity_hash=row_hash,
            prediction_provenance_hash=(
                f"{action}::{training_seed}::{generation_seed}::{PROBE_SEAL}"
            ),
            positive_class_probabilities=np.asarray(value, dtype=np.float32),
        )
        for (training_seed, generation_seed), value in zip(
            SEED_PAIRS, values, strict=True
        )
    )


def _shift(key: tuple[str, str, str], amount: float):
    base_values = tuple(
        (0.20 + 0.001 * index, 0.70 - 0.001 * index)
        for index in range(9)
    )
    tail_values = tuple(
        (
            base[0] + amount + 0.0001 * index,
            base[1] - amount - 0.0001 * index,
        )
        for index, base in enumerate(base_values)
    )
    identity = "support::" + "::".join(key)
    return build_support_action_shift(
        _vectors(base_values, row_hash=identity, action=f"base::{key}"),
        _vectors(tail_values, row_hash=identity, action=f"tail::{key}"),
    )


def _feature_row(
    *,
    role: str,
    query: str,
    source: str,
    training_seed: int,
    generation_seed: int,
) -> CandidateFeatureRow:
    sources = candidate_sources(OUTER)
    source_index = sources.index(source)
    query_index = 0 if role == TARGET_ROLE else sources.index(query)
    seed_index = SEED_PAIRS.index((training_seed, generation_seed))
    reconstruction = (
        0.65
        + 0.012 * source_index
        + 0.003 * query_index
        + 0.0001 * seed_index
    )
    kl = 0.20 + 0.006 * query_index + 0.002 * source_index
    return CandidateFeatureRow(
        role=role,
        outer_target_id=OUTER,
        query_id=query,
        candidate_source=source,
        training_seed=training_seed,
        generation_seed=generation_seed,
        candidate_source_count=7 if role == INNER_ROLE else 8,
        support_partition_hash=(
            f"support::{OUTER}::{query}"
            if role == INNER_ROLE
            else f"support::{OUTER}::target"
        ),
        support_case_count=2,
        reconstruction_mean=reconstruction,
        reconstruction_std=0.03 + 0.0001 * seed_index,
        reconstruction_q25=reconstruction - 0.04,
        reconstruction_q50=reconstruction,
        reconstruction_q75=reconstruction + 0.04,
        kl_mean=kl,
        kl_std=0.02,
        kl_q25=kl - 0.02,
        kl_q50=kl,
        kl_q75=kl + 0.02,
        replica_disagreement=0.01 + 0.0001 * seed_index,
        distribution_mmd=0.1 + 0.01 * abs(source_index - query_index),
        metadata_similarity=0.1 + 0.1 * source_index,
    )


def _heldout_inputs():
    sources = candidate_sources(OUTER)
    inner_rows: list[CandidateFeatureRow] = []
    target_rows: list[CandidateFeatureRow] = []
    inner_shifts = {}
    target_shifts = {}
    for query in sources:
        query_index = sources.index(query)
        for source in inner_candidate_sources(OUTER, query):
            source_index = sources.index(source)
            key = (OUTER, query, source)
            inner_shifts[key] = _shift(
                key,
                0.005
                + 0.001 * ((2 * query_index + 3 * source_index) % 7),
            )
            inner_rows.extend(
                _feature_row(
                    role=INNER_ROLE,
                    query=query,
                    source=source,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                )
                for training_seed, generation_seed in SEED_PAIRS
            )
    for source in sources:
        source_index = sources.index(source)
        key = (OUTER, OUTER, source)
        target_shifts[key] = _shift(key, 0.006 + 0.0015 * source_index)
        target_rows.extend(
            _feature_row(
                role=TARGET_ROLE,
                query=OUTER,
                source=source,
                training_seed=training_seed,
                generation_seed=generation_seed,
            )
            for training_seed, generation_seed in SEED_PAIRS
        )
    return tuple(inner_rows), tuple(target_rows), inner_shifts, target_shifts


def _heldout_utility() -> tuple[ScoredEnsembleUtilityResponse, ...]:
    sources = candidate_sources(OUTER)
    rows: list[ScoredEnsembleUtilityResponse] = []
    for query in sources:
        query_index = sources.index(query)
        for source in inner_candidate_sources(OUTER, query):
            source_index = sources.index(source)
            delta = (
                0.01
                + 0.003 * source_index
                + 0.004 * ((2 * query_index + 3 * source_index) % 7)
            )
            rows.append(
                ScoredEnsembleUtilityResponse(
                    outer_target_id=OUTER,
                    query_id=query,
                    candidate_source=source,
                    candidate_source_count=7,
                    support_partition_hash=f"support::{OUTER}::{query}",
                    evaluation_partition_hash=f"evaluation::{OUTER}::{query}",
                    prediction_seal_hash=f"development-seal::{OUTER}::{query}",
                    evaluation_row_identity_hash=f"evaluation-rows::{query}",
                    evaluation_label_hash=f"evaluation-labels::{query}",
                    base_endpoint_hash=f"base-endpoint::{query}",
                    tail_endpoint_hash=f"tail-endpoint::{query}::{source}",
                    base_probability_cell_hashes_hash=f"base-cells::{query}",
                    tail_probability_cell_hashes_hash=f"tail-cells::{query}::{source}",
                    base_ensemble_probability_hash=f"base-prob::{query}",
                    tail_ensemble_probability_hash=f"tail-prob::{query}::{source}",
                    base_ensemble_prediction_hash=f"base-pred::{query}",
                    tail_ensemble_prediction_hash=f"tail-pred::{query}::{source}",
                    source_response_hash=None,
                    source_endpoint_row_hash=None,
                    base_component_vector_hashes=tuple(
                        f"base::{query}::{index}" for index in range(9)
                    ),
                    tail_component_vector_hashes=tuple(
                        f"tail::{query}::{source}::{index}" for index in range(9)
                    ),
                    base_bacc=0.55,
                    tail_bacc=0.55 + delta,
                    support_eval_disjoint=True,
                    predictions_sealed_before_labels=True,
                    source_expert_frozen=True,
                )
            )
    return tuple(rows)


def test_frozen_stage90_ensemble_contract_and_workstation_counts() -> None:
    assert EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT == 504
    assert EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT == 4536
    assert len(INPUT_ARTIFACT_IDS) == 5
    assert all("exact_tail_router" not in value for value in INPUT_ARTIFACT_IDS)
    assert all("stage60" not in value and "stage70" not in value for value in INPUT_ARTIFACT_IDS)
    protocol = canonical_protocol_payload()
    model = canonical_model_payload()
    runtime = canonical_runtime_payload()
    claim = canonical_claim_boundary_payload()
    assert protocol["primary_development_response_count"] == 504
    assert protocol["descriptive_per_seed_rows_may_feed_model"] is False
    assert protocol["target_local_scalar_is_ensemble_first"] is True
    assert model["per_seed_utility_rows_may_feed_model"] is False
    assert runtime["target_unique_classifier_fit_count"] == 810
    assert runtime["maximum_total_classifier_fit_count"] == 5994
    assert claim["may_update_policy"] is False


def test_exact_nine_endpoint_means_probabilities_then_thresholds_once() -> None:
    values = tuple(
        (0.51, 0.51) if index < 5 else (0.0, 1.0) for index in range(9)
    )
    vectors = _vectors(values, row_hash="evaluation-rows", action="B")
    endpoint = score_nine_seed_probability_ensemble(vectors, [0, 1])
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
        score_nine_seed_probability_ensemble(vectors[:-1], [0, 1])


def test_support_shift_is_ensemble_first_and_seed_values_are_descriptive_only() -> None:
    base_values = tuple((float(index >= 5),) for index in range(9))
    tail_values = tuple((1.0 - value[0],) for value in base_values)
    shift = build_support_action_shift(
        _vectors(base_values, row_hash="support-row", action="base"),
        _vectors(tail_values, row_hash="support-row", action="tail"),
    )
    assert shift.value == pytest.approx(1.0 / 9.0)
    assert np.mean(shift.per_seed_mean_absolute_shifts) == pytest.approx(1.0)
    assert shift.to_payload()["technical_seed_values_may_feed_model"] is False


@pytest.fixture(scope="module")
def heldout_bundle():
    inner, target, inner_shifts, target_shifts = _heldout_inputs()
    features = build_heldout_ensemble_feature_surfaces(
        inner,
        target,
        inner_support_shift_by_candidate=inner_shifts,
        target_support_shift_by_candidate=target_shifts,
        inner_support_shift_lock_hash=LOCK,
        target_support_shift_lock_hash=TARGET_SHIFT_LOCK,
        target_probe_seal_hash=PROBE_SEAL,
        outer_target_id=OUTER,
    )
    models = fit_stage90_heldout_ensemble_models(
        features, _heldout_utility(), alphas=(1.0,)
    )
    plan = build_ensemble_endpoint_diagnostic_plan(models, features)
    return features, models, plan


def test_features_fit_only_56_candidate_responses_with_strict_H_q_e(heldout_bundle) -> None:
    features, models, _plan = heldout_bundle
    assert len(features.inner_m1.rows) == 56
    assert len(features.target_m1.rows) == 8
    assert features.target_probe_seal_hash == PROBE_SEAL
    assert features.inner_m1.feature_names == (
        "global_source_control",
        "target_local::mean_support_row_absolute_exact_nine_ensemble_probability_shift_v2",
    )
    for model in (models.global_model, models.routed_model, models.permutation_model):
        assert len(model.crossfit_row_keys) == 56
        for audit in model.fold_audits:
            outer, query, source = audit.predicted_row_key
            assert {outer, query, source}.issubset(audit.excluded_domain_ids)
            assert not {outer, query, source}.intersection(audit.training_query_ids)
            assert not {outer, query, source}.intersection(audit.training_source_ids)
    model_payload = models.to_payload()
    assert model_payload["inner_support_shift_lock_hash"] == LOCK
    assert model_payload["target_support_shift_lock_hash"] == TARGET_SHIFT_LOCK
    assert model_payload["target_probe_seal_hash"] == PROBE_SEAL


def test_two_case_plan_is_nonactionable_and_probe_locked(heldout_bundle) -> None:
    _features, _models, plan = heldout_bundle
    assert plan.routing_status == ROUTING_STATUS
    assert plan.support_case_count == 2
    assert plan.target_probe_seal_hash == PROBE_SEAL
    assert plan.may_update_policy is False
    assert plan.policy_authorized is False
    assert plan.fallback_authorized is False
    assert plan.promotion_authorized is False
    assert plan.deployment_authorized is False
    assert tuple(plan.proposed_source_by_router) == ("G_delta", "R2E", "P")
    assert plan.to_payload()["technical_seed_spread_values_may_feed_plan"] is False
    with pytest.raises(ProtocolError):
        replace(plan, may_update_policy=True)


def test_action_menu_is_modular_and_router_aliases_reuse_Hxe_geometry(heldout_bundle) -> None:
    _features, _models, plan = heldout_bundle
    inner = inner_action_library_for(OUTER, "1")
    target = build_target_ensemble_endpoint_actions(plan)
    assert len(inner) == 8
    assert inner[0].action_id == BASE_ACTION_ID
    assert len(target) == 13
    assert tuple(action.action_id for action in target[:5]) == (
        "B",
        "U",
        "G_delta",
        "R2E",
        "P",
    )
    routed = next(action for action in target if action.action_id == ROUTED_ENSEMBLE_ACTION_ID)
    selected_tail = next(
        action
        for action in target
        if action.action_id == f"Hxe::{routed.selected_source}"
    )
    assert routed.final_counts_by_class == selected_tail.final_counts_by_class
    assert routed.action_hash != selected_tail.action_hash
    routed_payload = routed.to_payload()
    assert routed_payload["may_update_policy"] is False
    assert routed_payload["inner_support_shift_lock_hash"] == LOCK
    assert routed_payload["target_support_shift_lock_hash"] == TARGET_SHIFT_LOCK
    assert routed_payload["target_probe_seal_hash"] == PROBE_SEAL


def test_terminal_scoring_requires_global_seal_and_exact_nine(heldout_bundle) -> None:
    _features, _models, plan = heldout_bundle
    action = next(
        value
        for value in build_target_ensemble_endpoint_actions(plan)
        if value.action_id == ROUTED_ENSEMBLE_ACTION_ID
    )
    values = tuple((0.2, 0.8) for _ in range(9))
    vectors = _vectors(values, row_hash="target-evaluation-rows", action="R2E")
    with pytest.raises(ProtocolError, match="global target seal"):
        score_target_action_ensemble_endpoint(
            target_id=OUTER,
            action_id=action.action_id,
            vectors=vectors,
            labels=[0, 1],
            action_hash=action.action_hash,
            router_plan_hash=plan.plan_hash,
            support_partition_hash="support-partition-hash",
            evaluation_partition_hash="evaluation-partition-hash",
            prediction_seal_hash="prediction-seal-hash",
            target_probe_seal_hash=PROBE_SEAL,
            evaluation_case_count=10,
            global_target_seal_verified=False,
        )
    score = score_target_action_ensemble_endpoint(
        target_id=OUTER,
        action_id=action.action_id,
        vectors=vectors,
        labels=[0, 1],
        action_hash=action.action_hash,
        router_plan_hash=plan.plan_hash,
        support_partition_hash="support-partition-hash",
        evaluation_partition_hash="evaluation-partition-hash",
        prediction_seal_hash="prediction-seal-hash",
        target_probe_seal_hash=PROBE_SEAL,
        evaluation_case_count=10,
        global_target_seal_verified=True,
    )
    assert score.seed_pair_count == 9
    assert score.to_payload()["terminal_target_labels_used_for_plan"] is False


def test_label_free_feature_api_has_no_label_or_utility_parameter() -> None:
    prohibited = {"label", "labels", "utility", "bacc", "oracle"}
    signature = inspect.signature(build_heldout_ensemble_feature_surfaces)
    assert not prohibited.intersection(signature.parameters)
