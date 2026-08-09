"""Post-seal exact response and isolated smooth-description production."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    EXACT_BACC_DELTA,
    EXACT_FAMILY_IDS,
    SMOOTH_BACC_DELTA,
    SMOOTH_DESCRIPTIVE_FAMILY_IDS,
    expected_row_keys,
)
from .execution_adapter import BASE_ACTION_ID, h_x_e_action_id
from .experiment_contracts import (
    CENTERS,
    EXPECTED_DESCRIPTIVE_SEED_ROW_COUNT,
    EXPECTED_RESPONSE_ROW_COUNT,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from .feature_production import FixedBankFeatureProduction
from .input_contracts import row_identity_hash
from .model_contracts import ExactCrossfitResult, SmoothCrossfitResult
from .row_contracts import FixedBankResponseRow
from .serialization import canonical_array_hash, canonical_hash


SEED_PAIRS = tuple(
    (training_seed, generation_seed)
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
)
_HASH_TOKEN = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")


@dataclass(frozen=True)
class FixedBankResponseProduction:
    rows: tuple[FixedBankResponseRow, ...]
    descriptive_seed_rows: tuple[Mapping[str, object], ...]
    response_surface_hash: str

    def __post_init__(self) -> None:
        if (
            tuple(row.row_key for row in self.rows) != expected_row_keys()
            or len(self.rows) != EXPECTED_RESPONSE_ROW_COUNT
            or len(self.descriptive_seed_rows) != EXPECTED_DESCRIPTIVE_SEED_ROW_COUNT
            or self.response_surface_hash != _response_surface_hash(self.rows)
        ):
            raise ProtocolError("Fixed-bank response surface drifted.")


def produce_fixed_bank_responses(
    features: FixedBankFeatureProduction,
    feature_lock: Mapping[str, object],
    development: object,
    labels: object,
    partitions: object,
) -> FixedBankResponseProduction:
    if not isinstance(features, FixedBankFeatureProduction):
        raise ProtocolError("Responses require the experiment-owned feature surface.")
    store, prediction_seal_hash = _validated_scoring_capability(
        development, labels=labels, partitions=partitions
    )
    feature_lock_hash = _validated_feature_lock(
        features,
        feature_lock,
        expected_prediction_seal_hash=prediction_seal_hash,
        expected_partition_lock_hash=str(getattr(partitions, "lock_hash", "")),
    )
    evaluation = getattr(partitions, "evaluation_rows_by_center", None)
    if not isinstance(evaluation, Mapping) or tuple(evaluation) != CENTERS:
        raise ProtocolError("Fixed-bank evaluation partitions are absent.")
    vector_cache: dict[tuple[str, str], tuple[object, ...]] = {}
    response_rows: list[FixedBankResponseRow] = []
    seed_rows: list[Mapping[str, object]] = []
    for feature in features.rows:
        outer, query, source = feature.row_key
        scope = f"{outer}::{query}"
        evaluation_rows = tuple(evaluation[query])
        evaluation_row_hash = row_identity_hash(evaluation_rows)
        truth = np.asarray(labels.labels_by_center[query], dtype=np.uint8)
        if (
            labels.evaluation_row_hash_by_center[query] != evaluation_row_hash
            or truth.shape != (len(evaluation_rows),)
            or set(truth.tolist()) != {0, 1}
        ):
            raise ProtocolError("Opened labels drifted from sealed evaluation rows.")
        base_vectors = _evaluation_vectors(
            store,
            scope=scope,
            action_id=BASE_ACTION_ID,
            expected_row_hash=evaluation_row_hash,
            cache=vector_cache,
        )
        tail_vectors = _evaluation_vectors(
            store,
            scope=scope,
            action_id=h_x_e_action_id(source),
            expected_row_hash=evaluation_row_hash,
            cache=vector_cache,
        )
        base_matrix = _probability_matrix(base_vectors)
        tail_matrix = _probability_matrix(tail_vectors)
        base_mean = np.mean(base_matrix, axis=0, dtype=np.float64)
        tail_mean = np.mean(tail_matrix, axis=0, dtype=np.float64)
        exact_base = balanced_accuracy(
            truth, (base_mean >= 0.5).astype(np.uint8)
        )
        exact_tail = balanced_accuracy(
            truth, (tail_mean >= 0.5).astype(np.uint8)
        )
        smooth_base = soft_balanced_accuracy(truth, base_mean)
        smooth_tail = soft_balanced_accuracy(truth, tail_mean)
        source_response_hash = canonical_hash(
            {
                "schema_version": "midogpp_fixed_bank_source_response_row_v1",
                "row_key": [outer, query, source],
                "evaluation_row_hash": evaluation_row_hash,
                "evaluation_label_sha256": canonical_array_hash(truth),
                "prediction_seal_hash": prediction_seal_hash,
                "fixed_bank_feature_lock_hash": feature_lock_hash,
                "base_vector_hashes": [str(value.vector_hash) for value in base_vectors],
                "tail_vector_hashes": [str(value.vector_hash) for value in tail_vectors],
                "probabilities_averaged_before_single_threshold": True,
                "exact_response_is_primary": True,
                "smooth_response_is_descriptive_only": True,
            }
        )
        response_rows.append(
            FixedBankResponseRow(
                outer_target_id=outer,
                query_id=query,
                candidate_source=source,
                feature_row_hash=feature.feature_row_hash,
                source_response_row_hash=source_response_hash,
                exact_bacc_delta=exact_tail - exact_base,
                smooth_bacc_delta=smooth_tail - smooth_base,
            )
        )
        seed_rows.extend(
            _descriptive_seed_rows(
                feature.row_key,
                base_vectors=base_vectors,
                tail_vectors=tail_vectors,
                labels=truth,
                evaluation_row_hash=evaluation_row_hash,
                feature_surface_hash=features.feature_surface_hash,
                feature_lock_hash=feature_lock_hash,
                prediction_seal_hash=prediction_seal_hash,
            )
        )
    frozen = tuple(response_rows)
    return FixedBankResponseProduction(
        rows=frozen,
        descriptive_seed_rows=tuple(seed_rows),
        response_surface_hash=_response_surface_hash(frozen),
    )


def build_fixed_bank_response_lock(
    responses: FixedBankResponseProduction,
    *,
    feature_lock_hash: str,
    prediction_seal_hash: str,
) -> dict[str, object]:
    if not isinstance(responses, FixedBankResponseProduction):
        raise ProtocolError("Response locking requires a typed surface.")
    exact_hash = canonical_hash(
        {
            "schema_version": "midogpp_fixed_bank_exact_response_surface_v1",
            "row_keys": [list(row.row_key) for row in responses.rows],
            "exact_bacc_delta": [row.exact_bacc_delta for row in responses.rows],
        }
    )
    smooth_hash = canonical_hash(
        {
            "schema_version": "midogpp_fixed_bank_smooth_response_surface_v1",
            "row_keys": [list(row.row_key) for row in responses.rows],
            "smooth_bacc_delta": [row.smooth_bacc_delta for row in responses.rows],
        }
    )
    unhashed = {
        "schema_version": "midogpp_stage90_fixed_bank_response_lock_v1",
        "status": "COMPLETE_AFTER_GLOBAL_PREDICTION_AND_FEATURE_SEALS",
        "response_surface_hash": responses.response_surface_hash,
        "ordered_response_row_hashes": [row.response_row_hash for row in responses.rows],
        "exact_response_surface_hash": exact_hash,
        "smooth_response_surface_hash": smooth_hash,
        "fixed_bank_feature_lock_hash": feature_lock_hash,
        "prediction_seal_hash": prediction_seal_hash,
        "exact_response_is_primary": True,
        "smooth_response_is_isolated_descriptive_only": True,
        "smooth_may_affect_exact_fit_selection_gate_or_decision": False,
        "support_labels_opened": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
    }
    return {**unhashed, "fixed_bank_response_lock_hash": canonical_hash(unhashed)}


def build_exact_crossfit_lock(result: ExactCrossfitResult) -> dict[str, object]:
    if (
        not isinstance(result, ExactCrossfitResult)
        or result.family_ids != EXACT_FAMILY_IDS
    ):
        raise ProtocolError("Exact crossfit lock requires the complete exact result.")
    unhashed = {
        "schema_version": "midogpp_stage90_fixed_bank_exact_crossfit_lock_v1",
        "exact_crossfit_hash": result.result_hash,
        "family_ids": list(result.family_ids),
        "prediction_row_hashes": [row.row_hash for row in result.predictions],
        "fold_hashes": [row.fold_hash for row in result.fold_audits],
        "response_name": EXACT_BACC_DELTA,
        "strict_H_q_all_role_exclusion": True,
        "candidate_e_history_retained_for_known_bank": True,
        "smooth_response_used": False,
        "decision_capability": False,
    }
    return {**unhashed, "exact_crossfit_lock_hash": canonical_hash(unhashed)}


def build_smooth_descriptive_crossfit_lock(
    result: SmoothCrossfitResult,
    *,
    exact_crossfit_hash: str,
) -> dict[str, object]:
    if (
        not isinstance(result, SmoothCrossfitResult)
        or result.family_ids != SMOOTH_DESCRIPTIVE_FAMILY_IDS
    ):
        raise ProtocolError("Smooth lock requires the complete descriptive result.")
    unhashed = {
        "schema_version": "midogpp_stage90_fixed_bank_smooth_crossfit_lock_v1",
        "smooth_crossfit_hash": result.result_hash,
        "family_ids": list(result.family_ids),
        "prediction_row_hashes": [row.row_hash for row in result.predictions],
        "fold_hashes": [row.fold_hash for row in result.fold_audits],
        "response_name": SMOOTH_BACC_DELTA,
        "exact_crossfit_hash_held_fixed": exact_crossfit_hash,
        "wholly_separate_models": True,
        "may_affect_exact_coefficients_selection_gate_or_decision": False,
        "decision_fields_exported": False,
        "terminal_decision_authorized": False,
    }
    return {
        **unhashed,
        "smooth_descriptive_crossfit_lock_hash": canonical_hash(unhashed),
    }


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.uint8)
    predicted = np.asarray(predictions, dtype=np.uint8)
    if (
        truth.shape != predicted.shape
        or truth.ndim != 1
        or set(truth.tolist()) != {0, 1}
        or not np.isin(predicted, (0, 1)).all()
    ):
        raise ProtocolError("Exact BACC inputs are malformed.")
    recalls = [float(np.mean(predicted[truth == label] == label)) for label in (0, 1)]
    return float(np.mean(recalls, dtype=np.float64))


def soft_balanced_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        truth.shape != values.shape
        or truth.ndim != 1
        or set(truth.tolist()) != {0, 1}
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ProtocolError("Smooth BACC inputs are malformed.")
    specificity = float(np.mean(1.0 - values[truth == 0], dtype=np.float64))
    recall = float(np.mean(values[truth == 1], dtype=np.float64))
    return 0.5 * (specificity + recall)


def _validated_scoring_capability(
    development: object, *, labels: object, partitions: object
) -> tuple[object, str]:
    store = getattr(development, "store", None)
    seal = getattr(development, "seal", None)
    prediction_hash = str(getattr(seal, "prediction_seal_hash", ""))
    if (
        store is None
        or seal is None
        or getattr(store, "role", None) != "development"
        or str(getattr(store, "partition_lock_hash", ""))
        != str(getattr(partitions, "lock_hash", ""))
        or str(getattr(labels, "prediction_seal_hash", "")) != prediction_hash
        or _HASH_TOKEN.fullmatch(prediction_hash) is None
    ):
        raise ProtocolError("Fixed-bank scoring lacks its sealed capability.")
    return store, prediction_hash


def _validated_feature_lock(
    features: FixedBankFeatureProduction,
    feature_lock: Mapping[str, object],
    *,
    expected_prediction_seal_hash: str,
    expected_partition_lock_hash: str,
) -> str:
    observed = dict(feature_lock)
    supplied_hash = str(observed.get("fixed_bank_feature_lock_hash", ""))
    unhashed = {
        key: value for key, value in observed.items() if key != "fixed_bank_feature_lock_hash"
    }
    if (
        observed.get("schema_version")
        != "midogpp_stage90_fixed_bank_feature_lock_v1"
        or observed.get("status") != "SEALED_BEFORE_TEST_LABEL_ACCESS"
        or observed.get("feature_surface_hash") != features.feature_surface_hash
        or observed.get("ordered_feature_row_hashes")
        != [row.feature_row_hash for row in features.rows]
        or observed.get("development_prediction_seal_hash")
        != expected_prediction_seal_hash
        or observed.get("support_partition_lock_hash")
        != expected_partition_lock_hash
        or observed.get("test_labels_opened") is not False
        or observed.get("support_labels_used") is not False
        or observed.get("evaluation_probabilities_used_as_features") is not False
        or supplied_hash != canonical_hash(unhashed)
    ):
        raise ProtocolError("Persisted fixed-bank pre-label feature lock drifted.")
    return supplied_hash


def _evaluation_vectors(
    store: object,
    *,
    scope: str,
    action_id: str,
    expected_row_hash: str,
    cache: dict[tuple[str, str], tuple[object, ...]],
) -> tuple[object, ...]:
    key = (scope, action_id)
    if key not in cache:
        values = tuple(
            sorted(store.vectors(scope, action_id, "evaluation"), key=lambda v: v.seed_key)
        )
        if (
            tuple(value.seed_key for value in values) != SEED_PAIRS
            or any(value.row_identity_hash != expected_row_hash for value in values)
        ):
            raise ProtocolError("Fixed-bank exact-nine evaluation binding drifted.")
        cache[key] = values
    return cache[key]


def _probability_matrix(vectors: Sequence[object]) -> np.ndarray:
    matrix = np.stack(
        [
            np.asarray(value.positive_class_probabilities, dtype=np.float64)
            for value in vectors
        ]
    )
    if matrix.shape[0] != len(SEED_PAIRS) or not np.isfinite(matrix).all():
        raise ProtocolError("Fixed-bank evaluation probability matrix drifted.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _descriptive_seed_rows(
    row_key: tuple[str, str, str],
    *,
    base_vectors: Sequence[object],
    tail_vectors: Sequence[object],
    labels: np.ndarray,
    evaluation_row_hash: str,
    feature_surface_hash: str,
    feature_lock_hash: str,
    prediction_seal_hash: str,
) -> tuple[Mapping[str, object], ...]:
    output: list[Mapping[str, object]] = []
    label_hash = canonical_array_hash(labels)
    for base, tail in zip(base_vectors, tail_vectors, strict=True):
        base_probability = np.asarray(base.positive_class_probabilities, dtype=np.float64)
        tail_probability = np.asarray(tail.positive_class_probabilities, dtype=np.float64)
        exact_base = balanced_accuracy(labels, (base_probability >= 0.5).astype(np.uint8))
        exact_tail = balanced_accuracy(labels, (tail_probability >= 0.5).astype(np.uint8))
        smooth_base = soft_balanced_accuracy(labels, base_probability)
        smooth_tail = soft_balanced_accuracy(labels, tail_probability)
        unhashed = {
            "schema_version": "midogpp_fixed_bank_seed_utility_diagnostic_v1",
            "outer_target_id": row_key[0],
            "query_id": row_key[1],
            "candidate_source": row_key[2],
            "training_seed": int(base.training_seed),
            "generation_seed": int(base.generation_seed),
            "evaluation_row_hash": evaluation_row_hash,
            "evaluation_label_sha256": label_hash,
            "feature_surface_hash": feature_surface_hash,
            "fixed_bank_feature_lock_hash": feature_lock_hash,
            "prediction_seal_hash": prediction_seal_hash,
            "base_vector_hash": str(base.vector_hash),
            "tail_vector_hash": str(tail.vector_hash),
            "exact_base_bacc": exact_base,
            "exact_tail_bacc": exact_tail,
            "exact_bacc_delta": exact_tail - exact_base,
            "smooth_base_bacc": smooth_base,
            "smooth_tail_bacc": smooth_tail,
            "smooth_bacc_delta": smooth_tail - smooth_base,
            "descriptive_only": True,
            "may_feed_exact_or_smooth_crossfit": False,
            "technical_seed_row_is_independent_observation": False,
        }
        output.append({**unhashed, "row_hash": canonical_hash(unhashed)})
    return tuple(output)


def _response_surface_hash(rows: Sequence[FixedBankResponseRow]) -> str:
    return canonical_hash(
        {
            "schema_version": "midogpp_fixed_bank_response_surface_v1",
            "row_keys": [list(row.row_key) for row in rows],
            "response_row_hashes": [row.response_row_hash for row in rows],
            "exact_response_is_primary": True,
            "smooth_response_is_descriptive_only": True,
        }
    )


__all__ = (
    "FixedBankResponseProduction",
    "balanced_accuracy",
    "build_exact_crossfit_lock",
    "build_fixed_bank_response_lock",
    "build_smooth_descriptive_crossfit_lock",
    "produce_fixed_bank_responses",
    "soft_balanced_accuracy",
)
