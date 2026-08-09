"""Experiment-owned label-free fixed-bank feature production."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .constants import candidate_sources, expected_row_keys
from .execution_adapter import BASE_ACTION_ID, h_x_e_action_id
from .experiment_contracts import (
    CENTERS,
    EXPECTED_FEATURE_ROW_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from .input_contracts import row_identity_hash
from .row_contracts import FixedBankFeatureRow
from .serialization import canonical_array_hash, canonical_hash


SEED_PAIRS = tuple(
    (training_seed, generation_seed)
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
)
_HASH_TOKEN = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")


@dataclass(frozen=True)
class FixedBankFeatureProduction:
    rows: tuple[FixedBankFeatureRow, ...]
    feature_surface_hash: str

    def __post_init__(self) -> None:
        if (
            tuple(row.row_key for row in self.rows) != expected_row_keys()
            or len(self.rows) != EXPECTED_FEATURE_ROW_COUNT
            or self.feature_surface_hash != _feature_surface_hash(self.rows)
        ):
            raise ProtocolError("Fixed-bank label-free feature surface drifted.")


def produce_label_free_fixed_bank_features(
    source_cache: object,
    frame: object,
    partitions: object,
    metadata_similarity: Mapping[str, Mapping[str, float]],
    development: object,
) -> FixedBankFeatureProduction:
    store, prediction_seal_hash = _validated_development(development, partitions)
    support = getattr(partitions, "support_rows_by_center", None)
    if not isinstance(support, Mapping) or tuple(support) != CENTERS:
        raise ProtocolError("Fixed-bank feature production lacks support rows.")
    if tuple(metadata_similarity) != CENTERS:
        raise ProtocolError("Fixed-bank metadata center coverage drifted.")

    case_rows = _support_cases_by_center(partitions)
    generated_means = _generated_stream_means(source_cache)
    components = _component_summaries(
        source_cache,
        frame=frame,
        partitions=partitions,
        case_rows=case_rows,
        generated_means=generated_means,
    )
    vector_cache: dict[tuple[str, str, str], tuple[object, ...]] = {}
    rows: list[FixedBankFeatureRow] = []
    for outer, query, source in expected_row_keys():
        scope = f"{outer}::{query}"
        base_vectors = _exact_nine_vectors(
            store,
            scope=scope,
            action_id=BASE_ACTION_ID,
            role="support",
            expected_row_hash=row_identity_hash(support[query]),
            cache=vector_cache,
        )
        tail_vectors = _exact_nine_vectors(
            store,
            scope=scope,
            action_id=h_x_e_action_id(source),
            role="support",
            expected_row_hash=row_identity_hash(support[query]),
            cache=vector_cache,
        )
        positions = _case_positions(support[query], case_rows[query])
        case_abs: list[float] = []
        case_signed: list[float] = []
        case_flips: list[float] = []
        case_entropy: list[float] = []
        case_reconstruction: list[float] = []
        case_kl: list[float] = []
        case_log_mmd: list[float] = []
        case_provenance: list[str] = []
        pooled_abs_sum = 0.0
        pooled_count = 0
        for case_id, rows_in_case in case_rows[query]:
            case_positions = positions[case_id]
            base_matrix = _slice_probability_matrix(base_vectors, case_positions)
            tail_matrix = _slice_probability_matrix(tail_vectors, case_positions)
            base = np.mean(base_matrix, axis=0, dtype=np.float64)
            tail = np.mean(tail_matrix, axis=0, dtype=np.float64)
            delta = tail - base
            absolute = np.abs(delta)
            pseudo_sign = np.where(base >= 0.5, 1.0, -1.0)
            reconstruction, kl, log_mmd = components[(query, source, case_id)]
            case_abs.append(float(np.mean(absolute, dtype=np.float64)))
            case_signed.append(float(np.mean(pseudo_sign * delta, dtype=np.float64)))
            case_flips.append(
                float(np.mean((base - 0.5) * (tail - 0.5) < 0.0, dtype=np.float64))
            )
            case_entropy.append(
                float(
                    np.mean(
                        _binary_entropy(tail) - _binary_entropy(base),
                        dtype=np.float64,
                    )
                )
            )
            case_reconstruction.append(reconstruction)
            case_kl.append(kl)
            case_log_mmd.append(log_mmd)
            pooled_abs_sum += float(np.sum(absolute, dtype=np.float64))
            pooled_count += len(case_positions)
            case_provenance.append(
                canonical_hash(
                    {
                        "schema_version": "midogpp_fixed_bank_support_case_provenance_v1",
                        "row_key": [outer, query, source],
                        "case_id": case_id,
                        "case_row_hash": row_identity_hash(rows_in_case),
                        "base_probability_hashes": [
                            canonical_array_hash(base_matrix[index])
                            for index in range(len(base_matrix))
                        ],
                        "tail_probability_hashes": [
                            canonical_array_hash(tail_matrix[index])
                            for index in range(len(tail_matrix))
                        ],
                        "reconstruction": reconstruction,
                        "kl": kl,
                        "log_mmd": log_mmd,
                        "labels_used": False,
                        "evaluation_probabilities_used": False,
                    }
                )
            )
        try:
            metadata_value = float(metadata_similarity[query][source])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Fixed-bank metadata query/source coverage drifted.") from exc
        if not np.isfinite(metadata_value):
            raise ProtocolError("Fixed-bank metadata similarity is not finite.")
        abs_values = np.asarray(case_abs, dtype=np.float64)
        source_feature_row_hash = canonical_hash(
            {
                "schema_version": "midogpp_fixed_bank_source_feature_row_v1",
                "row_key": [outer, query, source],
                "support_partition_hash": row_identity_hash(support[query]),
                "prediction_seal_hash": prediction_seal_hash,
                "metadata_similarity": metadata_value,
                "pooled_row_weighted_abs_shift": pooled_abs_sum / pooled_count,
                "case_provenance_hashes": case_provenance,
                "exact_nine_mean_before_case_aggregation": True,
                "whole_case_equal_weight": True,
                "support_labels_used": False,
            }
        )
        rows.append(
            FixedBankFeatureRow(
                outer_target_id=outer,
                query_id=query,
                candidate_source=source,
                source_feature_row_hash=source_feature_row_hash,
                metadata_similarity=metadata_value,
                pooled_row_weighted_abs_shift=pooled_abs_sum / pooled_count,
                equal_case_abs_shift=float(np.mean(abs_values, dtype=np.float64)),
                case_abs_shift_sd=float(np.std(abs_values, ddof=0)),
                equal_case_signed_margin=_mean(case_signed),
                case_balanced_flip_rate=_mean(case_flips),
                case_balanced_entropy_change=_mean(case_entropy),
                case_balanced_reconstruction=_mean(case_reconstruction),
                case_balanced_kl=_mean(case_kl),
                case_balanced_log_mmd=_mean(case_log_mmd),
            )
        )
    frozen = tuple(rows)
    return FixedBankFeatureProduction(
        rows=frozen,
        feature_surface_hash=_feature_surface_hash(frozen),
    )


def build_fixed_bank_feature_lock(
    features: FixedBankFeatureProduction,
    *,
    partition_lock_hash: str,
    development_prediction_seal_hash: str,
) -> dict[str, object]:
    if not isinstance(features, FixedBankFeatureProduction):
        raise ProtocolError("Feature locking requires a typed fixed-bank surface.")
    if _HASH_TOKEN.fullmatch(development_prediction_seal_hash) is None:
        raise ProtocolError("Fixed-bank prediction seal hash drifted.")
    unhashed = {
        "schema_version": "midogpp_stage90_fixed_bank_feature_lock_v1",
        "status": "SEALED_BEFORE_TEST_LABEL_ACCESS",
        "feature_surface_hash": features.feature_surface_hash,
        "ordered_feature_row_hashes": [row.feature_row_hash for row in features.rows],
        "feature_row_count": len(features.rows),
        "support_partition_lock_hash": partition_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "fixed_support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "support_probabilities_only": True,
        "exact_nine_mean_before_case_aggregation": True,
        "whole_case_equal_weight": True,
        "support_labels_used": False,
        "test_labels_opened": False,
        "evaluation_probabilities_used_as_features": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
    }
    return {**unhashed, "fixed_bank_feature_lock_hash": canonical_hash(unhashed)}


def _feature_surface_hash(rows: Sequence[FixedBankFeatureRow]) -> str:
    return canonical_hash(
        {
            "schema_version": "midogpp_fixed_bank_feature_surface_v1",
            "row_keys": [list(row.row_key) for row in rows],
            "feature_row_hashes": [row.feature_row_hash for row in rows],
            "labels_used": False,
            "evaluation_probabilities_used_as_features": False,
        }
    )


def _validated_development(
    development: object, partitions: object
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
        or str(getattr(seal, "partition_lock_hash", ""))
        != str(getattr(partitions, "lock_hash", ""))
        or _HASH_TOKEN.fullmatch(prediction_hash) is None
    ):
        raise ProtocolError("Fixed-bank features require a globally sealed store.")
    return store, prediction_hash


def _support_cases_by_center(
    partitions: object,
) -> dict[str, tuple[tuple[str, tuple[object, ...]], ...]]:
    output: dict[str, tuple[tuple[str, tuple[object, ...]], ...]] = {}
    for center in CENTERS:
        grouped: dict[str, list[object]] = {}
        for row in partitions.support_rows_by_center[center]:
            grouped.setdefault(str(row.case_id), []).append(row)
        if len(grouped) != FIXED_SUPPORT_CASE_COUNT_PER_CENTER:
            raise ProtocolError("Fixed-bank features require eight whole cases.")
        output[center] = tuple(
            (case_id, tuple(grouped[case_id])) for case_id in sorted(grouped)
        )
    return output


def _case_positions(
    all_rows: Sequence[object],
    cases: Sequence[tuple[str, Sequence[object]]],
) -> dict[str, np.ndarray]:
    position = {str(row.evaluation_row_id): index for index, row in enumerate(all_rows)}
    try:
        return {
            case_id: np.asarray(
                [position[str(row.evaluation_row_id)] for row in rows],
                dtype=np.int64,
            )
            for case_id, rows in cases
        }
    except KeyError as exc:
        raise ProtocolError("Fixed-bank support case alignment drifted.") from exc


def _exact_nine_vectors(
    store: object,
    *,
    scope: str,
    action_id: str,
    role: str,
    expected_row_hash: str,
    cache: dict[tuple[str, str, str], tuple[object, ...]],
) -> tuple[object, ...]:
    key = (scope, action_id, role)
    if key not in cache:
        values = tuple(sorted(store.vectors(scope, action_id, role), key=lambda v: v.seed_key))
        if (
            tuple(value.seed_key for value in values) != SEED_PAIRS
            or any(value.row_identity_hash != expected_row_hash for value in values)
        ):
            raise ProtocolError("Fixed-bank exact-nine vector binding drifted.")
        cache[key] = values
    return cache[key]


def _slice_probability_matrix(
    vectors: Sequence[object], positions: np.ndarray
) -> np.ndarray:
    values = np.stack(
        [
            np.asarray(vector.positive_class_probabilities, dtype=np.float64)[positions]
            for vector in vectors
        ]
    )
    if values.shape[0] != len(SEED_PAIRS) or not np.isfinite(values).all():
        raise ProtocolError("Fixed-bank support probability matrix drifted.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _generated_stream_means(
    source_cache: object,
) -> dict[tuple[str, int, int], np.ndarray]:
    output: dict[tuple[str, int, int], np.ndarray] = {}
    for source in CENTERS:
        for training_seed, generation_seed in SEED_PAIRS:
            block = np.asarray(
                source_cache.source_block(source, training_seed, generation_seed),
                dtype=np.float64,
            )
            if block.ndim != 2 or block.shape[1] != COMMON_OUTPUT_DIM:
                raise ProtocolError("Fixed-bank generated stream geometry drifted.")
            output[(source, training_seed, generation_seed)] = np.mean(
                block, axis=0, dtype=np.float64
            )
    return output


def _component_summaries(
    source_cache: object,
    *,
    frame: object,
    partitions: object,
    case_rows: Mapping[str, Sequence[tuple[str, Sequence[object]]]],
    generated_means: Mapping[tuple[str, int, int], np.ndarray],
) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    output: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for query in CENTERS:
        support_rows = tuple(partitions.support_rows_by_center[query])
        positions = _case_positions(support_rows, case_rows[query])
        case_means = {
            case_id: np.mean(
                np.asarray(frame.embeddings_for(rows), dtype=np.float64),
                axis=0,
                dtype=np.float64,
            )
            for case_id, rows in case_rows[query]
        }
        for source in (value for value in CENTERS if value != query):
            reconstruction_by_seed: dict[int, np.ndarray] = {}
            kl_by_seed: dict[int, np.ndarray] = {}
            for training_seed in TRAINING_SEEDS:
                reconstruction, kl = source_cache.component_arrays(
                    query_center=query,
                    source_center=source,
                    training_seed=training_seed,
                )
                reconstruction_by_seed[training_seed] = 0.5 * (
                    np.asarray(reconstruction[0], dtype=np.float64)
                    + np.asarray(reconstruction[1], dtype=np.float64)
                )
                kl_by_seed[training_seed] = 0.5 * (
                    np.asarray(kl[0], dtype=np.float64)
                    + np.asarray(kl[1], dtype=np.float64)
                )
                record = source_cache.component_by_key[(query, source, training_seed)]
                if (
                    record.support_partition_hash != row_identity_hash(support_rows)
                    or record.support_row_count != len(support_rows)
                    or record.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
                ):
                    raise ProtocolError("Fixed-bank source component binding drifted.")
            for case_id, _rows in case_rows[query]:
                indices = positions[case_id]
                reconstruction_summary = _mean(
                    np.mean(reconstruction_by_seed[seed][indices], dtype=np.float64)
                    for seed in TRAINING_SEEDS
                )
                kl_summary = _mean(
                    np.mean(kl_by_seed[seed][indices], dtype=np.float64)
                    for seed in TRAINING_SEEDS
                )
                real_mean = case_means[case_id]
                mmd = []
                for training_seed, generation_seed in SEED_PAIRS:
                    difference = real_mean - generated_means[
                        (source, training_seed, generation_seed)
                    ]
                    mmd.append(float(np.dot(difference, difference)))
                log_mmd_summary = _mean(np.log1p(mmd))
                numeric = np.asarray(
                    [reconstruction_summary, kl_summary, log_mmd_summary],
                    dtype=np.float64,
                )
                if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
                    raise ProtocolError("Fixed-bank component summary is invalid.")
                output[(query, source, case_id)] = tuple(numeric.tolist())  # type: ignore[assignment]
    return output


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    values = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    return -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))


def _mean(values: object) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)  # type: ignore[arg-type]
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("Fixed-bank case mean requires finite values.")
    return float(np.mean(array, dtype=np.float64))


__all__ = (
    "FixedBankFeatureProduction",
    "build_fixed_bank_feature_lock",
    "produce_label_free_fixed_bank_features",
)
