"""Label-free execution adapter for the case-aware feature surface.

This module is the only bridge from workstation prediction/cache objects into
the pure feature contracts.  It deliberately accepts no label capability and
never reads evaluation probability vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from ..utility_aligned_ensemble_endpoint_router.contracts import (
    BASE_ACTION_ID,
    h_x_e_action_id,
)
from .case_features import (
    build_case_aware_feature_row,
    build_case_aware_feature_surface,
)
from .contracts import (
    CENTERS,
    EXPECTED_FEATURE_ROW_COUNT,
    GENERATION_SEEDS,
    SEED_PAIRS,
    TRAINING_SEEDS,
    CaseAwareFeatureSurface,
    SupportCaseVectors,
    candidate_sources,
)
from .input_contracts import row_identity_hash


_HASH_TOKEN = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")


@dataclass(frozen=True)
class CaseAwareFeatureProduction:
    """Complete pre-label feature surface and its publication lock."""

    surface: CaseAwareFeatureSurface
    feature_lock: Mapping[str, object]

    @property
    def rows(self) -> tuple[object, ...]:
        return self.surface.rows


def produce_label_free_case_aware_features(
    source_cache: object,
    frame: object,
    partitions: object,
    metadata_similarity: Mapping[str, Mapping[str, float]],
    development: object,
) -> CaseAwareFeatureSurface:
    """Build all 504 H/q/e rows from support-only sealed information.

    Reconstruction and KL values are collapsed per whole case and then across
    the three training seeds.  Linear-kernel MMD is recomputed at the same
    whole-case granularity from each case mean and all exact-nine generated
    stream means.  Support probabilities are sliced from the globally sealed
    combined store; evaluation vectors are never requested here.
    """

    store, prediction_seal_hash = _validated_development(development, partitions)
    support = getattr(partitions, "support_rows_by_center", None)
    if not isinstance(support, Mapping) or tuple(support) != CENTERS:
        raise ProtocolError("Case-aware feature production lacks fixed support rows.")
    if tuple(metadata_similarity) != CENTERS:
        raise ProtocolError("Case-aware metadata center coverage drifted.")

    case_rows = _support_cases_by_center(partitions)
    generated_means = _generated_stream_means(source_cache)
    component_summaries = _component_summaries(
        source_cache,
        frame=frame,
        partitions=partitions,
        case_rows=case_rows,
        generated_means=generated_means,
    )

    vector_cache: dict[tuple[str, str, str], tuple[object, ...]] = {}
    rows: list[object] = []
    for outer in CENTERS:
        for query in (value for value in CENTERS if value != outer):
            scope = f"{outer}::{query}"
            base_vectors = _exact_nine_vectors(
                store,
                scope=scope,
                action_id=BASE_ACTION_ID,
                role="support",
                expected_row_hash=row_identity_hash(support[query]),
                cache=vector_cache,
            )
            case_positions = _case_positions(support[query], case_rows[query])
            for source in candidate_sources(outer, query):
                tail_vectors = _exact_nine_vectors(
                    store,
                    scope=scope,
                    action_id=h_x_e_action_id(source),
                    role="support",
                    expected_row_hash=row_identity_hash(support[query]),
                    cache=vector_cache,
                )
                typed_cases: list[SupportCaseVectors] = []
                for case_id, rows_in_case in case_rows[query]:
                    positions = case_positions[case_id]
                    case_row_hash = row_identity_hash(rows_in_case)
                    base_matrix = _slice_probability_matrix(base_vectors, positions)
                    tail_matrix = _slice_probability_matrix(tail_vectors, positions)
                    base_hashes = _case_vector_hashes(
                        base_vectors, base_matrix, case_row_hash=case_row_hash
                    )
                    tail_hashes = _case_vector_hashes(
                        tail_vectors, tail_matrix, case_row_hash=case_row_hash
                    )
                    reconstruction, kl, log_mmd = component_summaries[
                        (query, source, case_id)
                    ]
                    provenance = canonical_sha256(
                        {
                            "schema_version": (
                                "midogpp_stage90_case_aware_support_case_provenance_v1"
                            ),
                            "row_key": [outer, query, source],
                            "case_id": case_id,
                            "case_row_hash": case_row_hash,
                            "support_partition_hash": row_identity_hash(support[query]),
                            "prediction_seal_hash": prediction_seal_hash,
                            "base_case_vector_hashes": list(base_hashes),
                            "tail_case_vector_hashes": list(tail_hashes),
                            "source_stream_ids": [
                                _source_stream_id(source_cache, source, train, gen)
                                for train, gen in SEED_PAIRS
                            ],
                            "component_record_ids": [
                                _component_record_id(source_cache, query, source, train)
                                for train in TRAINING_SEEDS
                            ],
                            "reconstruction_summary": reconstruction,
                            "kl_summary": kl,
                            "log_mmd_summary": log_mmd,
                            "labels_used": False,
                            "evaluation_probabilities_used": False,
                        }
                    )
                    typed_cases.append(
                        SupportCaseVectors(
                            case_id=case_id,
                            case_hash=case_identity_hash(query, case_id),
                            row_hash=case_row_hash,
                            provenance_hash=provenance,
                            base_probabilities=base_matrix,
                            tail_probabilities=tail_matrix,
                            reconstruction_summary=reconstruction,
                            kl_summary=kl,
                            log_mmd_summary=log_mmd,
                            base_vector_hashes=base_hashes,
                            tail_vector_hashes=tail_hashes,
                        )
                    )
                try:
                    metadata_value = float(metadata_similarity[query][source])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ProtocolError(
                        "Case-aware metadata query/source coverage drifted."
                    ) from exc
                rows.append(
                    build_case_aware_feature_row(
                        outer_target_id=outer,
                        query_id=query,
                        candidate_source=source,
                        support_partition_hash=row_identity_hash(support[query]),
                        prediction_seal_hash=prediction_seal_hash,
                        metadata_similarity=metadata_value,
                        cases=typed_cases,
                    )
                )
    if len(rows) != EXPECTED_FEATURE_ROW_COUNT:
        raise ProtocolError("Case-aware feature execution coverage drifted.")
    return build_case_aware_feature_surface(rows)


def build_case_aware_feature_lock(
    surface: CaseAwareFeatureSurface,
    *,
    partition_lock_hash: str,
    development_prediction_seal_hash: str,
) -> dict[str, object]:
    """Bind the immutable label-free surface before labels are opened."""

    if not isinstance(surface, CaseAwareFeatureSurface):
        raise ProtocolError("Feature locking requires a typed complete surface.")
    if any(
        row.prediction_seal_hash != development_prediction_seal_hash
        or row.labels_used is not False
        or row.evaluation_probabilities_used_as_features is not False
        for row in surface.rows
    ):
        raise ProtocolError("Feature surface crossed its pre-label seal boundary.")
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_feature_lock_v1",
        "status": "SEALED_BEFORE_TEST_LABEL_ACCESS",
        "feature_surface_hash": surface.surface_hash,
        "ordered_feature_row_hashes": [
            row.feature_row_hash for row in surface.rows
        ],
        "feature_row_count": len(surface.rows),
        "support_partition_lock_hash": str(partition_lock_hash),
        "development_prediction_seal_hash": str(
            development_prediction_seal_hash
        ),
        "fixed_support_case_count_per_center": 8,
        "support_probabilities_only": True,
        "exact_nine_mean_before_case_aggregation": True,
        "whole_case_equal_weight_primary": True,
        "support_labels_used": False,
        "test_labels_opened": False,
        "evaluation_probabilities_used_as_features": False,
        "target_actions_built": False,
        "policy_update_authorized": False,
    }
    return {
        **unhashed,
        "case_aware_feature_lock_hash": canonical_sha256(unhashed),
    }


def produce_and_lock_label_free_case_aware_features(
    source_cache: object,
    frame: object,
    partitions: object,
    metadata_similarity: Mapping[str, Mapping[str, float]],
    development: object,
) -> CaseAwareFeatureProduction:
    surface = produce_label_free_case_aware_features(
        source_cache, frame, partitions, metadata_similarity, development
    )
    seal_hash = str(getattr(getattr(development, "seal", None), "prediction_seal_hash", ""))
    feature_lock = build_case_aware_feature_lock(
        surface,
        partition_lock_hash=str(getattr(partitions, "lock_hash", "")),
        development_prediction_seal_hash=seal_hash,
    )
    return CaseAwareFeatureProduction(surface=surface, feature_lock=feature_lock)


def case_identity_hash(center: str, case_id: str) -> str:
    """One shared support/evaluation case identity hash grammar."""

    return canonical_sha256(
        {
            "schema_version": "midogpp_stage90_case_aware_test_case_identity_v1",
            "split": "test",
            "center": str(center),
            "case_id": str(case_id),
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
        raise ProtocolError("Case-aware features require the globally sealed development store.")
    return store, prediction_hash


def _support_cases_by_center(
    partitions: object,
) -> dict[str, tuple[tuple[str, tuple[object, ...]], ...]]:
    output: dict[str, tuple[tuple[str, tuple[object, ...]], ...]] = {}
    for center in CENTERS:
        grouped: dict[str, list[object]] = {}
        for row in partitions.support_rows_by_center[center]:
            grouped.setdefault(str(row.case_id), []).append(row)
        if len(grouped) != 8:
            raise ProtocolError("Case-aware feature production requires eight whole cases.")
        output[center] = tuple(
            (case_id, tuple(grouped[case_id])) for case_id in sorted(grouped)
        )
    return output


def _case_positions(
    all_rows: Sequence[object],
    cases: Sequence[tuple[str, Sequence[object]]],
) -> dict[str, np.ndarray]:
    position = {str(row.evaluation_row_id): index for index, row in enumerate(all_rows)}
    output: dict[str, np.ndarray] = {}
    for case_id, rows in cases:
        try:
            values = np.asarray(
                [position[str(row.evaluation_row_id)] for row in rows], dtype=np.int64
            )
        except KeyError as exc:
            raise ProtocolError("Case-aware support case row alignment drifted.") from exc
        output[case_id] = values
    return output


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
        values = tuple(
            sorted(
                store.vectors(scope, action_id, role),
                key=lambda value: value.seed_key,
            )
        )
        if (
            tuple(value.seed_key for value in values) != SEED_PAIRS
            or any(value.row_identity_hash != expected_row_hash for value in values)
        ):
            raise ProtocolError("Case-aware exact-nine support vector binding drifted.")
        cache[key] = values
    return cache[key]


def _slice_probability_matrix(
    vectors: Sequence[object], positions: np.ndarray
) -> np.ndarray:
    matrix = np.stack(
        [
            np.asarray(vector.positive_class_probabilities, dtype=np.float64)[positions]
            for vector in vectors
        ]
    )
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _case_vector_hashes(
    vectors: Sequence[object], matrix: np.ndarray, *, case_row_hash: str
) -> tuple[str, ...]:
    return tuple(
        canonical_sha256(
            {
                "schema_version": "midogpp_stage90_case_probability_slice_v1",
                "parent_vector_hash": str(vector.vector_hash),
                "case_row_hash": case_row_hash,
                "probability_sha256": array_sha256(matrix[index]),
            }
        )
        for index, vector in enumerate(vectors)
    )


def _generated_stream_means(source_cache: object) -> dict[tuple[str, int, int], np.ndarray]:
    means: dict[tuple[str, int, int], np.ndarray] = {}
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                block = np.asarray(
                    source_cache.source_block(source, training_seed, generation_seed),
                    dtype=np.float64,
                )
                if block.ndim != 2 or block.shape[1] != 3_840:
                    raise ProtocolError("Case-aware generated stream geometry drifted.")
                means[(source, training_seed, generation_seed)] = np.mean(
                    block, axis=0, dtype=np.float64
                )
    return means


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
                    or record.support_case_count != 8
                ):
                    raise ProtocolError("Case-aware source component binding drifted.")
            for case_id, _rows in case_rows[query]:
                case_position = positions[case_id]
                reconstruction_summary = float(
                    np.mean(
                        [
                            np.mean(reconstruction_by_seed[seed][case_position])
                            for seed in TRAINING_SEEDS
                        ],
                        dtype=np.float64,
                    )
                )
                kl_summary = float(
                    np.mean(
                        [
                            np.mean(kl_by_seed[seed][case_position])
                            for seed in TRAINING_SEEDS
                        ],
                        dtype=np.float64,
                    )
                )
                mmd_values = []
                real_mean = case_means[case_id]
                for training_seed, generation_seed in SEED_PAIRS:
                    difference = real_mean - generated_means[
                        (source, training_seed, generation_seed)
                    ]
                    mmd_values.append(float(np.dot(difference, difference)))
                # Preserve the frozen feature grammar: transform each exact
                # (training, generation) case distance, then average the nine
                # transformed cells.  Pooling before log1p is not equivalent.
                log_mmd_summary = float(
                    np.mean(np.log1p(mmd_values), dtype=np.float64)
                )
                numeric = np.asarray(
                    [reconstruction_summary, kl_summary, log_mmd_summary],
                    dtype=np.float64,
                )
                if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
                    raise ProtocolError("Case-aware component summary is invalid.")
                output[(query, source, case_id)] = (
                    reconstruction_summary,
                    kl_summary,
                    log_mmd_summary,
                )
    return output


def _source_stream_id(
    source_cache: object, source: str, training_seed: int, generation_seed: int
) -> str:
    return str(
        source_cache.source_by_key[(source, training_seed, generation_seed)].stream_id
    )


def _component_record_id(
    source_cache: object, query: str, source: str, training_seed: int
) -> str:
    record = source_cache.component_by_key[(query, source, training_seed)]
    return canonical_sha256(record.to_row())


# Compact aliases for runner dependency injection and tests.
produce_case_aware_feature_surface = produce_label_free_case_aware_features
build_feature_lock = build_case_aware_feature_lock


__all__ = (
    "CaseAwareFeatureProduction",
    "build_case_aware_feature_lock",
    "build_feature_lock",
    "case_identity_hash",
    "produce_and_lock_label_free_case_aware_features",
    "produce_case_aware_feature_surface",
    "produce_label_free_case_aware_features",
)
