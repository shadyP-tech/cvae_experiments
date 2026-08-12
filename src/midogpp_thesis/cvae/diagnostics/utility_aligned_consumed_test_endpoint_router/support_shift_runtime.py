"""Label-free action-shift extraction from immutable prediction stores."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    SeedProbabilityVector,
    build_target_support_action_shift_case,
    support_action_probability_shift,
)
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    candidate_sources,
    h_x_e_action_id,
    inner_candidate_sources,
)
from .feature_runtime_contracts import (
    FeatureRuntimeProducts,
    SeedFeatureProduction,
    SupportShiftProduction,
)
from .partitions import ConsumedTestPartitionSurface
from .prediction_contracts import (
    DEVELOPMENT_ROLE,
    TARGET_ROLE,
    PredictionStore,
)
from .seals import DevelopmentPredictionCapability


def materialize_label_free_support_shifts(
    seed_features: SeedFeatureProduction,
    development: DevelopmentPredictionCapability,
    target_store: PredictionStore,
    partitions: ConsumedTestPartitionSurface,
    *,
    root: Path,
) -> SupportShiftProduction:
    """Derive the exact 504 global and 576 per-case local scalars.

    ``root`` is admitted only to keep the default runtime dependency-injection
    signature uniform.  The operation is pure over already sealed stores and
    intentionally leaves no additional artifact or checkpoint member behind.
    """

    if (
        not isinstance(seed_features, SeedFeatureProduction)
        or not isinstance(development, DevelopmentPredictionCapability)
        or development.store.phase != DEVELOPMENT_ROLE
        or not isinstance(target_store, PredictionStore)
        or target_store.phase != TARGET_ROLE
        or not isinstance(partitions, ConsumedTestPartitionSurface)
        or not isinstance(root, Path)
        or development.store.partition_lock_hash != partitions.lock_hash
        or target_store.partition_lock_hash != partitions.lock_hash
        or development.store.source_stream_lock_hash
        != target_store.source_stream_lock_hash
        or development.store.cache_binding_hash != target_store.cache_binding_hash
    ):
        raise ProtocolError("Endpoint-router support-shift runtime binding drifted.")
    source_shifts = {}
    for outer in CENTERS:
        for query in candidate_sources(outer):
            base = development.store.vectors(
                outer_target=outer,
                query_center=query,
                action_id=BASE_ACTION_ID,
                role="support",
            )
            for source in inner_candidate_sources(outer, query):
                tail = development.store.vectors(
                    outer_target=outer,
                    query_center=query,
                    action_id=h_x_e_action_id(source),
                    role="support",
                )
                source_shifts[(outer, query, source)] = (
                    support_action_probability_shift(base, tail)
                )

    target_case_rows = []
    for target in CENTERS:
        scope_case_ids = target_store.support_case_ids_by_scope[target]
        scope_row_ids = target_store.support_row_ids_by_scope[target]
        expected_cases = partitions.by_center[target].support_case_ids
        if (
            tuple(sorted(set(scope_case_ids))) != expected_cases
            or len(scope_case_ids) != len(scope_row_ids)
        ):
            raise ProtocolError("Endpoint-router target-store support case binding drifted.")
        base = target_store.vectors(
            outer_target=target,
            query_center=target,
            action_id=BASE_ACTION_ID,
            role="support",
        )
        for source in candidate_sources(target):
            tail = target_store.vectors(
                outer_target=target,
                query_center=target,
                action_id=h_x_e_action_id(source),
                role="support",
            )
            for case_id in expected_cases:
                indices = tuple(
                    index
                    for index, observed in enumerate(scope_case_ids)
                    if observed == case_id
                )
                if not indices:
                    raise ProtocolError("Endpoint-router target support case has no rows.")
                row_ids = tuple(scope_row_ids[index] for index in indices)
                case_row_hash = canonical_sha256(
                    {
                        "schema_version": "midogpp_endpoint_router_support_case_rows_v1",
                        "target_id": target,
                        "case_id": case_id,
                        "row_ids": list(row_ids),
                    }
                )
                target_case_rows.append(
                    build_target_support_action_shift_case(
                        target_id=target,
                        candidate_source=source,
                        case_id=case_id,
                        base_vectors=_slice_vectors(base, indices, case_row_hash),
                        tail_vectors=_slice_vectors(tail, indices, case_row_hash),
                    )
                )
    provisional = {
        "schema_version": "midogpp_endpoint_router_support_shift_production_v1",
        "seed_feature_production_hash": seed_features.production_hash,
        "development_prediction_seal_hash": development.seal_hash,
        "target_prediction_store_hash": target_store.store_hash,
        "partition_lock_hash": partitions.lock_hash,
        "source_inner_shift_count": len(source_shifts),
        "source_inner_shift_hashes": [value.shift_hash for value in source_shifts.values()],
        "target_case_shift_count": len(target_case_rows),
        "target_case_hashes": [value.case_hash for value in target_case_rows],
        "ensemble_first": True,
        "technical_seed_values_may_feed_model": False,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return SupportShiftProduction(
        source_inner_by_candidate=source_shifts,
        target_case_rows=tuple(target_case_rows),
        seed_feature_production_hash=seed_features.production_hash,
        development_prediction_seal_hash=development.seal_hash,
        target_prediction_store_hash=target_store.store_hash,
        partition_lock_hash=partitions.lock_hash,
        production_hash=canonical_sha256(provisional),
    )


def combine_feature_runtime(
    seed_features: SeedFeatureProduction,
    shifts: SupportShiftProduction,
) -> FeatureRuntimeProducts:
    provisional = {
        "schema_version": "midogpp_endpoint_router_feature_runtime_seal_v1",
        "seed_feature_production_hash": seed_features.production_hash,
        "feature_input_seal_hash": seed_features.feature_input_seal_hash,
        "support_shift_production_hash": shifts.production_hash,
        "development_prediction_seal_hash": shifts.development_prediction_seal_hash,
        "target_prediction_store_hash": shifts.target_prediction_store_hash,
        "labels_used": False,
        "evaluation_embeddings_used": False,
        "target_prediction_seal_binds_this_runtime_later": True,
    }
    return FeatureRuntimeProducts(
        seed_features=seed_features,
        support_shifts=shifts,
        runtime_seal_hash=canonical_sha256(provisional),
    )


def _slice_vectors(
    vectors: Sequence[SeedProbabilityVector],
    indices: Sequence[int],
    row_identity_hash: str,
) -> tuple[SeedProbabilityVector, ...]:
    selected = np.asarray(tuple(indices), dtype=np.int64)
    return tuple(
        SeedProbabilityVector(
            training_seed=vector.training_seed,
            generation_seed=vector.generation_seed,
            row_identity_hash=row_identity_hash,
            prediction_provenance_hash=canonical_sha256(
                {
                    "schema_version": "midogpp_endpoint_router_case_probability_slice_v1",
                    "parent_vector_hash": vector.vector_hash,
                    "row_identity_hash": row_identity_hash,
                }
            ),
            positive_class_probabilities=vector.positive_class_probabilities[selected],
        )
        for vector in vectors
    )


__all__ = (
    "combine_feature_runtime",
    "materialize_label_free_support_shifts",
)
