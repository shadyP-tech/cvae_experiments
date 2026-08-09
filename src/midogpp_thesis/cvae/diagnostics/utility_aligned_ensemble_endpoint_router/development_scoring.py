"""Primary 504 endpoint responses and descriptive 4,536 seed rows."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...metrics import balanced_accuracy
from ...protocol import ProtocolError
from .contracts import BASE_ACTION_ID, CENTERS, candidate_sources, h_x_e_action_id, inner_candidate_sources
from .development_label_access import OpenedDevelopmentLabels
from .development_seal import DevelopmentPredictionCapability
from .endpoint_scoring import (
    build_source_inner_ensemble_response,
    validate_source_inner_ensemble_responses,
)
from .input_contracts import row_identity_hash
from .prediction_contracts import array_sha256


DEVELOPMENT_SEED_COLUMNS = (
    "schema_version", "outer_target_id", "query_id", "candidate_source",
    "training_seed", "generation_seed", "base_bacc", "tail_bacc", "utility_delta",
    "base_probability_sha256", "tail_probability_sha256", "row_identity_hash",
    "descriptive_only", "technical_seed_row_may_feed_model",
)


def score_development_ensemble_endpoints(
    capability: DevelopmentPredictionCapability,
    labels: OpenedDevelopmentLabels,
    partitions: object,
) -> tuple[object, tuple[Mapping[str, object], ...]]:
    if capability.seal.prediction_seal_hash != labels.prediction_seal_hash:
        raise ProtocolError("Development scoring seal/label capability drifted.")
    primary: list[object] = []
    descriptive: list[Mapping[str, object]] = []
    for outer in CENTERS:
        for query in candidate_sources(outer):
            scope = f"{outer}::{query}"
            truth = np.asarray(labels.labels_by_center[query], dtype=np.uint8)
            support_hash = row_identity_hash(partitions.support_rows_by_center[query])
            evaluation_hash = row_identity_hash(partitions.evaluation_rows_by_center[query])
            base_vectors = capability.store.vectors(scope, BASE_ACTION_ID, "evaluation")
            for source in inner_candidate_sources(outer, query):
                tail_vectors = capability.store.vectors(scope, h_x_e_action_id(source), "evaluation")
                primary.append(build_source_inner_ensemble_response(
                    outer_target_id=outer, query_id=query, candidate_source=source,
                    base_vectors=base_vectors, tail_vectors=tail_vectors, labels=truth,
                    support_partition_hash=support_hash,
                    evaluation_partition_hash=evaluation_hash,
                    prediction_seal_hash=capability.seal.prediction_seal_hash,
                ))
                for base, tail in zip(base_vectors, tail_vectors, strict=True):
                    base_prediction = (base.positive_class_probabilities >= 0.5).astype(np.uint8)
                    tail_prediction = (tail.positive_class_probabilities >= 0.5).astype(np.uint8)
                    base_bacc = float(balanced_accuracy(truth.tolist(), base_prediction.tolist()))
                    tail_bacc = float(balanced_accuracy(truth.tolist(), tail_prediction.tolist()))
                    descriptive.append({
                        "schema_version": "midogpp_stage90_ensemble_endpoint_development_seed_diagnostic_v1",
                        "outer_target_id": outer, "query_id": query, "candidate_source": source,
                        "training_seed": base.training_seed, "generation_seed": base.generation_seed,
                        "base_bacc": base_bacc, "tail_bacc": tail_bacc,
                        "utility_delta": tail_bacc - base_bacc,
                        "base_probability_sha256": array_sha256(base.positive_class_probabilities),
                        "tail_probability_sha256": array_sha256(tail.positive_class_probabilities),
                        "row_identity_hash": evaluation_hash, "descriptive_only": True,
                        "technical_seed_row_may_feed_model": False,
                    })
    surface = validate_source_inner_ensemble_responses(primary)
    if len(descriptive) != 4536:
        raise ProtocolError("Development descriptive seed-row count drifted.")
    return surface, tuple(descriptive)


__all__ = ("DEVELOPMENT_SEED_COLUMNS", "score_development_ensemble_endpoints")
