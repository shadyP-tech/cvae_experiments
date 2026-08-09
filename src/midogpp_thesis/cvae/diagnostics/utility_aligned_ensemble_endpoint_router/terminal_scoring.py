"""Adapters from the sealed target store to terminal scoring contracts."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...metrics import balanced_accuracy
from ...protocol import ProtocolError
from .actions import FrozenEnsembleEndpointActionLibrary
from .contracts import CENTERS, expected_target_action_ids
from .diagnostic_plan import Stage90EnsembleDiagnosticPlanSet
from .input_contracts import row_identity_hash
from .prediction_contracts import CombinedPredictionStore, array_sha256
from .scoring import (
    build_terminal_hxe_oracle_diagnostics,
    score_target_action_ensemble_endpoint,
    validate_target_ensemble_endpoint_scores,
)


TARGET_SEED_COLUMNS = (
    "schema_version", "target_center", "action_id", "training_seed", "generation_seed",
    "balanced_accuracy", "probability_sha256", "row_identity_hash",
    "descriptive_only", "technical_seed_row_may_feed_inference",
)


def score_terminal_target_predictions(
    predictions: CombinedPredictionStore,
    labels_by_sample: Mapping[str, int],
    partitions: object,
    library: FrozenEnsembleEndpointActionLibrary,
    plans: Stage90EnsembleDiagnosticPlanSet,
    target_seal: Mapping[str, object],
) -> tuple[object, tuple[Mapping[str, object], ...], tuple[object, ...]]:
    if target_seal.get("target_prediction_store_hash") != predictions.store_hash:
        raise ProtocolError("Terminal scoring store differs from the global seal.")
    endpoint_rows: list[object] = []
    seed_rows: list[Mapping[str, object]] = []
    for target in CENTERS:
        evaluation_rows = partitions.evaluation_rows_by_center[target]
        truth = np.asarray([labels_by_sample[row.sample_id] for row in evaluation_rows], dtype=np.uint8)
        support_hash = row_identity_hash(partitions.support_rows_by_center[target])
        evaluation_hash = row_identity_hash(evaluation_rows)
        case_count = len({row.case_id for row in evaluation_rows})
        for action_id in expected_target_action_ids(target):
            action = library.action(target, action_id)
            vectors = predictions.vectors(target, action_id, "evaluation")
            endpoint_rows.append(score_target_action_ensemble_endpoint(
                target_id=target, action_id=action_id, vectors=vectors, labels=truth,
                action_hash=action.action_hash, router_plan_hash=str(action.router_plan_hash),
                support_partition_hash=support_hash, evaluation_partition_hash=evaluation_hash,
                prediction_seal_hash=str(target_seal["seal_hash"]),
                target_probe_seal_hash=str(target_seal["target_probe_seal_hash"]),
                evaluation_case_count=case_count, global_target_seal_verified=True,
            ))
            for vector in vectors:
                prediction = (vector.positive_class_probabilities >= 0.5).astype(np.uint8)
                seed_rows.append({
                    "schema_version": "midogpp_stage90_ensemble_endpoint_target_seed_diagnostic_v1",
                    "target_center": target, "action_id": action_id,
                    "training_seed": vector.training_seed, "generation_seed": vector.generation_seed,
                    "balanced_accuracy": float(balanced_accuracy(truth.tolist(), prediction.tolist())),
                    "probability_sha256": array_sha256(vector.positive_class_probabilities),
                    "row_identity_hash": evaluation_hash, "descriptive_only": True,
                    "technical_seed_row_may_feed_inference": False,
                })
    scores = validate_target_ensemble_endpoint_scores(endpoint_rows, library)
    if len(seed_rows) != 1053:
        raise ProtocolError("Target descriptive seed-row count drifted.")
    oracle = build_terminal_hxe_oracle_diagnostics(plans, scores)
    return scores, tuple(seed_rows), oracle


__all__ = ("TARGET_SEED_COLUMNS", "score_terminal_target_predictions")
