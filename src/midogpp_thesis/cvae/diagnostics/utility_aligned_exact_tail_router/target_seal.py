"""Durable all-target prediction seal and validation capability."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import FrozenExactTailActionLibrary
from .artifact_io import atomic_json, read_json, sha256_file
from .contracts import CENTERS
from .input_contracts import FixedPartitionSurface, row_identity_hash
from .partitions import CaseFoldSurface
from .target_prediction_contracts import (
    TARGET_PREDICTION_ARRAY_MEMBER,
    TARGET_PREDICTION_INDEX_MEMBER,
    TARGET_PREDICTION_SEAL_MEMBER,
    TargetPredictionStore,
    array_sha256,
)


def build_global_target_prediction_seal(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    partitions: FixedPartitionSurface,
    case_folds: CaseFoldSurface,
    library: FrozenExactTailActionLibrary,
    predictions: TargetPredictionStore,
) -> Mapping[str, object]:
    unhashed = _target_seal_unhashed_payload(
        root,
        config_contract_hash=config_contract_hash,
        source_cache_lock_hash=source_cache_lock_hash,
        partitions=partitions,
        case_folds=case_folds,
        library=library,
        predictions=predictions,
    )
    payload = {**unhashed, "seal_hash": stable_hash(unhashed)}
    atomic_json(root / TARGET_PREDICTION_SEAL_MEMBER, payload)
    return validate_global_target_prediction_seal(
        root,
        config_contract_hash=config_contract_hash,
        source_cache_lock_hash=source_cache_lock_hash,
        partitions=partitions,
        case_folds=case_folds,
        library=library,
        predictions=predictions,
    )


def _target_seal_unhashed_payload(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    partitions: FixedPartitionSurface,
    case_folds: CaseFoldSurface,
    library: FrozenExactTailActionLibrary,
    predictions: TargetPredictionStore,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_global_target_prediction_seal_v1",
        "status": "SEALED_ALL_TARGET_ACTIONS_BEFORE_TERMINAL_TARGET_SCORING",
        "config_contract_hash": config_contract_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "case_fold_lock_hash": case_folds.lock_hash,
        "action_library_hash": library.action_library_hash,
        "target_prediction_store_hash": predictions.store_hash,
        "prediction_array_sha256": sha256_file(root / TARGET_PREDICTION_ARRAY_MEMBER),
        "prediction_index_sha256": sha256_file(root / TARGET_PREDICTION_INDEX_MEMBER),
        "evaluation_row_identity_hash_by_target": {
            target: row_identity_hash(partitions.evaluation_rows_by_center[target])
            for target in CENTERS
        },
        "evaluation_case_ids_by_target": {
            target: sorted(
                {row.case_id for row in partitions.evaluation_rows_by_center[target]}
            )
            for target in CENTERS
        },
        "cell_count": len(predictions.cells),
        "cells": [
            {
                "key": list(cell.key),
                "action_hash": cell.action_hash,
                "evaluation_row_identity_hash": cell.evaluation_row_identity_hash,
                "prediction_sha256": array_sha256(cell.predictions),
                "probability_sha256": array_sha256(cell.probabilities),
                "composition_sha256": cell.composition_sha256,
            }
            for cell in predictions.cells
        ],
        "R2_support_case_count": 2,
        "R2_status": "INSUFFICIENT_SUPPORT_FOR_POLICY",
        "target_support_labels_opened": False,
        "development_crossfit_labels_previously_opened": True,
        "outer_H_development_rows_excluded_from_plan_H": True,
        "terminal_target_scoring_capability_opened": False,
        "target_evaluation_used_to_build_actions": False,
        "all_actions_frozen": True,
        "all_predictions_materialized": True,
        "may_update_policy": False,
    }


def validate_global_target_prediction_seal(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    partitions: FixedPartitionSurface,
    case_folds: CaseFoldSurface,
    library: FrozenExactTailActionLibrary,
    predictions: TargetPredictionStore,
) -> Mapping[str, object]:
    payload = read_json(root / TARGET_PREDICTION_SEAL_MEMBER)
    expected_unhashed = _target_seal_unhashed_payload(
        root,
        config_contract_hash=config_contract_hash,
        source_cache_lock_hash=source_cache_lock_hash,
        partitions=partitions,
        case_folds=case_folds,
        library=library,
        predictions=predictions,
    )
    expected = {**expected_unhashed, "seal_hash": stable_hash(expected_unhashed)}
    if payload != expected:
        raise ProtocolError("Utility-aligned global target prediction seal drifted.")
    return payload


__all__ = (
    "build_global_target_prediction_seal",
    "validate_global_target_prediction_seal",
)
