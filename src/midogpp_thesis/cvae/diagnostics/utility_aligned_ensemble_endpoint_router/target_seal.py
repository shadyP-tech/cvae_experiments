"""Durable all-action target probability seal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import FrozenEnsembleEndpointActionLibrary
from .artifact_io import atomic_json, read_json, sha256_file
from .contracts import CENTERS
from .input_contracts import row_identity_hash
from .prediction_contracts import CombinedPredictionStore, array_sha256
from .target_prediction_execution import (
    EXPECTED_FINAL_CELL_COUNT,
    EXPECTED_TARGET_UNIQUE_FIT_COUNT,
    TARGET_ARRAY_MEMBER,
    TARGET_INDEX_MEMBER,
    TARGET_INDEX_TABLE_MEMBER,
    TARGET_PROBE_SEAL_MEMBER,
)


TARGET_PREDICTION_SEAL_MEMBER = (
    "manifests/ensemble_endpoint_global_target_prediction_seal.json"
)


@dataclass(frozen=True)
class TargetScoringCapability:
    root: Path
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def seal_hash(self) -> str:
        return str(self.payload["seal_hash"])


def build_global_target_prediction_seal(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    partitions: object,
    case_folds: object,
    library: FrozenEnsembleEndpointActionLibrary,
    predictions: CombinedPredictionStore,
    target_support_shift_lock_hash: str,
) -> TargetScoringCapability:
    unhashed = _unhashed(
        root,
        config_contract_hash=config_contract_hash,
        source_cache_lock_hash=source_cache_lock_hash,
        partitions=partitions,
        case_folds=case_folds,
        library=library,
        predictions=predictions,
        target_support_shift_lock_hash=target_support_shift_lock_hash,
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
        target_support_shift_lock_hash=target_support_shift_lock_hash,
    )


def _unhashed(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    partitions: object,
    case_folds: object,
    library: FrozenEnsembleEndpointActionLibrary,
    predictions: CombinedPredictionStore,
    target_support_shift_lock_hash: str,
) -> dict[str, object]:
    probe = read_json(root / TARGET_PROBE_SEAL_MEMBER)
    if probe.get("status") != "SEALED_B_PLUS_EIGHT_HXE_BEFORE_TARGET_PLAN":
        raise ProtocolError("Global target seal requires a valid pre-plan probe seal.")
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_global_target_seal_v1",
        "status": "SEALED_ALL_TARGET_ACTIONS_BEFORE_TERMINAL_TARGET_SCORING",
        "config_contract_hash": config_contract_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "support_partition_lock_hash": str(partitions.lock_hash),
        "case_fold_lock_hash": str(case_folds.lock_hash),
        "target_probe_seal_hash": str(probe["probe_seal_hash"]),
        "target_support_shift_lock_hash": target_support_shift_lock_hash,
        "diagnostic_plan_set_hash": library.plan_set_hash,
        "action_library_hash": library.action_library_hash,
        "target_prediction_store_hash": predictions.store_hash,
        "prediction_array_sha256": sha256_file(root / TARGET_ARRAY_MEMBER),
        "prediction_index_sha256": sha256_file(root / TARGET_INDEX_MEMBER),
        "prediction_index_table_sha256": sha256_file(root / TARGET_INDEX_TABLE_MEMBER),
        "support_row_identity_hash_by_target": {
            target: row_identity_hash(partitions.support_rows_by_center[target]) for target in CENTERS
        },
        "evaluation_row_identity_hash_by_target": {
            target: row_identity_hash(partitions.evaluation_rows_by_center[target]) for target in CENTERS
        },
        "cell_count": len(predictions.cells),
        "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
        "ordered_cells": [
            {
                "key": list(cell.key), "action_hash": cell.action_hash,
                "support_probability_sha256": array_sha256(cell.support_probabilities),
                "evaluation_probability_sha256": array_sha256(cell.evaluation_probabilities),
                "fit_provenance_hash": cell.fit_provenance_hash,
                "aliased_from_action_id": cell.aliased_from_action_id,
            }
            for cell in predictions.cells
        ],
        "support_case_count": 2,
        "routing_status": "INSUFFICIENT_SUPPORT_FOR_POLICY",
        "development_labels_previously_opened": True,
        "target_support_labels_opened": False,
        "target_evaluation_labels_opened": False,
        "target_evaluation_used_to_build_plan": False,
        "all_actions_frozen": True,
        "all_predictions_materialized": True,
        "policy_update_authorized": False,
    }


def validate_global_target_prediction_seal(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    partitions: object,
    case_folds: object,
    library: FrozenEnsembleEndpointActionLibrary,
    predictions: CombinedPredictionStore,
    target_support_shift_lock_hash: str,
) -> TargetScoringCapability:
    if (
        len(predictions.cells) != EXPECTED_FINAL_CELL_COUNT
        or predictions.unique_classifier_fit_count != EXPECTED_TARGET_UNIQUE_FIT_COUNT
    ):
        raise ProtocolError("Global target seal prediction coverage drifted.")
    unhashed = _unhashed(
        root,
        config_contract_hash=config_contract_hash,
        source_cache_lock_hash=source_cache_lock_hash,
        partitions=partitions,
        case_folds=case_folds,
        library=library,
        predictions=predictions,
        target_support_shift_lock_hash=target_support_shift_lock_hash,
    )
    expected = {**unhashed, "seal_hash": stable_hash(unhashed)}
    observed = read_json(root / TARGET_PREDICTION_SEAL_MEMBER)
    if observed != expected:
        raise ProtocolError("Global ensemble-endpoint target seal drifted.")
    return TargetScoringCapability(root=root, payload=observed)


__all__ = (
    "TARGET_PREDICTION_SEAL_MEMBER",
    "TargetScoringCapability",
    "build_global_target_prediction_seal",
    "validate_global_target_prediction_seal",
)
