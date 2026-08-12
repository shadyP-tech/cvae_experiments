"""Durable capabilities for prediction-before-label phase boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import atomic_json, read_json, sha256_file
from .contracts import CENTERS
from .prediction_contracts import (
    DEVELOPMENT_ARRAY_MEMBER,
    DEVELOPMENT_CELL_COUNT,
    DEVELOPMENT_INDEX_MEMBER,
    DEVELOPMENT_ROLE,
    DEVELOPMENT_SEAL_MEMBER,
    DEVELOPMENT_TASK_COUNT,
    TARGET_ARRAY_MEMBER,
    TARGET_CELL_COUNT,
    TARGET_INDEX_MEMBER,
    TARGET_ROLE,
    TARGET_SEAL_MEMBER,
    TARGET_TASK_COUNT,
    PredictionStore,
)


@dataclass(frozen=True)
class DevelopmentPredictionCapability:
    store: PredictionStore
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    def __post_init__(self) -> None:
        payload = dict(self.seal_payload)
        unhashed = {key: value for key, value in payload.items() if key != "development_prediction_seal_hash"}
        if (
            self.store.phase != DEVELOPMENT_ROLE
            or payload.get("development_prediction_seal_hash") != canonical_sha256(unhashed)
            or payload.get("schema_version")
            != "midogpp_endpoint_router_development_prediction_seal_v1"
            or payload.get("status") != "SEALED_ALL_5184_DEVELOPMENT_CELLS"
            or payload.get("prediction_store_hash") != self.store.store_hash
            or payload.get("cell_count") != DEVELOPMENT_CELL_COUNT
            or payload.get("task_count") != DEVELOPMENT_TASK_COUNT
            or payload.get("cross_center_evaluation_labels_opened") is not False
            or payload.get("same_outer_H_evaluation_labels_opened") is not False
            or payload.get("support_labels_opened") is not False
        ):
            raise ProtocolError("Endpoint-router development seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["development_prediction_seal_hash"])


@dataclass(frozen=True)
class TargetPredictionCapability:
    store: PredictionStore
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    def __post_init__(self) -> None:
        payload = dict(self.seal_payload)
        unhashed = {key: value for key, value in payload.items() if key != "target_prediction_seal_hash"}
        plan_hashes = payload.get("target_policy_plan_hashes_by_center")
        if (
            self.store.phase != TARGET_ROLE
            or payload.get("target_prediction_seal_hash") != canonical_sha256(unhashed)
            or payload.get("schema_version")
            != "midogpp_endpoint_router_target_prediction_seal_v1"
            or payload.get("status")
            != "SEALED_ALL_TARGET_PROBABILITIES_POLICIES_AND_ACTIONS"
            or payload.get("prediction_store_hash") != self.store.store_hash
            or payload.get("cell_count") != TARGET_CELL_COUNT
            or payload.get("task_count") != TARGET_TASK_COUNT
            or payload.get("physical_action_identity_count") != 90
            or not isinstance(plan_hashes, Mapping)
            or tuple(map(str, plan_hashes)) != CENTERS
            or payload.get("all_nine_target_policy_plans_frozen") is not True
            or payload.get("all_target_actions_frozen") is not True
            or payload.get("support_labels_opened") is not False
            or payload.get("same_outer_H_evaluation_labels_opened") is not False
        ):
            raise ProtocolError("Endpoint-router target prediction seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["target_prediction_seal_hash"])


def seal_development_predictions(
    store: PredictionStore, *, root: Path
) -> DevelopmentPredictionCapability:
    if store.phase != DEVELOPMENT_ROLE:
        raise ProtocolError("Cannot issue development capability for another store.")
    arrays = root / DEVELOPMENT_ARRAY_MEMBER
    index = root / DEVELOPMENT_INDEX_MEMBER
    seal = root / DEVELOPMENT_SEAL_MEMBER
    unhashed = {
        "schema_version": "midogpp_endpoint_router_development_prediction_seal_v1",
        "status": "SEALED_ALL_5184_DEVELOPMENT_CELLS",
        "prediction_store_hash": store.store_hash,
        "source_stream_lock_hash": store.source_stream_lock_hash,
        "partition_lock_hash": store.partition_lock_hash,
        "cache_binding_hash": store.cache_binding_hash,
        "action_library_hash": store.action_library_hash,
        "arrays_member": DEVELOPMENT_ARRAY_MEMBER,
        "arrays_sha256": sha256_file(arrays),
        "index_member": DEVELOPMENT_INDEX_MEMBER,
        "index_sha256": sha256_file(index),
        "cell_count": DEVELOPMENT_CELL_COUNT,
        "task_count": DEVELOPMENT_TASK_COUNT,
        "unique_classifier_fit_count": DEVELOPMENT_CELL_COUNT,
        "expected_development_endpoint_response_count": 504,
        "descriptive_seed_row_count": 4_536,
        "all_probabilities_materialized": True,
        "cross_center_evaluation_labels_opened": False,
        "same_outer_H_evaluation_labels_opened": False,
        "support_labels_opened": False,
        "labels_stored": False,
        "storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
    }
    payload = {**unhashed, "development_prediction_seal_hash": canonical_sha256(unhashed)}
    _persist_or_validate_seal(seal, payload)
    return DevelopmentPredictionCapability(store, payload, arrays, index, seal)


def seal_target_predictions(
    store: PredictionStore,
    *,
    root: Path,
    target_policy_plan_hashes_by_center: Mapping[str, str],
    target_policy_plan_set_hash: str,
    frozen_action_set_hash: str,
    global_prelabel_seal_hash: str,
) -> TargetPredictionCapability:
    if store.phase != TARGET_ROLE:
        raise ProtocolError("Cannot issue target capability for another store.")
    plan_hashes = {str(key): str(value) for key, value in target_policy_plan_hashes_by_center.items()}
    if tuple(plan_hashes) != CENTERS:
        raise ProtocolError("Endpoint-router target plan coverage is incomplete.")
    arrays = root / TARGET_ARRAY_MEMBER
    index = root / TARGET_INDEX_MEMBER
    seal = root / TARGET_SEAL_MEMBER
    unhashed = {
        "schema_version": "midogpp_endpoint_router_target_prediction_seal_v1",
        "status": "SEALED_ALL_TARGET_PROBABILITIES_POLICIES_AND_ACTIONS",
        "prediction_store_hash": store.store_hash,
        "source_stream_lock_hash": store.source_stream_lock_hash,
        "partition_lock_hash": store.partition_lock_hash,
        "cache_binding_hash": store.cache_binding_hash,
        "action_library_hash": store.action_library_hash,
        "arrays_member": TARGET_ARRAY_MEMBER,
        "arrays_sha256": sha256_file(arrays),
        "index_member": TARGET_INDEX_MEMBER,
        "index_sha256": sha256_file(index),
        "cell_count": TARGET_CELL_COUNT,
        "task_count": TARGET_TASK_COUNT,
        "physical_action_identity_count": 90,
        "reported_method_count": 117,
        "unique_classifier_fit_count": TARGET_CELL_COUNT,
        "target_policy_plan_hashes_by_center": plan_hashes,
        "target_policy_plan_set_hash": str(target_policy_plan_set_hash),
        "frozen_action_set_hash": str(frozen_action_set_hash),
        "global_prelabel_seal_hash": str(global_prelabel_seal_hash),
        "all_nine_target_policy_plans_frozen": True,
        "all_target_actions_frozen": True,
        "all_support_and_evaluation_probabilities_materialized": True,
        "support_labels_opened": False,
        "same_outer_H_evaluation_labels_opened": False,
        "labels_stored": False,
        "storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
    }
    payload = {**unhashed, "target_prediction_seal_hash": canonical_sha256(unhashed)}
    _persist_or_validate_seal(seal, payload)
    return TargetPredictionCapability(store, payload, arrays, index, seal)


def _persist_or_validate_seal(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise ProtocolError("Endpoint-router persisted prediction seal drifted.")
        return
    atomic_json(path, payload)


__all__ = (
    "DevelopmentPredictionCapability",
    "TargetPredictionCapability",
    "seal_development_predictions",
    "seal_target_predictions",
)
