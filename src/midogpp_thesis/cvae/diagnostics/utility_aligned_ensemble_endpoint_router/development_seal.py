"""Durable global development-probability seal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation.config import MANIFEST_SHA256
from ...protocol import ProtocolError
from .artifact_io import atomic_json, read_json, sha256_file
from .contracts import CENTERS
from .development_prediction_execution import (
    DEVELOPMENT_ARRAY_MEMBER,
    DEVELOPMENT_INDEX_MEMBER,
    EXPECTED_DEVELOPMENT_CELL_COUNT,
    materialize_development_predictions as _materialize,
)
from .input_contracts import row_identity_hash
from .prediction_contracts import CombinedPredictionStore


GLOBAL_DEVELOPMENT_SEAL_MEMBER = (
    "manifests/ensemble_endpoint_global_development_prediction_seal.json"
)


@dataclass(frozen=True)
class GlobalDevelopmentPredictionSeal:
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        value = dict(self.payload)
        unhashed = {key: item for key, item in value.items() if key != "prediction_seal_hash"}
        if (
            value.get("schema_version") != "midogpp_stage90_ensemble_endpoint_development_seal_v1"
            or value.get("status") != "SEALED_ALL_H_Q_ACTION_PROBABILITIES_BEFORE_LABEL_ACCESS"
            or value.get("prediction_seal_hash") != stable_hash(unhashed)
            or value.get("cell_count") != EXPECTED_DEVELOPMENT_CELL_COUNT
            or value.get("endpoint_response_count") != 504
            or value.get("descriptive_seed_row_count") != 4536
            or value.get("support_labels_opened") is not False
            or value.get("evaluation_labels_opened") is not False
        ):
            raise ProtocolError("Ensemble-endpoint development seal drifted.")
        object.__setattr__(self, "payload", MappingProxyType(value))

    @property
    def prediction_seal_hash(self) -> str:
        return str(self.payload["prediction_seal_hash"])

    @property
    def partition_lock_hash(self) -> str:
        return str(self.payload["partition_lock_hash"])

    @property
    def evaluation_row_hash_by_center(self) -> Mapping[str, str]:
        return MappingProxyType(
            {str(key): str(value) for key, value in self.payload["evaluation_row_hash_by_center"].items()}
        )

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True)
class DevelopmentPredictionCapability:
    store: CombinedPredictionStore
    seal: GlobalDevelopmentPredictionSeal
    seal_path: Path
    prediction_index_path: Path
    prediction_arrays_path: Path


def materialize_development_predictions(*args: object, **kwargs: object) -> DevelopmentPredictionCapability:
    root = Path(kwargs["root"])
    partitions = args[4]
    store = _materialize(*args, **kwargs)
    payload = _seal_payload(root, store=store, partitions=partitions)
    path = root / GLOBAL_DEVELOPMENT_SEAL_MEMBER
    if path.is_file():
        if read_json(path) != payload:
            raise ProtocolError("Persisted ensemble-endpoint development seal drifted.")
    else:
        atomic_json(path, payload)
    seal = GlobalDevelopmentPredictionSeal(read_json(path))
    return DevelopmentPredictionCapability(
        store=store,
        seal=seal,
        seal_path=path,
        prediction_index_path=root / DEVELOPMENT_INDEX_MEMBER,
        prediction_arrays_path=root / DEVELOPMENT_ARRAY_MEMBER,
    )


def _seal_payload(root: Path, *, store: CombinedPredictionStore, partitions: object) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_development_seal_v1",
        "status": "SEALED_ALL_H_Q_ACTION_PROBABILITIES_BEFORE_LABEL_ACCESS",
        "store_hash": store.store_hash,
        "source_cache_lock_hash": store.source_cache_lock_hash,
        "partition_lock_hash": store.partition_lock_hash,
        "action_library_hash": store.action_library_hash,
        "prediction_array_sha256": sha256_file(root / DEVELOPMENT_ARRAY_MEMBER),
        "prediction_index_sha256": sha256_file(root / DEVELOPMENT_INDEX_MEMBER),
        "development_manifest_sha256": MANIFEST_SHA256,
        "support_row_hash_by_center": {
            center: row_identity_hash(partitions.support_rows_by_center[center]) for center in CENTERS
        },
        "evaluation_row_hash_by_center": {
            center: row_identity_hash(partitions.evaluation_rows_by_center[center]) for center in CENTERS
        },
        "cell_count": len(store.cells),
        "endpoint_response_count": 504,
        "descriptive_seed_row_count": 4536,
        "exact_nine_seed_vectors_per_endpoint": True,
        "support_and_evaluation_predicted_by_same_fit": True,
        "all_predictions_materialized": True,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "technical_seed_rows_may_feed_model": False,
        "prior_stage90_output_consumed": False,
    }
    return {**unhashed, "prediction_seal_hash": stable_hash(unhashed)}


def validate_global_development_seal(capability: DevelopmentPredictionCapability) -> Mapping[str, object]:
    if not isinstance(capability, DevelopmentPredictionCapability):
        raise ProtocolError("Development seal validation requires a typed capability.")
    observed = read_json(capability.seal_path)
    if (
        observed != capability.seal.to_payload()
        or observed.get("prediction_array_sha256") != sha256_file(capability.prediction_arrays_path)
        or observed.get("prediction_index_sha256") != sha256_file(capability.prediction_index_path)
        or observed.get("store_hash") != capability.store.store_hash
    ):
        raise ProtocolError("Development sealed prediction bytes drifted.")
    return observed


__all__ = (
    "DevelopmentPredictionCapability",
    "GLOBAL_DEVELOPMENT_SEAL_MEMBER",
    "GlobalDevelopmentPredictionSeal",
    "materialize_development_predictions",
    "validate_global_development_seal",
)
