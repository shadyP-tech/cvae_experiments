"""Complete prediction orchestration and the durable pre-label global seal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping, Protocol

from ....common.hashing import stable_hash
from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from .actions import build_inner_exact_tail_action_library
from .contracts import CENTERS
from .development_prediction_contracts import (
    EXPECTED_COARSE_TASK_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    PredictionCheckpointRecord,
    expected_prediction_keys,
)
from .development_prediction_planning import (
    build_prediction_worker_inputs,
    execute_pending_prediction_tasks,
    validate_runtime,
    write_evaluation_scratch,
)
from .development_prediction_store import (
    DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
    DEVELOPMENT_PREDICTION_INDEX_MEMBER,
    DevelopmentPredictionStore,
    consolidate_prediction_records,
    load_development_prediction_store,
    load_prediction_checkpoint,
)
from .input_contracts import row_identity_hash
from .source_cache_contracts import SOURCE_CACHE_LOCK_MEMBER, SourceCache
from .source_cache_store import atomic_write_json, read_json, sha256_file


GLOBAL_DEVELOPMENT_SEAL_MEMBER = (
    "manifests/utility_aligned_global_development_prediction_seal.json"
)


class DevelopmentExecutionConfig(Protocol):
    contract_hash: str
    expected_manifest_sha256: str
    runtime: Mapping[str, object]
    classifier: object


@dataclass(frozen=True)
class GlobalDevelopmentPredictionSeal:
    config_contract_hash: str
    generation_lock_hash: str
    source_cache_lock_hash: str
    partition_lock_hash: str
    support_partition_hash_by_center: Mapping[str, str]
    evaluation_row_hash_by_center: Mapping[str, str]
    development_manifest_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    prediction_index_hash: str
    canonical_inner_action_library_hash: str
    cell_binding_hash: str
    cell_count: int
    prediction_seal_hash: str

    def __post_init__(self) -> None:
        support = {str(key): str(value) for key, value in self.support_partition_hash_by_center.items()}
        evaluation = {str(key): str(value) for key, value in self.evaluation_row_hash_by_center.items()}
        if tuple(support) != CENTERS or tuple(evaluation) != CENTERS:
            raise ProtocolError("Stage-90 prediction seal center coverage drifted.")
        if self.cell_count != EXPECTED_PREDICTION_CELL_COUNT:
            raise ProtocolError("Stage-90 prediction seal cell count drifted.")
        if self.canonical_inner_action_library_hash != (
            build_inner_exact_tail_action_library().action_library_hash
        ):
            raise ProtocolError("Stage-90 prediction seal action-library hash drifted.")
        if self.prediction_seal_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Stage-90 prediction seal hash drifted.")
        object.__setattr__(
            self, "support_partition_hash_by_center", MappingProxyType(support)
        )
        object.__setattr__(
            self, "evaluation_row_hash_by_center", MappingProxyType(evaluation)
        )

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_utility_aligned_global_prediction_seal_v1",
            "status": "COMPLETE_BEFORE_ANY_DEVELOPMENT_LABEL_ACCESS",
            "config_contract_hash": self.config_contract_hash,
            "generation_lock_hash": self.generation_lock_hash,
            "source_cache_lock_hash": self.source_cache_lock_hash,
            "partition_lock_hash": self.partition_lock_hash,
            "support_partition_hash_by_center": dict(self.support_partition_hash_by_center),
            "evaluation_row_hash_by_center": dict(self.evaluation_row_hash_by_center),
            "development_manifest_sha256": self.development_manifest_sha256,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "prediction_index_hash": self.prediction_index_hash,
            "canonical_inner_action_library_hash": self.canonical_inner_action_library_hash,
            "cell_binding_hash": self.cell_binding_hash,
            "cell_count": self.cell_count,
            "coarse_task_count": EXPECTED_COARSE_TASK_COUNT,
            "all_base_and_tail_predictions_materialized": True,
            "strict_H_q_e_exclusion": True,
            "labels_opened": False,
            "target_labels_used_for_routing": False,
            "diagnostic_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "prediction_seal_hash": self.prediction_seal_hash}


@dataclass(frozen=True)
class DevelopmentPredictionCapability:
    store: DevelopmentPredictionStore
    seal: GlobalDevelopmentPredictionSeal
    seal_path: Path
    prediction_index_path: Path
    prediction_arrays_path: Path


def build_global_development_seal(
    config: DevelopmentExecutionConfig,
    generation_lock: GenerationLock,
    source_cache: SourceCache,
    partitions: object,
    store: DevelopmentPredictionStore,
    *,
    root: Path,
) -> GlobalDevelopmentPredictionSeal:
    source_lock = read_json(source_cache.root / SOURCE_CACHE_LOCK_MEMBER)
    support_by_center = getattr(partitions, "support_rows_by_center", {})
    evaluation_by_center = getattr(partitions, "evaluation_rows_by_center", {})
    support_hashes = {
        center: row_identity_hash(tuple(support_by_center[center])) for center in CENTERS
    }
    evaluation_hashes = {
        center: row_identity_hash(tuple(evaluation_by_center[center])) for center in CENTERS
    }
    cell_binding_hash = stable_hash([dict(row) for row in store.index_rows])
    values: dict[str, object] = {
        "config_contract_hash": str(config.contract_hash),
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "source_cache_lock_hash": str(source_lock["source_cache_lock_hash"]),
        "partition_lock_hash": str(getattr(partitions, "lock_hash", "")),
        "support_partition_hash_by_center": support_hashes,
        "evaluation_row_hash_by_center": evaluation_hashes,
        "development_manifest_sha256": str(config.expected_manifest_sha256),
        "prediction_index_sha256": sha256_file(
            root / DEVELOPMENT_PREDICTION_INDEX_MEMBER
        ),
        "prediction_arrays_sha256": sha256_file(
            root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER
        ),
        "prediction_index_hash": store.prediction_index_hash,
        "canonical_inner_action_library_hash": (
            build_inner_exact_tail_action_library().action_library_hash
        ),
        "cell_binding_hash": cell_binding_hash,
        "cell_count": len(store.index_rows),
        "prediction_seal_hash": "",
    }
    provisional = GlobalDevelopmentPredictionSeal.__new__(
        GlobalDevelopmentPredictionSeal
    )
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    values["prediction_seal_hash"] = stable_hash(provisional._unhashed_payload())
    return GlobalDevelopmentPredictionSeal(**values)  # type: ignore[arg-type]


def validate_global_development_seal(
    capability: DevelopmentPredictionCapability,
    *,
    config: DevelopmentExecutionConfig,
    generation_lock: GenerationLock,
    source_cache: SourceCache,
    partitions: object,
    root: Path,
) -> None:
    observed = read_json(capability.seal_path)
    if observed != capability.seal.to_payload():
        raise ProtocolError("Persisted Stage-90 prediction seal drifted.")
    expected = build_global_development_seal(
        config,
        generation_lock,
        source_cache,
        partitions,
        capability.store,
        root=root,
    )
    if expected.to_payload() != capability.seal.to_payload():
        raise ProtocolError("Stage-90 prediction seal binding drifted.")


def load_global_development_seal(path: Path) -> GlobalDevelopmentPredictionSeal:
    raw = read_json(path)
    expected = set(GlobalDevelopmentPredictionSeal.__dataclass_fields__)
    if set(raw) != expected.union(
        {
            "schema_version", "status", "coarse_task_count",
            "all_base_and_tail_predictions_materialized", "strict_H_q_e_exclusion",
            "labels_opened", "target_labels_used_for_routing", "diagnostic_only",
        }
    ):
        raise ProtocolError("Stage-90 prediction seal schema drifted.")
    if (
        raw.get("schema_version")
        != "midogpp_stage90_utility_aligned_global_prediction_seal_v1"
        or raw.get("status") != "COMPLETE_BEFORE_ANY_DEVELOPMENT_LABEL_ACCESS"
        or int(raw.get("coarse_task_count", -1)) != EXPECTED_COARSE_TASK_COUNT
        or raw.get("all_base_and_tail_predictions_materialized") is not True
        or raw.get("strict_H_q_e_exclusion") is not True
        or raw.get("labels_opened") is not False
        or raw.get("target_labels_used_for_routing") is not False
        or raw.get("diagnostic_only") is not True
    ):
        raise ProtocolError("Stage-90 prediction seal claim boundary drifted.")
    return GlobalDevelopmentPredictionSeal(
        config_contract_hash=str(raw["config_contract_hash"]),
        generation_lock_hash=str(raw["generation_lock_hash"]),
        source_cache_lock_hash=str(raw["source_cache_lock_hash"]),
        partition_lock_hash=str(raw["partition_lock_hash"]),
        support_partition_hash_by_center=dict(raw["support_partition_hash_by_center"]),
        evaluation_row_hash_by_center=dict(raw["evaluation_row_hash_by_center"]),
        development_manifest_sha256=str(raw["development_manifest_sha256"]),
        prediction_index_sha256=str(raw["prediction_index_sha256"]),
        prediction_arrays_sha256=str(raw["prediction_arrays_sha256"]),
        prediction_index_hash=str(raw["prediction_index_hash"]),
        canonical_inner_action_library_hash=str(
            raw["canonical_inner_action_library_hash"]
        ),
        cell_binding_hash=str(raw["cell_binding_hash"]),
        cell_count=int(raw["cell_count"]),
        prediction_seal_hash=str(raw["prediction_seal_hash"]),
    )


def materialize_development_predictions(
    config: DevelopmentExecutionConfig,
    generation_lock: GenerationLock,
    source_cache: SourceCache,
    frame: object,
    partitions: object,
    *,
    root: Path,
) -> DevelopmentPredictionCapability:
    """Resume 648 atomic tasks, consolidate, and seal before labels."""

    validate_runtime(config)  # type: ignore[arg-type]
    arrays_path = root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER
    index_path = root / DEVELOPMENT_PREDICTION_INDEX_MEMBER
    seal_path = root / GLOBAL_DEVELOPMENT_SEAL_MEMBER
    if arrays_path.is_file() and index_path.is_file() and seal_path.is_file():
        capability = DevelopmentPredictionCapability(
            store=load_development_prediction_store(root),
            seal=load_global_development_seal(seal_path),
            seal_path=seal_path,
            prediction_index_path=index_path,
            prediction_arrays_path=arrays_path,
        )
        validate_global_development_seal(
            capability,
            config=config,
            generation_lock=generation_lock,
            source_cache=source_cache,
            partitions=partitions,
            root=root,
        )
        return capability

    checkpoint_root = root / "checkpoints/utility_aligned_development_predictions"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    scratch = write_evaluation_scratch(
        checkpoint_root, frame=frame, partitions=partitions
    )
    inputs = build_prediction_worker_inputs(
        config,  # type: ignore[arg-type]
        source_cache,
        partitions,
        scratch=scratch,
        checkpoint_root=checkpoint_root,
        generation_lock_hash=generation_lock.generation_lock_hash,
    )
    completed: dict[tuple[str, str, int, int], PredictionCheckpointRecord] = {}
    pending = []
    for item in inputs:
        record = load_prediction_checkpoint(item)
        if record is None:
            pending.append(item)
        else:
            completed[item.task.key] = record
    pending_by_key = {item.task.key: item for item in pending}
    for new_count, returned in enumerate(
        execute_pending_prediction_tasks(pending), start=1
    ):
        item = pending_by_key[returned.task.key]
        verified = load_prediction_checkpoint(item)
        if verified is None or verified.checkpoint_hash != returned.checkpoint_hash:
            raise ProtocolError("Stage-90 prediction checkpoint return drifted.")
        completed[item.task.key] = verified
        print(
            f"[utility-exact-tail] inner jobs {len(completed)}/{EXPECTED_COARSE_TASK_COUNT} "
            f"(new {new_count}/{len(pending)})",
            flush=True,
        )
    if len(completed) != EXPECTED_COARSE_TASK_COUNT:
        raise ProtocolError("Stage-90 prediction task coverage is incomplete.")
    records = tuple(completed[item.task.key] for item in inputs)
    store = consolidate_prediction_records(records, root=root)
    seal = build_global_development_seal(
        config, generation_lock, source_cache, partitions, store, root=root
    )
    atomic_write_json(seal_path, seal.to_payload())
    capability = DevelopmentPredictionCapability(
        store=store,
        seal=seal,
        seal_path=seal_path,
        prediction_index_path=index_path,
        prediction_arrays_path=arrays_path,
    )
    validate_global_development_seal(
        capability,
        config=config,
        generation_lock=generation_lock,
        source_cache=source_cache,
        partitions=partitions,
        root=root,
    )
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return capability


__all__ = (
    "GLOBAL_DEVELOPMENT_SEAL_MEMBER",
    "DevelopmentPredictionCapability",
    "GlobalDevelopmentPredictionSeal",
    "build_global_development_seal",
    "load_global_development_seal",
    "materialize_development_predictions",
    "validate_global_development_seal",
)
