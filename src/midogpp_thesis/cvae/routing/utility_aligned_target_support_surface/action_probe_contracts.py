"""Typed contracts for the label-free target-support action probe."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned.ensemble_contracts import (
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
)
from ..utility_aligned_identities import CENTERS


ACTION_PROBE_SCHEMA = "midogpp_target_support_action_probe_task_v1"
ACTION_PROBE_RUNTIME_SCHEMA = "midogpp_target_support_action_probe_runtime_v1"
ACTION_SHIFT_ROW_SCHEMA = "midogpp_utility_aligned_target_support_action_shift_row_v2"
ACTION_SHIFT_LOCK_SCHEMA = (
    "midogpp_utility_aligned_target_support_action_shifts_lock_v2"
)
ACTION_SHIFT_TABLE_MEMBER = "tables/target_support_action_shifts.csv"
ACTION_SHIFT_LOCK_MEMBER = "manifests/target_support_action_shifts_lock.json"
ACTION_SHIFT_ROW_SCALAR_SEMANTICS = (
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
)
ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS = (
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS
)
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
TARGET_BASE_PER_SOURCE = 128
TARGET_TAIL_PER_SELECTED_SOURCE = 128
SOURCE_ROWS_PER_CLASS = 270


@dataclass(frozen=True)
class ActionProbeRuntime:
    """Operational CPU topology for the post-generation action-probe phase."""

    classifier_workers: int
    threads_per_worker: int
    task_count: int
    fit_count: int
    multiprocessing_start_method: str
    execution_order: str = "source_generation_then_cpu_action_probe"
    gpu_cpu_overlap_allowed: bool = False

    def __post_init__(self) -> None:
        try:
            integer_values = tuple(
                int(value)
                for value in (
                    self.classifier_workers,
                    self.threads_per_worker,
                    self.task_count,
                    self.fit_count,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Target-support action-probe runtime drifted.") from exc
        workers, threads, task_count, fit_count = integer_values
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    self.classifier_workers,
                    self.threads_per_worker,
                    self.task_count,
                    self.fit_count,
                )
            )
            or workers != 4
            or threads != 3
            or task_count
            != len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
            or fit_count != task_count * (len(CENTERS) - 1 + 1)
            or str(self.multiprocessing_start_method) != "spawn"
            or str(self.execution_order) != "source_generation_then_cpu_action_probe"
            or self.gpu_cpu_overlap_allowed is not False
        ):
            raise ProtocolError("Target-support action-probe runtime drifted.")
        object.__setattr__(self, "classifier_workers", workers)
        object.__setattr__(self, "threads_per_worker", threads)
        object.__setattr__(self, "task_count", task_count)
        object.__setattr__(self, "fit_count", fit_count)
        object.__setattr__(
            self,
            "multiprocessing_start_method",
            str(self.multiprocessing_start_method),
        )

    @property
    def fits_per_task(self) -> int:
        return self.fit_count // self.task_count

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": ACTION_PROBE_RUNTIME_SCHEMA,
            "classifier_workers": self.classifier_workers,
            "threads_per_worker": self.threads_per_worker,
            "task_count": self.task_count,
            "fits_per_task": self.fits_per_task,
            "fit_count": self.fit_count,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "execution_order": self.execution_order,
            "gpu_cpu_overlap_allowed": False,
        }


def action_probe_runtime_from_config(
    runtime: Mapping[str, object],
) -> ActionProbeRuntime:
    """Construct the typed execution contract from the strict workspace config."""

    try:
        result = ActionProbeRuntime(
            classifier_workers=int(runtime["action_probe_classifier_workers"]),
            threads_per_worker=int(runtime["action_probe_threads_per_worker"]),
            task_count=int(runtime["action_probe_task_count"]),
            fit_count=int(runtime["action_probe_fit_count"]),
            multiprocessing_start_method=str(
                runtime["multiprocessing_start_method"]
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Target-support action-probe runtime is malformed.") from exc
    return result


def workstation_action_probe_runtime() -> ActionProbeRuntime:
    """Canonical CPU allocation for the declared 12-core workstation phase."""

    return ActionProbeRuntime(
        classifier_workers=4,
        threads_per_worker=3,
        task_count=81,
        fit_count=729,
        multiprocessing_start_method="spawn",
    )


@dataclass(frozen=True)
class ActionProbeTask:
    """One target and seed-pair task; B and all eight tails share one fit batch."""

    task_ordinal: int
    target_id: str
    training_seed: int
    generation_seed: int
    candidate_sources: tuple[str, ...]
    support_array_path: str
    support_file_sha256: str
    support_partition_hash: str
    support_case_ids: tuple[str, ...]
    support_sample_ids: tuple[str, ...]
    source_array_path_by_source: Mapping[str, str]
    source_file_sha256_by_source: Mapping[str, str]
    generated_cache_hash: str
    classifier_payload: Mapping[str, object]
    runtime: ActionProbeRuntime
    checkpoint_root: str
    task_hash: str = ""

    def __post_init__(self) -> None:
        sources = tuple(str(value) for value in self.candidate_sources)
        expected = tuple(value for value in CENTERS if value != str(self.target_id))
        paths = {str(key): str(value) for key, value in self.source_array_path_by_source.items()}
        hashes = {str(key): str(value) for key, value in self.source_file_sha256_by_source.items()}
        if (
            isinstance(self.task_ordinal, bool)
            or int(self.task_ordinal) < 0
            or str(self.target_id) not in CENTERS
            or sources != expected
            or int(self.training_seed) not in TRAINING_SEEDS
            or int(self.generation_seed) not in GENERATION_SEEDS
            or tuple(paths) != sources
            or tuple(hashes) != sources
            or not self.support_array_path
            or not _is_sha(self.support_file_sha256)
            or not self.support_partition_hash
            or not self.generated_cache_hash
            or len(self.support_case_ids) != len(self.support_sample_ids)
            or not self.support_case_ids
            or not isinstance(self.runtime, ActionProbeRuntime)
        ):
            raise ProtocolError("Target-support action-probe task drifted.")
        payload = self.identity_payload()
        expected_hash = canonical_sha256(payload)
        if self.task_hash and self.task_hash != expected_hash:
            raise ProtocolError("Target-support action-probe task hash drifted.")
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(self, "generation_seed", int(self.generation_seed))
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "source_array_path_by_source", MappingProxyType(paths))
        object.__setattr__(self, "source_file_sha256_by_source", MappingProxyType(hashes))
        object.__setattr__(self, "classifier_payload", MappingProxyType(dict(self.classifier_payload)))
        object.__setattr__(self, "task_hash", expected_hash)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": ACTION_PROBE_SCHEMA,
            "task_ordinal": int(self.task_ordinal),
            "target_id": str(self.target_id),
            "training_seed": int(self.training_seed),
            "generation_seed": int(self.generation_seed),
            "candidate_sources": list(self.candidate_sources),
            "support_array_path": str(self.support_array_path),
            "support_file_sha256": str(self.support_file_sha256),
            "support_partition_hash": str(self.support_partition_hash),
            "support_case_ids": list(self.support_case_ids),
            "support_sample_ids": list(self.support_sample_ids),
            "source_array_path_by_source": dict(self.source_array_path_by_source),
            "source_file_sha256_by_source": dict(self.source_file_sha256_by_source),
            "generated_cache_hash": str(self.generated_cache_hash),
            "classifier_payload": dict(self.classifier_payload),
            "runtime": self.runtime.to_payload(),
            "checkpoint_root": str(self.checkpoint_root),
            "action_geometry": action_geometry_payload(self.candidate_sources),
            "labels_used": False,
        }

    @property
    def checkpoint_stem(self) -> str:
        return (
            f"target_{self.target_id}_train_{self.training_seed}_"
            f"generation_{self.generation_seed}"
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.task_ordinal,
                self.target_id,
                self.training_seed,
                self.generation_seed,
                self.candidate_sources,
                self.support_array_path,
                self.support_file_sha256,
                self.support_partition_hash,
                self.support_case_ids,
                self.support_sample_ids,
                dict(self.source_array_path_by_source),
                dict(self.source_file_sha256_by_source),
                self.generated_cache_hash,
                dict(self.classifier_payload),
                self.runtime,
                self.checkpoint_root,
                self.task_hash,
            ),
        )


@dataclass(frozen=True)
class TargetSupportActionShiftRow:
    """One descriptive seed row bound to one case-level ensemble scalar."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    training_seed: int
    generation_seed: int
    case_id: str
    support_partition_hash: str
    case_row_identity_hash: str
    support_row_count: int
    base_probability_sha256: str
    tail_probability_sha256: str
    base_component_vector_hash: str
    tail_component_vector_hash: str
    descriptive_seed_mean_absolute_positive_probability_shift: float
    case_ensemble_mean_absolute_positive_probability_shift: float
    case_base_ensemble_probability_sha256: str
    case_tail_ensemble_probability_sha256: str
    case_ensemble_absolute_difference_sha256: str
    case_ensemble_shift_hash: str
    scalar_name: str = SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
    scalar_semantics: str = ACTION_SHIFT_ROW_SCALAR_SEMANTICS
    labels_used: bool = False
    row_hash: str = ""

    def __post_init__(self) -> None:
        descriptive_value = float(
            self.descriptive_seed_mean_absolute_positive_probability_shift
        )
        ensemble_value = float(
            self.case_ensemble_mean_absolute_positive_probability_shift
        )
        if (
            self.outer_target_id not in CENTERS
            or self.query_id != self.outer_target_id
            or self.candidate_source not in CENTERS
            or self.candidate_source == self.outer_target_id
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or not self.case_id
            or not all(
                _is_sha(value)
                for value in (
                    self.support_partition_hash,
                    self.case_row_identity_hash,
                    self.base_probability_sha256,
                    self.tail_probability_sha256,
                    self.base_component_vector_hash,
                    self.tail_component_vector_hash,
                    self.case_base_ensemble_probability_sha256,
                    self.case_tail_ensemble_probability_sha256,
                    self.case_ensemble_absolute_difference_sha256,
                    self.case_ensemble_shift_hash,
                )
            )
            or isinstance(self.support_row_count, bool)
            or int(self.support_row_count) <= 0
            or not isfinite(descriptive_value)
            or descriptive_value < 0.0
            or descriptive_value > 1.0
            or not isfinite(ensemble_value)
            or ensemble_value < 0.0
            or ensemble_value > 1.0
            or self.scalar_name != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
            or self.scalar_semantics != ACTION_SHIFT_ROW_SCALAR_SEMANTICS
            or self.labels_used is not False
        ):
            raise ProtocolError("Target-support action-shift row drifted.")
        payload = self.payload_without_hash()
        expected = canonical_sha256(payload)
        if self.row_hash and self.row_hash != expected:
            raise ProtocolError("Target-support action-shift row hash drifted.")
        object.__setattr__(
            self,
            "descriptive_seed_mean_absolute_positive_probability_shift",
            descriptive_value,
        )
        object.__setattr__(
            self,
            "case_ensemble_mean_absolute_positive_probability_shift",
            ensemble_value,
        )
        object.__setattr__(self, "support_row_count", int(self.support_row_count))
        object.__setattr__(self, "row_hash", expected)

    @property
    def row_key(self) -> tuple[str, str, int, int, str]:
        return (
            self.outer_target_id,
            self.candidate_source,
            self.training_seed,
            self.generation_seed,
            self.case_id,
        )

    def payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": ACTION_SHIFT_ROW_SCHEMA,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "case_id": self.case_id,
            "support_partition_hash": self.support_partition_hash,
            "case_row_identity_hash": self.case_row_identity_hash,
            "support_row_count": self.support_row_count,
            "base_probability_sha256": self.base_probability_sha256,
            "tail_probability_sha256": self.tail_probability_sha256,
            "base_component_vector_hash": self.base_component_vector_hash,
            "tail_component_vector_hash": self.tail_component_vector_hash,
            "descriptive_seed_mean_absolute_positive_probability_shift": (
                self.descriptive_seed_mean_absolute_positive_probability_shift
            ),
            "case_ensemble_mean_absolute_positive_probability_shift": (
                self.case_ensemble_mean_absolute_positive_probability_shift
            ),
            "case_base_ensemble_probability_sha256": (
                self.case_base_ensemble_probability_sha256
            ),
            "case_tail_ensemble_probability_sha256": (
                self.case_tail_ensemble_probability_sha256
            ),
            "case_ensemble_absolute_difference_sha256": (
                self.case_ensemble_absolute_difference_sha256
            ),
            "case_ensemble_shift_hash": self.case_ensemble_shift_hash,
            "scalar_name": self.scalar_name,
            "scalar_semantics": self.scalar_semantics,
            "descriptive_seed_value_may_feed_model": False,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self.payload_without_hash(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class ActionProbeCheckpoint:
    task_hash: str
    checkpoint_hash: str
    probability_member: str
    probability_file_sha256: str
    action_ids: tuple[str, ...]
    support_row_count: int


@dataclass(frozen=True)
class TargetSupportActionShiftSurface:
    rows: tuple[TargetSupportActionShiftRow, ...]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.rows, key=lambda row: row.row_key))
        if ordered != self.rows or len({row.row_key for row in ordered}) != len(ordered):
            raise ProtocolError("Target-support action-shift row order drifted.")
        object.__setattr__(self, "lock_payload", MappingProxyType(dict(self.lock_payload)))


def action_geometry_payload(candidate_sources: tuple[str, ...]) -> dict[str, object]:
    return {
        "source_order": list(candidate_sources),
        "source_count": len(candidate_sources),
        "base_per_source_per_class": TARGET_BASE_PER_SOURCE,
        "single_source_tail_per_class": TARGET_TAIL_PER_SELECTED_SOURCE,
        "final_total_per_class": (
            len(candidate_sources) * TARGET_BASE_PER_SOURCE
            + TARGET_TAIL_PER_SELECTED_SOURCE
        ),
        "source_rows_available_per_class": SOURCE_ROWS_PER_CLASS,
        "class_labels": [0, 1],
        "action_ids": ["B", *[f"Hxe::{source}" for source in candidate_sources]],
    }


def action_probe_topology_payload(
    runtime: ActionProbeRuntime,
) -> dict[str, object]:
    """Canonical topology binding shared by tasks and the final shift lock."""

    if not isinstance(runtime, ActionProbeRuntime):
        raise ProtocolError("Target-support action-probe runtime drifted.")
    return {
        "task_count": runtime.task_count,
        "fits_per_task": runtime.fits_per_task,
        "fit_count": runtime.fit_count,
        "cpu_workers": runtime.classifier_workers,
        "blas_threads_per_worker": runtime.threads_per_worker,
        "multiprocessing_start_method": runtime.multiprocessing_start_method,
        "execution_order": runtime.execution_order,
        "gpu_cpu_overlap_allowed": False,
    }


def _is_sha(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(character in "0123456789abcdef" for character in rendered)


__all__ = (
    "ACTION_PROBE_SCHEMA",
    "ACTION_PROBE_RUNTIME_SCHEMA",
    "ACTION_SHIFT_LOCK_MEMBER",
    "ACTION_SHIFT_LOCK_SCHEMA",
    "ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS",
    "ACTION_SHIFT_ROW_SCALAR_SEMANTICS",
    "ACTION_SHIFT_ROW_SCHEMA",
    "ACTION_SHIFT_TABLE_MEMBER",
    "ActionProbeCheckpoint",
    "ActionProbeRuntime",
    "ActionProbeTask",
    "TargetSupportActionShiftRow",
    "TargetSupportActionShiftSurface",
    "action_geometry_payload",
    "action_probe_runtime_from_config",
    "action_probe_topology_payload",
    "workstation_action_probe_runtime",
)
