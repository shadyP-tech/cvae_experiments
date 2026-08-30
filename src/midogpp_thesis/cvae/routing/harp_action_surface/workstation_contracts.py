"""Neutral workstation DTOs shared by HARP plan construction and execution.

This module deliberately has no dependency on either workstation producer or
runtime.  Keeping the serializable task graph here makes both sides importable
in isolation and preserves the parent-process CUDA boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import (
    DEVELOPMENT_SURFACE,
    TARGET_SURFACE,
    HarpPredictionMenuSeal,
)
from ..harp_protocol.hashing import canonical_hash, require_sha256


@dataclass(frozen=True, kw_only=True)
class HarpGenerationTask:
    """One persistent-GPU-worker source job (three generation seeds)."""

    ordinal: int
    source_center: str
    training_seed: int
    generation_seeds: tuple[int, ...]
    device: str
    checkpoint_array_member: str
    checkpoint_receipt_member: str
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or not self.source_center
            or self.training_seed not in (17, 42, 101)
            or self.generation_seeds != (17, 42, 101)
            or self.device not in ("cuda:0", "cuda:1")
            or not self.checkpoint_array_member.endswith(".npy")
            or not self.checkpoint_receipt_member.endswith(".json")
        ):
            raise ProtocolError("HARP generation task topology drifted.")
        object.__setattr__(self, "task_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_generation_task_v1",
            "ordinal": self.ordinal,
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "generation_seeds": list(self.generation_seeds),
            "device": self.device,
            "checkpoint_array_member": self.checkpoint_array_member,
            "checkpoint_receipt_member": self.checkpoint_receipt_member,
            "persistent_worker_per_device": True,
            "transport_dtype": "float32",
            "late_torch_interop_setter_used": False,
        }


@dataclass(frozen=True, kw_only=True)
class HarpClassifierTask:
    """One query/seed task; a worker fits every action in that query menu."""

    ordinal: int
    surface_kind: str
    outer_target_id: str
    query_center_id: str
    training_seed: int
    generation_seed: int
    action_hashes: tuple[str, ...]
    checkpoint_array_member: str
    checkpoint_receipt_member: str
    threads_per_worker: int = 3
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected_action_count = 9 if self.surface_kind == DEVELOPMENT_SURFACE else 10
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or self.surface_kind not in (DEVELOPMENT_SURFACE, TARGET_SURFACE)
            or not self.outer_target_id
            or not self.query_center_id
            or (
                self.surface_kind == DEVELOPMENT_SURFACE
                and self.outer_target_id == self.query_center_id
            )
            or (
                self.surface_kind == TARGET_SURFACE
                and self.outer_target_id != self.query_center_id
            )
            or self.training_seed not in (17, 42, 101)
            or self.generation_seed not in (17, 42, 101)
            or len(self.action_hashes) != expected_action_count
            or len(set(self.action_hashes)) != len(self.action_hashes)
            or any(len(value) != 64 for value in self.action_hashes)
            or self.threads_per_worker != 3
            or not self.checkpoint_array_member.endswith(".npz")
            or not self.checkpoint_receipt_member.endswith(".json")
        ):
            raise ProtocolError("HARP classifier task topology drifted.")
        object.__setattr__(self, "task_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_classifier_task_v2",
            "ordinal": self.ordinal,
            "surface_kind": self.surface_kind,
            "outer_target_id": self.outer_target_id,
            "query_center_id": self.query_center_id,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "action_hashes": list(self.action_hashes),
            "checkpoint_array_member": self.checkpoint_array_member,
            "checkpoint_receipt_member": self.checkpoint_receipt_member,
            "threads_per_worker": self.threads_per_worker,
            "classifier_pool_size": 4,
            "nested_process_pools": False,
            "target_or_source_labels_available": False,
        }


@dataclass(frozen=True, kw_only=True)
class HarpWorkstationPlan:
    surface_kind: str
    config_contract_hash: str
    readiness_binding_hash: str
    generation_devices: tuple[str, ...]
    generation_tasks: tuple[HarpGenerationTask, ...]
    classifier_tasks: tuple[HarpClassifierTask, ...]
    action_hashes: tuple[str, ...]
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.config_contract_hash, name="HARP config contract hash")
        require_sha256(self.readiness_binding_hash, name="HARP readiness binding hash")
        expected_classifier_count = 648 if self.surface_kind == DEVELOPMENT_SURFACE else 81
        expected_action_count = 648 if self.surface_kind == DEVELOPMENT_SURFACE else 90
        expected_cell_count = 5832 if self.surface_kind == DEVELOPMENT_SURFACE else 810
        if (
            self.surface_kind not in (DEVELOPMENT_SURFACE, TARGET_SURFACE)
            or self.generation_devices != ("cuda:0", "cuda:1")
            or len(self.generation_tasks) != 27
            or tuple(task.ordinal for task in self.generation_tasks) != tuple(range(27))
            or len(self.classifier_tasks) != expected_classifier_count
            or tuple(task.ordinal for task in self.classifier_tasks)
            != tuple(range(expected_classifier_count))
            or len(self.action_hashes) != expected_action_count
            or sum(len(task.action_hashes) for task in self.classifier_tasks)
            != expected_cell_count
        ):
            raise ProtocolError("HARP workstation plan coverage drifted.")
        object.__setattr__(self, "plan_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_workstation_plan_v2",
            "surface_kind": self.surface_kind,
            "config_contract_hash": self.config_contract_hash,
            "readiness_binding_hash": self.readiness_binding_hash,
            "generation_devices": list(self.generation_devices),
            "generation_workers": 2,
            "persistent_workers_per_gpu": 1,
            "generation_tasks": [
                task.to_payload() | {"task_hash": task.task_hash}
                for task in self.generation_tasks
            ],
            "classifier_workers": 4,
            "classifier_threads_per_worker": 3,
            "classifier_tasks": [
                task.to_payload() | {"task_hash": task.task_hash}
                for task in self.classifier_tasks
            ],
            "action_hashes": list(self.action_hashes),
            "multiprocessing_start_method": "spawn",
            "parent_cuda_context_created": False,
            "gpu_and_cpu_phases_disjoint": True,
            "nested_process_pools": False,
            "late_torch_interop_setter_used": False,
            "probability_transport": "float32_memmap",
            "exact_nine_reduction_dtype": "float64",
            "labels_available_to_workers": False,
        }


class HarpPrimitivePredictor(Protocol):
    """Test/provider seam; production must use a checked-in physical loader."""

    def materialize(self, plan: HarpWorkstationPlan) -> HarpPredictionMenuSeal: ...


__all__ = (
    "HarpClassifierTask",
    "HarpGenerationTask",
    "HarpPrimitivePredictor",
    "HarpWorkstationPlan",
)
