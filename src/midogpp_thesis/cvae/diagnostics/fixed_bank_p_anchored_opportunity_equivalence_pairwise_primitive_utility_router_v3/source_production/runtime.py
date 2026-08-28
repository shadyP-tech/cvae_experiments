"""Workstation-locked runtime identity for source-supervision production."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..workstation import CPU_WORKER_ENVIRONMENT


def source_production_runtime_payload() -> dict[str, object]:
    return {
        "generation_devices": ["cuda:0", "cuda:1"],
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "generated_cache_format": "float32_npy_memmap",
        "source_prefix_rows_per_class": 270,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 1,
        "cpu_worker_environment": dict(CPU_WORKER_ENVIRONMENT),
        "phase_disjoint_gpu_and_cpu_pools": True,
        "unordered_held_pair_count": 36,
        "seed_pair_count": 9,
        "cpu_task_count": 324,
        "oriented_output_block_count": 72,
        "actions_per_task": 9,
        "classifier_fit_count": 2916,
        "prediction_storage_dtype": "<f4",
        "scientific_reductions_dtype": "<f8",
        "worker_transport": "pickle_plain_mapping_only",
        "nested_process_pools_allowed": False,
        "labels_visible_to_prediction_workers": False,
    }


@dataclass(frozen=True, slots=True)
class SourceProductionRuntimeConfig:
    expert_bank_root: Path
    classifier: ClassifierSpec = field(
        default_factory=lambda: ClassifierSpec(
            C=0.01,
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            class_weight=None,
            random_state=23,
            l1_ratio=None,
            threshold_policy="predict",
            scaler_fit="synthetic_train_only",
        )
    )
    runtime: Mapping[str, object] = field(default_factory=dict)
    contract_hash: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.expert_bank_root)
        expected = source_production_runtime_payload()
        supplied = expected if not self.runtime else dict(self.runtime)
        if (
            not root.is_absolute()
            or root == Path(root.anchor)
            or type(self.classifier) is not ClassifierSpec
            or supplied != expected
        ):
            raise ProtocolError("OE-PPUR v3 source-production runtime drifted.")
        object.__setattr__(self, "expert_bank_root", root)
        object.__setattr__(self, "runtime", MappingProxyType(supplied))
        object.__setattr__(
            self,
            "contract_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_source_production_runtime_v1",
                    "classifier": self.classifier.to_payload(),
                    "runtime": expected,
                    "expert_bank_role": "direct_input_1",
                    "source_cache_role": "canonical_source_train_cache",
                    "labels_visible_to_prediction_workers": False,
                    "paths_persisted_to_direct_input_3": False,
                }
            ),
        )


__all__ = ("SourceProductionRuntimeConfig", "source_production_runtime_payload")
