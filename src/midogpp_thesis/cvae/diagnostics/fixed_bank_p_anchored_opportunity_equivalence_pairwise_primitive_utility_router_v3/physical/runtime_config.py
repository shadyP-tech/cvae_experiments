"""V3-owned immutable runtime identity for fixed-bank probability work."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
)

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..workstation import (
    BLAS_THREADS_PER_CPU_WORKER,
    CPU_SPAWN_WORKER_COUNT,
    MULTIPROCESSING_START_METHOD,
)


@dataclass(frozen=True, slots=True)
class PhysicalRuntimeConfig:
    """Only the fields consumed by the sealed neutral lower-level runtime."""

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
        if not root.is_absolute() or root == Path(root.anchor):
            raise ProtocolError("OE-PPUR v3 expert-bank root is not an admitted path.")
        expected = physical_runtime_payload()
        supplied = dict(self.runtime) if self.runtime else expected
        if type(self.classifier) is not ClassifierSpec or supplied != expected:
            raise ProtocolError("OE-PPUR v3 physical runtime contract drifted.")
        object.__setattr__(self, "expert_bank_root", root)
        object.__setattr__(self, "runtime", MappingProxyType(supplied))
        object.__setattr__(
            self,
            "contract_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_physical_runtime_config_v1",
                    "classifier": self.classifier.to_payload(),
                    "runtime": expected,
                    "expert_bank_role": "direct_input_1",
                    "paths_persisted": False,
                }
            ),
        )


def physical_runtime_payload() -> dict[str, object]:
    return {
        "generation_devices": ["cuda:0", "cuda:1"],
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "multiprocessing_start_method": MULTIPROCESSING_START_METHOD,
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "generated_cache_format": "float32_npy_memmap",
        "source_prefix_rows_per_class": 270,
        "classifier_workers": CPU_SPAWN_WORKER_COUNT,
        "classifier_threads_per_worker": BLAS_THREADS_PER_CPU_WORKER,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "scientific_reductions_dtype": "float64",
        "target_task_count": 81,
        "target_probability_cell_count": 810,
        "maximum_total_classifier_fit_count": 810,
        "worker_transport": "pickle_plain_mapping_only",
    }


__all__ = ("PhysicalRuntimeConfig", "physical_runtime_payload")
