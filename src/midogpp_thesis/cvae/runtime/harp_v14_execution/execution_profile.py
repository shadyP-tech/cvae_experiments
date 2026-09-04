"""Typed, fail-closed workstation topology for HARP v14 execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .resident_stream_contracts import SOURCE_ROWS_PER_CLASS


@dataclass(frozen=True, slots=True)
class WorkstationProfile:
    """Process, queue, precision, and CUDA ownership constants."""

    profile_name: str = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
    gpu_devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    persistent_gpu_workers: int = 2
    cpu_fit_workers: int = 4
    blas_threads_per_worker: int = 3
    science_workers: int = 4
    science_blas_threads_per_worker: int = 1
    multiprocessing_start_method: str = "spawn"
    parent_cuda_context_created: bool = False
    late_torch_interop_setter_used: bool = False
    probability_transport_dtype: str = "float32"
    scientific_reduction_dtype: str = "float64"
    memory_mapped_surfaces: bool = True
    bounded_inflight_batches_per_gpu: int = 2
    bounded_inflight_tasks_per_cpu_worker: int = 2

    def __post_init__(self) -> None:
        if (
            self.profile_name != "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
            or self.gpu_devices != tuple(f"cuda:{index}" for index in range(2))
            or type(self.persistent_gpu_workers) is not int
            or self.persistent_gpu_workers != len(self.gpu_devices)
            or type(self.cpu_fit_workers) is not int
            or self.cpu_fit_workers != 4
            or type(self.blas_threads_per_worker) is not int
            or self.blas_threads_per_worker != 3
            or type(self.science_workers) is not int
            or self.science_workers != 4
            or type(self.science_blas_threads_per_worker) is not int
            or self.science_blas_threads_per_worker != 1
            or self.multiprocessing_start_method != "spawn"
            or self.parent_cuda_context_created is not False
            or self.late_torch_interop_setter_used is not False
            or self.probability_transport_dtype != "float32"
            or self.scientific_reduction_dtype != "float64"
            or self.memory_mapped_surfaces is not True
            or type(self.bounded_inflight_batches_per_gpu) is not int
            or self.bounded_inflight_batches_per_gpu != 2
            or type(self.bounded_inflight_tasks_per_cpu_worker) is not int
            or self.bounded_inflight_tasks_per_cpu_worker != 2
        ):
            raise ProtocolError("HARP v14 workstation execution profile drifted.")

    @classmethod
    def from_runtime(cls, runtime: Mapping[str, object]) -> WorkstationProfile:
        raw_devices = runtime.get("gpu_devices")
        devices = tuple(raw_devices) if isinstance(raw_devices, (list, tuple)) else ()
        return cls(
            profile_name=runtime.get("profile"),  # type: ignore[arg-type]
            gpu_devices=devices,  # type: ignore[arg-type]
            persistent_gpu_workers=runtime.get("persistent_gpu_workers"),  # type: ignore[arg-type]
            cpu_fit_workers=runtime.get("classifier_workers"),  # type: ignore[arg-type]
            blas_threads_per_worker=runtime.get(
                "classifier_blas_threads_per_worker"
            ),  # type: ignore[arg-type]
            science_workers=runtime.get("science_workers"),  # type: ignore[arg-type]
            science_blas_threads_per_worker=runtime.get(
                "science_blas_threads_per_worker"
            ),  # type: ignore[arg-type]
            multiprocessing_start_method=runtime.get(
                "multiprocessing_start_method"
            ),  # type: ignore[arg-type]
            parent_cuda_context_created=runtime.get(
                "parent_cuda_context_created"
            ),  # type: ignore[arg-type]
            late_torch_interop_setter_used=runtime.get(
                "late_torch_interop_setter_used"
            ),  # type: ignore[arg-type]
            probability_transport_dtype=runtime.get(
                "probability_transport_dtype"
            ),  # type: ignore[arg-type]
            scientific_reduction_dtype=runtime.get(
                "scientific_reduction_dtype"
            ),  # type: ignore[arg-type]
            memory_mapped_surfaces=runtime.get(
                "memory_mapped_surfaces"
            ),  # type: ignore[arg-type]
            bounded_inflight_batches_per_gpu=runtime.get(
                "bounded_inflight_batches_per_gpu"
            ),  # type: ignore[arg-type]
            bounded_inflight_tasks_per_cpu_worker=runtime.get(
                "bounded_inflight_classifier_tasks_per_worker"
            ),  # type: ignore[arg-type]
        )

    @property
    def profile_hash(self) -> str:
        return canonical_hash(self.public_payload())

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v14_workstation_profile_v1",
            "profile": self.profile_name,
            "gpu_devices": list(self.gpu_devices),
            "persistent_gpu_workers": self.persistent_gpu_workers,
            "classifier_workers": self.cpu_fit_workers,
            "classifier_blas_threads_per_worker": self.blas_threads_per_worker,
            "science_workers": self.science_workers,
            "science_blas_threads_per_worker": self.science_blas_threads_per_worker,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "parent_cuda_context_created": self.parent_cuda_context_created,
            "late_torch_interop_setter_used": self.late_torch_interop_setter_used,
            "probability_transport_dtype": self.probability_transport_dtype,
            "scientific_reduction_dtype": self.scientific_reduction_dtype,
            "memory_mapped_surfaces": self.memory_mapped_surfaces,
            "bounded_inflight_batches_per_gpu": self.bounded_inflight_batches_per_gpu,
            "bounded_inflight_classifier_tasks_per_worker": (
                self.bounded_inflight_tasks_per_cpu_worker
            ),
        }

    def source_runtime(self) -> Mapping[str, object]:
        """Return the narrow runtime view owned by persistent GPU workers."""

        return MappingProxyType(
            {
                "generation_devices": list(self.gpu_devices),
                "source_workers_per_device": 1,
                "generation_workers_per_device": 1,
                "persistent_source_workers": True,
                "multiprocessing_start_method": self.multiprocessing_start_method,
                "parent_cuda_context_forbidden": not self.parent_cuda_context_created,
                "tf32_enabled": False,
                "amp_enabled": False,
                "generated_cache_format": (
                    f"{self.probability_transport_dtype}_npy_memmap"
                ),
                "source_prefix_rows_per_class": SOURCE_ROWS_PER_CLASS,
                "bounded_inflight_batches_per_gpu": (
                    self.bounded_inflight_batches_per_gpu
                ),
                "workstation_profile_hash": self.profile_hash,
            }
        )


DEFAULT_WORKSTATION_PROFILE = WorkstationProfile()


__all__ = ("DEFAULT_WORKSTATION_PROFILE", "WorkstationProfile")
