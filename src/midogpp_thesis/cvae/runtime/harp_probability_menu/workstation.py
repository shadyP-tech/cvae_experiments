"""Fail-closed workstation execution contract for HARP probability cells."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .hashing import canonical_sha256


@dataclass(frozen=True, kw_only=True)
class HarpWorkstationContract:
    """Execution facts that every physical materializer must attest.

    CUDA work belongs to spawned children.  The orchestration parent remains
    CUDA-free, and workers must not change PyTorch inter-op thread state after
    any parallel work has begun.  Probability transport is float32; only the
    exact-nine scientific reduction is float64.
    """

    multiprocessing_start_method: str = "spawn"
    parent_cuda_context_created: bool = False
    late_torch_interop_setter_used: bool = False
    transport_dtype: str = "float32"
    scientific_reduction_dtype: str = "float64"
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.multiprocessing_start_method != "spawn"
            or self.parent_cuda_context_created is not False
            or self.late_torch_interop_setter_used is not False
            or self.transport_dtype != "float32"
            or self.scientific_reduction_dtype != "float64"
        ):
            raise ProtocolError("HARP workstation execution contract drifted.")
        object.__setattr__(self, "runtime_hash", canonical_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_workstation_contract_v1",
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "parent_cuda_context_created": self.parent_cuda_context_created,
            "late_torch_interop_setter_used": self.late_torch_interop_setter_used,
            "transport_dtype": self.transport_dtype,
            "scientific_reduction_dtype": self.scientific_reduction_dtype,
            "gpu_process_semantics": "spawned_children_only",
        }


DEFAULT_WORKSTATION_CONTRACT = HarpWorkstationContract()


__all__ = ("DEFAULT_WORKSTATION_CONTRACT", "HarpWorkstationContract")
