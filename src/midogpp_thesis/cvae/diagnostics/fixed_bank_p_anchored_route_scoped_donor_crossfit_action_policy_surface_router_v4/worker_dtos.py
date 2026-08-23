"""Plain, spawn-pickle-safe DTOs for one complete outer-center task."""

from __future__ import annotations

from dataclasses import dataclass, field
import pickle

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    OuterActionPolicyResult,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    POSTERIOR_CONTROL_IDS,
)
from .identity import canonical_hash, require_sha256
from .method_runtime import OuterMethodRuntimeResult


WORKER_DEPTH_ENV = "MIDOGPP_PDCAPS_V4_WORKER_DEPTH"
WORKER_DTO_KIND = "PLAIN_PICKLE_SAFE_SCIENCE_DTOS_NO_MAPPINGPROXY"


def assert_pickle_safe(value: object, *, role: str) -> None:
    """Fail before pool creation if a DTO contains an unpicklable capability."""

    try:
        pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except (AttributeError, ImportError, pickle.PickleError, TypeError) as exc:
        raise ProtocolError(f"P-DCAPS v4 {role} is not pickle-safe.") from exc


@dataclass(frozen=True)
class OuterRuntimeRequest:
    outer_center: str
    ordinal: int
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        ordinal = int(self.ordinal)
        if outer not in CENTERS or ordinal < 0:
            raise ProtocolError("P-DCAPS v4 outer worker request drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(
            self,
            "request_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_outer_runtime_request_v1",
                    "outer_center": outer,
                    "ordinal": ordinal,
                    "controls_in_one_task": POSTERIOR_CONTROL_IDS,
                    "worker_dto_kind": WORKER_DTO_KIND,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True)
class OuterControlPair:
    request_hash: str
    outer_center: str
    identity_result: OuterActionPolicyResult
    cyclic_result: OuterActionPolicyResult
    pair_hash: str = field(init=False)

    def __post_init__(self) -> None:
        request_hash = require_sha256(self.request_hash, "v4 outer request")
        outer = str(self.outer_center)
        if (
            self.identity_result.outer_center != outer
            or self.cyclic_result.outer_center != outer
            or self.identity_result.posterior_control_id
            != POSTERIOR_CONTROL_IDS[0]
            or self.cyclic_result.posterior_control_id != POSTERIOR_CONTROL_IDS[1]
            or self.identity_result.physical_surface_hash
            != self.cyclic_result.physical_surface_hash
            or self.identity_result.action_surface_seal_hash
            == self.cyclic_result.action_surface_seal_hash
        ):
            raise ProtocolError("P-DCAPS v4 paired outer result drifted.")
        object.__setattr__(self, "request_hash", request_hash)
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(
            self,
            "pair_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_outer_control_pair_v1",
                    "request_hash": request_hash,
                    "outer_center": outer,
                    "identity_result_hash": self.identity_result.result_hash,
                    "cyclic_result_hash": self.cyclic_result.result_hash,
                    "controls_fit_sequentially_in_one_h_task": True,
                    "worker_dto_kind": WORKER_DTO_KIND,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_outer_control_pair_v1",
            "request_hash": self.request_hash,
            "outer_center": self.outer_center,
            "identity_result": self.identity_result.to_payload(),
            "cyclic_result": self.cyclic_result.to_payload(),
            "controls_fit_sequentially_in_one_h_task": True,
            "worker_dto_kind": WORKER_DTO_KIND,
            "target_labels_used": False,
            "pair_hash": self.pair_hash,
        }


@dataclass(frozen=True)
class OuterRuntimeResult:
    request: OuterRuntimeRequest
    control_pair: OuterControlPair
    methods: OuterMethodRuntimeResult
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.control_pair.request_hash != self.request.request_hash
            or self.control_pair.outer_center != self.request.outer_center
            or self.methods.outer_center != self.request.outer_center
            or self.methods.identity_result.result_hash
            != self.control_pair.identity_result.result_hash
            or self.methods.cyclic_result.result_hash
            != self.control_pair.cyclic_result.result_hash
        ):
            raise ProtocolError("P-DCAPS v4 outer runtime result drifted.")
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_outer_runtime_result_v1",
                    "request_hash": self.request.request_hash,
                    "control_pair_hash": self.control_pair.pair_hash,
                    "method_runtime_hash": self.methods.runtime_hash,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def outer_center(self) -> str:
        return self.request.outer_center

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_outer_runtime_result_v1",
            "request_hash": self.request.request_hash,
            "control_pair": self.control_pair.to_payload(),
            "methods": self.methods.to_payload(),
            "target_labels_used": False,
            "result_hash": self.result_hash,
        }


__all__ = (
    "OuterControlPair",
    "OuterRuntimeRequest",
    "OuterRuntimeResult",
    "WORKER_DEPTH_ENV",
    "WORKER_DTO_KIND",
    "assert_pickle_safe",
)
