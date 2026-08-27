"""Nominal source-sealed scientific service boundary for OE-PPUR v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NoReturn, Sequence

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..source_supervision import SourceTrainingSurface
from .dto import PrimitiveWorkerResult, PrimitiveWorkerTask


_SERVICE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ServicePreflightRequest:
    seven_input_contract_hash: str
    protocol_hash: str
    source_seal_hash: str
    workstation_receipt_hash: str

    def __post_init__(self) -> None:
        for role in (
            "seven_input_contract_hash",
            "protocol_hash",
            "source_seal_hash",
            "workstation_receipt_hash",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )


@dataclass(frozen=True, slots=True)
class ServicePreflightReceipt:
    request_hash: str
    implementation_source_hash: str
    exact_nominal_service: bool
    source_supervision_is_direct_input_three: bool
    labels_opened: bool
    mutation_performed: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_hash", require_sha256(self.request_hash, "service preflight request hash"))
        object.__setattr__(self, "implementation_source_hash", require_sha256(self.implementation_source_hash, "service implementation source hash"))
        if (
            self.exact_nominal_service is not True
            or self.source_supervision_is_direct_input_three is not True
            or self.labels_opened is not False
            or self.mutation_performed is not False
        ):
            raise ProtocolError("OE-PPUR v3 service preflight boundary drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload(include_hash=False)))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        body = {
            "schema_version": "oe_ppur_v3_service_preflight_receipt_v1",
            "request_hash": self.request_hash,
            "implementation_source_hash": self.implementation_source_hash,
            "exact_nominal_service": True,
            "source_supervision_is_direct_input_three": True,
            "labels_opened": False,
            "mutation_performed": False,
        }
        return {**body, "receipt_hash": self.receipt_hash} if include_hash else body


class CanonicalScientificRouterService:
    """Concrete nominal service bound to one parsed source-training surface."""

    __slots__ = (
        "_factory_identity_hash",
        "_source_seal_hash",
        "_source_surface",
        "_seven_input_contract_hash",
    )

    def __init__(
        self,
        *,
        source_surface: SourceTrainingSurface,
        source_seal_hash: str,
        seven_input_contract_hash: str,
        factory_identity_hash: str,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _SERVICE_TOKEN or type(source_surface) is not SourceTrainingSurface:
            raise ProtocolError("OE-PPUR v3 canonical service bypassed its factory.")
        self._source_surface = source_surface
        self._source_seal_hash = require_sha256(source_seal_hash, "service source seal hash")
        self._seven_input_contract_hash = require_sha256(
            seven_input_contract_hash, "service seven-input contract hash"
        )
        self._factory_identity_hash = require_sha256(
            factory_identity_hash, "service factory identity hash"
        )

    @property
    def source_training_surface_receipt_hash(self) -> str:
        return self._source_surface.receipt.receipt_hash

    def preflight(self, request: ServicePreflightRequest) -> ServicePreflightReceipt:
        """Validate the sealed implementation without opening terminal labels."""

        if type(request) is not ServicePreflightRequest:
            raise ProtocolError("OE-PPUR v3 service preflight request is untyped.")
        request_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v3_service_preflight_request_v1",
                "seven_input_contract_hash": request.seven_input_contract_hash,
                "protocol_hash": request.protocol_hash,
                "source_seal_hash": request.source_seal_hash,
                "workstation_receipt_hash": request.workstation_receipt_hash,
            }
        )
        if (
            request.seven_input_contract_hash != self._seven_input_contract_hash
            or request.source_seal_hash != self._source_seal_hash
        ):
            raise ProtocolError("OE-PPUR v3 canonical service lineage drifted.")
        return ServicePreflightReceipt(
            request_hash=request_hash,
            implementation_source_hash=self._source_seal_hash,
            exact_nominal_service=True,
            source_supervision_is_direct_input_three=True,
            labels_opened=False,
            mutation_performed=False,
        )

    def build_worker_tasks(self) -> NoReturn:
        """Fail closed until the sealed target-probability adapter is supplied."""

        raise ProtocolError(
            "OE-PPUR v3 target probability materialization is not implemented "
            "inside the current source seal."
        )

    def execute_worker(self, task: PrimitiveWorkerTask) -> NoReturn:
        """Fail closed until the science worker API is source-sealed."""

        if type(task) is not PrimitiveWorkerTask:
            raise ProtocolError("OE-PPUR v3 primitive task is untyped.")
        raise ProtocolError(
            "OE-PPUR v3 outer source-science execution is not yet sealed."
        )

    def seal_preterminal(
        self, results: Sequence[PrimitiveWorkerResult]
    ) -> NoReturn:
        """Reject partial results rather than fabricate a decision ledger."""

        if any(type(value) is not PrimitiveWorkerResult for value in results):
            raise ProtocolError("OE-PPUR v3 primitive result inventory is untyped.")
        raise ProtocolError(
            "OE-PPUR v3 preterminal sealing requires the complete outer service."
        )

    def __reduce__(self):  # pragma: no cover - explicit process boundary
        raise TypeError("OE-PPUR v3 scientific services cannot be serialized.")


def _build_canonical_scientific_router_service(
    *,
    source_surface: SourceTrainingSurface,
    source_seal_hash: str,
    seven_input_contract_hash: str,
    factory_identity_hash: str,
) -> CanonicalScientificRouterService:
    return CanonicalScientificRouterService(
        source_surface=source_surface,
        source_seal_hash=source_seal_hash,
        seven_input_contract_hash=seven_input_contract_hash,
        factory_identity_hash=factory_identity_hash,
        _factory_token=_SERVICE_TOKEN,
    )


__all__ = (
    "CanonicalScientificRouterService",
    "ServicePreflightReceipt",
    "ServicePreflightRequest",
)
