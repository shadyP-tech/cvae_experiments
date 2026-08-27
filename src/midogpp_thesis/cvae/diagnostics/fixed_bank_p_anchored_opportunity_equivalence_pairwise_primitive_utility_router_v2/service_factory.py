"""Nominal, source-bound construction seam for OE-PPUR v2 services.

The public runner must never accept a structurally compatible caller object as
its scientific implementation.  This module therefore owns the only canonical
service-construction seam.  It snapshots the exact workspace-rendered six-input
bundle without reading those inputs, proves that the real factory implementation
is inside the admitted v2 source closure, and currently fails closed because the
scientific phase implementation has not yet been supplied.

Importantly, preparing or building this factory does not import the authorization
lease module and does not create output, scratch, or lease state.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import inspect
from pathlib import Path
from typing import NoReturn

from ...protocol import ProtocolError
from .config import ResolvedConfigBundle, RouterV2Config
from .execution_admission import SixInputAdmissionReceipt
from .hashing import canonical_hash, require_sha256
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
    INPUT_RELATIVE_MEMBERS,
)
from .source_seal import (
    SourceContractReceipt,
    validate_source_contract_receipt,
)
from .workspace_inputs import (
    WorkspaceInputBinding,
    hash_ordered_input_locations,
)


_FACTORY_CONSTRUCTION_TOKEN = object()
_IDENTITY_CONSTRUCTION_TOKEN = object()
_SOURCE_RELATIVE_DIRECTORY = Path(
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_"
    "utility_router_v2"
)


@dataclass(frozen=True, slots=True)
class ImmutableInputBinding:
    """Path-syntax-only copy of one workspace-rendered direct input."""

    role: str
    artifact_id: str
    path: Path
    kind: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            not path.is_absolute()
            or path == Path(path.anchor)
            or ".." in path.parts
            or str(path).startswith(("artifact://", "output://", "file://"))
        ):
            raise ProtocolError("OE-PPUR v2 service input path is unsafe.")
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "artifact_id", str(self.artifact_id))
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "kind", str(self.kind))

    def _payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "path": self.path.as_posix(),
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class CanonicalServiceFactoryIdentity:
    """Guarded identity of the actual nominal factory source."""

    factory_module: str
    factory_qualified_name: str
    factory_function_qualified_name: str
    source_relative_path: str
    source_file_sha256: str
    source_contract_hash: str
    source_contract_receipt_hash: str
    resolved_config_contract_hash: str
    six_input_admission_hash: str
    admitted_input_binding_hash: str
    admitted_input_location_binding_sha256: str
    resolved_input_location_binding_sha256: str
    immutable_input_binding_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _IDENTITY_CONSTRUCTION_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 service identity requires canonical inspection."
            )
        if (
            self.factory_module != __name__
            or self.factory_qualified_name
            != "CanonicalExecutionServiceFactory"
            or self.factory_function_qualified_name
            != "prepare_canonical_execution_service_factory"
            or Path(self.source_relative_path)
            != _SOURCE_RELATIVE_DIRECTORY / "service_factory.py"
        ):
            raise ProtocolError("OE-PPUR v2 service factory identity drifted.")
        for role in (
            "source_file_sha256",
            "source_contract_hash",
            "source_contract_receipt_hash",
            "resolved_config_contract_hash",
            "six_input_admission_hash",
            "admitted_input_binding_hash",
            "admitted_input_location_binding_sha256",
            "resolved_input_location_binding_sha256",
            "immutable_input_binding_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        if (
            self.admitted_input_location_binding_sha256
            != self.resolved_input_location_binding_sha256
        ):
            raise ProtocolError(
                "OE-PPUR v2 service input-location identity drifted."
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_canonical_service_factory_identity_v1",
            "factory_module": self.factory_module,
            "factory_qualified_name": self.factory_qualified_name,
            "factory_function_qualified_name": (
                self.factory_function_qualified_name
            ),
            "source_relative_path": self.source_relative_path,
            "source_file_sha256": self.source_file_sha256,
            "source_contract_hash": self.source_contract_hash,
            "source_contract_receipt_hash": self.source_contract_receipt_hash,
            "resolved_config_contract_hash": self.resolved_config_contract_hash,
            "six_input_admission_hash": self.six_input_admission_hash,
            "admitted_input_binding_hash": self.admitted_input_binding_hash,
            "admitted_input_location_binding_sha256": (
                self.admitted_input_location_binding_sha256
            ),
            "resolved_input_location_binding_sha256": (
                self.resolved_input_location_binding_sha256
            ),
            "immutable_input_binding_hash": self.immutable_input_binding_hash,
            "input_location_binding_exact_match": True,
            "admitted_direct_input_count": 6,
            "preterminal_path_binding_count": 3,
            "preterminal_path_roles": list(DIRECT_INPUT_ROLES[:3]),
            "resolved_config_path_retained": False,
            "canonical_manifest_path_retained": False,
            "parent_ledger_path_retained": False,
            "authorization_amendment_path_retained": False,
            "nominal_factory_required": True,
            "structural_service_injection_allowed": False,
            "scientific_service_implemented": False,
            "mutation_performed": False,
            "authorization_lease_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


class CanonicalExecutionServiceFactory:
    """Prepared nominal factory exposing only label-free preterminal paths.

    The exact-six snapshot is absorbed into ``identity`` and then discarded.
    A future scientific service can receive the bank, GenerationLock, and
    label-free cache paths only.  In particular, neither ``config.resolved``
    nor the canonical annotation manifest path can be retained by this object.
    """

    __slots__ = ("_artifact_root", "_bindings", "_identity")

    def __init__(
        self,
        *,
        artifact_root: Path,
        bindings: tuple[ImmutableInputBinding, ...],
        identity: CanonicalServiceFactoryIdentity,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _FACTORY_CONSTRUCTION_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 canonical service factory bypassed resolved admission."
            )
        if (
            not isinstance(identity, CanonicalServiceFactoryIdentity)
            or tuple(row.role for row in bindings) != DIRECT_INPUT_ROLES[:3]
            or tuple(row.artifact_id for row in bindings)
            != DIRECT_INPUT_ARTIFACT_IDS[:3]
            or tuple(row.kind for row in bindings) != EXPECTED_INPUT_KINDS[:3]
        ):
            raise ProtocolError("OE-PPUR v2 canonical service binding drifted.")
        self._artifact_root = Path(artifact_root)
        self._bindings = tuple(bindings)
        self._identity = identity

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @property
    def label_free_input_bindings(self) -> tuple[ImmutableInputBinding, ...]:
        return self._bindings

    @property
    def identity(self) -> CanonicalServiceFactoryIdentity:
        return self._identity

    def build(self) -> NoReturn:
        """Refuse execution until a real source-sealed science service exists."""

        raise ProtocolError(
            "OE-PPUR v2 canonical scientific execution service is not "
            "implemented; refusing before authorization lease."
        )

    def __reduce__(self):  # pragma: no cover - safety seam exercised indirectly
        raise TypeError("OE-PPUR v2 service factories cannot be serialized.")


def prepare_canonical_execution_service_factory(
    resolved: ResolvedConfigBundle,
    *,
    admission: SixInputAdmissionReceipt,
    source: SourceContractReceipt,
) -> CanonicalExecutionServiceFactory:
    """Prepare the only nominal factory from an exact resolved-config bundle.

    This function is deliberately path-syntax-only for the six inputs.  Content,
    topology, amendment, and overlap validation remain the responsibility of the
    read-only six-input admission phase.
    """

    if type(resolved) is not ResolvedConfigBundle:
        raise ProtocolError(
            "OE-PPUR v2 canonical service factory requires ResolvedConfigBundle."
        )
    if type(resolved.config) is not RouterV2Config:
        raise ProtocolError("OE-PPUR v2 resolved service config is untyped.")
    if type(admission) is not SixInputAdmissionReceipt:
        raise ProtocolError("OE-PPUR v2 canonical service admission is untyped.")
    if (
        resolved.config.execution_authorized is not True
        or resolved.config.source_contract_hash is None
        or not resolved.source_path.is_absolute()
        or resolved.source_path.name != "config.resolved.yaml"
        or not resolved.artifact_root.is_absolute()
        or resolved.artifact_root == Path(resolved.artifact_root.anchor)
    ):
        raise ProtocolError("OE-PPUR v2 resolved service bundle drifted.")
    if (
        admission.config_contract_hash != resolved.config.contract_hash
        or admission.source_contract_hash
        != resolved.config.source_contract_hash
        or admission.artifact_root != resolved.artifact_root.as_posix()
        or admission.input_roles != DIRECT_INPUT_ROLES
        or admission.input_artifact_ids != DIRECT_INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("OE-PPUR v2 canonical service admission drifted.")

    raw_bindings = tuple(resolved.input_bindings)
    if (
        len(raw_bindings) != 6
        or any(type(row) is not WorkspaceInputBinding for row in raw_bindings)
        or tuple(row.role for row in raw_bindings) != DIRECT_INPUT_ROLES
        or tuple(row.artifact_id for row in raw_bindings)
        != DIRECT_INPUT_ARTIFACT_IDS
        or tuple(row.kind for row in raw_bindings) != EXPECTED_INPUT_KINDS
    ):
        raise ProtocolError(
            "OE-PPUR v2 canonical service factory requires six exact bindings."
        )
    bindings = tuple(
        ImmutableInputBinding(row.role, row.artifact_id, row.path, row.kind)
        for row in raw_bindings
    )
    if len({row.path for row in bindings}) != 6:
        raise ProtocolError("OE-PPUR v2 canonical service paths are duplicated.")
    if any(
        member and not _has_relative_suffix(row.path, Path(member))
        for row, member in zip(bindings, INPUT_RELATIVE_MEMBERS, strict=True)
    ):
        raise ProtocolError("OE-PPUR v2 canonical service input member drifted.")

    resolved_input_location_binding_sha256 = hash_ordered_input_locations(
        raw_bindings
    )
    if (
        resolved_input_location_binding_sha256
        != admission.input_location_binding_sha256
    ):
        raise ProtocolError(
            "OE-PPUR v2 canonical service input-location admission drifted."
        )

    validated_source = validate_source_contract_receipt(
        source,
        expected_source_contract_hash=resolved.config.source_contract_hash,
    )
    binding_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v2_immutable_service_input_binding_v1",
            "resolved_config_contract_hash": resolved.config.contract_hash,
            "source_path": resolved.source_path.as_posix(),
            "artifact_root": resolved.artifact_root.as_posix(),
            "input_bindings": [row._payload() for row in bindings],
            "direct_input_count": 6,
            "order_is_semantic": True,
        }
    )
    identity = _inspect_factory_identity(
        source=validated_source,
        resolved_config_contract_hash=resolved.config.contract_hash,
        six_input_admission_hash=admission.receipt_hash,
        admitted_input_binding_hash=admission.input_binding_hash,
        admitted_input_location_binding_sha256=(
            admission.input_location_binding_sha256
        ),
        resolved_input_location_binding_sha256=(
            resolved_input_location_binding_sha256
        ),
        immutable_input_binding_hash=binding_hash,
    )
    return CanonicalExecutionServiceFactory(
        artifact_root=resolved.artifact_root,
        bindings=bindings[:3],
        identity=identity,
        _factory_token=_FACTORY_CONSTRUCTION_TOKEN,
    )


def build_canonical_execution_services(
    resolved: ResolvedConfigBundle,
    *,
    admission: SixInputAdmissionReceipt,
    source: SourceContractReceipt,
) -> NoReturn:
    """Fail-closed production entry point until science phases are implemented."""

    factory = prepare_canonical_execution_service_factory(
        resolved, admission=admission, source=source
    )
    return factory.build()


def _inspect_factory_identity(
    *,
    source: SourceContractReceipt,
    resolved_config_contract_hash: str,
    six_input_admission_hash: str,
    admitted_input_binding_hash: str,
    admitted_input_location_binding_sha256: str,
    resolved_input_location_binding_sha256: str,
    immutable_input_binding_hash: str,
) -> CanonicalServiceFactoryIdentity:
    class_file = inspect.getsourcefile(CanonicalExecutionServiceFactory)
    function_file = inspect.getsourcefile(prepare_canonical_execution_service_factory)
    if class_file is None or function_file is None:
        raise ProtocolError("OE-PPUR v2 service factory source is not inspectable.")
    class_path = Path(class_file)
    function_path = Path(function_file)
    if class_path != function_path or class_path.is_symlink():
        raise ProtocolError("OE-PPUR v2 service factory source identity drifted.")
    try:
        root = Path(source.repository_root).resolve(strict=True)
        source_path = class_path.resolve(strict=True)
        relative = source_path.relative_to(root)
        source_bytes = source_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise ProtocolError(
            "OE-PPUR v2 service factory escaped its source contract."
        ) from exc
    if relative != _SOURCE_RELATIVE_DIRECTORY / "service_factory.py":
        raise ProtocolError("OE-PPUR v2 service factory source path drifted.")
    return CanonicalServiceFactoryIdentity(
        factory_module=CanonicalExecutionServiceFactory.__module__,
        factory_qualified_name=CanonicalExecutionServiceFactory.__qualname__,
        factory_function_qualified_name=(
            prepare_canonical_execution_service_factory.__qualname__
        ),
        source_relative_path=relative.as_posix(),
        source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_contract_hash=source.combined_source_sha256,
        source_contract_receipt_hash=source.receipt_hash,
        resolved_config_contract_hash=resolved_config_contract_hash,
        six_input_admission_hash=six_input_admission_hash,
        admitted_input_binding_hash=admitted_input_binding_hash,
        admitted_input_location_binding_sha256=(
            admitted_input_location_binding_sha256
        ),
        resolved_input_location_binding_sha256=(
            resolved_input_location_binding_sha256
        ),
        immutable_input_binding_hash=immutable_input_binding_hash,
        _factory_token=_IDENTITY_CONSTRUCTION_TOKEN,
    )


def _has_relative_suffix(path: Path, suffix: Path) -> bool:
    return len(path.parts) >= len(suffix.parts) and path.parts[
        -len(suffix.parts) :
    ] == suffix.parts


__all__ = (
    "CanonicalExecutionServiceFactory",
    "CanonicalServiceFactoryIdentity",
    "ImmutableInputBinding",
    "build_canonical_execution_services",
    "prepare_canonical_execution_service_factory",
)
