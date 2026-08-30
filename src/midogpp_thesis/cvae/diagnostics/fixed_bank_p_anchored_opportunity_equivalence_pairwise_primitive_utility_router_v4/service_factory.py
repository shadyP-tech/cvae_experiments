"""Exact source-bound nominal-service factory for OE-PPUR v4.

The factory consumes a parsed :class:`SourceTrainingSurface`; hashes alone are
insufficient.  Construction is label-free with respect to the consumed target
test set and performs no filesystem mutation or authorization action.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import inspect
from pathlib import Path

from ...protocol import ProtocolError
from .config import (
    ResolvedV4ConfigBundle,
    validate_workspace_sealed_config,
)
from .execution.services import (
    CanonicalScientificRouterService,
    _build_canonical_scientific_router_service,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
    EXPECTED_SOURCE_RECEIPT_SHA256,
    EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256,
    EXPECTED_SOURCE_ROW_ORDER_SHA256,
    EXPECTED_SOURCE_SURFACE_SHA256,
    PACKAGE_NAME,
    SOURCE_CONTENT_LINEAGE_ARTIFACT_ID,
)
from .execution.inputs import hash_resolved_input_locations
from .run_admission import SevenInputRunAdmission
from .source_supervision import SourceTrainingSurface


_FACTORY_TOKEN = object()
_IDENTITY_TOKEN = object()
_SOURCE_RELATIVE_ROOT = Path("src/midogpp_thesis/cvae/diagnostics") / PACKAGE_NAME


@dataclass(frozen=True, slots=True)
class CanonicalServiceFactoryIdentity:
    factory_module: str
    factory_qualified_name: str
    service_module: str
    service_qualified_name: str
    factory_source_relative_path: str
    factory_source_file_sha256: str
    service_source_relative_path: str
    service_source_file_sha256: str
    source_seal_hash: str
    source_seal_receipt_hash: str
    config_contract_hash: str
    seven_input_contract_hash: str
    source_training_surface_receipt_hash: str
    source_training_surface_hash: str
    run_admission_receipt_hash: str
    input_location_binding_hash: str
    execution_launch_authority_sha256: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _IDENTITY_TOKEN:
            raise ProtocolError("OE-PPUR v4 service identity bypassed inspection.")
        if (
            self.factory_module != __name__
            or self.factory_qualified_name != "CanonicalScientificServiceFactory"
            or self.service_module
            != CanonicalScientificRouterService.__module__
            or self.service_qualified_name
            != CanonicalScientificRouterService.__qualname__
            or Path(self.factory_source_relative_path)
            != _SOURCE_RELATIVE_ROOT / "service_factory.py"
            or Path(self.service_source_relative_path)
            != _SOURCE_RELATIVE_ROOT / "execution/services.py"
        ):
            raise ProtocolError("OE-PPUR v4 nominal service identity drifted.")
        for role in (
            "factory_source_file_sha256",
            "service_source_file_sha256",
            "source_seal_hash",
            "source_seal_receipt_hash",
            "config_contract_hash",
            "seven_input_contract_hash",
            "source_training_surface_receipt_hash",
            "source_training_surface_hash",
            "run_admission_receipt_hash",
            "input_location_binding_hash",
            "execution_launch_authority_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_canonical_service_factory_identity_v1",
            "factory_module": self.factory_module,
            "factory_qualified_name": self.factory_qualified_name,
            "service_module": self.service_module,
            "service_qualified_name": self.service_qualified_name,
            "factory_source_relative_path": self.factory_source_relative_path,
            "factory_source_file_sha256": self.factory_source_file_sha256,
            "service_source_relative_path": self.service_source_relative_path,
            "service_source_file_sha256": self.service_source_file_sha256,
            "source_seal_hash": self.source_seal_hash,
            "source_seal_receipt_hash": self.source_seal_receipt_hash,
            "config_contract_hash": self.config_contract_hash,
            "seven_input_contract_hash": self.seven_input_contract_hash,
            "source_training_surface_receipt_hash": self.source_training_surface_receipt_hash,
            "source_training_surface_hash": self.source_training_surface_hash,
            "run_admission_receipt_hash": self.run_admission_receipt_hash,
            "input_location_binding_hash": self.input_location_binding_hash,
            "execution_launch_authority_sha256": (
                self.execution_launch_authority_sha256
            ),
            "source_supervision_consumed_as_typed_surface": True,
            "source_supervision_direct_input_ordinal": 3,
            "service_shell_implemented": True,
            "canonical_scientific_execution_service_implemented": True,
            "end_to_end_preterminal_scientific_execution_implemented": True,
            "terminal_evaluation_is_separate_capability": True,
            "caller_service_injection_allowed": False,
            "target_labels_opened": False,
            "mutation_performed": False,
            "authorization_consumed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


class CanonicalScientificServiceFactory:
    """Prepared factory retaining a typed source surface, never target labels."""

    __slots__ = ("_identity", "_source_surface")

    def __init__(
        self,
        *,
        identity: CanonicalServiceFactoryIdentity,
        source_surface: SourceTrainingSurface,
        _factory_token: object | None = None,
    ) -> None:
        if (
            _factory_token is not _FACTORY_TOKEN
            or type(identity) is not CanonicalServiceFactoryIdentity
            or type(source_surface) is not SourceTrainingSurface
            or identity.source_training_surface_receipt_hash
            != source_surface.receipt.receipt_hash
            or identity.source_training_surface_hash != source_surface.surface_hash
        ):
            raise ProtocolError("OE-PPUR v4 canonical service factory drifted.")
        self._identity = identity
        self._source_surface = source_surface

    @property
    def identity(self) -> CanonicalServiceFactoryIdentity:
        return self._identity

    def build(self) -> CanonicalScientificRouterService:
        return _build_canonical_scientific_router_service(
            source_surface=self._source_surface,
            source_seal_hash=self._identity.source_seal_hash,
            seven_input_contract_hash=self._identity.seven_input_contract_hash,
            factory_identity_hash=self._identity.receipt_hash,
        )

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v4 service factories cannot be serialized.")


def prepare_canonical_scientific_service_factory(
    config_bundle: ResolvedV4ConfigBundle,
    *,
    source_seal: object,
    source_surface: SourceTrainingSurface,
    admission: SevenInputRunAdmission,
) -> CanonicalScientificServiceFactory:
    """Build from exact v4 bindings; launch authority stays an explicit input.

    ``source_seal`` is intentionally structural.  The execution package owns
    its sealed concrete receipt, while this module requires only the exact
    source-tree and receipt hashes.  This avoids coupling science to a mutable
    authority/lifecycle implementation.
    """

    if (
        type(config_bundle) is not ResolvedV4ConfigBundle
        or type(admission) is not SevenInputRunAdmission
    ):
        raise ProtocolError("OE-PPUR v4 service admission is untyped.")
    config = validate_workspace_sealed_config(config_bundle.config)
    seal_hash = require_sha256(
        getattr(source_seal, "combined_source_sha256", None),
        "source seal hash",
    )
    seal_receipt_hash = require_sha256(
        getattr(source_seal, "receipt_hash", None),
        "source seal receipt hash",
    )
    if (
        type(source_surface) is not SourceTrainingSurface
        or source_surface.receipt.artifact_id
        != SOURCE_CONTENT_LINEAGE_ARTIFACT_ID
        or source_surface.receipt.target_rows_present
        or source_surface.receipt.target_labels_used
        or source_surface.receipt.receipt_hash != EXPECTED_SOURCE_RECEIPT_SHA256
        or source_surface.surface_hash != EXPECTED_SOURCE_SURFACE_SHA256
        or source_surface.receipt.row_order_sha256
        != EXPECTED_SOURCE_ROW_ORDER_SHA256
        or source_surface.receipt.contract.producer_source_seal_sha256
        != EXPECTED_SOURCE_PRODUCER_SEAL_SHA256
        or source_surface.receipt.compiler_recomputation_receipt_sha256
        != EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256
    ):
        raise ProtocolError("OE-PPUR v4 source-training surface identity drifted.")
    if (
        admission.config_contract_hash != config.contract_hash
        or admission.protocol_hash != config.protocol_hash
        or admission.seven_input_contract_hash != config.seven_input_contract_hash
        or admission.source_seal_hash != seal_hash
        or admission.source_seal_receipt_hash != seal_receipt_hash
        or admission.source_training_surface_receipt_hash
        != source_surface.receipt.receipt_hash
        or admission.source_training_surface_hash != source_surface.surface_hash
        or admission.input_location_binding_hash
        != hash_resolved_input_locations(config_bundle.input_bindings)
        or admission.authorization_amendment_sha256
        != config.authorization_amendment_sha256
        or admission.workspace_plan_sha256 != config_bundle.workspace_plan_sha256
        or admission.final_envelope_sha256 != config_bundle.final_envelope_sha256
        or admission.execution_launch_authority_sha256
        != config_bundle.execution_launch_authority_sha256
        or admission.artifact_root != config_bundle.artifact_root
    ):
        raise ProtocolError("OE-PPUR v4 service admission lineage drifted.")
    repository_root = _resolve_repository_root(source_seal)
    factory_path = Path(inspect.getsourcefile(CanonicalScientificServiceFactory) or "")
    service_path = Path(inspect.getsourcefile(CanonicalScientificRouterService) or "")
    factory_relative, factory_hash = _inspect_source_member(
        factory_path, repository_root=repository_root
    )
    service_relative, service_hash = _inspect_source_member(
        service_path, repository_root=repository_root
    )
    identity = CanonicalServiceFactoryIdentity(
        factory_module=CanonicalScientificServiceFactory.__module__,
        factory_qualified_name=CanonicalScientificServiceFactory.__qualname__,
        service_module=CanonicalScientificRouterService.__module__,
        service_qualified_name=CanonicalScientificRouterService.__qualname__,
        factory_source_relative_path=factory_relative,
        factory_source_file_sha256=factory_hash,
        service_source_relative_path=service_relative,
        service_source_file_sha256=service_hash,
        source_seal_hash=seal_hash,
        source_seal_receipt_hash=seal_receipt_hash,
        config_contract_hash=config.contract_hash,
        seven_input_contract_hash=config.seven_input_contract_hash,
        source_training_surface_receipt_hash=source_surface.receipt.receipt_hash,
        source_training_surface_hash=source_surface.surface_hash,
        run_admission_receipt_hash=admission.receipt_hash,
        input_location_binding_hash=admission.input_location_binding_hash,
        execution_launch_authority_sha256=(
            admission.execution_launch_authority_sha256
        ),
        _factory_token=_IDENTITY_TOKEN,
    )
    return CanonicalScientificServiceFactory(
        identity=identity,
        source_surface=source_surface,
        _factory_token=_FACTORY_TOKEN,
    )


def build_canonical_scientific_service(
    config_bundle: ResolvedV4ConfigBundle,
    *,
    source_seal: object,
    source_surface: SourceTrainingSurface,
    admission: SevenInputRunAdmission,
) -> CanonicalScientificRouterService:
    return prepare_canonical_scientific_service_factory(
        config_bundle,
        source_seal=source_seal,
        source_surface=source_surface,
        admission=admission,
    ).build()


def _resolve_repository_root(source_seal: object) -> Path:
    supplied = getattr(source_seal, "repository_root", None)
    if supplied is not None:
        candidate = Path(str(supplied))
        try:
            root = candidate.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError("OE-PPUR v4 source-seal repository is absent.") from exc
        if root.is_symlink() or not root.is_dir():
            raise ProtocolError("OE-PPUR v4 source-seal repository is unsafe.")
        return root
    path = Path(__file__).resolve(strict=True)
    for parent in path.parents:
        if (parent / "src/midogpp_thesis").is_dir():
            return parent
    raise ProtocolError("OE-PPUR v4 repository root could not be resolved.")


def _inspect_source_member(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[str, str]:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(repository_root)
        payload = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v4 service source escaped its seal.") from exc
    if resolved.is_symlink() or _SOURCE_RELATIVE_ROOT not in relative.parents:
        raise ProtocolError("OE-PPUR v4 service source is outside its adapter seal.")
    return relative.as_posix(), hashlib.sha256(payload).hexdigest()


__all__ = (
    "CanonicalScientificServiceFactory",
    "CanonicalServiceFactoryIdentity",
    "build_canonical_scientific_service",
    "prepare_canonical_scientific_service_factory",
)
