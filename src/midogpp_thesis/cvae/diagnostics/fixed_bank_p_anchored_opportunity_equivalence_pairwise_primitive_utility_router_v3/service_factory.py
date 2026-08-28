"""Exact source-bound nominal-service factory for OE-PPUR v3.

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
    RouterV3Config,
    validate_authorization_ready_config,
    validate_planned_config,
)
from .execution.services import (
    CanonicalScientificRouterService,
    _build_canonical_scientific_router_service,
)
from .hashing import canonical_hash, require_sha256
from .identity import PACKAGE_NAME, SOURCE_SUPERVISION_ARTIFACT_ID
from .source_seal import (
    SourceSealReceipt,
    validate_live_producer_seal_binding,
    validate_source_seal,
)
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
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _IDENTITY_TOKEN:
            raise ProtocolError("OE-PPUR v3 service identity bypassed inspection.")
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
            raise ProtocolError("OE-PPUR v3 nominal service identity drifted.")
        for role in (
            "factory_source_file_sha256",
            "service_source_file_sha256",
            "source_seal_hash",
            "source_seal_receipt_hash",
            "config_contract_hash",
            "seven_input_contract_hash",
            "source_training_surface_receipt_hash",
            "source_training_surface_hash",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_canonical_service_factory_identity_v1",
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

    __slots__ = ("_execution_authorized", "_identity", "_source_surface")

    def __init__(
        self,
        *,
        identity: CanonicalServiceFactoryIdentity,
        source_surface: SourceTrainingSurface,
        execution_authorized: bool,
        _factory_token: object | None = None,
    ) -> None:
        if (
            _factory_token is not _FACTORY_TOKEN
            or type(identity) is not CanonicalServiceFactoryIdentity
            or type(source_surface) is not SourceTrainingSurface
            or identity.source_training_surface_receipt_hash
            != source_surface.receipt.receipt_hash
            or identity.source_training_surface_hash != source_surface.surface_hash
            or type(execution_authorized) is not bool
        ):
            raise ProtocolError("OE-PPUR v3 canonical service factory drifted.")
        self._identity = identity
        self._source_surface = source_surface
        self._execution_authorized = execution_authorized

    @property
    def identity(self) -> CanonicalServiceFactoryIdentity:
        return self._identity

    def build(self) -> CanonicalScientificRouterService:
        return _build_canonical_scientific_router_service(
            source_surface=self._source_surface,
            source_seal_hash=self._identity.source_seal_hash,
            seven_input_contract_hash=self._identity.seven_input_contract_hash,
            factory_identity_hash=self._identity.receipt_hash,
            execution_authorized=self._execution_authorized,
        )

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v3 service factories cannot be serialized.")


def prepare_canonical_scientific_service_factory(
    config: RouterV3Config,
    *,
    source_seal: SourceSealReceipt,
    source_surface: SourceTrainingSurface,
) -> CanonicalScientificServiceFactory:
    config = (
        validate_authorization_ready_config(config)
        if getattr(config, "execution_authorized", False) is True
        else validate_planned_config(config)
    )
    seal = validate_source_seal(source_seal)
    if (
        type(source_surface) is not SourceTrainingSurface
        or source_surface.receipt.artifact_id != SOURCE_SUPERVISION_ARTIFACT_ID
        or source_surface.receipt.target_rows_present
        or source_surface.receipt.target_labels_used
    ):
        raise ProtocolError("OE-PPUR v3 source-training surface identity drifted.")
    if config.execution_authorized:
        validate_live_producer_seal_binding(
            configured_sha256=config.source_supervision_producer_seal_sha256,
            parsed_sha256=(
                source_surface.receipt.contract.producer_source_seal_sha256
            ),
            source_seal=seal,
        )
    if config.execution_authorized and (
        source_surface.receipt.receipt_hash
        != config.source_supervision_content_sha256
        or source_surface.receipt.row_order_sha256
        != config.source_supervision_row_order_sha256
        or source_surface.receipt.contract.producer_source_seal_sha256
        != config.source_supervision_producer_seal_sha256
        or source_surface.receipt.compiler_recomputation_receipt_sha256
        != config.source_supervision_recomputation_receipt_sha256
    ):
        raise ProtocolError(
            "OE-PPUR v3 parsed source supervision drifted from external admission."
        )
    repository_root = Path(seal.repository_root)
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
        source_seal_hash=seal.combined_source_sha256,
        source_seal_receipt_hash=seal.receipt_hash,
        config_contract_hash=config.contract_hash,
        seven_input_contract_hash=config.seven_input_contract_hash,
        source_training_surface_receipt_hash=source_surface.receipt.receipt_hash,
        source_training_surface_hash=source_surface.surface_hash,
        _factory_token=_IDENTITY_TOKEN,
    )
    return CanonicalScientificServiceFactory(
        identity=identity,
        source_surface=source_surface,
        execution_authorized=config.execution_authorized,
        _factory_token=_FACTORY_TOKEN,
    )


def build_canonical_scientific_service(
    config: RouterV3Config,
    *,
    source_seal: SourceSealReceipt,
    source_surface: SourceTrainingSurface,
) -> CanonicalScientificRouterService:
    return prepare_canonical_scientific_service_factory(
        config,
        source_seal=source_seal,
        source_surface=source_surface,
    ).build()


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
        raise ProtocolError("OE-PPUR v3 service source escaped its seal.") from exc
    if resolved.is_symlink() or _SOURCE_RELATIVE_ROOT not in relative.parents:
        raise ProtocolError("OE-PPUR v3 service source is outside its adapter seal.")
    return relative.as_posix(), hashlib.sha256(payload).hexdigest()


__all__ = (
    "CanonicalScientificServiceFactory",
    "CanonicalServiceFactoryIdentity",
    "build_canonical_scientific_service",
    "prepare_canonical_scientific_service_factory",
)
