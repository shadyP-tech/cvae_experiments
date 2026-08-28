"""Mutation-free preflight for irreversible OE-PPUR v3 authorization."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from ....workspace import MidogppWorkspace, WorkspaceError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_contract import (
    authorization_amendment_bytes,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lifecycle_source_seal import (
    LifecycleSourceSealReceipt,
    validate_lifecycle_source_seal,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.contracts import (
    SourceTrainingSurface,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    build_source_seal,
    validate_live_producer_seal_binding,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_admission import (
    validate_prospective_input_paths,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal.authority import (
    validate_prospective_terminal_authority,
)
from .amendment_publisher import assert_unissued_authorization_topology
from .envelope_plan import build_authorization_envelope_plan
from .input_manifest import validate_preissuance_input_inventory
from .paths import CanonicalPreparationPaths


@dataclass(frozen=True, slots=True)
class AuthorizationPreflightReceipt:
    protocol_hash: str
    source_contract_hash: str
    source_surface_hash: str
    lifecycle_source_seal_sha256: str
    lifecycle_source_seal_receipt_hash: str
    prospective_amendment_sha256: str
    prospective_config_contract_hash: str
    existing_input_inventory_receipt_hash: str
    authorized_semantics_hash: str
    prospective_envelope_content_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "protocol_hash",
            "source_contract_hash",
            "source_surface_hash",
            "lifecycle_source_seal_sha256",
            "lifecycle_source_seal_receipt_hash",
            "prospective_amendment_sha256",
            "prospective_config_contract_hash",
            "existing_input_inventory_receipt_hash",
            "authorized_semantics_hash",
            "prospective_envelope_content_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_authorization_preflight_receipt_v1",
            "protocol_hash": self.protocol_hash,
            "source_contract_hash": self.source_contract_hash,
            "source_surface_hash": self.source_surface_hash,
            "lifecycle_source_seal_sha256": self.lifecycle_source_seal_sha256,
            "lifecycle_source_seal_receipt_hash": (
                self.lifecycle_source_seal_receipt_hash
            ),
            "prospective_amendment_sha256": self.prospective_amendment_sha256,
            "prospective_config_contract_hash": (
                self.prospective_config_contract_hash
            ),
            "existing_input_inventory_receipt_hash": (
                self.existing_input_inventory_receipt_hash
            ),
            "authorized_semantics_hash": self.authorized_semantics_hash,
            "prospective_envelope_content_hash": (
                self.prospective_envelope_content_hash
            ),
            "amendment_issued": False,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def preflight_authorization_issuance(
    paths: CanonicalPreparationPaths,
    *,
    source_surface: SourceTrainingSurface,
    protocol_hash: str,
    lifecycle_source_seal: LifecycleSourceSealReceipt,
) -> AuthorizationPreflightReceipt:
    """Validate every feasible input/config fact before creating input #7."""

    if not isinstance(paths, CanonicalPreparationPaths):
        raise ProtocolError("OE-PPUR v3 preflight paths are untyped.")
    if type(source_surface) is not SourceTrainingSurface:
        raise ProtocolError("OE-PPUR v3 preflight source surface is untyped.")
    protocol = require_sha256(protocol_hash, "protocol hash")
    lifecycle = validate_lifecycle_source_seal(lifecycle_source_seal)
    assert_unissued_authorization_topology(paths)

    source_receipt = source_surface.receipt
    source_seal = build_source_seal(paths.repository_root)
    validate_live_producer_seal_binding(
        configured_sha256=source_receipt.contract.producer_source_seal_sha256,
        parsed_sha256=source_receipt.contract.producer_source_seal_sha256,
        source_seal=source_seal,
    )
    amendment_raw = authorization_amendment_bytes(
        source_contract_hash=source_receipt.receipt_hash,
        protocol_hash=protocol,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
    )
    plan = build_authorization_envelope_plan(
        paths,
        source_surface=source_surface,
        protocol_hash=protocol,
        lifecycle_source_seal=lifecycle,
        amendment_raw=amendment_raw,
        prospective_amendment=True,
    )
    validate_prospective_input_paths(
        plan.candidate_bundle,
        artifact_root=paths.artifact_root,
        scratch_root=paths.scratch_root,
    )
    validate_prospective_terminal_authority(
        plan.candidate_bundle,
        source_training_surface_receipt_hash=source_receipt.receipt_hash,
        amendment_raw=amendment_raw,
        lifecycle_source_seal=lifecycle,
    )
    semantics = plan.authorized_semantics_payload()
    try:
        workspace = MidogppWorkspace.load(paths.repository_root)
    except WorkspaceError as exc:
        raise ProtocolError("OE-PPUR v3 preflight workspace could not load.") from exc
    inventory = validate_preissuance_input_inventory(
        workspace,
        paths,
        authorized_semantics=semantics,
    )
    return AuthorizationPreflightReceipt(
        protocol_hash=protocol,
        source_contract_hash=source_receipt.receipt_hash,
        source_surface_hash=source_surface.surface_hash,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
        lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
        prospective_amendment_sha256=plan.amendment_sha256,
        prospective_config_contract_hash=plan.config.contract_hash,
        existing_input_inventory_receipt_hash=inventory.receipt_hash,
        authorized_semantics_hash=inventory.authorized_semantics_hash,
        prospective_envelope_content_hash=plan.content_hash,
    )


__all__ = (
    "AuthorizationPreflightReceipt",
    "preflight_authorization_issuance",
)
