"""Authorized OE-PPUR v3 amendment and launch-envelope orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.protocol import (
    frozen_protocol_payload,
)
from .amendment_publisher import (
    AmendmentPublicationReceipt,
    publish_authorization_amendment,
    validate_existing_authorization_amendment,
)
from .authorization_preflight import (
    AuthorizationPreflightReceipt,
    preflight_authorization_issuance,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lifecycle_source_seal import (
    build_lifecycle_source_seal,
)
from .paths import DEFAULT_SCRATCH_ROOT, resolve_canonical_preparation_paths
from .resolved_config_renderer import (
    ResolvedEnvelopeReceipt,
    render_authorization_ready_envelope,
)
from .source_receipt import load_materialized_source_surface


@dataclass(frozen=True, slots=True)
class AuthorizationPreparationReceipt:
    amendment: AmendmentPublicationReceipt
    envelope: ResolvedEnvelopeReceipt
    preflight: AuthorizationPreflightReceipt | None
    recovered_existing_amendment: bool
    protocol_hash: str
    source_surface_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.amendment, AmendmentPublicationReceipt)
            or not isinstance(self.envelope, ResolvedEnvelopeReceipt)
            or self.amendment.amendment_sha256 != self.envelope.amendment_sha256
            or type(self.recovered_existing_amendment) is not bool
            or (
                self.recovered_existing_amendment
                and self.preflight is not None
            )
            or (
                not self.recovered_existing_amendment
                and not isinstance(self.preflight, AuthorizationPreflightReceipt)
            )
            or self.amendment.lifecycle_source_seal_sha256
            != self.envelope.lifecycle_source_seal_sha256
            or self.amendment.lifecycle_source_seal_receipt_hash
            != self.envelope.lifecycle_source_seal_receipt_hash
            or (
                self.preflight is not None
                and (
                    self.preflight.prospective_amendment_sha256
                    != self.amendment.amendment_sha256
                    or self.preflight.prospective_config_contract_hash
                    != self.envelope.config_contract_hash
                    or self.preflight.lifecycle_source_seal_sha256
                    != self.amendment.lifecycle_source_seal_sha256
                    or self.preflight.authorized_semantics_hash
                    != self.envelope.authorized_semantics_hash
                    or self.preflight.prospective_envelope_content_hash
                    != self.envelope.envelope_content_hash
                )
            )
        ):
            raise ProtocolError("OE-PPUR v3 preparation receipt topology drifted.")
        for role in ("protocol_hash", "source_surface_hash"):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_authorization_preparation_receipt_v1",
                    "amendment_publication_receipt_hash": self.amendment.receipt_hash,
                    "resolved_envelope_receipt_hash": self.envelope.receipt_hash,
                    "authorization_preflight_receipt_hash": (
                        None if self.preflight is None else self.preflight.receipt_hash
                    ),
                    "recovered_existing_amendment": (
                        self.recovered_existing_amendment
                    ),
                    "protocol_hash": self.protocol_hash,
                    "source_surface_hash": self.source_surface_hash,
                    "authorization_consumed": False,
                    "target_labels_opened": False,
                    "experiment_launched": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_authorization_preparation_receipt_v1",
            "amendment": self.amendment.to_payload(),
            "envelope": self.envelope.to_payload(),
            "preflight": (
                None if self.preflight is None else self.preflight.to_payload()
            ),
            "recovered_existing_amendment": self.recovered_existing_amendment,
            "protocol_hash": self.protocol_hash,
            "source_surface_hash": self.source_surface_hash,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "experiment_launched": False,
            "receipt_hash": self.receipt_hash,
        }


def authorize_and_render(
    repository_root: str | Path,
    *,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
) -> AuthorizationPreparationReceipt:
    """Issue input #7 once, then atomically render the resolved envelope."""

    paths = resolve_canonical_preparation_paths(
        repository_root,
        scratch_root=scratch_root,
        require_source=True,
        require_amendment=False,
    )
    source_surface = load_materialized_source_surface(paths.input_bindings[2].path)
    protocol_hash = str(frozen_protocol_payload()["protocol_hash"])
    lifecycle = build_lifecycle_source_seal(paths.repository_root)
    preflight = preflight_authorization_issuance(
        paths,
        source_surface=source_surface,
        protocol_hash=protocol_hash,
        lifecycle_source_seal=lifecycle,
    )
    amendment = publish_authorization_amendment(
        paths,
        source_surface=source_surface,
        protocol_hash=protocol_hash,
        lifecycle_source_seal=lifecycle,
    )
    if amendment.amendment_sha256 != preflight.prospective_amendment_sha256:
        raise ProtocolError("OE-PPUR v3 issued amendment drifted from preflight.")
    envelope = render_authorization_ready_envelope(
        paths,
        source_surface=source_surface,
        protocol_hash=protocol_hash,
        lifecycle_source_seal=lifecycle,
    )
    return AuthorizationPreparationReceipt(
        amendment=amendment,
        envelope=envelope,
        preflight=preflight,
        recovered_existing_amendment=False,
        protocol_hash=protocol_hash,
        source_surface_hash=source_surface.surface_hash,
    )


def render_existing_authorization(
    repository_root: str | Path,
    *,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
) -> AuthorizationPreparationReceipt:
    """Recover envelope rendering without reissuing direct input #7."""

    paths = resolve_canonical_preparation_paths(
        repository_root,
        scratch_root=scratch_root,
        require_source=True,
        require_amendment=True,
    )
    source_surface = load_materialized_source_surface(paths.input_bindings[2].path)
    protocol_hash = str(frozen_protocol_payload()["protocol_hash"])
    lifecycle = build_lifecycle_source_seal(paths.repository_root)
    amendment = validate_existing_authorization_amendment(
        paths,
        source_surface=source_surface,
        protocol_hash=protocol_hash,
        lifecycle_source_seal=lifecycle,
    )
    envelope = render_authorization_ready_envelope(
        paths,
        source_surface=source_surface,
        protocol_hash=protocol_hash,
        lifecycle_source_seal=lifecycle,
        allow_existing_envelope=True,
    )
    return AuthorizationPreparationReceipt(
        amendment=amendment,
        envelope=envelope,
        preflight=None,
        recovered_existing_amendment=True,
        protocol_hash=protocol_hash,
        source_surface_hash=source_surface.surface_hash,
    )


__all__ = (
    "AuthorizationPreparationReceipt",
    "authorize_and_render",
    "render_existing_authorization",
)
