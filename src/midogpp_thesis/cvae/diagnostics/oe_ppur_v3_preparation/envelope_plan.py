"""Pure, path-bearing launch-envelope planning for OE-PPUR v3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json

import yaml

from ...protocol import ProtocolError
from ....workspace import MidogppWorkspace, WorkspaceError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_contract import (
    authorization_amendment_bytes,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    ResolvedV3ConfigBundle,
    RouterV3Config,
    build_authorization_ready_config,
    parse_resolved_config_payload,
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
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workspace_provenance import (
    build_authorized_input_semantics,
    validate_workspace_input_provenance_payload,
)
from .input_manifest import build_exact_input_manifest
from .paths import CanonicalPreparationPaths


@dataclass(frozen=True, slots=True)
class AuthorizationEnvelopePlan:
    config: RouterV3Config
    candidate_bundle: ResolvedV3ConfigBundle
    amendment_sha256: str
    config_raw: bytes
    manifest_raw: bytes
    authorized_semantics: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    lifecycle_source_seal_sha256: str
    prospective_amendment: bool
    content_hash: str = field(init=False)
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.config_raw) is not bytes
            or type(self.manifest_raw) is not bytes
            or type(self.prospective_amendment) is not bool
            or self.candidate_bundle.config != self.config
        ):
            raise ProtocolError("OE-PPUR v3 envelope plan topology drifted.")
        for role in (
            "amendment_sha256",
            "lifecycle_source_seal_sha256",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        normalized = tuple(
            (
                str(artifact_id),
                tuple((str(key), str(value)) for key, value in identities),
            )
            for artifact_id, identities in self.authorized_semantics
        )
        if (
            normalized != self.authorized_semantics
            or tuple(artifact_id for artifact_id, _items in normalized)
            != tuple(sorted(artifact_id for artifact_id, _items in normalized))
        ):
            raise ProtocolError("OE-PPUR v3 envelope semantics topology drifted.")
        content = {
            "schema_version": "oe_ppur_v3_authorization_envelope_content_v1",
            "config_contract_hash": self.config.contract_hash,
            "amendment_sha256": self.amendment_sha256,
            "resolved_config_sha256": hashlib.sha256(self.config_raw).hexdigest(),
            "input_manifest_sha256": hashlib.sha256(self.manifest_raw).hexdigest(),
            "authorized_semantics_hash": canonical_hash(
                self.authorized_semantics_payload()
            ),
            "lifecycle_source_seal_sha256": self.lifecycle_source_seal_sha256,
            "target_labels_opened": False,
        }
        object.__setattr__(self, "content_hash", canonical_hash(content))
        object.__setattr__(
            self,
            "plan_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_authorization_envelope_plan_v1",
                    "content_hash": self.content_hash,
                    "prospective_amendment": self.prospective_amendment,
                }
            ),
        )

    def authorized_semantics_payload(self) -> dict[str, dict[str, str]]:
        return {
            artifact_id: dict(identities)
            for artifact_id, identities in self.authorized_semantics
        }

    def manifest_payload(self) -> dict[str, object]:
        try:
            payload = json.loads(self.manifest_raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise ProtocolError("OE-PPUR v3 planned manifest became unreadable.") from exc
        if not isinstance(payload, dict):  # pragma: no cover
            raise ProtocolError("OE-PPUR v3 planned manifest became malformed.")
        return payload


def build_authorization_envelope_plan(
    paths: CanonicalPreparationPaths,
    *,
    source_surface: SourceTrainingSurface,
    protocol_hash: str,
    lifecycle_source_seal: LifecycleSourceSealReceipt,
    amendment_raw: bytes,
    prospective_amendment: bool,
) -> AuthorizationEnvelopePlan:
    """Construct and parse the exact envelope before any publication edge."""

    if type(paths) is not CanonicalPreparationPaths:
        raise ProtocolError("OE-PPUR v3 envelope-plan paths are untyped.")
    if type(source_surface) is not SourceTrainingSurface:
        raise ProtocolError("OE-PPUR v3 envelope-plan source is untyped.")
    if type(amendment_raw) is not bytes or type(prospective_amendment) is not bool:
        raise ProtocolError("OE-PPUR v3 envelope-plan amendment is untyped.")
    protocol = require_sha256(protocol_hash, "protocol hash")
    lifecycle = validate_lifecycle_source_seal(lifecycle_source_seal)
    source_receipt = source_surface.receipt
    expected_amendment = authorization_amendment_bytes(
        source_contract_hash=source_receipt.receipt_hash,
        protocol_hash=protocol,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
    )
    if amendment_raw != expected_amendment:
        raise ProtocolError("OE-PPUR v3 envelope-plan amendment bytes drifted.")
    amendment_sha256 = hashlib.sha256(amendment_raw).hexdigest()
    config = build_authorization_ready_config(
        source_supervision_content_sha256=source_receipt.receipt_hash,
        source_supervision_row_order_sha256=source_receipt.row_order_sha256,
        source_supervision_producer_seal_sha256=(
            source_receipt.contract.producer_source_seal_sha256
        ),
        source_supervision_recomputation_receipt_sha256=(
            source_receipt.compiler_recomputation_receipt_sha256
        ),
        authorization_amendment_sha256=amendment_sha256,
    )
    if config.protocol_hash != protocol:
        raise ProtocolError("OE-PPUR v3 envelope-plan protocol drifted.")
    semantics = build_authorized_input_semantics(
        source_contract_hash=source_receipt.receipt_hash,
        source_row_order_sha256=source_receipt.row_order_sha256,
        source_producer_seal_sha256=(
            source_receipt.contract.producer_source_seal_sha256
        ),
        source_recomputation_receipt_sha256=(
            source_receipt.compiler_recomputation_receipt_sha256
        ),
        authorization_amendment_sha256=amendment_sha256,
        protocol_hash=protocol,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
    )
    try:
        workspace = MidogppWorkspace.load(paths.repository_root)
    except WorkspaceError as exc:
        raise ProtocolError("OE-PPUR v3 envelope-plan workspace could not load.") from exc
    manifest = build_exact_input_manifest(
        workspace,
        paths,
        authorized_semantics=semantics,
        prospective_amendment_bytes=(
            amendment_raw if prospective_amendment else None
        ),
    )
    config_payload = resolved_config_payload(config.to_payload(), paths)
    config_raw = yaml.safe_dump(config_payload, sort_keys=False).encode("utf-8")
    manifest_raw = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    candidate = parse_resolved_config_payload(
        yaml.safe_load(config_raw.decode("utf-8")),
        source_path=paths.artifact_root / "config.resolved.yaml",
    )
    if (
        candidate.artifact_root != paths.artifact_root
        or candidate.input_bindings != paths.input_bindings
    ):
        raise ProtocolError("OE-PPUR v3 envelope-plan path binding drifted.")
    validate_workspace_input_provenance_payload(
        manifest,
        candidate.input_bindings,
        expected_authorized_semantics=semantics,
        allow_missing_amendment=prospective_amendment,
    )
    frozen_semantics = tuple(
        (artifact_id, tuple(sorted(identities.items())))
        for artifact_id, identities in sorted(semantics.items())
    )
    return AuthorizationEnvelopePlan(
        config=config,
        candidate_bundle=candidate,
        amendment_sha256=amendment_sha256,
        config_raw=config_raw,
        manifest_raw=manifest_raw,
        authorized_semantics=frozen_semantics,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
        prospective_amendment=prospective_amendment,
    )


def resolved_config_payload(
    path_free: Mapping[str, object],
    paths: CanonicalPreparationPaths,
) -> dict[str, object]:
    experiment = dict(path_free["experiment"])  # type: ignore[arg-type]
    inputs = dict(path_free["inputs"])  # type: ignore[arg-type]
    experiment["artifact_root"] = paths.artifact_root.as_posix()
    inputs["direct_input_locations"] = {
        row.role: row.path.as_posix() for row in paths.input_bindings
    }
    return {**path_free, "experiment": experiment, "inputs": inputs}


__all__ = (
    "AuthorizationEnvelopePlan",
    "build_authorization_envelope_plan",
    "resolved_config_payload",
)
