"""One-shot publisher for the externally authorized OE-PPUR v3 amendment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_contract import (
    authorization_amendment_bytes,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
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
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_paths import (
    is_exact_workspace_launch_envelope,
)
from .durable_io import (
    fsync_directory,
    hash_unique_regular_file,
    read_bounded_unique_file,
    write_bytes_exclusive,
)
from .paths import CanonicalPreparationPaths


@dataclass(frozen=True, slots=True)
class AmendmentPublicationReceipt:
    amendment_path: Path
    amendment_sha256: str
    source_contract_hash: str
    protocol_hash: str
    lifecycle_source_seal_sha256: str
    lifecycle_source_seal_receipt_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.amendment_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ProtocolError("OE-PPUR v3 published amendment path is unsafe.")
        for role in (
            "amendment_sha256",
            "source_contract_hash",
            "protocol_hash",
            "lifecycle_source_seal_sha256",
            "lifecycle_source_seal_receipt_hash",
        ):
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
                    "schema_version": "oe_ppur_v3_amendment_publication_receipt_v1",
                    "amendment_path": path.as_posix(),
                    "amendment_sha256": self.amendment_sha256,
                    "source_contract_hash": self.source_contract_hash,
                    "protocol_hash": self.protocol_hash,
                    "lifecycle_source_seal_sha256": (
                        self.lifecycle_source_seal_sha256
                    ),
                    "lifecycle_source_seal_receipt_hash": (
                        self.lifecycle_source_seal_receipt_hash
                    ),
                    "published_no_overwrite": True,
                    "authorization_consumed": False,
                    "target_labels_opened": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_amendment_publication_receipt_v1",
            "amendment_path": self.amendment_path.as_posix(),
            "amendment_sha256": self.amendment_sha256,
            "source_contract_hash": self.source_contract_hash,
            "protocol_hash": self.protocol_hash,
            "lifecycle_source_seal_sha256": self.lifecycle_source_seal_sha256,
            "lifecycle_source_seal_receipt_hash": (
                self.lifecycle_source_seal_receipt_hash
            ),
            "published_no_overwrite": True,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "receipt_hash": self.receipt_hash,
        }


def publish_authorization_amendment(
    paths: CanonicalPreparationPaths,
    *,
    source_surface: SourceTrainingSurface,
    protocol_hash: str,
    lifecycle_source_seal: LifecycleSourceSealReceipt,
) -> AmendmentPublicationReceipt:
    """Issue direct input #7 once without claiming the execution lease."""

    if not isinstance(paths, CanonicalPreparationPaths):
        raise ProtocolError("OE-PPUR v3 amendment paths are untyped.")
    if not isinstance(source_surface, SourceTrainingSurface):
        raise ProtocolError("OE-PPUR v3 amendment source surface is untyped.")
    protocol = require_sha256(protocol_hash, "protocol hash")
    lifecycle = validate_lifecycle_source_seal(lifecycle_source_seal)
    assert_unissued_authorization_topology(paths)

    receipt = source_surface.receipt
    live_seal = build_source_seal(paths.repository_root)
    validate_live_producer_seal_binding(
        configured_sha256=receipt.contract.producer_source_seal_sha256,
        parsed_sha256=receipt.contract.producer_source_seal_sha256,
        source_seal=live_seal,
    )
    parent_digest, _ = hash_unique_regular_file(
        paths.input_bindings[5].path,
        role="parent ledger",
    )
    if parent_digest != EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256:
        raise ProtocolError("OE-PPUR v3 parent ledger drifted before amendment.")

    raw = authorization_amendment_bytes(
        source_contract_hash=receipt.receipt_hash,
        protocol_hash=protocol,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
    )
    digest = hashlib.sha256(raw).hexdigest()
    try:
        os.mkdir(paths.amendment_root, 0o700)
        fsync_directory(paths.amendment_root.parent)
        write_bytes_exclusive(
            paths.amendment_path,
            raw,
            role="authorization amendment",
        )
        fsync_directory(paths.amendment_root)
    except BaseException as exc:
        # A partially created authorization root is intentionally left in
        # place.  Its presence blocks any automatic second issuance.
        raise ProtocolError(
            "OE-PPUR v3 amendment publication failed closed."
        ) from exc
    observed, _ = hash_unique_regular_file(
        paths.amendment_path,
        role="authorization amendment",
    )
    if observed != digest:
        raise ProtocolError("OE-PPUR v3 amendment changed after publication.")
    return AmendmentPublicationReceipt(
        amendment_path=paths.amendment_path,
        amendment_sha256=digest,
        source_contract_hash=receipt.receipt_hash,
        protocol_hash=protocol,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
        lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
    )


def validate_existing_authorization_amendment(
    paths: CanonicalPreparationPaths,
    *,
    source_surface: SourceTrainingSurface,
    protocol_hash: str,
    lifecycle_source_seal: LifecycleSourceSealReceipt,
) -> AmendmentPublicationReceipt:
    """Validate an already issued amendment without rewriting its bytes."""

    if not isinstance(paths, CanonicalPreparationPaths):
        raise ProtocolError("OE-PPUR v3 recovery paths are untyped.")
    if type(source_surface) is not SourceTrainingSurface:
        raise ProtocolError("OE-PPUR v3 recovery source surface is untyped.")
    protocol = require_sha256(protocol_hash, "protocol hash")
    lifecycle = validate_lifecycle_source_seal(lifecycle_source_seal)
    for role, candidate in (
        ("scratch", paths.scratch_root),
        ("lease", paths.lease_root),
    ):
        if candidate.exists() or candidate.is_symlink():
            raise ProtocolError(
                f"OE-PPUR v3 prior {role} state forbids amendment recovery."
            )
    if (paths.artifact_root.exists() or paths.artifact_root.is_symlink()) and not (
        paths.artifact_root.is_dir()
        and not paths.artifact_root.is_symlink()
        and is_exact_workspace_launch_envelope(paths.artifact_root)
    ):
        raise ProtocolError("OE-PPUR v3 recovery output topology drifted.")
    if (
        not paths.amendment_root.is_dir()
        or paths.amendment_root.is_symlink()
        or not paths.amendment_path.is_file()
        or paths.amendment_path.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v3 recovery amendment is absent or unsafe.")

    source_receipt = source_surface.receipt
    source_seal = build_source_seal(paths.repository_root)
    validate_live_producer_seal_binding(
        configured_sha256=source_receipt.contract.producer_source_seal_sha256,
        parsed_sha256=source_receipt.contract.producer_source_seal_sha256,
        source_seal=source_seal,
    )
    parent_digest, _ = hash_unique_regular_file(
        paths.input_bindings[5].path,
        role="parent ledger",
    )
    if parent_digest != EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256:
        raise ProtocolError("OE-PPUR v3 parent ledger drifted during recovery.")
    expected = authorization_amendment_bytes(
        source_contract_hash=source_receipt.receipt_hash,
        protocol_hash=protocol,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
    )
    observed, digest = read_bounded_unique_file(
        paths.amendment_path,
        maximum_bytes=1024 * 1024,
        role="authorization amendment",
    )
    if observed != expected or hashlib.sha256(observed).hexdigest() != digest:
        raise ProtocolError("OE-PPUR v3 recovery amendment bytes drifted.")
    return AmendmentPublicationReceipt(
        amendment_path=paths.amendment_path,
        amendment_sha256=digest,
        source_contract_hash=source_receipt.receipt_hash,
        protocol_hash=protocol,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
        lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
    )


def assert_unissued_authorization_topology(
    paths: CanonicalPreparationPaths,
) -> None:
    for role, candidate in (
        ("output", paths.artifact_root),
        ("scratch", paths.scratch_root),
        ("lease", paths.lease_root),
        ("amendment root", paths.amendment_root),
    ):
        if candidate.exists() or candidate.is_symlink():
            raise ProtocolError(
                f"OE-PPUR v3 prior {role} state forbids amendment issuance."
            )
    if not paths.amendment_root.parent.is_dir() or paths.amendment_root.parent.is_symlink():
        raise ProtocolError("OE-PPUR v3 amendment parent is unsafe.")


__all__ = (
    "AmendmentPublicationReceipt",
    "assert_unissued_authorization_topology",
    "publish_authorization_amendment",
    "validate_existing_authorization_amendment",
)
