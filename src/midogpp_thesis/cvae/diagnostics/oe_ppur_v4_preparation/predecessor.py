"""Read-only preservation witness for the stranded OE-PPUR v3 authority.

The v3 amendment is historical evidence only.  It is never inserted into the
v4 input inventory and never grants v4 launch authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import json
import os
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import bytes_sha256, payload_sha256, require_sha256


PRESERVED_V3_AMENDMENT_SHA256 = (
    "56269322ead01ef683c985d8f295b0369fb35ddef04d12115704f1df18a0c425"
)
PRESERVED_V3_STATUS = "AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
PRESERVED_V3_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router.v3"
)
_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PredecessorPreservationWitness:
    amendment_path: Path
    amendment_sha256: str
    output_root: Path
    lease_path: Path
    scratch_root: Path
    status: str
    amendment_issued: bool
    envelope_rendered: bool
    authorization_claimed: bool
    experiment_launched: bool
    authority_inherited_by_v4: bool
    direct_input_to_v4: bool
    _factory_token: InitVar[object | None] = None
    witness_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise ProtocolError("OE-PPUR v3 preservation witness bypassed capture.")
        for role in ("amendment_path", "output_root", "lease_path", "scratch_root"):
            value = getattr(self, role)
            if (
                not isinstance(value, Path)
                or not value.is_absolute()
                or value != Path(os.path.normpath(value.as_posix()))
                or ".." in value.parts
            ):
                raise ProtocolError(f"OE-PPUR v4 predecessor {role} drifted.")
        object.__setattr__(
            self,
            "amendment_sha256",
            require_sha256(self.amendment_sha256, "preserved v3 amendment"),
        )
        if (
            self.amendment_sha256 != PRESERVED_V3_AMENDMENT_SHA256
            or self.status != PRESERVED_V3_STATUS
            or self.amendment_issued is not True
            or self.envelope_rendered is not False
            or self.authorization_claimed is not False
            or self.experiment_launched is not False
            or self.authority_inherited_by_v4 is not False
            or self.direct_input_to_v4 is not False
        ):
            raise ProtocolError("OE-PPUR v3 preservation state drifted.")
        object.__setattr__(self, "witness_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_predecessor_preservation_witness_v1",
            "predecessor_experiment_id": PRESERVED_V3_EXPERIMENT_ID,
            "amendment_path": self.amendment_path.as_posix(),
            "amendment_sha256": self.amendment_sha256,
            "output_root": self.output_root.as_posix(),
            "lease_path": self.lease_path.as_posix(),
            "scratch_root": self.scratch_root.as_posix(),
            "status": self.status,
            "amendment_issued": True,
            "envelope_rendered": False,
            "authorization_claimed": False,
            "experiment_launched": False,
            "authority_inherited_by_v4": False,
            "direct_input_to_v4": False,
        }


def capture_predecessor_preservation(
    *,
    amendment_path: Path,
    output_root: Path,
    lease_path: Path,
    scratch_root: Path,
) -> PredecessorPreservationWitness:
    """Verify exact v3 bytes and absence of every operational surface."""

    for role, path in (
        ("v3 output", output_root),
        ("v3 lease", lease_path),
        ("v3 scratch", scratch_root),
    ):
        if os.path.lexists(path):
            raise ProtocolError(
                f"OE-PPUR v4 cannot preserve {role} as unrendered/unclaimed/no-run."
            )
    if amendment_path.is_symlink() or not amendment_path.is_file():
        raise ProtocolError("OE-PPUR v3 preserved amendment is absent or unsafe.")
    try:
        before = amendment_path.stat()
        raw = amendment_path.read_bytes()
        after = amendment_path.stat()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v3 preserved amendment is unreadable.") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or bytes_sha256(raw) != PRESERVED_V3_AMENDMENT_SHA256
        or not isinstance(payload, dict)
        or payload.get("schema_version")
        != "oe_ppur_v3_single_use_authorization_amendment_v1"
        or payload.get("consumer_experiment_id") != PRESERVED_V3_EXPERIMENT_ID
        or payload.get("status") != PRESERVED_V3_STATUS
        or payload.get("authorization_exhausted") is not False
        or payload.get("execution_authorized") is not True
        or payload.get("authorized_run_count") != 1
    ):
        raise ProtocolError("OE-PPUR v3 preserved amendment bytes drifted.")
    return PredecessorPreservationWitness(
        amendment_path=amendment_path,
        amendment_sha256=PRESERVED_V3_AMENDMENT_SHA256,
        output_root=output_root,
        lease_path=lease_path,
        scratch_root=scratch_root,
        status=PRESERVED_V3_STATUS,
        amendment_issued=True,
        envelope_rendered=False,
        authorization_claimed=False,
        experiment_launched=False,
        authority_inherited_by_v4=False,
        direct_input_to_v4=False,
        _factory_token=_FACTORY_TOKEN,
    )


__all__ = (
    "PRESERVED_V3_AMENDMENT_SHA256",
    "PRESERVED_V3_EXPERIMENT_ID",
    "PRESERVED_V3_STATUS",
    "PredecessorPreservationWitness",
    "capture_predecessor_preservation",
)
